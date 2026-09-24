"""GR operator DAGs, serialized as Chakra execution traces (one NPU).

Compute coefficients describe the whole model and are divided over layers.
Each memory transfer is an ASTRA memory node, not folded into compute runtime.
This fork interprets Chakra duration_micros as nanoseconds, like LLMConverter.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path


LOCATIONS = {"local": 1, "remote": 2, "cxl": 3, "storage": 4}
DEFAULT_LOCATIONS = {"HBM": "local", "CPU": "remote", "CXL": "cxl", "HBF": "storage"}


def tier_location(tier):
    location = tier.astra_location or DEFAULT_LOCATIONS.get(tier.name.upper())
    if location is None:
        raise ValueError(f"tier {tier.name} requires astra_location")
    return location


@dataclass(frozen=True)
class GROperator:
    id: int
    name: str
    kind: str
    dependencies: tuple
    duration_ns: int = 0
    flops: float = 0
    bytes: int = 0
    location: str = "local"


@dataclass
class GRTrace:
    phase: str
    operators: list

    def to_dict(self):
        return {"phase": self.phase, "operators": [asdict(op) for op in self.operators]}


class GraphBuilder:
    def __init__(self, phase, config):
        self.trace = GRTrace(phase, [])
        self.flops_per_ns = config.peak_flops * config.compute_efficiency / 1e9

    def add(self, name, kind, deps=(), *, flops=0, size=0, location="local", duration=None):
        node_id = len(self.trace.operators)
        runtime = max(1, math.ceil(flops / self.flops_per_ns)) if kind == "compute" else 0
        if duration is not None:
            runtime = duration
        self.trace.operators.append(GROperator(
            node_id, name, kind, tuple(deps), runtime, flops, math.ceil(size), location))
        return node_id

    def memory(self, name, size, location, deps=(), write=False):
        if not size:
            return list(deps)
        return [self.add(name, "store" if write else "load", deps, size=size, location=location)]


def idle_trace(config, duration):
    graph = GraphBuilder("idle", config)
    graph.add("idle", "compute", duration=duration)
    return graph.trace


def recall_trace(config, req, location):
    graph = GraphBuilder("recall", config)
    size = req.history_tokens * config.model.kv_bytes_per_token
    load = graph.memory("recall_history", size, location)
    graph.memory("stage_history", size, "local", load, write=True)
    return graph.trace


def operator_trace(config, req, stage, source_index, plans):
    graph = GraphBuilder(stage.name, config)
    model = config.model
    source = config.tiers[source_index] if source_index is not None else None
    # Remote/CPU histories are recalled once; HBF/CXL histories stream directly.
    direct = source is not None and tier_location(source) in ("storage", "cxl")
    predecessor = []
    for layer in range(model.num_layers):
        prefix = f"layer{layer}"
        weight_bytes = (model.weight_bytes * (layer + 1) // model.num_layers
                        - model.weight_bytes * layer // model.num_layers)
        weight = graph.memory(f"{prefix}.weights", weight_bytes,
                              config.astra.weight_location, predecessor)
        projection = graph.add(f"{prefix}.projection", "compute", weight,
                               flops=stage.linear_flops / model.num_layers * model.projection_fraction)
        # KV bytes are divided without losing integer remainder across layers.
        kv = (model.kv_bytes_per_token * (layer + 1) // model.num_layers
              - model.kv_bytes_per_token * layer // model.num_layers)
        new_kv = graph.memory(f"{prefix}.working_kv_append", stage.queries * kv,
                             "local", [projection], write=True)
        history = req.history_tokens if direct and stage.name != "history_prefill" else 0
        history_reads = graph.memory(f"{prefix}.user_history", history * kv,
                                     tier_location(source) if direct else "local", predecessor)
        local_tokens = stage.context_tokens - history
        if stage.name.endswith("prefill"):
            local_tokens += stage.queries
        else:
            # Semantic prefixes are private per beam; history reads are shared.
            local_tokens += stage.semantic_prefix_tokens
        local_reads = graph.memory(f"{prefix}.working_kv_read", local_tokens * kv,
                                   "local", new_kv)
        attention = graph.add(f"{prefix}.attention", "compute",
                              sorted(set([projection] + new_kv + history_reads + local_reads)),
                              flops=stage.attention_flops / model.num_layers)
        output = graph.add(f"{prefix}.output_ffn", "compute", [attention],
                           flops=stage.linear_flops / model.num_layers * (1 - model.projection_fraction))
        predecessor = [output]
    if stage.head_flops:
        graph.add("candidate_scores" if model.paradigm == "hstu" else "semantic_logits_topk",
                  "compute", predecessor, flops=stage.head_flops)
    if stage.decode_step == 1:
        # Prefill is complete. Writeback can overlap scoring; the stage barrier
        # waits for both branches. User-visible publication is after request end.
        for tier, plan in zip(config.tiers, plans):
            if not plan.write_bytes:
                continue
            copy = graph.memory(f"publish.{tier.name}.source", plan.write_bytes, "local")
            graph.memory(f"publish.{tier.name}.destination",
                         plan.write_bytes * tier.write_amplification,
                         tier_location(tier), copy, write=True)
    return graph.trace


@lru_cache(maxsize=1)
def chakra_schema():
    # Use the exact schema pinned by this repository's ASTRA submodule. No
    # dependency on HTA/PyTorch or an unrelated globally installed Chakra fork.
    path = (Path(__file__).resolve().parents[2] / "astra-sim/extern/graph_frontend/"
            "chakra/schema/protobuf/et_def_pb2.py")
    if not path.is_file():
        raise RuntimeError("Chakra schema is missing; run scripts/compile-gr.sh first")
    spec = importlib.util.spec_from_file_location("gr_chakra_et_def_pb2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_graph(trace, prefix, save_trace_text=False):
    schema = chakra_schema()
    path = Path(str(prefix) + ".0.et")
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(stream, message):
        payload = message.SerializeToString()
        size = len(payload)
        while size > 127:
            stream.write(bytes([(size & 127) | 128]))
            size >>= 7
        stream.write(bytes([size]))
        stream.write(payload)

    kinds = {"compute": schema.COMP_NODE, "load": schema.MEM_LOAD_NODE,
             "store": schema.MEM_STORE_NODE}
    with path.open("wb") as stream:
        encode(stream, schema.GlobalMetadata(version="0.0.4"))
        for op in trace.operators:
            node = schema.Node(id=op.id, name=op.name, type=kinds[op.kind],
                               duration_micros=op.duration_ns, data_deps=op.dependencies)
            node.attr.add(name="is_cpu_op", bool_val=False)
            node.attr.add(name="num_ops", uint64_val=math.ceil(op.flops))
            node.attr.add(name="tensor_size", uint64_val=op.bytes)
            if op.kind != "compute":
                node.attr.add(name="tensor_loc", uint32_val=LOCATIONS[op.location])
                node.attr.add(name="tensor_device", uint32_val=0)
            encode(stream, node)
    if save_trace_text:
        path.with_suffix(".json").write_text(json.dumps(trace.to_dict(), indent=2) + "\n")
    return str(prefix)
