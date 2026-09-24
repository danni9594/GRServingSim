"""ASTRA subprocess/Chakra adapter for the GR stage scheduler."""

from collections import defaultdict, deque
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time

from ..core.controller import Controller
from ..core.graph_generator import generate_graph
from .scheduler import GRScheduler
from .trace import LOCATIONS, idle_trace, tier_location


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"


def memory_description(read_bw, write_bw, read_latency=0, write_latency=0):
    return {"memory-type": "PER_NODE_MEMORY_EXPANSION", "num-devices": 1,
            "mem-bw": max(1, int(read_bw / 1e9)), "mem-latency": 0,
            "read-bw": read_bw / 1e9, "write-bw": write_bw / 1e9,
            "read-latency": read_latency, "write-latency": write_latency}


def build_configs(config, directory):
    memory = {"local_mem": memory_description(
        config.astra.workspace_read_bandwidth_bytes_per_s,
        config.astra.workspace_write_bandwidth_bytes_per_s)}
    used = set()
    for tier in config.tiers:
        location = tier_location(tier)
        if location in used:
            raise ValueError("ASTRA GR supports one cache tier per memory location")
        used.add(location)
        read_bw, write_bw = tier.read_bandwidth_bytes_per_s, tier.write_bandwidth_bytes_per_s
        if tier.link_bandwidth_bytes_per_s:
            read_bw = min(read_bw, tier.link_bandwidth_bytes_per_s)
            write_bw = min(write_bw, tier.link_bandwidth_bytes_per_s)
        if location == "local" and (
            read_bw != config.astra.workspace_read_bandwidth_bytes_per_s or
            write_bw != config.astra.workspace_write_bandwidth_bytes_per_s
        ):
            raise ValueError("HBM cache and workspace share a queue and must use identical bandwidths")
        memory[f"{location}_mem"] = memory_description(
            read_bw, write_bw, tier.read_latency_ns, tier.write_latency_ns)
    if config.model.weight_bytes and f"{config.astra.weight_location}_mem" not in memory:
        raise ValueError("weight_location has no configured memory endpoint")
    system = {"scheduling-policy": "LIFO", "endpoint-delay": 10,
              "active-chunks-per-dimension": 1, "preferred-dataset-splits": 1,
              "all-reduce-implementation": ["ring"], "all-gather-implementation": ["ring"],
              "reduce-scatter-implementation": ["ring"], "all-to-all-implementation": ["ring"],
              "collective-optimization": "localBWAware", "boost-mode": 0,
              "local-mem-bw": config.astra.workspace_read_bandwidth_bytes_per_s / 1e9}
    for name, data in (("memory", memory), ("system", system)):
        (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    (directory / "network.yml").write_text(
        "topology: [FullyConnected]\nnpus_count: [1]\nbandwidth: [112.0]\nlatency: [0.0]\n")
    return memory


class AstraExecutor:
    """Keep one ASTRA process alive; every reported stage cycle is authoritative."""

    def __init__(self, config, binary=None, keep_inputs=False, save_trace_text=False, timeout_s=60):
        self.config = config
        self.binary = Path(binary or DEFAULT_BINARY).resolve()
        if not self.binary.is_file():
            raise ValueError(f"ASTRA binary is missing: {self.binary}; run scripts/compile-gr.sh")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("astra-timeout must be positive")
        self.timeout_s = timeout_s
        self.keep_inputs = keep_inputs or save_trace_text
        self.save_trace_text = save_trace_text
        root = REPO_ROOT / "astra-sim/inputs/runs"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="gr_", dir=root))
        self.memory = build_configs(config, self.directory)
        self.scheduler = GRScheduler(config)
        self.controller = Controller(1)
        self.process = None
        self.lines = queue.Queue()
        self.tail = deque(maxlen=12)
        self.sequence = 0
        self.clock = 0
        self.origin = 0
        self.closed = False
        self.log_stream = None
        self.reader = None
        self.metadata = {"backend": "astra_sim", "num_npus": 1,
                         "binary_sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
                         "inputs_root": str(self.directory) if self.keep_inputs else None,
                         "memory_queue_policy": "one shared FIFO read/write queue per tier"}

    def _graph(self, trace):
        name = f"stage{self.sequence:06d}_{trace.phase}"
        self.sequence += 1
        return generate_graph(None, "gr", 1, inputs_root=str(self.directory),
                              workload_name=name, save_trace_text=self.save_trace_text, trace=trace)

    def _reader(self):
        try:
            for line in self.process.stdout:
                self.log_stream.write(line)
                self.log_stream.flush()
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def _wait(self):
        deadline = time.monotonic() + self.timeout_s
        report = None
        while True:
            try:
                line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise RuntimeError(f"ASTRA timed out; recent output: {''.join(self.tail)}") from exc
            if line is None:
                raise RuntimeError(f"ASTRA exited before completing a stage: {''.join(self.tail)}")
            self.tail.append(line)
            parsed = self.controller.parse_output(line)
            if parsed is not None:
                report = parsed
            if "Waiting" in line:
                if report is None or report["sys"] != 0:
                    raise RuntimeError("ASTRA prompt has no NPU 0 completion report")
                return report["cycle"]

    def __enter__(self):
        try:
            bootstrap = self._graph(idle_trace(self.config, 1))
            self.log_stream = (self.directory / "astra.log").open("w")
            command = [str(self.binary), f"--workload-configuration={bootstrap}",
                       f"--system-configuration={self.directory / 'system.json'}",
                       f"--network-configuration={self.directory / 'network.yml'}",
                       f"--memory-configuration={self.directory / 'memory.json'}",
                       "--start-npu-ids=0", "--end-npu-ids=0"]
            self.process = subprocess.Popen(command, cwd=self.directory, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1)
            self.reader = threading.Thread(target=self._reader, daemon=True)
            self.reader.start()
            self.origin = self._wait()
            self.metadata["bootstrap_cycles"] = self.origin
            # Verify the binary honors read/write direction. A stale binary can
            # accept the JSON but silently use symmetric mem-bw instead.
            self._probe_directional_memory()
            return self
        except BaseException:
            self.close(failed=True)
            raise

    def _probe_directional_memory(self):
        from .trace import GraphBuilder
        graph = GraphBuilder("memory_probe", self.config)
        expected = 0
        predecessor = []
        size = 1048576
        for location in LOCATIONS:
            if f"{location}_mem" not in self.memory:
                continue
            spec = self.memory[f"{location}_mem"]
            for write in (False, True):
                predecessor = graph.memory(f"probe.{location}.{write}", size, location,
                                           predecessor, write=write)
                prefix = "write" if write else "read"
                expected += math.ceil(spec[f"{prefix}-latency"] + size / spec[f"{prefix}-bw"])
        elapsed = self.execute(graph.trace)
        if elapsed != expected:
            raise RuntimeError(f"ASTRA memory probe returned {elapsed} cycles; expected {expected}. "
                               "Rebuild with scripts/compile-gr.sh (directional memory patches required).")
        # Probe/initialization are outside the simulated workload timeline.
        self.origin += self.clock
        self.clock = 0
        self.metadata["memory_probe_cycles"] = elapsed
        self.metadata["cycle_origin"] = self.origin

    def execute(self, trace):
        if not trace.operators:
            return 0
        prefix = self._graph(trace)
        self.controller.write_flush(self.process, prefix)
        finished = self._wait() - self.origin
        duration = finished - self.clock
        if duration < 0:
            raise RuntimeError("ASTRA clock moved backwards")
        self.clock = finished
        return duration

    def service(self, req, source_index, plans, cost, start, observe_until):
        start = math.ceil(start)
        if start > self.clock:
            self.execute(idle_trace(self.config, start - self.clock))
        if self.clock != start:
            raise RuntimeError("frontend and ASTRA clocks differ")
        details = []
        reads, writes = defaultdict(int), defaultdict(int)
        compute_ns = 0
        read_ns = write_ns = 0
        for stage, trace in self.scheduler.traces(req, source_index, plans):
            before = self.clock
            duration = self.execute(trace)
            observe_until(self.clock, inclusive=False)
            for op in trace.operators:
                compute_ns += op.duration_ns
                if op.kind == "load":
                    reads[op.location] += op.bytes
                    spec = self.memory[f"{op.location}_mem"]
                    read_ns += math.ceil(spec["read-latency"] + op.bytes / spec["read-bw"])
                if op.kind == "store":
                    writes[op.location] += op.bytes
                    spec = self.memory[f"{op.location}_mem"]
                    write_ns += math.ceil(spec["write-latency"] + op.bytes / spec["write-bw"])
            details.append({"phase": trace.phase, "stage": asdict(stage) if stage else None,
                            "start_ns": before, "finish_ns": self.clock, "cycles": duration,
                            "operator_count": len(trace.operators)})
        tier_reads = {tier.name: reads[tier_location(tier)] for tier in self.config.tiers}
        decode_source = "HBM_workspace"
        if source_index is not None and tier_location(self.config.tiers[source_index]) != "remote":
            decode_source = self.config.tiers[source_index].name
        steps = tuple(replace(step, history_read_bytes=req.history_tokens * self.config.model.kv_bytes_per_token,
                              history_read_tier=decode_source) for step in cost.steps)
        cost = replace(cost, service_ns=self.clock - start, compute_ns=compute_ns,
                       read_ns=read_ns, write_ns=write_ns,
                       steps=steps,
                       read_bytes=sum(reads.values()), reads_by_tier=tier_reads,
                       stage_service_ns=tuple(row["cycles"] for row in details))
        return cost, {"backend": "astra_sim", "astra_stages": details,
                      "memory_read_bytes_by_location": dict(reads),
                      "memory_write_bytes_by_location": dict(writes),
                      "read_write_ns_note": "Summed memory-node busy cycles, excluding queue wait; service_ns comes from ASTRA"}

    def close(self, failed=False):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                if not failed:
                    try:
                        self.controller.write_flush(self.process, "exit")
                        self.process.wait(timeout=5)
                    except (BrokenPipeError, subprocess.TimeoutExpired):
                        self.process.kill()
                else:
                    self.process.kill()
                self.process.wait()
            if self.reader is not None:
                self.reader.join(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
        if self.log_stream is not None:
            self.log_stream.close()
        if not self.keep_inputs and not failed:
            shutil.rmtree(self.directory)

    def __exit__(self, exc_type, exc, traceback):
        self.close(failed=exc_type is not None)
