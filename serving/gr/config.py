"""Explicit units and calibration inputs for GR simulation."""

from dataclasses import dataclass, field
import json
import math


def number(name, value, minimum=0, integer=False, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    paradigm: str
    kv_bytes_per_token: int
    linear_flops_per_token: float
    attention_flops_per_pair: float
    score_flops_per_candidate: float = 0
    semantic_vocab_sizes: tuple = (8192, 8192, 8192)
    semantic_head_flops_per_logit: float = 0
    calibration_note: str = "Uncalibrated analytical coefficients"
    num_layers: int = 1
    projection_fraction: float = 0.5
    weight_bytes: int = 0

    def __post_init__(self):
        if self.paradigm not in ("hstu", "openonerec"):
            raise ValueError("model.paradigm must be hstu or openonerec")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model.name must be non-empty")
        number("kv_bytes_per_token", self.kv_bytes_per_token, integer=True, strict=True)
        number("linear_flops_per_token", self.linear_flops_per_token, strict=True)
        number("attention_flops_per_pair", self.attention_flops_per_pair, strict=True)
        number("score_flops_per_candidate", self.score_flops_per_candidate)
        number("num_layers", self.num_layers, integer=True, strict=True)
        number("projection_fraction", self.projection_fraction, strict=True)
        if self.projection_fraction >= 1:
            raise ValueError("projection_fraction must be < 1")
        number("weight_bytes", self.weight_bytes, integer=True)
        number("semantic_head_flops_per_logit", self.semantic_head_flops_per_logit)
        if len(self.semantic_vocab_sizes) != 3:
            raise ValueError("OpenOneRec items have exactly three semantic IDs")
        for size in self.semantic_vocab_sizes:
            number("semantic_vocab_size", size, integer=True, strict=True)
        if self.paradigm == "openonerec" and self.semantic_head_flops_per_logit <= 0:
            raise ValueError("OpenOneRec requires semantic_head_flops_per_logit > 0")


@dataclass(frozen=True)
class TierConfig:
    name: str
    capacity_bytes: int
    read_bandwidth_bytes_per_s: float
    write_bandwidth_bytes_per_s: float
    link_bandwidth_bytes_per_s: float = 0
    read_latency_ns: float = 0
    write_latency_ns: float = 0
    endurance_cycles: int = 0
    write_amplification: float = 1
    astra_location: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tier.name must be non-empty")
        number("capacity_bytes", self.capacity_bytes, integer=True)
        for name in ("read_bandwidth_bytes_per_s", "write_bandwidth_bytes_per_s"):
            number(name, getattr(self, name), strict=True)
        for name in ("link_bandwidth_bytes_per_s", "read_latency_ns", "write_latency_ns"):
            number(name, getattr(self, name))
        number("endurance_cycles", self.endurance_cycles, integer=True)
        number("write_amplification", self.write_amplification, minimum=1)
        if self.astra_location not in ("", "local", "remote", "cxl", "storage"):
            raise ValueError("unknown astra_location")


@dataclass(frozen=True)
class AstraConfig:
    prefill_chunk_tokens: int = 2048
    workspace_capacity_bytes: int = 36000000000
    workspace_read_bandwidth_bytes_per_s: float = 1600000000000
    workspace_write_bandwidth_bytes_per_s: float = 1600000000000
    weight_location: str = "local"

    def __post_init__(self):
        number("prefill_chunk_tokens", self.prefill_chunk_tokens, integer=True, strict=True)
        number("workspace_capacity_bytes", self.workspace_capacity_bytes, integer=True, strict=True)
        number("workspace_read_bandwidth_bytes_per_s", self.workspace_read_bandwidth_bytes_per_s, strict=True)
        number("workspace_write_bandwidth_bytes_per_s", self.workspace_write_bandwidth_bytes_per_s, strict=True)
        if self.weight_location not in ("local", "remote", "cxl", "storage"):
            raise ValueError("unknown weight_location")


@dataclass(frozen=True)
class GRConfig:
    model: ModelConfig
    tiers: tuple
    peak_flops: float
    cache_k: int = 1
    block_tokens: int = 16
    compute_efficiency: float = 1
    latency_mode: str = "paper_roofline"
    workload_qps: float = None
    metadata: dict = field(default_factory=dict)
    astra: AstraConfig = field(default_factory=AstraConfig)

    def __post_init__(self):
        number("peak_flops", self.peak_flops, strict=True)
        number("cache_k", self.cache_k, integer=True, strict=True)
        number("block_tokens", self.block_tokens, integer=True, strict=True)
        number("compute_efficiency", self.compute_efficiency, strict=True)
        if self.compute_efficiency > 1:
            raise ValueError("compute_efficiency must be <= 1")
        if self.latency_mode not in ("paper_roofline", "sequential_roofline"):
            raise ValueError("unknown latency_mode")
        if not self.tiers or len({tier.name for tier in self.tiers}) != len(self.tiers):
            raise ValueError("tiers must be non-empty with unique names")
        if self.workload_qps is not None:
            number("workload_qps", self.workload_qps, strict=True)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["model"] = ModelConfig(**data["model"])
        data["tiers"] = tuple(TierConfig(**tier) for tier in data["tiers"])
        data["astra"] = AstraConfig(**data.get("astra", {}))
        return cls(**data)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
