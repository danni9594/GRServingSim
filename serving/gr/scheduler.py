"""One in-flight GR request, chunked history/increment and dependent decode.

A stage is dispatched only after ASTRA completes its predecessor. Candidate
parallelism is within a request; this is not cross-request continuous batching.
"""

from dataclasses import dataclass

from .model import GRPerformanceModel
from .trace import operator_trace, recall_trace, tier_location


@dataclass(frozen=True)
class GRStage:
    name: str
    queries: int
    context_tokens: int
    linear_flops: float
    attention_flops: float
    head_flops: float = 0
    decode_step: int = 0
    semantic_prefix_tokens: int = 0


class GRScheduler:
    def __init__(self, config):
        self.config = config
        self.performance = GRPerformanceModel(config)

    def traces(self, req, source_index, plans):
        model = self.config.model
        hit = source_index is not None
        steps = self.performance.decode_steps(req, hit)
        # History and maximum live semantic/candidate KV must fit the active
        # workspace. Cache capacities exclude this separately reserved region.
        scratch_tokens = max(step.query_count * step.step for step in steps)
        required = (req.updated_tokens + scratch_tokens) * model.kv_bytes_per_token
        if required > self.config.astra.workspace_capacity_bytes:
            raise ValueError(f"request {req.request_id}: active KV needs {required} bytes, "
                             "exceeding astra.workspace_capacity_bytes")
        if hit and req.history_tokens and tier_location(self.config.tiers[source_index]) == "remote":
            yield None, recall_trace(self.config, req, "remote")
        for name, first, last in (("history_prefill", 0, 0 if hit else req.history_tokens),
                                  ("incremental_prefill", req.history_tokens, req.updated_tokens)):
            for begin in range(first, last, self.config.astra.prefill_chunk_tokens):
                count = min(last - begin, self.config.astra.prefill_chunk_tokens)
                stage = GRStage(name, count, begin, count * model.linear_flops_per_token,
                                (count * begin + count * (count + 1) / 2)
                                * model.attention_flops_per_pair)
                yield stage, operator_trace(self.config, req, stage, source_index, plans)
        for step in steps:
            prefix = (step.step - 1) * step.query_count if model.paradigm == "openonerec" else 0
            head = (step.logits_scored * model.semantic_head_flops_per_logit
                    if model.paradigm == "openonerec" else
                    step.query_count * model.score_flops_per_candidate)
            stage = GRStage("candidate_decode", step.query_count, req.updated_tokens,
                            step.query_count * model.linear_flops_per_token,
                            step.query_count * step.context_tokens * model.attention_flops_per_pair,
                            head, step.step, prefix)
            yield stage, operator_trace(self.config, req, stage, source_index, plans)
