"""FLOP accounting and Eqs. (1)/(6), with explicit decode dependencies.

The manuscript does not publish operator dimensions. Coefficients below are
configuration inputs, not inferred HSTU/OpenOneRec measurements. Attention
counts causal history pairs and candidate-to-history pairs, never a causal
500-token LLM completion for HSTU.
"""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class DecodeStep:
    step: int
    query_count: int
    context_tokens: int
    logits_scored: int
    flops: float
    history_read_bytes: int
    history_read_tier: str = "none"


@dataclass(frozen=True)
class ServiceCost:
    service_ns: float
    compute_ns: float
    read_ns: float
    write_ns: float
    historical_flops: float
    incremental_flops: float
    decode_flops: float
    saved_historical_flops: float
    read_bytes: int
    reads_by_tier: dict
    writes_by_tier: dict
    physical_writes_by_tier: dict
    source_tier: str
    steps: tuple
    stage_service_ns: tuple

    def to_dict(self):
        return asdict(self)


class GRPerformanceModel:
    def __init__(self, config):
        self.config = config
        self.model = config.model
        self.flops_per_ns = config.peak_flops * config.compute_efficiency / 1e9

    def historical_flops(self, tokens):
        return (tokens * self.model.linear_flops_per_token
                + tokens * (tokens + 1) / 2 * self.model.attention_flops_per_pair)

    def decode_steps(self, req, hit):
        model = self.model
        context = req.updated_tokens
        read_bytes = req.history_tokens * model.kv_bytes_per_token if hit else 0
        if model.paradigm == "hstu":
            if req.decode_widths:
                raise ValueError("HSTU uses candidate_count, not decode_widths")
            count = req.candidate_count
            flops = count * (model.linear_flops_per_token
                             + context * model.attention_flops_per_pair
                             + model.score_flops_per_candidate)
            return (DecodeStep(1, count, context, count, flops, read_bytes),)

        steps = []
        width = 1
        possible_items = 1
        for size in model.semantic_vocab_sizes:
            possible_items *= size
        if req.candidate_count > possible_items:
            raise ValueError("candidate_count exceeds the three-codebook item space")
        for index, vocab_size in enumerate(model.semantic_vocab_sizes):
            query_count = req.decode_widths[index] if req.decode_widths else width
            logits = query_count * vocab_size
            flops = (query_count * (model.linear_flops_per_token
                                   + (context + index) * model.attention_flops_per_pair)
                     + logits * model.semantic_head_flops_per_logit)
            steps.append(DecodeStep(index + 1, query_count, context + index,
                                    logits, flops, read_bytes))
            width = min(req.candidate_count, logits)
        return tuple(steps)

    @staticmethod
    def transfer_ns(size, tier, write=False):
        if not size:
            return 0.0
        bandwidth = (tier.write_bandwidth_bytes_per_s if write
                     else tier.read_bandwidth_bytes_per_s)
        if tier.link_bandwidth_bytes_per_s:
            bandwidth = min(bandwidth, tier.link_bandwidth_bytes_per_s)
        startup = tier.write_latency_ns if write else tier.read_latency_ns
        return size / bandwidth * 1e9 + startup

    def evaluate(self, req, source_index, plans):
        hit = source_index is not None
        model = self.model
        full_history_flops = self.historical_flops(req.history_tokens)
        hist_flops = 0 if hit else full_history_flops
        inc = req.incremental_tokens
        inc_flops = (inc * model.linear_flops_per_token
                     + (inc * req.history_tokens + inc * (inc + 1) / 2)
                     * model.attention_flops_per_pair)
        steps = self.decode_steps(req, hit)
        decode_flops = sum(step.flops for step in steps)
        compute_ns = (hist_flops + inc_flops + decode_flops) / self.flops_per_ns
        reads = sum(step.history_read_bytes for step in steps)
        source = self.config.tiers[source_index] if hit else None
        # If a lower-tier hit is promoted, later semantic steps can read its
        # private staged copy from the faster tier before publishing it to other
        # requests. HBF direct reads still stream HBF once per semantic pass.
        promoted_index = (next((index for index in range(source_index)
                               if plans[index].accepted), source_index) if hit else None)
        read_times = []
        reads_by_tier = {tier.name: 0 for tier in self.config.tiers}
        located_steps = []
        for index, step in enumerate(steps):
            read_tier = (source if index == 0 else self.config.tiers[promoted_index]) if hit else None
            read_times.append(self.transfer_ns(step.history_read_bytes, read_tier) if hit else 0)
            if hit:
                reads_by_tier[read_tier.name] += step.history_read_bytes
            located_steps.append(replace(step, history_read_tier=read_tier.name if hit else "none"))
        steps = tuple(located_steps)
        writes = {tier.name: plan.write_bytes
                  for tier, plan in zip(self.config.tiers, plans)}
        physical = {tier.name: writes[tier.name] * tier.write_amplification
                    for tier in self.config.tiers}
        write_ns = max(self.transfer_ns(physical[tier.name], tier, write=True)
                       for tier in self.config.tiers)
        # Incremental prefill and the first scoring pass share the history read.
        # Repeated semantic decoding streams history once per step, shared across
        # beams. New KV and semantic prefixes are in the active HBM workspace.
        stage_times = []
        for index, step in enumerate(steps):
            stage_flops = step.flops + (hist_flops + inc_flops if index == 0 else 0)
            stage_times.append(max(stage_flops / self.flops_per_ns, read_times[index],
                                   write_ns if index == 0 else 0))
        if self.config.latency_mode == "paper_roofline":
            service_ns = max(compute_ns, sum(read_times), write_ns)
        else:
            # Sum step rooflines to prohibit overlap across semantic dependencies.
            service_ns = sum(stage_times)
        return ServiceCost(service_ns, compute_ns, sum(read_times), write_ns,
                           hist_flops, inc_flops, decode_flops,
                           full_history_flops if hit else 0, reads, reads_by_tier, writes, physical,
                           source.name if source is not None else "none", steps,
                           tuple(stage_times))
