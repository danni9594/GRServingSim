"""Single-device FCFS GR service, preserving arrival-time cache observations.

One request is in flight, as assumed by the paper's QPS = 1 / mean service time.
Candidate parallelism is within a request. This is not continuous batching or a
multi-device network model. Requests arriving during service see only committed
KV; LRU-K reference histories still update at their actual arrival times.
"""

from collections import Counter
from dataclasses import asdict
import math
import statistics

from .cache import AccessHistory, UserKVCacheManager
from .model import GRPerformanceModel


class GRSimulator:
    def __init__(self, config, executor=None):
        self.config = config
        self.executor = executor
        self.history = AccessHistory(config.cache_k)
        self.caches = [UserKVCacheManager(tier.capacity_bytes,
                                        config.model.kv_bytes_per_token,
                                        config.block_tokens, self.history)
                       for tier in config.tiers]
        self.performance = GRPerformanceModel(config)
        self._has_run = False

    def _source(self, req, now):
        return next((index for index, cache in enumerate(self.caches)
                     if cache.lookup(req, now) is not None), None)

    def run(self, requests, warmup_requests=0):
        if self._has_run:
            raise RuntimeError("create a new simulator for each run")
        if not requests:
            raise ValueError("GR workload is empty")
        if len({req.request_id for req in requests}) != len(requests):
            raise ValueError("request_id must be unique")
        if not isinstance(warmup_requests, int) or not 0 <= warmup_requests < len(requests):
            raise ValueError("warmup_requests must leave at least one measured request")
        self._has_run = True
        requests = sorted(requests, key=lambda req: req.arrival_time_ns)
        arrival_index = 0
        arrival_hits = {}
        records, stages = [], []
        clock = 0

        def observe_until(time, inclusive):
            nonlocal arrival_index
            while arrival_index < len(requests):
                arriving = requests[arrival_index]
                arrival = arriving.arrival_time_ns
                if arrival > time or (arrival == time and not inclusive):
                    break
                arrival_hits[arriving.request_id] = self._source(arriving, arrival) is not None
                self.history.record(arriving.user_key, arrival)
                arrival_index += 1

        for request_index, req in enumerate(requests):
            start = max(clock, req.arrival_time_ns)
            observe_until(start, inclusive=True)
            source = self._source(req, start)
            plans = [cache.plan(req, start) for cache in self.caches]
            cost = self.performance.evaluate(req, source, plans)
            execution = {}
            if self.executor is not None:
                cost, execution = self.executor.service(req, source, plans, cost, start, observe_until)
            if not math.isfinite(cost.service_ns) or cost.service_ns <= 0:
                raise ValueError("service time must be finite and positive")
            finish = start + cost.service_ns
            # Completion wins ties with arrival. No future KV is exposed early.
            observe_until(finish, inclusive=False)
            for cache, plan in zip(self.caches, plans):
                cache.commit(req, plan, finish)
            path = ("hit" if source is not None else
                    "miss_write" if any(plan.write_bytes for plan in plans)
                    else "miss_no_write")
            records.append({
                "request_id": req.request_id, "user_id": req.user_id,
                "is_warmup": request_index < warmup_requests,
                "model_version": req.model_version,
                "history_version": req.history_version,
                "next_history_version": req.next_history_version,
                "history_tokens": req.history_tokens,
                "incremental_tokens": req.incremental_tokens,
                "candidate_count": req.candidate_count,
                "paradigm": self.config.model.paradigm,
                "arrival_time_ns": req.arrival_time_ns,
                "start_time_ns": start, "finish_time_ns": finish,
                "queue_ns": start - req.arrival_time_ns,
                "latency_ns": finish - req.arrival_time_ns,
                "service_ns": cost.service_ns,
                "path": path, "source_tier": cost.source_tier,
                "arrival_cache_hit": arrival_hits[req.request_id],
                "decode_steps": len(cost.steps),
                "decode_queries": sum(step.query_count for step in cost.steps),
                "compute_ns": cost.compute_ns, "read_ns": cost.read_ns,
                "write_ns": cost.write_ns,
                "historical_flops": cost.historical_flops,
                "incremental_flops": cost.incremental_flops,
                "decode_flops": cost.decode_flops,
                "saved_historical_flops": cost.saved_historical_flops,
                "read_bytes": cost.read_bytes,
                "reads_by_tier": cost.reads_by_tier,
                "write_bytes": sum(cost.writes_by_tier.values()),
                "physical_write_bytes": sum(cost.physical_writes_by_tier.values()),
                "writes_by_tier": cost.writes_by_tier,
                "physical_writes_by_tier": cost.physical_writes_by_tier,
                "retained_tiers": [tier.name for tier, plan in zip(self.config.tiers, plans)
                                   if plan.accepted],
                "admission_reasons": {tier.name: plan.reason
                                      for tier, plan in zip(self.config.tiers, plans)},
                "evicted_users": {tier.name: [list(key) for key in plan.victims]
                                  for tier, plan in zip(self.config.tiers, plans)},
            })
            stages.append({"request_id": req.request_id, **cost.to_dict(), **execution})
            clock = finish
        summary = self._summary(records[warmup_requests:])
        summary["warmup_requests"] = warmup_requests
        summary["total_requests"] = len(records)
        summary["cache_counters_include_warmup"] = True
        if self.executor is not None:
            summary["backend"] = "astra_sim"
            summary["astra"] = self.executor.metadata
            summary.pop("roofline_qps")
        return records, stages, summary

    def _summary(self, records):
        count = len(records)
        counts = Counter(row["path"] for row in records)
        mean_service = statistics.fmean(row["service_ns"] for row in records)
        busy_ns = sum(row["service_ns"] for row in records)
        elapsed_ns = records[-1]["finish_time_ns"] - records[0]["arrival_time_ns"]
        achieved_qps = count * 1e9 / elapsed_ns
        # Lifetime uses workload arrivals, never blindly the saturation QPS.
        rate = self.config.workload_qps
        rate_source = "configured" if rate is not None else "trace_arrival_span"
        arrival_span = records[-1]["arrival_time_ns"] - records[0]["arrival_time_ns"]
        if rate is None and count > 1 and arrival_span > 0:
            rate = (count - 1) * 1e9 / arrival_span
        if rate is None:
            rate_source = "unknown_single_arrival_time"
        paths = {}
        for path in ("hit", "miss_write", "miss_no_write"):
            subset = [row for row in records if row["path"] == path]
            paths[path] = {
                "count": len(subset), "probability": len(subset) / count,
                "mean_service_ns": statistics.fmean(row["service_ns"] for row in subset) if subset else 0,
                "mean_history_tokens": statistics.fmean(row["history_tokens"] for row in subset) if subset else 0,
                "mean_write_bytes": statistics.fmean(row["write_bytes"] for row in subset) if subset else 0,
            }
        tiers = {}
        for tier, cache in zip(self.config.tiers, self.caches):
            total = sum(row["writes_by_tier"][tier.name] for row in records)
            physical = sum(row["physical_writes_by_tier"][tier.name] for row in records)
            lifetime = None
            if tier.endurance_cycles and physical and rate:
                lifetime = tier.capacity_bytes * tier.endurance_cycles / (rate * physical / count)
            tiers[tier.name] = {
                "capacity_bytes": tier.capacity_bytes,
                "resident_bytes": cache.used_bytes, "peak_resident_bytes": cache.peak_bytes,
                "resident_users": len(cache.entries), "evictions": cache.evictions,
                "invalidations": cache.invalidations,
                "write_bytes": total, "physical_write_bytes": physical,
                "read_bytes": sum(row["reads_by_tier"][tier.name] for row in records),
                "lifetime_seconds": lifetime,
                "lifetime_years": lifetime / (365.25 * 86400) if lifetime is not None else None,
                "lifetime_status": ("not_flash" if not tier.endurance_cycles else
                                    "no_writes" if not physical else
                                    "unknown_arrival_rate" if not rate else "estimated"),
            }
        misses = count - counts["hit"]
        return {
            "backend": "gr_analytical", "config": asdict(self.config),
            "requests": count, "paths": paths,
            "hit_rate": counts["hit"] / count,
            "arrival_hit_rate": sum(row["arrival_cache_hit"] for row in records) / count,
            "admission_given_miss": counts["miss_write"] / misses if misses else None,
            "mean_service_ns": mean_service,
            "path_weighted_mean_service_ns": sum(
                path["probability"] * path["mean_service_ns"] for path in paths.values()),
            "mean_latency_ns": statistics.fmean(row["latency_ns"] for row in records),
            "mean_queue_ns": statistics.fmean(row["queue_ns"] for row in records),
            "roofline_qps": 1e9 / mean_service, "achieved_qps": achieved_qps,
            "service_capacity_qps": 1e9 / mean_service,
            "busy_fraction": busy_ns / elapsed_ns,
            "workload_qps": rate, "workload_qps_source": rate_source,
            "mean_write_bytes": statistics.fmean(row["write_bytes"] for row in records),
            "total_saved_historical_flops": sum(row["saved_historical_flops"] for row in records),
            "tiers": tiers,
        }
