from collections import OrderedDict
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from serving.gr.cache import AccessHistory, UserKVCacheManager
from serving.gr.config import GRConfig, ModelConfig, TierConfig
from serving.gr.model import GRPerformanceModel
from serving.gr.request import GRRequest, load_requests
from serving.gr.simulator import GRSimulator


def request(user="a", time=0, history=4, inc=0, version="0", next_version=None,
            candidates=2, model="v1", rid=None, widths=()):
    if next_version is None:
        next_version = str(int(version) + 1) if inc else version
    return GRRequest(rid or f"{user}-{time}", user, time, history, inc, version,
                     next_version, model, candidates, widths)


def config(k=1, paradigm="hstu", capacity=1024, peak=1e9, read_bw=1e9, write_bw=1e9):
    model = ModelConfig("test", paradigm, 4, 100, 10, 2,
                        (4, 4, 4), 2, "Exact small test coefficients")
    return GRConfig(model, (TierConfig("HBF", capacity, read_bw, write_bw,
                                      endurance_cycles=100000),), peak,
                    cache_k=k, block_tokens=1)


class CacheTests(unittest.TestCase):
    def cache(self, k=1, capacity=8, block=1):
        history = AccessHistory(k)
        return UserKVCacheManager(capacity, 1, block, history)

    def apply(self, cache, req):
        cache.history.record(req.user_key, req.arrival_time_ns)
        plan = cache.plan(req, req.arrival_time_ns)
        cache.commit(req, plan, req.arrival_time_ns)
        return plan

    def test_k1_matches_independent_lru_reference(self):
        cache = self.cache(capacity=3)
        reference = OrderedDict()
        rng = random.Random(17)
        for time in range(300):
            user = str(rng.randrange(8))
            req = request(user, time, history=1)
            self.assertEqual(cache.lookup(req, time) is not None, user in reference)
            if user in reference:
                del reference[user]
            elif len(reference) == 3:
                reference.popitem(last=False)
            reference[user] = True
            self.assertTrue(self.apply(cache, req).accepted)
            self.assertEqual({key[1] for key in cache.entries}, set(reference))

    def test_cold_k2_does_not_admit_even_with_free_space(self):
        cache = self.cache(k=2)
        self.assertEqual(self.apply(cache, request(time=1)).reason, "fewer_than_k_references")
        self.assertEqual(cache.used_bytes, 0)
        self.assertTrue(self.apply(cache, request(time=2)).accepted)

    def test_evicted_reference_history_survives_and_competes(self):
        cache = self.cache(k=2, capacity=4)
        for user, time in [("a", 1), ("a", 2), ("b", 3), ("b", 4)]:
            self.apply(cache, request(user, time))
        self.assertNotIn(("v1", "a"), cache.entries)
        rejected = self.apply(cache, request("a", 5))
        self.assertEqual(rejected.reason, "colder_than_victims")
        self.assertEqual(rejected.write_bytes, 0)
        self.assertTrue(self.apply(cache, request("a", 6)).accepted)
        self.assertIn(("v1", "a"), cache.entries)

    def test_multiple_victim_failure_is_atomic(self):
        cache = self.cache(k=2, capacity=8)
        for user, time in [("a", 1), ("a", 2), ("c", 5), ("b", 8), ("b", 9)]:
            self.apply(cache, request(user, time))
        cache.history.record(("v1", "c"), 10)
        before = dict(cache.entries)
        plan = cache.plan(request("c", 10, history=8), 10)
        self.assertFalse(plan.accepted)
        self.assertEqual(plan.victims, ())
        self.assertEqual(cache.entries, before)
        self.assertEqual(cache.used_bytes, 8)

    def test_multiple_victim_success_evicts_whole_objects(self):
        cache = self.cache(capacity=8)
        self.apply(cache, request("a", 1))
        self.apply(cache, request("b", 2))
        plan = self.apply(cache, request("c", 3, history=7))
        self.assertEqual(len(plan.victims), 2)
        self.assertEqual(list(cache.entries), [("v1", "c")])
        self.assertEqual(cache.used_bytes, 7)

    def test_partial_tail_is_retained(self):
        cache = self.cache(capacity=64, block=16)
        plan = self.apply(cache, request(history=17))
        self.assertEqual(plan.allocated_bytes, 32)
        self.assertEqual(plan.write_bytes, 17)
        self.assertIsNotNone(cache.lookup(request(history=17), 1))
        self.assertIsNone(cache.lookup(request(history=16), 1))

    def test_new_version_and_model_require_exact_match(self):
        cache = self.cache()
        self.apply(cache, request())
        self.assertIsNone(cache.lookup(request(version="1"), 1))
        self.assertIsNone(cache.lookup(request(model="v2"), 1))

    def test_publish_only_after_completion(self):
        cache = self.cache()
        req = request()
        cache.history.record(req.user_key, 0)
        plan = cache.plan(req, 0)
        self.assertIsNone(cache.lookup(req, 0))
        cache.commit(req, plan, 100)
        self.assertIsNone(cache.lookup(req, 99))
        self.assertIsNotNone(cache.lookup(req, 100))

    def test_hit_appends_incremental_only(self):
        cache = self.cache(capacity=16)
        self.apply(cache, request(time=0, inc=1))
        plan = self.apply(cache, request(time=1, history=5, inc=2, version="1"))
        self.assertTrue(plan.local_hit)
        self.assertEqual(plan.write_bytes, 2)
        self.assertEqual(cache.entries[("v1", "a")].valid_tokens, 7)
        self.assertEqual(cache.entries[("v1", "a")].version, "2")

    def test_oversized_miss_never_evicts_or_writes(self):
        cache = self.cache(capacity=4)
        self.apply(cache, request("a", 0))
        plan = self.apply(cache, request("b", 1, history=5))
        self.assertFalse(plan.accepted)
        self.assertEqual(plan.write_bytes, 0)
        self.assertEqual(cache.evictions, 0)

    def test_growth_over_capacity_invalidates_old_snapshot(self):
        cache = self.cache(capacity=4)
        self.apply(cache, request(time=0))
        plan = self.apply(cache, request(time=1, inc=1))
        self.assertTrue(plan.local_hit)
        self.assertFalse(plan.accepted)
        self.assertEqual(cache.used_bytes, 0)
        self.assertIsNone(cache.lookup(request(time=2), 2))

    def test_random_variable_sizes_preserve_capacity(self):
        rng = random.Random(9)
        for k in (1, 2, 5):
            cache = self.cache(k=k, capacity=128, block=16)
            for time in range(500):
                user = str(rng.randrange(12))
                self.apply(cache, request(user, time, history=1 + int(user) * 3))
                self.assertEqual(cache.used_bytes, sum(e.allocated_bytes for e in cache.entries.values()))
                self.assertLessEqual(cache.used_bytes, 128)


class SimulationTests(unittest.TestCase):
    def test_three_paths_and_exact_write_equations(self):
        cfg = config(k=2)
        requests = [request(time=i * 10000, history=4 + i, inc=1, version=str(i))
                    for i in range(3)]
        rows, _, summary = GRSimulator(cfg).run(requests)
        self.assertEqual([row["path"] for row in rows], ["miss_no_write", "miss_write", "hit"])
        self.assertEqual([row["write_bytes"] for row in rows], [0, 24, 4])
        self.assertEqual(summary["admission_given_miss"], 0.5)
        self.assertAlmostEqual(summary["mean_service_ns"], summary["path_weighted_mean_service_ns"])
        for row in rows:
            self.assertEqual(row["service_ns"], max(row["compute_ns"], row["read_ns"], row["write_ns"]))
        self.assertEqual(rows[0]["service_ns"], rows[0]["compute_ns"])

    def test_hstu_one_parallel_candidate_step_with_known_flops(self):
        rows, stages, _ = GRSimulator(config()).run([request(inc=1)])
        self.assertEqual(rows[0]["decode_steps"], 1)
        self.assertEqual(stages[0]["steps"][0]["query_count"], 2)
        self.assertEqual(rows[0]["historical_flops"], 500)
        self.assertEqual(rows[0]["incremental_flops"], 150)
        self.assertEqual(rows[0]["decode_flops"], 304)
        self.assertEqual(rows[0]["compute_ns"], 954)

    def test_candidate_count_changes_compute_not_persistent_kv(self):
        first = GRSimulator(config())
        second = GRSimulator(config())
        a, _, _ = first.run([request(candidates=1)])
        b, _, _ = second.run([request(candidates=500)])
        self.assertGreater(b[0]["decode_flops"], a[0]["decode_flops"])
        self.assertEqual(first.caches[0].used_bytes, second.caches[0].used_bytes)
        self.assertEqual(a[0]["write_bytes"], b[0]["write_bytes"])

    def test_openonerec_three_steps_and_three_history_reads(self):
        cfg = config(paradigm="openonerec")
        sim = GRSimulator(cfg)
        rows, stages, _ = sim.run([request(time=0), request(time=10000)])
        steps = stages[1]["steps"]
        self.assertEqual(len(steps), 3)
        self.assertEqual([step["query_count"] for step in steps], [1, 2, 2])
        self.assertEqual([step["context_tokens"] for step in steps], [4, 5, 6])
        self.assertEqual([step["logits_scored"] for step in steps], [4, 8, 8])
        self.assertEqual(rows[1]["read_bytes"], 4 * 4 * 3)
        self.assertEqual(sim.caches[0].entries[("v1", "a")].valid_tokens, 4)

    def test_explicit_candidate_scoring_widths(self):
        cfg = config(paradigm="openonerec")
        _, stages, _ = GRSimulator(cfg).run([request(widths=(3, 3, 3))])
        self.assertEqual([step["query_count"] for step in stages[0]["steps"]], [3, 3, 3])

    def test_sequential_decode_cannot_overlap_dependent_steps(self):
        cfg = config(paradigm="openonerec", peak=1e11, read_bw=1e8)
        cfg = replace(cfg, model=replace(cfg.model, semantic_vocab_sizes=(1, 10000, 1)))
        requests = [request(time=0), request(time=100000)]
        aggregate, _, _ = GRSimulator(cfg).run(requests)
        serial, stages, _ = GRSimulator(replace(cfg, latency_mode="sequential_roofline")).run(requests)
        self.assertGreater(serial[1]["service_ns"], aggregate[1]["service_ns"])
        self.assertEqual(serial[1]["service_ns"], sum(stages[1]["stage_service_ns"]))

    def test_hstu_roofline_modes_agree(self):
        cfg = config()
        a, _, _ = GRSimulator(cfg).run([request()])
        b, _, _ = GRSimulator(replace(cfg, latency_mode="sequential_roofline")).run([request()])
        self.assertEqual(a[0]["service_ns"], b[0]["service_ns"])

    def test_arrival_during_build_cannot_hit_future_kv(self):
        rows, _, summary = GRSimulator(config()).run([request(time=0), request(time=1)])
        self.assertEqual(rows[1]["path"], "hit")
        self.assertFalse(rows[1]["arrival_cache_hit"])
        self.assertGreater(rows[1]["queue_ns"], 0)
        self.assertEqual(summary["arrival_hit_rate"], 0)

    def test_no_lookahead_in_admission(self):
        cfg = config(k=2)
        rows, _, _ = GRSimulator(cfg).run([request(time=0), request(time=1)])
        self.assertEqual(rows[0]["path"], "miss_no_write")
        self.assertEqual(rows[1]["path"], "miss_write")

    def test_queued_k1_misses_are_always_admitted(self):
        cfg = config(capacity=16)
        rows, _, _ = GRSimulator(cfg).run([request("a", 0), request("b", 1), request("a", 2)])
        self.assertEqual([row["path"] for row in rows], ["miss_write"] * 3)

    def test_warmup_updates_cache_but_not_summary(self):
        cfg = config(k=2)
        rows, _, summary = GRSimulator(cfg).run(
            [request(time=0), request(time=10000), request(time=20000)], warmup_requests=2)
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["hit_rate"], 1)
        self.assertEqual(summary["mean_write_bytes"], 0)
        self.assertEqual([row["is_warmup"] for row in rows], [True, True, False])

    def test_tiered_recall_promotion_and_write_through(self):
        cfg = replace(config(), tiers=(TierConfig("HBM", 16, 1e12, 1e12),
                                      TierConfig("CPU", 64, 1e10, 1e10, 1e8)))
        rows, _, _ = GRSimulator(cfg).run([request("a", 0), request("b", 10000), request("a", 20000)])
        self.assertEqual(rows[2]["source_tier"], "CPU")
        self.assertEqual(rows[2]["read_ns"], 160)
        self.assertEqual(rows[2]["writes_by_tier"], {"HBM": 16, "CPU": 0})

    def test_slow_memory_hit_can_be_slower_than_recompute(self):
        cfg = config(peak=1e12, read_bw=1e6)
        rows, _, _ = GRSimulator(cfg).run([request(time=0), request(time=100000)])
        self.assertGreater(rows[1]["service_ns"], rows[0]["service_ns"])

    def test_openonerec_promotion_avoids_repeated_pcie_recall(self):
        cfg = replace(config(paradigm="openonerec"),
                      tiers=(TierConfig("HBM", 16, 1e12, 1e12),
                             TierConfig("CPU", 64, 1e10, 1e10, 1e8)))
        rows, stages, _ = GRSimulator(cfg).run(
            [request("a", 0), request("b", 10000), request("a", 20000)])
        self.assertEqual([step["history_read_tier"] for step in stages[2]["steps"]],
                         ["CPU", "HBM", "HBM"])
        self.assertEqual(rows[2]["reads_by_tier"], {"HBM": 32, "CPU": 16})
        self.assertAlmostEqual(rows[2]["read_ns"], 160.032)

    def test_lifetime_uses_workload_rate_and_amplification(self):
        cfg = replace(config(), workload_qps=10)
        cfg = replace(cfg, tiers=(replace(cfg.tiers[0], write_amplification=2),))
        _, _, summary = GRSimulator(cfg).run([request()])
        lifetime = summary["tiers"]["HBF"]["lifetime_seconds"]
        self.assertEqual(lifetime, 1024 * 100000 / (10 * 16 * 2))

    def test_unknown_rate_and_no_write_do_not_emit_infinity(self):
        for k, status in [(1, "unknown_arrival_rate"), (2, "no_writes")]:
            _, _, summary = GRSimulator(config(k=k)).run([request()])
            self.assertEqual(summary["tiers"]["HBF"]["lifetime_status"], status)
            json.dumps(summary, allow_nan=False)

    def test_heterogeneous_history_uses_conditional_latency(self):
        reqs = [request("long", 0, history=100), request("short", 100000, history=1),
                request("long", 200000, history=100)]
        _, _, summary = GRSimulator(config()).run(reqs)
        self.assertEqual(summary["paths"]["hit"]["mean_history_tokens"], 100)
        self.assertAlmostEqual(summary["mean_service_ns"], summary["path_weighted_mean_service_ns"])


class InputAndCLITests(unittest.TestCase):
    def test_invalid_numeric_inputs_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                replace(config(), cache_k=value)
        with self.assertRaises(ValueError):
            replace(config(), peak_flops=float("nan"))
        with self.assertRaises(ValueError):
            request(inc=1, next_version="0")
        with self.assertRaises(ValueError):
            request(widths=(1, 2))

    def test_loader_catches_duplicate_ids_and_bad_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"input_toks":4}\n')
            with self.assertRaisesRegex(ValueError, ":1:"):
                load_requests(path)
            row = json.dumps(asdict(request()))
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaisesRegex(ValueError, "request_id must be unique"):
                load_requests(path)

    def test_generator_is_reproducible_and_advances_histories(self):
        from serving.gr.generate import main as generate
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("a.jsonl", "b.jsonl")]
            for path in paths:
                with redirect_stdout(io.StringIO()):
                    generate(["--users", "3", "--duration-s", "100", "--seed", "7",
                              "--min-rate-per-hour", "1000", "--max-rate-per-hour", "1000",
                              "--incremental-tokens", "2", "--output", str(path)])
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            previous = {}
            for req in load_requests(paths[0]):
                if req.user_id in previous:
                    old = previous[req.user_id]
                    self.assertEqual(req.history_tokens, old.updated_tokens)
                    self.assertEqual(req.history_version, old.next_history_version)
                previous[req.user_id] = req

    def test_cli_dispatch_and_output_bundle(self):
        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.csv"
            proc = subprocess.run([
                sys.executable, "-m", "serving", "gr", "--backend", "analytical",
                "--config", "configs/gr/hstu_hbf.json",
                "--dataset", "workloads/gr_example.jsonl", "--sweep-k", "1", "2",
                "--output", str(output),
            ], cwd=repo, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for k in (1, 2):
                summary = json.loads((Path(directory) / f"run_k{k}.summary.json").read_text())
                self.assertEqual(summary["requests"], 12)
                self.assertTrue((Path(directory) / f"run_k{k}.stages.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
