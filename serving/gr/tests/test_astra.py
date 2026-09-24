"""Real subprocess tests: skipped only when the backend has not been built."""

from dataclasses import replace
import unittest

from serving.gr.astra import AstraExecutor, DEFAULT_BINARY
from serving.gr.config import AstraConfig, TierConfig
from serving.gr.simulator import GRSimulator
from serving.gr.trace import GraphBuilder
from .test_gr import config, request


@unittest.skipUnless(DEFAULT_BINARY.is_file(), "run scripts/compile-gr.sh")
class AstraTests(unittest.TestCase):
    def run_sim(self, cfg, requests):
        with AstraExecutor(cfg, timeout_s=10) as executor:
            return GRSimulator(cfg, executor).run(requests)

    def test_directional_memory_runs_in_backend(self):
        cfg = config(read_bw=2e9, write_bw=0.5e9)
        with AstraExecutor(cfg, timeout_s=10) as executor:
            graph = GraphBuilder("test", cfg)
            load = graph.memory("load", 100, "storage")
            graph.memory("store", 100, "storage", load, write=True)
            self.assertEqual(executor.execute(graph.trace), 50 + 200)

    def test_independent_tiers_overlap_and_same_tier_queues(self):
        cfg = replace(config(read_bw=1e9, write_bw=1e9), astra=AstraConfig(
            workspace_read_bandwidth_bytes_per_s=1e9,
            workspace_write_bandwidth_bytes_per_s=1e9))
        with AstraExecutor(cfg, timeout_s=10) as executor:
            graph = GraphBuilder("independent", cfg)
            graph.memory("local", 100, "local")
            graph.memory("storage", 100, "storage")
            self.assertEqual(executor.execute(graph.trace), 100)
            graph = GraphBuilder("contended", cfg)
            graph.memory("first", 100, "storage")
            graph.memory("second", 100, "storage", write=True)
            self.assertEqual(executor.execute(graph.trace), 200)

    def test_hstu_hit_skips_history_and_batches_candidates_once(self):
        reqs = [request(time=0), request(time=100000, history=4, inc=2)]
        rows, stages, summary = self.run_sim(config(), reqs)
        self.assertEqual([row["path"] for row in rows], ["miss_write", "hit"])
        phases = [row["phase"] for row in stages[1]["astra_stages"]]
        self.assertEqual(phases, ["incremental_prefill", "candidate_decode"])
        decode = stages[1]["astra_stages"][-1]
        self.assertEqual(decode["stage"]["queries"], reqs[1].candidate_count)
        self.assertEqual(rows[1]["write_bytes"], 2 * 4)
        self.assertEqual(rows[1]["historical_flops"], 0)
        self.assertEqual(summary["backend"], "astra_sim")
        self.assertNotIn("roofline_qps", summary)
        self.assertEqual(rows[1]["start_time_ns"], 100000)
        for row, detail in zip(rows, stages):
            self.assertEqual(row["service_ns"], sum(s["cycles"] for s in detail["astra_stages"]))

    def test_semantic_steps_depend_on_previous_completion(self):
        rows, stages, _ = self.run_sim(config(paradigm="openonerec"), [request(candidates=3)])
        decode = [stage for stage in stages[0]["astra_stages"] if stage["phase"] == "candidate_decode"]
        self.assertEqual([stage["stage"]["decode_step"] for stage in decode], [1, 2, 3])
        self.assertEqual([stage["stage"]["queries"] for stage in decode], [1, 3, 3])
        self.assertEqual(rows[0]["decode_steps"], 3)
        for before, after in zip(decode, decode[1:]):
            self.assertEqual(before["finish_ns"], after["start_ns"])

    def test_write_bandwidth_affects_admitted_miss_but_not_no_write(self):
        reqs = [request()]
        fast, _, _ = self.run_sim(config(write_bw=1e9), reqs)
        slow, _, _ = self.run_sim(config(write_bw=1e6), reqs)
        self.assertGreater(slow[0]["service_ns"], fast[0]["service_ns"])
        fast, _, _ = self.run_sim(config(k=2, write_bw=1e9), reqs)
        slow, stages, _ = self.run_sim(config(k=2, write_bw=1e6), reqs)
        self.assertEqual(slow[0]["path"], "miss_no_write")
        self.assertEqual(slow[0]["write_bytes"], 0)
        self.assertEqual(slow[0]["service_ns"], fast[0]["service_ns"])
        self.assertEqual(stages[0]["memory_write_bytes_by_location"].get("storage", 0), 0)
        self.assertGreater(stages[0]["memory_write_bytes_by_location"]["local"], 0)

    def test_lruk_admission_and_completion_visibility(self):
        reqs = [request(time=0), request(time=1), request(time=100000)]
        rows, _, _ = self.run_sim(config(k=2), reqs)
        self.assertEqual([row["path"] for row in rows], ["miss_no_write", "miss_write", "hit"])
        self.assertFalse(rows[1]["arrival_cache_hit"])
        self.assertTrue(rows[2]["arrival_cache_hit"])
        rows, _, _ = self.run_sim(config(), reqs[:2])
        self.assertEqual(rows[1]["path"], "hit")
        self.assertFalse(rows[1]["arrival_cache_hit"])
        self.assertEqual(rows[1]["start_time_ns"], rows[0]["finish_time_ns"])

    def test_cpu_recall_once_without_hbm_cache_admission(self):
        cfg = replace(config(paradigm="openonerec"), tiers=(
            TierConfig("HBM", 0, 1e9, 1e9), TierConfig("CPU", 1000, 1e9, 1e9)),
            astra=AstraConfig(workspace_read_bandwidth_bytes_per_s=1e9,
                              workspace_write_bandwidth_bytes_per_s=1e9))
        rows, stages, _ = self.run_sim(cfg, [request(), request(time=100000)])
        self.assertEqual(rows[1]["source_tier"], "CPU")
        phases = [stage["phase"] for stage in stages[1]["astra_stages"]]
        self.assertEqual(phases, ["recall"] + ["candidate_decode"] * 3)
        self.assertEqual(stages[1]["memory_read_bytes_by_location"]["remote"], 4 * 4)
        self.assertNotIn("HBM", rows[1]["retained_tiers"])

    def test_chunked_history_preserves_total_flops(self):
        cfg = replace(config(), astra=AstraConfig(prefill_chunk_tokens=2))
        rows, stages, _ = self.run_sim(cfg, [request(history=5, inc=3)])
        phases = [stage["phase"] for stage in stages[0]["astra_stages"]]
        self.assertEqual(phases, ["history_prefill"] * 3 + ["incremental_prefill"] * 2 + ["candidate_decode"])
        self.assertEqual(rows[0]["historical_flops"] + rows[0]["incremental_flops"], 8*100 + 8*9/2*10)

    def test_reject_workspace_overflow(self):
        cfg = replace(config(), astra=AstraConfig(workspace_capacity_bytes=1))
        with self.assertRaisesRegex(ValueError, "active KV needs"):
            self.run_sim(cfg, [request()])


if __name__ == "__main__":
    unittest.main()
