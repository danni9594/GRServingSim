"""CLI for the manuscript's trace-driven GR model and ASTRA executor."""

import argparse
import csv
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path

from .config import GRConfig
from .request import load_requests
from .simulator import GRSimulator


def write_outputs(records, stages, summary, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})
    summary_path = output.with_suffix(".summary.json")
    stages_path = output.with_suffix(".stages.jsonl")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with stages_path.open("w", encoding="utf-8") as stream:
        for stage in stages:
            stream.write(json.dumps(stage, allow_nan=False) + "\n")
    return str(summary_path), str(stages_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GR whole-user KV/LRU-K simulation (HSTU or OpenOneRec). "
                    "ASTRA-Sim operator DAG execution, single-device FCFS.")
    parser.add_argument("--backend", choices=("astra", "analytical"), default="astra")
    parser.add_argument("--astra-binary", help="override the ASTRA executable")
    parser.add_argument("--astra-timeout", type=float, default=60, help="seconds per backend handshake")
    parser.add_argument("--keep-inputs", action="store_true")
    parser.add_argument("--save-trace-text", action="store_true", help="keep operator DAG JSON and Chakra traces")
    parser.add_argument("--config", required=True, help="GR JSON configuration")
    parser.add_argument("--dataset", required=True, help="versioned GR request JSONL")
    parser.add_argument("--output", default="outputs/gr_requests.csv", help="CSV; also writes .summary.json/.stages.jsonl")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--cache-k", type=int, help="override admission-controlled LRU-K")
    policy.add_argument("--sweep-k", type=int, nargs="+", help="run independent cold-cache experiments, suffixing outputs with _kN")
    parser.add_argument("--latency-mode", choices=("paper_roofline", "sequential_roofline"))
    parser.add_argument("--workload-qps", type=float, help="arrival rate used for flash lifetime; not saturation QPS")
    parser.add_argument("--num-reqs", type=int, default=0, help="0 loads all requests")
    parser.add_argument("--warmup-requests", type=int, default=0,
                        help="initial requests update cache but are excluded from summary rates/latencies")
    args = parser.parse_args(argv)
    try:
        if args.num_reqs < 0:
            raise ValueError("num-reqs must be nonnegative")
        config = GRConfig.load(args.config)
        if args.backend == "astra" and args.latency_mode is not None:
            raise ValueError("latency-mode applies only to --backend analytical")
        if args.latency_mode is not None:
            config = replace(config, latency_mode=args.latency_mode)
        if args.workload_qps is not None:
            config = replace(config, workload_qps=args.workload_qps)
        requests = load_requests(args.dataset, args.num_reqs)
        values = args.sweep_k or [args.cache_k if args.cache_k is not None else config.cache_k]
        if len(set(values)) != len(values):
            raise ValueError("sweep-k values must be unique")
        for k in values:
            run_config = replace(config, cache_k=k)
            output = Path(args.output)
            if args.sweep_k:
                output = output.with_name(f"{output.stem}_k{k}{output.suffix}")
            if output.suffix.lower() != ".csv":
                raise ValueError("output must have a .csv suffix")
            targets = [output, output.with_suffix(".summary.json"), output.with_suffix(".stages.jsonl")]
            if any(target.resolve() in (Path(args.config).resolve(), Path(args.dataset).resolve())
                   for target in targets):
                raise ValueError("output must not overwrite an input")
            context = nullcontext(None)
            if args.backend == "astra":
                from .astra import AstraExecutor
                context = AstraExecutor(run_config, args.astra_binary, args.keep_inputs,
                                        args.save_trace_text, args.astra_timeout)
            with context as executor:
                records, stages, summary = GRSimulator(run_config, executor).run(requests, args.warmup_requests)
            summary_path, stages_path = write_outputs(records, stages, summary, output)
            print(json.dumps({
                "cache_k": k, "requests": summary["requests"],
                "backend": summary["backend"], "hit_rate": summary["hit_rate"],
                "service_capacity_qps": summary["service_capacity_qps"],
                "mean_service_ms": summary["mean_service_ns"] / 1e6,
                "mean_write_bytes": summary["mean_write_bytes"],
                "calibration_note": config.model.calibration_note,
                "output": str(output), "summary": summary_path, "stages": stages_path,
            }, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0
