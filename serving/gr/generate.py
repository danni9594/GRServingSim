"""Synthetic Poisson traffic with activity/history correlation; not paper data."""

import argparse
import heapq
import json
import math
from pathlib import Path
import random


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--duration-s", type=float, default=3600)
    parser.add_argument("--min-rate-per-hour", type=float, default=1)
    parser.add_argument("--max-rate-per-hour", type=float, default=200)
    parser.add_argument("--min-history", type=int, default=32)
    parser.add_argument("--max-initial-history", type=int, default=8192)
    parser.add_argument("--incremental-tokens", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (args.users < 1 or not math.isfinite(args.duration_s) or args.duration_s <= 0
            or not math.isfinite(args.min_rate_per_hour) or args.min_rate_per_hour <= 0
            or not math.isfinite(args.max_rate_per_hour) or args.max_rate_per_hour < args.min_rate_per_hour
            or args.min_history < 1 or args.max_initial_history < args.min_history
            or args.incremental_tokens < 0 or args.candidates < 1):
        parser.error("invalid workload size/rate/history arguments")
    rng = random.Random(args.seed)
    pending = []
    users = {}
    for user in range(args.users):
        activity = rng.random()
        rate = (args.min_rate_per_hour
                * (args.max_rate_per_hour / args.min_rate_per_hour) ** activity / 3600)
        history = round(args.min_history + activity * (args.max_initial_history - args.min_history))
        users[user] = [rate, history, 0]
        heapq.heappush(pending, (rng.expovariate(rate), user))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        while pending:
            time_s, user = heapq.heappop(pending)
            if time_s > args.duration_s:
                break
            rate, history, version = users[user]
            next_version = version + (1 if args.incremental_tokens else 0)
            stream.write(json.dumps({
                "request_id": str(count), "user_id": f"u{user}",
                "arrival_time_ns": round(time_s * 1e9),
                "history_tokens": history, "incremental_tokens": args.incremental_tokens,
                "history_version": str(version), "next_history_version": str(next_version),
                "candidate_count": args.candidates,
            }) + "\n")
            users[user] = [rate, history + args.incremental_tokens, next_version]
            count += 1
            heapq.heappush(pending, (time_s + rng.expovariate(rate), user))
    print(json.dumps({"requests": count, "output": str(output), "seed": args.seed,
                      "expected_workload_qps": sum(state[0] for state in users.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
