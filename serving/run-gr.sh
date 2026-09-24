#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
for model in hstu openonerec; do
  python3 -m serving gr \
    --config "configs/gr/${model}_hbf.json" \
    --dataset workloads/gr_example.jsonl \
    --sweep-k 1 2 10 \
    --output "outputs/gr_${model}.csv"
done
