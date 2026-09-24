#!/usr/bin/env bash
# Keep integration edits reproducible without changing pinned upstream commits.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apply_patch() {
  local tree="$1" patch="$2"
  if git -C "$tree" apply --check "$patch" 2>/dev/null; then
    git -C "$tree" apply "$patch"
  elif git -C "$tree" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: $(basename "$patch")"
  else
    echo "Cannot apply $patch cleanly; inspect local changes in $tree" >&2
    exit 1
  fi
}
apply_patch "$REPO_ROOT/astra-sim" "$REPO_ROOT/patches/gr-astra.patch"
apply_patch "$REPO_ROOT/astra-sim/extern/memory_backend/analytical" "$REPO_ROOT/patches/gr-memory.patch"
