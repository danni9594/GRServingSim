"""Whole-object, byte-capacity LRU-K with admission and completion separated.

Reference histories survive eviction. References are recorded once at request
arrival, including misses that are not admitted. K=1 is ordinary LRU. With K>1,
an object needs K references even during cold start. To admit a variable-sized
object, its K-th reference must be newer than *every* required victim's.
"""

from collections import deque
from dataclasses import dataclass


class AccessHistory:
    def __init__(self, k):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.references = {}
        self.sequence = 0
        self.last_time = -1

    def record(self, key, arrival_ns):
        if arrival_ns < self.last_time:
            raise ValueError("references must be recorded in arrival order")
        self.last_time = arrival_ns
        self.sequence += 1
        history = self.references.setdefault(key, deque(maxlen=self.k))
        # Sequence breaks equal timestamps, making LRU-1 admit every feasible miss.
        history.append((arrival_ns, self.sequence))

    def priority(self, key):
        history = self.references.get(key, ())
        return history[0] if len(history) == self.k else None


@dataclass(frozen=True)
class UserKVEntry:
    user_key: tuple
    version: str
    valid_tokens: int
    allocated_bytes: int
    ready_at_ns: float


@dataclass(frozen=True)
class CachePlan:
    accepted: bool
    local_hit: bool
    allocated_bytes: int
    write_bytes: int
    victims: tuple = ()
    reason: str = ""


class UserKVCacheManager:
    """One memory tier; entries are indivisible complete user KV snapshots.

    Blocks are charged by ceil(tokens / block_tokens), including partial tails.
    No physical tensors are stored. A plan reserves no data and is committed
    only after service completes. The GR scheduler has one in-flight request.
    """

    def __init__(self, capacity_bytes, bytes_per_token, block_tokens, history):
        self.capacity_bytes = capacity_bytes
        self.bytes_per_token = bytes_per_token
        self.block_tokens = block_tokens
        self.history = history
        self.entries = {}
        self.used_bytes = 0
        self.peak_bytes = 0
        self.evictions = 0
        self.invalidations = 0

    def size(self, tokens):
        return ((tokens + self.block_tokens - 1) // self.block_tokens
                * self.block_tokens * self.bytes_per_token)

    def lookup(self, req, now):
        entry = self.entries.get(req.user_key)
        if (entry is not None and entry.version == req.history_version
                and entry.valid_tokens == req.history_tokens
                and entry.ready_at_ns <= now):
            return entry
        return None

    def plan(self, req, now):
        existing = self.entries.get(req.user_key)
        local_hit = self.lookup(req, now) is not None
        required = self.size(req.updated_tokens)
        if required == 0:
            return CachePlan(False, local_hit, 0, 0, reason="empty_history")
        if required > self.capacity_bytes:
            return CachePlan(False, local_hit, required, 0, reason="object_too_large")
        priority = self.history.priority(req.user_key)
        if priority is None:
            return CachePlan(False, local_hit, required, 0, reason="fewer_than_k_references")

        # The old version will be replaced or invalidated, never partly reused.
        available = self.capacity_bytes - self.used_bytes
        if existing is not None:
            available += existing.allocated_bytes
        candidates = sorted(
            (entry for key, entry in self.entries.items() if key != req.user_key),
            key=lambda entry: (self.history.priority(entry.user_key), entry.user_key),
        ) if available < required else ()
        victims = []
        for entry in candidates:
            if available >= required:
                break
            # All-or-nothing: a rejected plan evicts nobody.
            # LRU-1 admits every feasible miss, including a queued request
            # whose arrival predates another resident user's queued reference.
            if self.history.k > 1 and priority <= self.history.priority(entry.user_key):
                return CachePlan(False, local_hit, required, 0, reason="colder_than_victims")
            victims.append(entry.user_key)
            available += entry.allocated_bytes
        if available < required:
            return CachePlan(False, local_hit, required, 0, reason="insufficient_capacity")
        tokens_written = req.incremental_tokens if local_hit else req.updated_tokens
        return CachePlan(True, local_hit, required, tokens_written * self.bytes_per_token,
                         tuple(victims), "append" if local_hit else "admit")

    def commit(self, req, plan, finish_ns):
        old = self.entries.pop(req.user_key, None)
        if old is not None:
            self.used_bytes -= old.allocated_bytes
            if not plan.local_hit or not plan.accepted:
                self.invalidations += 1
        if not plan.accepted:
            # Dropping stale/unretainable data does not write it anywhere.
            return
        for key in plan.victims:
            victim = self.entries.pop(key)
            self.used_bytes -= victim.allocated_bytes
            self.evictions += 1
        entry = UserKVEntry(req.user_key, req.next_history_version, req.updated_tokens,
                            plan.allocated_bytes, finish_ns)
        self.entries[req.user_key] = entry
        self.used_bytes += entry.allocated_bytes
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        if not 0 <= self.used_bytes <= self.capacity_bytes:
            raise RuntimeError("whole-user cache capacity invariant violated")
