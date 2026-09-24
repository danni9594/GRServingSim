"""Versioned user histories; candidate/semantic-token KV is request-local."""

from dataclasses import dataclass
import json

from .config import number


@dataclass(frozen=True)
class GRRequest:
    request_id: str
    user_id: str
    arrival_time_ns: int
    history_tokens: int
    incremental_tokens: int
    history_version: str
    next_history_version: str
    model_version: str = "default"
    candidate_count: int = 500
    decode_widths: tuple = ()

    def __post_init__(self):
        for name in ("request_id", "user_id", "history_version", "next_history_version", "model_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("arrival_time_ns", "history_tokens", "incremental_tokens"):
            number(name, getattr(self, name), integer=True)
        number("candidate_count", self.candidate_count, integer=True, strict=True)
        if self.incremental_tokens and self.next_history_version == self.history_version:
            raise ValueError("new interactions must advance next_history_version")
        if not self.incremental_tokens and self.next_history_version != self.history_version:
            raise ValueError("without new interactions the history version must stay unchanged")
        if self.decode_widths and len(self.decode_widths) != 3:
            raise ValueError("decode_widths must contain exactly three OpenOneRec widths")
        for width in self.decode_widths:
            number("decode_width", width, integer=True, strict=True)

    @property
    def user_key(self):
        return self.model_version, self.user_id

    @property
    def updated_tokens(self):
        return self.history_tokens + self.incremental_tokens


def load_requests(path, limit=0):
    requests = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if limit and len(requests) >= limit:
                break
            try:
                row = json.loads(line)
                row.setdefault("request_id", str(line_number))
                row["decode_widths"] = tuple(row.get("decode_widths", ()))
                requests.append(GRRequest(**row))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not requests:
        raise ValueError("GR workload is empty")
    if len({req.request_id for req in requests}) != len(requests):
        raise ValueError("request_id must be unique")
    # Stable order breaks simultaneous-arrival ties deterministically.
    return sorted(requests, key=lambda req: req.arrival_time_ns)
