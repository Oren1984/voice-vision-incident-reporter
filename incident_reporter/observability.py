"""Structured JSON logs, request ids and in-process metrics.

Logs carry ids, sizes, durations, states and error types — never transcripts, summaries, passwords,
session tokens or file contents.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED = set(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(out, default=str)


def setup_logging(path: Path | None) -> logging.Logger:
    log = logging.getLogger("vvir")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
    return log


class Metrics:
    """Counters and duration samples, kept in memory since process start."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.durations: dict[str, list[float]] = defaultdict(list)
        self.started = time.time()

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            d = self.durations[name]
            d.append(seconds)
            if len(d) > 1000:
                del d[: len(d) - 1000]

    def snapshot(self) -> dict:
        with self._lock:
            dur = {}
            for k, v in self.durations.items():
                s = sorted(v)
                dur[k] = {
                    "count": len(s),
                    "p50_ms": round(1000 * s[len(s) // 2], 1),
                    "p95_ms": round(1000 * s[min(len(s) - 1, int(0.95 * len(s)))], 1),
                    "max_ms": round(1000 * s[-1], 1),
                }
            return {"uptime_s": round(time.time() - self.started), "counters": dict(self.counters), "durations": dur}


class timed:
    """Context manager: records a duration sample and returns elapsed seconds in .elapsed."""

    def __init__(self, metrics: Metrics, name: str) -> None:
        self.metrics, self.name, self.elapsed = metrics, name, 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        self.metrics.observe(self.name, self.elapsed)
        if exc_type:
            self.metrics.inc(f"{self.name}.failures")
        return False
