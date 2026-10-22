"""In-process metrics registry with counters, gauges, and timing observations."""

from __future__ import annotations

import math
import time
from threading import Lock
from typing import Any, cast

LabelValues = dict[str, Any]


def _now() -> float:
    return time.monotonic()


class _Metric:
    __slots__ = ("name", "labels")

    def __init__(self, name: str, labels: LabelValues | None = None) -> None:
        self.name = name
        self.labels = labels or {}

    @property
    def key(self) -> tuple[str, str, tuple[tuple[str, Any], ...]]:
        return (
            type(self).__name__.lower(),
            self.name,
            tuple(sorted(self.labels.items())),
        )

    def render_labels(self) -> str:
        if not self.labels:
            return ""
        joined = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        return "{" + joined + "}"


class Counter(_Metric):
    __slots__ = ("value", "_lock")

    def __init__(self, name: str, labels: LabelValues | None = None) -> None:
        super().__init__(name, labels)
        self.value = 0
        self._lock = Lock()

    def add(self, amount: float = 1.0) -> float:
        with self._lock:
            self.value += amount
            return self.value


class Gauge(_Metric):
    __slots__ = ("value", "_lock")

    def __init__(self, name: str, labels: LabelValues | None = None) -> None:
        super().__init__(name, labels)
        self.value = 0.0
        self._lock = Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self.value = value

    def add(self, amount: float) -> float:
        with self._lock:
            self.value += amount
            return self.value


class Summary(_Metric):
    __slots__ = ("count", "total", "minimum", "maximum", "_lock")

    def __init__(self, name: str, labels: LabelValues | None = None) -> None:
        super().__init__(name, labels)
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self._lock = Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self.count += 1
            self.total += value
            if value < self.minimum:
                self.minimum = value
            if value > self.maximum:
                self.maximum = value

    @property
    def mean(self) -> float:
        with self._lock:
            return self.total / self.count if self.count else 0.0


class MetricsRegistry:
    """Thread-safe registry of counters, gauges, and summaries.

    Metrics are keyed by name plus sorted label pairs, so the same name
    can carry multiple labeled series. Supports Prometheus-style text
    export and a timed context manager for latency histograms.
    """

    def __init__(self) -> None:
        self._metrics: dict[tuple[str, str, tuple[tuple[str, Any], ...]], _Metric] = {}
        self._lock = Lock()

    def _get(self, sample: _Metric) -> _Metric:
        with self._lock:
            existing = self._metrics.get(sample.key)
            if existing is not None:
                return existing
            self._metrics[sample.key] = sample
            return sample

    def counter(self, name: str, labels: LabelValues | None = None) -> Counter:
        return cast(Counter, self._get(Counter(name, labels)))

    def gauge(self, name: str, labels: LabelValues | None = None) -> Gauge:
        return cast(Gauge, self._get(Gauge(name, labels)))

    def summary(self, name: str, labels: LabelValues | None = None) -> Summary:
        return cast(Summary, self._get(Summary(name, labels)))

    def increment(
        self, name: str, value: float = 1.0, labels: LabelValues | None = None
    ) -> float:
        return self.counter(name, labels).add(value)

    def set(self, name: str, value: float, labels: LabelValues | None = None) -> None:
        self.gauge(name, labels).set(value)

    def observe(
        self, name: str, value: float, labels: LabelValues | None = None
    ) -> None:
        self.summary(name, labels).observe(value)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            metrics = list(self._metrics.values())
        out: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            if isinstance(metric, Counter):
                kind = "counter"
                body: dict[str, Any] = {"type": kind, "value": metric.value}
            elif isinstance(metric, Gauge):
                kind = "gauge"
                body = {"type": kind, "value": metric.value}
            elif isinstance(metric, Summary):
                kind = "summary"
                body = {
                    "type": kind,
                    "count": metric.count,
                    "total": metric.total,
                    "mean": metric.mean,
                    "min": metric.minimum if metric.count else None,
                    "max": metric.maximum if metric.count else None,
                }
            else:
                continue
            out[f"{metric.name}{metric.render_labels()}"] = body
        return out

    def export(self) -> str:
        lines = []
        for name, body in sorted(self.snapshot().items()):
            kind = body["type"]
            if kind == "summary":
                line = (
                    f"{name} count={body['count']} "
                    f"total={body['total']} mean={body['mean']}"
                )
                lines.append(line)
            else:
                lines.append(f"{name} {body['value']}")
        return "\n".join(lines) + ("\n" if lines else "")

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()

    def timed(self, name: str, labels: LabelValues | None = None) -> _Timer:
        return _Timer(self, name, labels)


class _Timer:
    __slots__ = ("registry", "name", "labels", "_start")

    def __init__(
        self, registry: MetricsRegistry, name: str, labels: LabelValues | None
    ) -> None:
        self.registry = registry
        self.name = name
        self.labels = labels
        self._start: float | None = None

    def __enter__(self) -> _Timer:
        self._start = _now()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._start is not None:
            elapsed = _now() - self._start
            self.registry.observe(self.name, elapsed, self.labels)
