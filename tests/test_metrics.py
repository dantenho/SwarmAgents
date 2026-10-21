import threading

from swarmagents.metrics import MetricsRegistry


def test_counter_increment_and_labels() -> None:
    reg = MetricsRegistry()
    assert reg.increment("requests") == 1
    assert reg.increment("requests") == 2
    reg.increment("requests", value=5, labels={"path": "/api"})
    snap = reg.snapshot()
    assert snap["requests"]["value"] == 2
    assert snap['requests{path="/api"}']["value"] == 5


def test_gauge_set() -> None:
    reg = MetricsRegistry()
    reg.set("queue_depth", 7)
    reg.set("queue_depth", 3)
    assert reg.snapshot()["queue_depth"]["value"] == 3


def test_summary_stats() -> None:
    reg = MetricsRegistry()
    for value in (1.0, 2.0, 3.0):
        reg.observe("latency", value)
    body = reg.snapshot()["latency"]
    assert body["count"] == 3
    assert body["total"] == 6.0
    assert body["mean"] == 2.0
    assert body["min"] == 1.0
    assert body["max"] == 3.0


def test_timed_context_manager_records_duration() -> None:
    reg = MetricsRegistry()
    with reg.timed("query_duration"):
        pass
    assert reg.snapshot()["query_duration"]["count"] == 1


def test_thread_safety_under_concurrency() -> None:
    reg = MetricsRegistry()
    barrier = threading.Barrier(8)

    def bump() -> None:
        barrier.wait()
        for _ in range(1000):
            reg.increment("hits")

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert reg.snapshot()["hits"]["value"] == 8000


def test_export_format() -> None:
    reg = MetricsRegistry()
    reg.increment("requests")
    reg.observe("latency", 0.5)
    exported = reg.export()
    assert "requests 1" in exported
    assert "latency count=1 total=0.5" in exported


def test_clear() -> None:
    reg = MetricsRegistry()
    reg.increment("x")
    reg.clear()
    assert reg.snapshot() == {}


def test_same_name_different_types_coexist() -> None:
    reg = MetricsRegistry()
    reg.increment("thing")
    reg.observe("thing", 1.0)
    reg.observe("thing", 2.0)
    snap = reg.snapshot()
    assert snap["thing"]["type"] == "summary"
    assert snap["thing"]["count"] == 2
