"""Unit tests for HTTP run event bus lifecycle behavior."""

from src.server.event_bus import MAX_CLOSED_RUN_IDS, RunEventBus


def test_close_tracks_only_bounded_recent_run_ids():
    """Closed run tracking should stay bounded in memory."""
    bus = RunEventBus()
    total_closed = MAX_CLOSED_RUN_IDS + 128

    for index in range(total_closed):
        bus.close(f"run_{index}")

    assert len(bus._closed_runs) == MAX_CLOSED_RUN_IDS
    assert "run_0" not in bus._closed_runs
    assert f"run_{total_closed - 1}" in bus._closed_runs


def test_subscribe_rejects_recently_closed_run():
    """Subscriptions should be rejected for recently closed runs."""
    bus = RunEventBus()
    bus.close("run_closed")

    events = list(bus.subscribe("run_closed"))

    assert events == []


def test_close_removes_subscriber_and_ends_stream():
    """Closing one run should drain and end active subscriber iterators."""
    bus = RunEventBus()
    events_iter = bus.subscribe("run_live", heartbeat_seconds=60)

    assert "run_live" in bus._subscribers
    bus.close("run_live")

    assert list(events_iter) == []
    assert "run_live" not in bus._subscribers
