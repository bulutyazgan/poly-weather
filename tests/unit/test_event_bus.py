"""Tests for the EventBus pub/sub system."""
from __future__ import annotations

import asyncio

import pytest

from src.api.event_bus import EventBus


def test_subscribe_creates_queue():
    bus = EventBus()
    q = bus.subscribe()
    assert isinstance(q, asyncio.Queue)
    assert len(bus._subscribers) == 1


def test_unsubscribe_removes_queue():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    assert len(bus._subscribers) == 0


def test_unsubscribe_missing_is_safe():
    bus = EventBus()
    q = asyncio.Queue()
    bus.unsubscribe(q)  # should not raise
    assert len(bus._subscribers) == 0


def test_publish_delivers_to_subscriber():
    bus = EventBus()
    q = bus.subscribe()
    bus.publish("test_event", {"key": "value"})
    assert not q.empty()
    msg = q.get_nowait()
    assert msg["event"] == "test_event"
    assert msg["data"] == {"key": "value"}
    assert "timestamp" in msg


def test_publish_delivers_to_multiple_subscribers():
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.publish("evt", {"n": 1})
    assert not q1.empty()
    assert not q2.empty()
    assert q1.get_nowait()["data"] == {"n": 1}
    assert q2.get_nowait()["data"] == {"n": 1}


def test_publish_no_subscribers_is_safe():
    bus = EventBus()
    bus.publish("evt", {})  # should not raise


def test_queue_full_drops_oldest():
    bus = EventBus()
    q = bus.subscribe()
    # Fill the queue
    for i in range(256):
        bus.publish("evt", {"i": i})
    assert q.full()
    # Publish one more — should drop oldest and add new
    bus.publish("evt", {"i": 256})
    # Queue should still be full but contain the newest message
    msgs = []
    while not q.empty():
        msgs.append(q.get_nowait())
    assert msgs[-1]["data"]["i"] == 256
    assert msgs[0]["data"]["i"] == 1  # 0 was dropped
    assert len(msgs) == 256
