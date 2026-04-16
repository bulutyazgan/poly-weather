"""Async event bus for broadcasting real-time events to SSE clients."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class EventBus:
    """Publish/subscribe event bus backed by asyncio queues.

    Each SSE client subscribes and gets its own queue. Publishers
    fire-and-forget into all queues. Full queues drop the oldest
    message to prevent backpressure from slow clients.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def publish(self, event: str, data: dict) -> None:
        msg = {
            "event": event,
            "data": data,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass
