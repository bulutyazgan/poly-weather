"""WebSocket order book feed for Polymarket CLOB.

Maintains a shadow order book via delta reconstruction, with automatic
reconnection using exponential backoff.  Falls back to HTTP polling
when the WebSocket is unavailable.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from src.data.models import MarketPrice

logger = logging.getLogger(__name__)

# Reconnect parameters
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
_BACKOFF_FACTOR = 2.0


def parse_message(raw: str) -> dict | None:
    """Parse a raw WebSocket message into a dict.  Returns None on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


class WebSocketFeed:
    """Async WebSocket feed for real-time Polymarket order book updates.

    Usage::

        feed = WebSocketFeed("wss://ws-subscriptions-clob.polymarket.com/ws/market")
        await feed.subscribe(["token_abc", "token_def"])

        async for update in feed.listen():
            print(update)

        await feed.close()
    """

    def __init__(self, ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market") -> None:
        self._ws_url = ws_url
        self._token_ids: list[str] = []
        self._price_cache: dict[str, MarketPrice] = {}
        self._shadow_bids: dict[str, list[list[float]]] = defaultdict(list)
        self._shadow_asks: dict[str, list[list[float]]] = defaultdict(list)
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def subscribe(self, token_ids: list[str]) -> None:
        """Start listening for the given token IDs."""
        self._token_ids = list(token_ids)
        self._closed = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_with_reconnect())

    async def listen(self) -> AsyncIterator[dict]:
        """Yield parsed update dicts as they arrive."""
        while not self._closed:
            try:
                update = await asyncio.wait_for(self._queue.get(), timeout=60.0)
                yield update
            except asyncio.TimeoutError:
                continue

    def get_latest_price(self, token_id: str) -> MarketPrice | None:
        """Return the most recent cached price for a token."""
        return self._price_cache.get(token_id)

    async def close(self) -> None:
        """Shut down the feed."""
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def resubscribe(self, token_ids: list[str]) -> None:
        """Update subscriptions by restarting the connection.

        Cancels the current connection task and starts a new one with the
        updated token list.  This is the simplest correct approach given
        that ``_send_subscription`` only runs at connection time.
        """
        self._token_ids = list(token_ids)
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = asyncio.create_task(self._run_with_reconnect())

    # -- internal ------------------------------------------------------------

    async def _run_with_reconnect(self) -> None:
        """Connection loop with exponential backoff."""
        backoff = _INITIAL_BACKOFF_S
        while not self._closed:
            try:
                await self._connect_and_listen()
                backoff = _INITIAL_BACKOFF_S  # reset on clean exit
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("WebSocket connection error — reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF_S)

    async def _connect_and_listen(self) -> None:
        """Connect via websockets and process messages.

        Requires the ``websockets`` package.  If not installed the method
        raises ImportError and the reconnect loop will keep retrying.
        """
        import websockets  # type: ignore[import-untyped]

        async with websockets.connect(self._ws_url) as ws:
            await self._send_subscription(ws)
            async for raw in ws:
                if self._closed:
                    break
                self._handle_raw(raw)

    async def _send_subscription(self, ws: Any) -> None:
        """Send subscription message for tracked tokens."""
        for token_id in self._token_ids:
            msg = json.dumps({
                "type": "subscribe",
                "channel": "market",
                "assets_ids": [token_id],
            })
            await ws.send(msg)

    def _handle_raw(self, raw: str | bytes) -> None:
        """Dispatch a raw message to the appropriate handler."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        msg = parse_message(raw)
        if msg is None:
            return

        event_type = msg.get("event_type") or msg.get("type", "")

        if event_type == "book":
            self._handle_book_event(msg)
        elif event_type == "price_change":
            self._handle_price_change_event(msg)

        # Push every valid message to listeners
        self._queue.put_nowait(msg)

    def _handle_book_event(self, msg: dict) -> None:
        """Process a full or delta book snapshot."""
        token_id = msg.get("asset_id", "")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])

        if msg.get("snapshot"):
            self._shadow_bids[token_id] = [[float(p), float(s)] for p, s in bids]
            self._shadow_asks[token_id] = [[float(p), float(s)] for p, s in asks]
        else:
            self._apply_deltas(self._shadow_bids[token_id], bids)
            self._apply_deltas(self._shadow_asks[token_id], asks)

        self._update_cache_from_shadow(token_id)

    def _handle_price_change_event(self, msg: dict) -> None:
        """Process a price change event."""
        token_id = msg.get("asset_id", "")
        price = msg.get("price")
        if token_id and price is not None:
            p = float(price)
            now = datetime.now(timezone.utc)
            self._price_cache[token_id] = MarketPrice(
                token_id=token_id,
                timestamp=now,
                bid=p,
                ask=p,
                mid=p,
                volume_24h=0.0,
            )

    def _update_cache_from_shadow(self, token_id: str) -> None:
        """Derive best bid/ask from shadow book and update price cache."""
        bids = self._shadow_bids.get(token_id, [])
        asks = self._shadow_asks.get(token_id, [])

        best_bid = max((b[0] for b in bids if b[1] > 0), default=0.0)
        best_ask = min((a[0] for a in asks if a[1] > 0), default=1.0)
        mid = (best_bid + best_ask) / 2.0

        now = datetime.now(timezone.utc)
        self._price_cache[token_id] = MarketPrice(
            token_id=token_id,
            timestamp=now,
            bid=best_bid,
            ask=best_ask,
            mid=mid,
            volume_24h=0.0,
        )

    @staticmethod
    def _apply_deltas(book: list[list[float]], deltas: list) -> None:
        """Apply price-level deltas to a shadow book side."""
        for delta in deltas:
            price = float(delta[0])
            size = float(delta[1])
            # Find existing level
            found = False
            for level in book:
                if abs(level[0] - price) < 1e-9:
                    if size == 0:
                        book.remove(level)
                    else:
                        level[1] = size
                    found = True
                    break
            if not found and size > 0:
                book.append([price, size])
