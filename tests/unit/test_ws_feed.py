"""Tests for the WebSocket order book feed."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.ws_feed import WebSocketFeed, parse_message


# ===================================================================
# parse_message
# ===================================================================

class TestParseMessage:
    def test_valid_json(self):
        msg = parse_message('{"type": "book", "asset_id": "abc"}')
        assert msg == {"type": "book", "asset_id": "abc"}

    def test_invalid_json(self):
        assert parse_message("not json") is None

    def test_none_input(self):
        assert parse_message(None) is None

    def test_empty_string(self):
        assert parse_message("") is None


# ===================================================================
# _handle_raw
# ===================================================================

class TestHandleRaw:
    def test_price_change_updates_cache(self):
        feed = WebSocketFeed()
        raw = json.dumps({
            "event_type": "price_change",
            "asset_id": "tok_1",
            "price": "0.65",
        })
        feed._handle_raw(raw)
        price = feed.get_latest_price("tok_1")
        assert price is not None
        assert price.mid == pytest.approx(0.65)

    def test_book_snapshot_updates_cache(self):
        feed = WebSocketFeed()
        raw = json.dumps({
            "event_type": "book",
            "asset_id": "tok_1",
            "snapshot": True,
            "bids": [["0.58", "100"], ["0.55", "200"]],
            "asks": [["0.62", "150"], ["0.65", "100"]],
        })
        feed._handle_raw(raw)
        price = feed.get_latest_price("tok_1")
        assert price is not None
        assert price.bid == pytest.approx(0.58)
        assert price.ask == pytest.approx(0.62)
        assert price.mid == pytest.approx(0.60)

    def test_book_delta_updates(self):
        feed = WebSocketFeed()
        # First: snapshot
        snap = json.dumps({
            "event_type": "book",
            "asset_id": "tok_1",
            "snapshot": True,
            "bids": [["0.58", "100"]],
            "asks": [["0.62", "150"]],
        })
        feed._handle_raw(snap)

        # Then: delta adding a better bid
        delta = json.dumps({
            "event_type": "book",
            "asset_id": "tok_1",
            "bids": [["0.59", "50"]],
            "asks": [],
        })
        feed._handle_raw(delta)
        price = feed.get_latest_price("tok_1")
        assert price is not None
        assert price.bid == pytest.approx(0.59)

    def test_book_delta_removes_level(self):
        feed = WebSocketFeed()
        snap = json.dumps({
            "event_type": "book",
            "asset_id": "tok_1",
            "snapshot": True,
            "bids": [["0.58", "100"], ["0.55", "200"]],
            "asks": [["0.62", "150"]],
        })
        feed._handle_raw(snap)

        # Remove the 0.58 bid (size=0)
        delta = json.dumps({
            "event_type": "book",
            "asset_id": "tok_1",
            "bids": [["0.58", "0"]],
            "asks": [],
        })
        feed._handle_raw(delta)
        price = feed.get_latest_price("tok_1")
        assert price is not None
        assert price.bid == pytest.approx(0.55)

    def test_bytes_input(self):
        feed = WebSocketFeed()
        raw = json.dumps({
            "event_type": "price_change",
            "asset_id": "tok_1",
            "price": "0.50",
        }).encode("utf-8")
        feed._handle_raw(raw)
        assert feed.get_latest_price("tok_1") is not None

    def test_invalid_message_ignored(self):
        feed = WebSocketFeed()
        feed._handle_raw("not valid json")
        assert feed.get_latest_price("tok_1") is None

    def test_unknown_event_still_queued(self):
        feed = WebSocketFeed()
        raw = json.dumps({"type": "heartbeat"})
        feed._handle_raw(raw)
        assert not feed._queue.empty()


# ===================================================================
# get_latest_price
# ===================================================================

class TestGetLatestPrice:
    def test_returns_none_when_empty(self):
        feed = WebSocketFeed()
        assert feed.get_latest_price("tok_nonexistent") is None

    def test_returns_cached_price(self):
        feed = WebSocketFeed()
        raw = json.dumps({
            "event_type": "price_change",
            "asset_id": "tok_1",
            "price": "0.72",
        })
        feed._handle_raw(raw)
        p = feed.get_latest_price("tok_1")
        assert p is not None
        assert p.mid == pytest.approx(0.72)


# ===================================================================
# subscribe / close
# ===================================================================

class TestSubscribeStartsTask:
    @pytest.mark.asyncio
    async def test_subscribe_creates_task(self):
        feed = WebSocketFeed()
        # Patch _run_with_reconnect to avoid real connections
        with patch.object(feed, "_run_with_reconnect", new_callable=AsyncMock):
            await feed.subscribe(["tok_1", "tok_2"])
            assert feed._task is not None
            assert feed._token_ids == ["tok_1", "tok_2"]
            await feed.close()


class TestClose:
    @pytest.mark.asyncio
    async def test_close_sets_flag(self):
        feed = WebSocketFeed()
        await feed.close()
        assert feed._closed is True

    @pytest.mark.asyncio
    async def test_close_cancels_task(self):
        feed = WebSocketFeed()
        with patch.object(feed, "_run_with_reconnect", new_callable=AsyncMock):
            await feed.subscribe(["tok_1"])
            assert feed._task is not None
            await feed.close()
            assert feed._closed is True


# ===================================================================
# listen queue
# ===================================================================

class TestListenQueue:
    @pytest.mark.asyncio
    async def test_listen_yields_from_queue(self):
        feed = WebSocketFeed()
        feed._queue.put_nowait({"type": "test", "data": 42})
        feed._queue.put_nowait({"type": "test", "data": 43})

        results = []
        count = 0
        async for update in feed.listen():
            results.append(update)
            count += 1
            if count >= 2:
                feed._closed = True
                break

        assert len(results) == 2
        assert results[0]["data"] == 42
        assert results[1]["data"] == 43


# ===================================================================
# Reconnect backoff
# ===================================================================

class TestReconnectBackoff:
    @pytest.mark.asyncio
    async def test_backoff_increases_on_failure(self):
        feed = WebSocketFeed()
        call_count = 0

        async def _failing_connect():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                feed._closed = True
                return
            raise ConnectionError("test")

        with patch.object(feed, "_connect_and_listen", side_effect=_failing_connect):
            with patch("src.data.ws_feed.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await feed._run_with_reconnect()
                # Should have slept twice (before 2nd and 3rd attempts)
                assert mock_sleep.await_count == 2
                # Backoff should increase: 1.0, 2.0
                calls = [c.args[0] for c in mock_sleep.await_args_list]
                assert calls[0] == pytest.approx(1.0)
                assert calls[1] == pytest.approx(2.0)


# ===================================================================
# Connection errors
# ===================================================================

class TestResubscribe:
    @pytest.mark.asyncio
    async def test_resubscribe_updates_token_ids(self):
        feed = WebSocketFeed()
        feed._token_ids = ["old_token"]
        feed._task = asyncio.create_task(asyncio.sleep(999))
        await feed.resubscribe(["new_token_1", "new_token_2"])
        assert feed._token_ids == ["new_token_1", "new_token_2"]
        await feed.close()

    @pytest.mark.asyncio
    async def test_resubscribe_cancels_old_task(self):
        feed = WebSocketFeed()
        old_task = asyncio.create_task(asyncio.sleep(999))
        feed._task = old_task
        with patch.object(feed, "_run_with_reconnect", new_callable=AsyncMock):
            await feed.resubscribe(["tok_1"])
        assert old_task.cancelled() or old_task.done()
        await feed.close()

    @pytest.mark.asyncio
    async def test_resubscribe_starts_new_task(self):
        feed = WebSocketFeed()
        assert feed._task is None
        with patch.object(feed, "_run_with_reconnect", new_callable=AsyncMock):
            await feed.resubscribe(["tok_1"])
        assert feed._task is not None
        await feed.close()


# ===================================================================
# Connection errors
# ===================================================================

class TestConnectionErrors:
    @pytest.mark.asyncio
    async def test_import_error_triggers_reconnect(self):
        """If websockets not installed, ImportError is caught by reconnect loop."""
        feed = WebSocketFeed()
        call_count = 0

        async def _import_fail():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                feed._closed = True
                return
            raise ImportError("No module named 'websockets'")

        with patch.object(feed, "_connect_and_listen", side_effect=_import_fail):
            with patch("src.data.ws_feed.asyncio.sleep", new_callable=AsyncMock):
                await feed._run_with_reconnect()
                assert call_count == 2
