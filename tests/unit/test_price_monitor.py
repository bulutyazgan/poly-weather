"""Tests for PriceMonitor and WebSocketFeed."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.ws_feed import WebSocketFeed

from src.data.models import (
    MarketContract,
    MarketPrice,
    RegimeClassification,
    TradingSignal,
)
from src.orchestrator.price_monitor import PriceMonitor
from src.orchestrator.signal_cache import CachedSignal, SignalCache
from src.trading.exposure_tracker import ExposureTracker


def _make_contract(token_id="tok_yes", no_token_id="tok_no"):
    return MarketContract(
        token_id=token_id,
        no_token_id=no_token_id,
        condition_id="cond_1",
        question="Will NYC high be 72-73°F?",
        city="NYC",
        resolution_date=date(2026, 4, 16),
        end_date_utc=datetime(2026, 4, 16, 23, 59, 59, tzinfo=timezone.utc),
        temp_bucket_low=72.0,
        temp_bucket_high=73.0,
        outcome="Yes",
        volume_24h=5000.0,
    )


def _make_cached_signal(token_id="tok_yes", model_prob=0.40):
    return CachedSignal(
        model_prob=model_prob,
        regime=RegimeClassification(
            station_id="KNYC",
            valid_date=date(2026, 4, 15),
            regime="normal",
            confidence="HIGH",
            ensemble_spread_percentile=30.0,
        ),
        contract=_make_contract(token_id),
        station_id="KNYC",
        forecast_time=datetime.now(timezone.utc),
    )


def _make_price(token_id="tok_yes", bid=0.30, ask=0.35, mid=0.325):
    return MarketPrice(
        token_id=token_id,
        timestamp=datetime.now(timezone.utc),
        bid=bid,
        ask=ask,
        mid=mid,
        volume_24h=0.0,
    )


def _build_monitor(**overrides):
    """Build a PriceMonitor with mocked dependencies."""
    ws_feed = MagicMock()
    signal_cache = SignalCache()
    edge_detector = MagicMock()
    position_sizer = MagicMock()
    exposure_tracker = ExposureTracker()
    executor = MagicMock()
    prediction_log = MagicMock()
    paper_trader = MagicMock()

    defaults = dict(
        ws_feed=ws_feed,
        signal_cache=signal_cache,
        edge_detector=edge_detector,
        position_sizer=position_sizer,
        exposure_tracker=exposure_tracker,
        executor=executor,
        prediction_log=prediction_log,
        paper_trader=paper_trader,
        debounce_seconds=0.0,
        cooldown_seconds=900.0,
        bankroll=300.0,
    )
    defaults.update(overrides)

    monitor = PriceMonitor(**defaults)
    return monitor, defaults


class TestHandlePriceUpdate:
    @pytest.mark.asyncio
    async def test_skip_when_no_cached_signal(self):
        monitor, deps = _build_monitor()
        result = await monitor._handle_price_update("unknown_token")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_forecast_too_old(self):
        monitor, deps = _build_monitor(max_forecast_age_s=100.0)
        sig = _make_cached_signal()
        deps["signal_cache"].update({"tok_yes": sig})
        deps["signal_cache"]._last_update = datetime.now(timezone.utc) - timedelta(hours=3)
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_price_too_old(self):
        """Cached WS price older than max_price_age_s is rejected."""
        monitor, deps = _build_monitor(max_price_age_s=15.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        # Price is 60 seconds old — older than 15s limit
        stale_price = MarketPrice(
            token_id="tok_yes",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
            bid=0.30, ask=0.35, mid=0.325, volume_24h=0.0,
        )
        deps["ws_feed"].get_latest_price.return_value = stale_price
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_edge_detector_says_skip(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="SKIP",
            edge=0.02, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_trade_executes_on_edge(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        result = await monitor._handle_price_update("tok_yes")
        assert result is not None
        deps["executor"].execute.assert_called_once()
        call_args = deps["executor"].execute.call_args
        assert call_args[0][0].kelly_size == 2.50

    @pytest.mark.asyncio
    async def test_exposure_tracker_updated_after_trade(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")
        assert deps["exposure_tracker"].current == 2.50

    @pytest.mark.asyncio
    async def test_buy_no_routes_to_no_token(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.side_effect = lambda tid: {
            "tok_yes": _make_price("tok_yes"),
            "tok_no": _make_price("tok_no", bid=0.65, ask=0.70, mid=0.675),
        }.get(tid)
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_NO", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.00
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")
        call_args = deps["executor"].execute.call_args
        assert call_args[0][1] == "tok_no"

    @pytest.mark.asyncio
    async def test_cusum_alarm_blocks_trade(self):
        cusum = MagicMock()
        cusum.alarm = True
        monitor, deps = _build_monitor(cusum=cusum)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_volume_uses_contract_not_ws(self):
        monitor, deps = _build_monitor()
        cached = _make_cached_signal()
        deps["signal_cache"].update({"tok_yes": cached})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="SKIP",
            edge=0.0, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        await monitor._handle_price_update("tok_yes")
        call_kwargs = deps["edge_detector"].evaluate.call_args[1]
        assert call_kwargs["volume_24h"] == 5000.0


    @pytest.mark.asyncio
    async def test_drawdown_halt_blocks_trade(self):
        """When exposure tracker is halted, price monitor must not trade."""
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        tracker.record_pnl(-50.0)  # Exceed 15% of $300 = $45
        assert tracker.is_halted is True

        monitor, deps = _build_monitor(exposure_tracker=tracker)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_paper_trade_uses_ask_not_mid(self):
        """Paper trade entry_price must be the ask price, not mid.

        Mock price has bid=0.30, ask=0.35, mid=0.325.
        Entry price must be 0.35 (ask).
        """
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")
        call_kwargs = deps["paper_trader"].record_trade.call_args[1]
        assert call_kwargs["entry_price"] == pytest.approx(0.35), (
            f"Expected ask price 0.35, got {call_kwargs['entry_price']}"
        )


    @pytest.mark.asyncio
    async def test_no_token_update_triggers_evaluation_via_yes_price(self):
        """When a NO token update arrives, look up the YES token's live price
        for edge evaluation (since model_prob is always for the YES outcome)."""
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})

        # WS feed returns prices for both tokens; the monitor should
        # request the YES token's price for edge eval, not the NO token.
        yes_price = _make_price("tok_yes", bid=0.30, ask=0.35, mid=0.325)
        no_price = _make_price("tok_no", bid=0.65, ask=0.70, mid=0.675)
        deps["ws_feed"].get_latest_price.side_effect = lambda tid: {
            "tok_yes": yes_price,
            "tok_no": no_price,
        }.get(tid)
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_NO", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.00
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        # Trigger via NO token update
        result = await monitor._handle_price_update("tok_no")
        assert result is not None

        # Edge detector should receive the YES price, not the NO price
        eval_call = deps["edge_detector"].evaluate.call_args
        assert eval_call.kwargs.get("market_prob") == pytest.approx(0.325)
        assert eval_call.kwargs.get("market_bid") == pytest.approx(0.30)
        assert eval_call.kwargs.get("market_ask") == pytest.approx(0.35)


class TestDebounce:
    @pytest.mark.asyncio
    async def test_debounce_blocks_immediate_trade(self):
        monitor, deps = _build_monitor(debounce_seconds=5.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_debounce_clears_on_skip(self):
        monitor, deps = _build_monitor(debounce_seconds=5.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()

        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        await monitor._handle_price_update("tok_yes")
        assert "tok_yes" in monitor._pending_edges

        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="SKIP",
            edge=0.02, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        await monitor._handle_price_update("tok_yes")
        assert "tok_yes" not in monitor._pending_edges


class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_second_trade(self):
        monitor, deps = _build_monitor(cooldown_seconds=900.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        result1 = await monitor._handle_price_update("tok_yes")
        assert result1 is not None

        result2 = await monitor._handle_price_update("tok_yes")
        assert result2 is None
        assert deps["executor"].execute.call_count == 1


    @pytest.mark.asyncio
    async def test_cooldown_covers_both_yes_and_no_tokens(self):
        """Trading via YES token should also cooldown the NO token."""
        monitor, deps = _build_monitor(cooldown_seconds=900.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.side_effect = lambda tid: {
            "tok_yes": _make_price("tok_yes"),
            "tok_no": _make_price("tok_no", bid=0.65, ask=0.70, mid=0.675),
        }.get(tid)
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        # Trade via YES token
        result1 = await monitor._handle_price_update("tok_yes")
        assert result1 is not None

        # NO token should also be cooled down
        assert "tok_no" in monitor._cooldowns
        assert "tok_yes" in monitor._cooldowns


class TestActiveStationCount:
    @pytest.mark.asyncio
    async def test_correlation_discount_applied_with_active_cooldowns(self):
        """When other tokens have active cooldowns, active_station_count > 1."""
        monitor, deps = _build_monitor(cooldown_seconds=900.0)

        # Simulate an existing cooldown from a previous trade on another token
        monitor._cooldowns["other_tok"] = datetime.now(timezone.utc) + timedelta(seconds=600)

        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")

        # active_station_count should be 2 (1 active cooldown + 1 current trade)
        call_kwargs = deps["position_sizer"].compute.call_args[1]
        assert call_kwargs["active_station_count"] == 2

    @pytest.mark.asyncio
    async def test_expired_cooldowns_not_counted(self):
        """Expired cooldowns should not inflate the active station count."""
        monitor, deps = _build_monitor(cooldown_seconds=900.0)

        # Add an expired cooldown (in the past)
        monitor._cooldowns["expired_tok"] = datetime.now(timezone.utc) - timedelta(seconds=10)

        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")

        # Only the current trade counts (expired cooldown excluded)
        call_kwargs = deps["position_sizer"].compute.call_args[1]
        assert call_kwargs["active_station_count"] == 1


class TestWebSocketFeedHandleRaw:
    def test_handle_raw_dict_message(self):
        """Normal dict messages are processed without error."""
        feed = WebSocketFeed()
        feed._handle_raw('{"event_type": "price_change", "asset_id": "tok1", "price": "0.55"}')
        assert feed._queue.qsize() == 1

    def test_handle_raw_list_message_skipped(self):
        """JSON array messages (batch payloads) must not crash."""
        feed = WebSocketFeed()
        feed._handle_raw('[{"event_type": "book"}, {"event_type": "price_change"}]')
        assert feed._queue.qsize() == 0

    def test_handle_raw_invalid_json_skipped(self):
        """Malformed JSON is silently skipped."""
        feed = WebSocketFeed()
        feed._handle_raw("not json at all")
        assert feed._queue.qsize() == 0


class TestWebSocketFeedPhantomPrices:
    """Empty order books must not produce phantom 0.0-bid / 1.0-ask prices."""

    def test_empty_bids_skips_cache_update(self):
        """No bids → best_bid defaults to 0.0 → cache must NOT be updated."""
        feed = WebSocketFeed()
        feed._shadow_bids["tok1"] = []
        feed._shadow_asks["tok1"] = [[0.55, 10.0]]
        feed._update_cache_from_shadow("tok1")
        assert "tok1" not in feed._price_cache

    def test_empty_asks_skips_cache_update(self):
        """No asks → best_ask defaults to 1.0 → cache must NOT be updated."""
        feed = WebSocketFeed()
        feed._shadow_bids["tok1"] = [[0.45, 10.0]]
        feed._shadow_asks["tok1"] = []
        feed._update_cache_from_shadow("tok1")
        assert "tok1" not in feed._price_cache

    def test_crossed_book_skips_cache_update(self):
        """Crossed book (bid >= ask) must not produce a cached price."""
        feed = WebSocketFeed()
        feed._shadow_bids["tok1"] = [[0.60, 10.0]]
        feed._shadow_asks["tok1"] = [[0.55, 10.0]]
        feed._update_cache_from_shadow("tok1")
        assert "tok1" not in feed._price_cache

    def test_valid_book_updates_cache(self):
        """Normal uncrossed book updates the price cache."""
        feed = WebSocketFeed()
        feed._shadow_bids["tok1"] = [[0.48, 10.0]]
        feed._shadow_asks["tok1"] = [[0.52, 10.0]]
        feed._update_cache_from_shadow("tok1")
        assert "tok1" in feed._price_cache
        assert feed._price_cache["tok1"].bid == 0.48
        assert feed._price_cache["tok1"].ask == 0.52
