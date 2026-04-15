"""Tests for PriceMonitor."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

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
