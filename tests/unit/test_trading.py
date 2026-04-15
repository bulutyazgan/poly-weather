"""Tests for the trading engine: EdgeDetector, PositionSizer, OrderExecutor."""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import (
    MarketPrice,
    RegimeClassification,
    TradingSignal,
    TradeRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regime(
    confidence: str = "HIGH",
    confidence_score: float = 0.9,
    ensemble_spread_percentile: float = 50.0,
) -> RegimeClassification:
    return RegimeClassification(
        station_id="KNYC",
        valid_date=date(2026, 4, 16),
        regime="zonal",
        confidence=confidence,
        confidence_score=confidence_score,
        ensemble_spread_percentile=ensemble_spread_percentile,
    )


def _make_market_price(
    token_id: str = "tok_abc",
    bid: float = 0.58,
    ask: float = 0.62,
    volume_24h: float = 5000.0,
) -> MarketPrice:
    return MarketPrice(
        token_id=token_id,
        timestamp=datetime.now(timezone.utc),
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        volume_24h=volume_24h,
    )


# ===================================================================
# EdgeDetector tests
# ===================================================================

class TestEdgeDetector:
    """Test suite for EdgeDetector.evaluate()."""

    def _make_detector(self):
        from src.trading.edge_detector import EdgeDetector
        return EdgeDetector()

    def test_detect_edge_trade_signal(self):
        """model_prob=0.75, market_prob=0.60, HIGH regime -> edge=0.15, TRADE, BUY_YES."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.75,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "TRADE"
        assert sig.direction == "BUY_YES"
        assert abs(sig.edge - 0.15) < 1e-9

    def test_detect_edge_skip_insufficient(self):
        """model_prob=0.65, market_prob=0.60, HIGH regime -> edge=0.05 < 0.08 -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.65,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"
        assert abs(sig.edge - 0.05) < 1e-9

    def test_detect_edge_skip_low_regime(self):
        """Even with large edge, LOW regime -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.90,
            market_prob=0.50,
            regime=_make_regime(confidence="LOW"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"

    def test_detect_edge_medium_regime_higher_threshold(self):
        """model_prob=0.70, market_prob=0.62, MEDIUM -> edge=0.08 < 0.12 -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.70,
            market_prob=0.62,
            regime=_make_regime(confidence="MEDIUM"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"
        assert abs(sig.edge - 0.08) < 1e-9

    def test_detect_edge_buy_no(self):
        """model_prob=0.30, market_prob=0.55 -> BUY_NO, edge=0.25."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.30,
            market_prob=0.55,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "TRADE"
        assert sig.direction == "BUY_NO"
        assert abs(sig.edge - 0.25) < 1e-9

    def test_detect_edge_skip_low_volume(self):
        """volume < MIN_MARKET_VOLUME -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.80,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=500.0,  # below 2000
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"

    def test_detect_edge_skip_too_close_to_resolution(self):
        """hours_to_resolution < 2 -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.80,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=1.5,
        )
        assert sig.action == "SKIP"


# ===================================================================
# PositionSizer tests
# ===================================================================

class TestPositionSizer:
    """Test suite for PositionSizer.compute()."""

    def _make_sizer(self):
        from src.trading.position_sizer import PositionSizer
        return PositionSizer()

    def test_kelly_sizing_basic(self):
        """edge=0.10, bankroll=300 -> kelly = 0.10/(1-0.60)=0.25, raw=0.08*0.25*300=6.0, capped at $3."""
        sizer = self._make_sizer()
        size = sizer.compute(edge=0.10, market_prob=0.60, bankroll=300.0)
        # kelly = 0.10 / 0.40 = 0.25
        # raw = 0.08 * 0.25 * 300 = 6.0
        # capped by MAX_TRADE_USD = 3.0
        assert size == pytest.approx(3.0)

    def test_kelly_sizing_capped_by_max_trade(self):
        """Large edge -> capped at $3."""
        sizer = self._make_sizer()
        size = sizer.compute(edge=0.50, market_prob=0.50, bankroll=1000.0)
        assert size == pytest.approx(3.0)

    def test_kelly_sizing_capped_by_bankroll_pct(self):
        """Position capped at 3% of bankroll when bankroll is small."""
        sizer = self._make_sizer()
        # bankroll=50 -> 3% = 1.5
        # kelly = 0.10/0.40 = 0.25, raw = 0.08*0.25*50 = 1.0
        # 1.0 < 1.5 and < 3.0, so not capped
        # Use a scenario where bankroll_pct is the binding constraint:
        # bankroll=20 -> 3% = 0.60
        # kelly = 0.30/0.40 = 0.75, raw = 0.08*0.75*20 = 1.2
        # min(1.2, 3.0, 0.60) = 0.60
        size = sizer.compute(edge=0.30, market_prob=0.60, bankroll=20.0)
        assert size == pytest.approx(0.60)

    def test_kelly_sizing_zero_edge(self):
        """edge=0 -> position=0."""
        sizer = self._make_sizer()
        size = sizer.compute(edge=0.0, market_prob=0.60, bankroll=300.0)
        assert size == 0.0

    def test_kelly_sizing_negative_edge(self):
        """edge<0 -> position=0."""
        sizer = self._make_sizer()
        size = sizer.compute(edge=-0.05, market_prob=0.60, bankroll=300.0)
        assert size == 0.0

    def test_portfolio_exposure_check(self):
        """$50 already exposed, $300 bankroll, 20% max -> only $10 more allowed."""
        sizer = self._make_sizer()
        # max_portfolio_exposure * bankroll - current_exposure = 0.20*300 - 50 = 10
        # kelly = 0.30/0.40 = 0.75, raw = 0.08*0.75*300 = 18.0
        # min(18.0, 3.0, 9.0, 10.0) = 3.0  (MAX_TRADE_USD still binding)
        # Need a scenario where exposure is the binding constraint:
        # current_exposure=57 -> remaining = 60-57 = 3.0 ... still tied with max_trade
        # current_exposure=58 -> remaining = 2.0
        size = sizer.compute(
            edge=0.30, market_prob=0.60, bankroll=300.0, current_exposure=58.0
        )
        assert size == pytest.approx(2.0)

    def test_kelly_spread_scaling(self):
        """Low ensemble spread -> 1.2x multiplier; high spread -> 0.8x."""
        sizer = self._make_sizer()
        # Low spread (pctile=10): kelly=0.10/0.40=0.25, raw=0.08*0.25*300=6.0, *1.2=7.2, capped at 3.0
        size_low = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=300.0, ensemble_spread_pctile=10.0
        )
        # High spread (pctile=60): raw=6.0, *0.8=4.8, capped at 3.0
        size_high = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=300.0, ensemble_spread_pctile=60.0
        )
        # Both capped at 3.0 in this case. Use smaller bankroll:
        # bankroll=30, 3%=0.90
        # low spread: raw=0.08*0.25*30=0.60, *1.2=0.72
        # high spread: raw=0.60, *0.8=0.48
        size_low = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, ensemble_spread_pctile=10.0
        )
        size_high = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, ensemble_spread_pctile=60.0
        )
        assert size_low > size_high
        assert size_low == pytest.approx(0.72)
        assert size_high == pytest.approx(0.48)


# ===================================================================
# OrderExecutor tests
# ===================================================================

class TestOrderExecutor:
    """Test suite for OrderExecutor (mocked CLOBClient)."""

    def _make_executor(self, mock_clob=None):
        from src.trading.executor import OrderExecutor
        if mock_clob is None:
            mock_clob = AsyncMock()
            mock_clob.paper_trading = True
            mock_clob.place_limit_order = AsyncMock(return_value="order-123")
            mock_clob.cancel_all_orders = AsyncMock(return_value=2)
        return OrderExecutor(clob_client=mock_clob, paper_trading=True), mock_clob

    @pytest.mark.asyncio
    async def test_execute_trade_paper_mode(self):
        """TRADE signal -> places order via CLOBClient, returns TradeRecord."""
        executor, mock_clob = self._make_executor()
        signal = TradingSignal(
            market_id="market_1",
            direction="BUY_YES",
            action="TRADE",
            edge=0.15,
            kelly_size=2.50,
            timestamp=datetime.now(timezone.utc),
        )
        price = _make_market_price()
        record = await executor.execute(signal, token_id="tok_abc", market_price=price)

        assert record is not None
        assert isinstance(record, TradeRecord)
        assert record.direction == "BUY_YES"
        assert record.amount_usd == 2.50
        mock_clob.place_limit_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_skip_signal(self):
        """SKIP signal -> no order placed, returns None."""
        executor, mock_clob = self._make_executor()
        signal = TradingSignal(
            market_id="market_1",
            direction="BUY_YES",
            action="SKIP",
            edge=0.02,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        price = _make_market_price()
        record = await executor.execute(signal, token_id="tok_abc", market_price=price)

        assert record is None
        mock_clob.place_limit_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_quote_detection(self):
        """Model update >3h old and market moved >3% -> cancel all orders."""
        executor, mock_clob = self._make_executor()
        old_time = datetime.now(timezone.utc) - timedelta(hours=4)
        current_prices = {"tok_abc": 0.65}
        previous_prices = {"tok_abc": 0.60}  # 8.3% move

        cancelled = await executor.check_stale_quotes(
            last_model_update=old_time,
            current_prices=current_prices,
            previous_prices=previous_prices,
        )
        assert cancelled is True
        mock_clob.cancel_all_orders.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_before_resolution(self):
        """<4h to resolution -> cancel all orders."""
        executor, mock_clob = self._make_executor()
        cancelled = await executor.check_resolution_proximity(hours_to_resolution=3.5)
        assert cancelled is True
        mock_clob.cancel_all_orders.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_cancel_when_not_stale(self):
        """Recent model update -> no cancellation."""
        executor, mock_clob = self._make_executor()
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        current_prices = {"tok_abc": 0.65}
        previous_prices = {"tok_abc": 0.60}

        cancelled = await executor.check_stale_quotes(
            last_model_update=recent_time,
            current_prices=current_prices,
            previous_prices=previous_prices,
        )
        assert cancelled is False
        mock_clob.cancel_all_orders.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_cancel_far_from_resolution(self):
        """Far from resolution -> no cancellation."""
        executor, mock_clob = self._make_executor()
        cancelled = await executor.check_resolution_proximity(hours_to_resolution=10.0)
        assert cancelled is False
        mock_clob.cancel_all_orders.assert_not_awaited()
