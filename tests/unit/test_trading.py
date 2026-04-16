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

    def _make_detector(self, **kwargs):
        from src.trading.edge_detector import EdgeDetector
        # Default to zero fees and wide spread for legacy tests so hand-calculations remain valid
        kwargs.setdefault("taker_fee_rate", 0.0)
        kwargs.setdefault("max_spread", 1.0)
        return EdgeDetector(**kwargs)

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
        """model_prob=0.63, market_prob=0.60, HIGH regime -> edge=0.03 < 0.05 -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.63,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"
        assert abs(sig.edge - 0.03) < 1e-9

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
        """model_prob=0.35, market_prob=0.55 -> BUY_NO, edge=0.20."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.35,
            market_prob=0.55,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "TRADE"
        assert sig.direction == "BUY_NO"
        assert abs(sig.edge - 0.20) < 1e-9

    def test_detect_edge_skip_low_volume(self):
        """volume < MIN_MARKET_VOLUME -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.80,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=100.0,  # below 500
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

    def test_skip_foregone_conclusion_high_market_prob_model_agrees(self):
        """market_prob=0.95, model=0.96 -> both agree near-certain YES -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.96,
            market_prob=0.95,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.94,
            market_ask=0.96,
        )
        assert sig.action == "SKIP"
        assert sig.skip_reason == "foregone_conclusion"

    def test_trade_foregone_high_market_prob_model_disagrees(self):
        """market_prob=0.75, model=0.60 -> model disagrees with high market -> allow BUY_NO."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.60,
            market_prob=0.75,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.74,
            market_ask=0.76,
        )
        assert sig.action == "TRADE"
        assert sig.direction == "BUY_NO"

    def test_skip_foregone_conclusion_low_market_prob_model_agrees(self):
        """market_prob=0.04, model=0.03 -> both agree near-zero -> SKIP."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.03,
            market_prob=0.04,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.03,
            market_ask=0.05,
        )
        assert sig.action == "SKIP"
        assert sig.skip_reason == "foregone_conclusion"

    def test_trade_foregone_low_market_prob_model_disagrees(self):
        """market_prob=0.25, model=0.40 -> model disagrees with low market -> allow BUY_YES."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.40,
            market_prob=0.25,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.24,
            market_ask=0.26,
        )
        assert sig.action == "TRADE"
        assert sig.direction == "BUY_YES"

    def test_trade_allowed_at_boundary(self):
        """market_prob=0.85 -> boundary test with bid/ask, edge above threshold -> TRADE."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.70,
            market_prob=0.85,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.84,
            market_ask=0.86,
        )
        # BUY_NO: exec_price = 1.0 - 0.84 = 0.16 > 0.10
        # edge = bid - model - fee = 0.84 - 0.70 - 0.003 = 0.137 >= 0.08
        assert sig.action == "TRADE"

    def test_trade_allowed_moderate_market_prob(self):
        """market_prob=0.80 is well within range -> normal edge logic applies."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.65,
            market_prob=0.80,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "TRADE"
        assert sig.direction == "BUY_NO"

    def test_custom_certainty_threshold(self):
        """Custom max_market_certainty=0.85 -> tighter guard, but only when model agrees."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(max_market_certainty=0.85, taker_fee_rate=0.0)
        # Model agrees with extreme market → SKIP
        sig = det.evaluate(
            model_prob=0.90,
            market_prob=0.90,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.89,
            market_ask=0.91,
        )
        assert sig.action == "SKIP"
        assert sig.skip_reason == "foregone_conclusion"
        # Model disagrees with extreme market → TRADE (BUY_NO)
        sig2 = det.evaluate(
            model_prob=0.70,
            market_prob=0.90,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.89,
            market_ask=0.91,
        )
        assert sig2.action == "TRADE"
        assert sig2.direction == "BUY_NO"

    def test_skip_implausible_edge(self):
        """Edge > max_edge (default 25%) is rejected as model failure."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(taker_fee_rate=0.0)
        # 0.005 model_prob vs 0.50 market = 49.5% edge -> SKIP
        sig = det.evaluate(
            model_prob=0.005,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"
        assert sig.edge > 0.25

    def test_allow_plausible_edge(self):
        """Edge within max_edge passes through to normal threshold check."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(taker_fee_rate=0.0)
        # 0.40 model vs 0.50 market = 10% edge -> TRADE (HIGH regime, threshold 8%)
        sig = det.evaluate(
            model_prob=0.40,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "TRADE"

    def test_custom_max_edge(self):
        """Custom max_edge=0.15 -> tighter implausibility guard."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(max_edge=0.15, taker_fee_rate=0.0)
        # 0.30 model vs 0.50 market = 20% edge -> SKIP with max_edge=0.15
        sig = det.evaluate(
            model_prob=0.30,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "SKIP"

    def test_implausible_raw_edge_masked_by_fees(self):
        """Raw edge > max_edge must be caught even when fees reduce net edge below cap.

        Without this fix, a raw edge of 0.27 with fee ~0.01 would yield a net
        edge of 0.26 and sneak past a max_edge=0.25 check on net edge.
        """
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(max_edge=0.25, taker_fee_rate=0.02)
        # model=0.77, ask=0.50 -> raw_edge = 0.27, fee ≈ 0.02*0.50 = 0.01
        # net_edge = 0.26 (would pass a net-edge check at 0.25)
        # raw_edge = 0.27 (must trigger implausible_edge)
        sig = det.evaluate(
            model_prob=0.77,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.48,
            market_ask=0.50,
        )
        assert sig.action == "SKIP"
        assert sig.skip_reason == "implausible_edge"

    def test_wide_spread_blocks_trade(self):
        """Spread > max_spread (default 10%) skips to avoid illiquid slippage."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(taker_fee_rate=0.0, max_spread=0.10)
        # bid=0.30, ask=0.45 -> spread=0.15 > 0.10
        sig = det.evaluate(
            model_prob=0.60,
            market_prob=0.375,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.30,
            market_ask=0.45,
        )
        assert sig.action == "SKIP"
        assert sig.skip_reason == "wide_spread"

    def test_narrow_spread_allows_trade(self):
        """Spread within max_spread allows normal evaluation."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(taker_fee_rate=0.0, max_spread=0.10)
        # bid=0.48, ask=0.52 -> spread=0.04 < 0.10
        sig = det.evaluate(
            model_prob=0.65,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.48,
            market_ask=0.52,
        )
        assert sig.action == "TRADE"

    def test_spread_kills_edge_skip_reason(self):
        """When raw edge > 0 but fees make net edge negative, skip_reason is 'spread_kills_edge'."""
        from src.trading.edge_detector import EdgeDetector
        det = EdgeDetector(taker_fee_rate=0.02, max_spread=1.0)
        # model=0.51, ask=0.52: raw_edge = 0.51 - 0.52 = -0.01
        # But model > mid (0.50), so direction is BUY_YES
        # raw_edge is negative → spread_killed is True (raw_edge < 0 means
        # raw_edge not > 0, so spread_killed is False)
        # Actually: model=0.535, ask=0.53: raw_edge = 0.005, fee = 0.02*0.47 = 0.0094
        # net_edge = 0.005 - 0.0094 = -0.0044 → clamped to 0
        # spread_killed = True (raw > 0 but net < 0)
        sig = det.evaluate(
            model_prob=0.535,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.47,
            market_ask=0.53,
        )
        assert sig.action == "SKIP"
        assert sig.skip_reason == "spread_kills_edge"

    def test_spread_aware_edge_buy_yes(self):
        """BUY_YES edge computed against ask, not mid."""
        det = self._make_detector()
        # model=0.40, bid=0.28, ask=0.32, mid=0.30
        # Old: edge = |0.40 - 0.30| = 0.10
        # New: edge = 0.40 - 0.32 = 0.08 (against ask)
        sig = det.evaluate(
            model_prob=0.40,
            market_prob=0.30,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.28,
            market_ask=0.32,
        )
        assert sig.direction == "BUY_YES"
        assert abs(sig.edge - 0.08) < 1e-9
        assert sig.action == "TRADE"

    def test_spread_aware_edge_buy_no(self):
        """BUY_NO edge computed against bid, not mid."""
        det = self._make_detector()
        # model=0.30, bid=0.40, ask=0.44, mid=0.42
        # Old: edge = |0.30 - 0.42| = 0.12
        # New: edge = 0.40 - 0.30 = 0.10 (against bid)
        sig = det.evaluate(
            model_prob=0.30,
            market_prob=0.42,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.40,
            market_ask=0.44,
        )
        assert sig.direction == "BUY_NO"
        assert abs(sig.edge - 0.10) < 1e-9
        assert sig.action == "TRADE"

    def test_spread_eats_edge(self):
        """When spread consumes the entire edge, signal is SKIP."""
        det = self._make_detector()
        # model=0.33, bid=0.30, ask=0.36, mid=0.33
        # BUY_YES: model > mid, but model < ask -> edge = 0
        sig = det.evaluate(
            model_prob=0.34,
            market_prob=0.33,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.30,
            market_ask=0.36,
        )
        assert sig.edge == 0.0
        assert sig.action == "SKIP"

    # --- Taker fee tests ---

    def test_fee_reduces_buy_yes_edge(self):
        """2% taker fee reduces BUY_YES edge by fee_rate * min(ask, 1-ask)."""
        det = self._make_detector(taker_fee_rate=0.02)
        # model=0.40, ask=0.32, mid=0.30
        # raw_edge = 0.40 - 0.32 = 0.08
        # fee = 0.02 * min(0.32, 0.68) = 0.02 * 0.32 = 0.0064
        # net_edge = 0.08 - 0.0064 = 0.0736
        sig = det.evaluate(
            model_prob=0.40,
            market_prob=0.30,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.28,
            market_ask=0.32,
        )
        assert sig.direction == "BUY_YES"
        assert abs(sig.edge - 0.0736) < 1e-9
        # 0.0736 > 0.05 HIGH threshold -> TRADE (fee reduced but still above)
        assert sig.action == "TRADE"

    def test_fee_reduces_buy_no_edge(self):
        """2% taker fee reduces BUY_NO edge by fee_rate * min(bid, 1-bid)."""
        det = self._make_detector(taker_fee_rate=0.02)
        # model=0.30, bid=0.40, ask=0.44, mid=0.42
        # raw_edge = 0.40 - 0.30 = 0.10
        # fee = 0.02 * min(0.40, 0.60) = 0.02 * 0.40 = 0.008
        # net_edge = 0.10 - 0.008 = 0.092
        sig = det.evaluate(
            model_prob=0.30,
            market_prob=0.42,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.40,
            market_ask=0.44,
        )
        assert sig.direction == "BUY_NO"
        assert abs(sig.edge - 0.092) < 1e-9
        assert sig.action == "TRADE"

    def test_fee_turns_marginal_trade_to_skip(self):
        """A trade barely above threshold without fees becomes SKIP with fees."""
        det = self._make_detector(taker_fee_rate=0.02)
        # model=0.66, ask=0.62, mid=0.60
        # raw_edge = 0.66 - 0.62 = 0.04
        # fee = 0.02 * min(0.62, 0.38) = 0.02 * 0.38 = 0.0076
        # net_edge = 0.04 - 0.0076 = 0.0324 < 0.05 -> SKIP
        sig = det.evaluate(
            model_prob=0.66,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.58,
            market_ask=0.62,
        )
        assert sig.action == "SKIP"
        assert sig.edge < 0.05

    def test_fee_on_fallback_mid_price(self):
        """When no bid/ask, fee is based on market_prob (mid)."""
        det = self._make_detector(taker_fee_rate=0.02)
        # model=0.75, market_prob=0.60, no bid/ask
        # raw_edge = |0.75 - 0.60| = 0.15
        # fee = 0.02 * min(0.60, 0.40) = 0.02 * 0.40 = 0.008
        # net_edge = 0.15 - 0.008 = 0.142
        sig = det.evaluate(
            model_prob=0.75,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert abs(sig.edge - 0.142) < 1e-9
        assert sig.action == "TRADE"

    def test_fee_plus_spread_eats_edge(self):
        """When fees + spread together consume the edge, result is zero."""
        det = self._make_detector(taker_fee_rate=0.10)  # exaggerated fee
        # model=0.35, ask=0.34, mid=0.33
        # raw_edge = 0.35 - 0.34 = 0.01
        # fee = 0.10 * min(0.34, 0.66) = 0.034
        # net_edge = 0.01 - 0.034 = -0.024 -> clamped to 0
        sig = det.evaluate(
            model_prob=0.35,
            market_prob=0.33,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.32,
            market_ask=0.34,
        )
        assert sig.edge == 0.0
        assert sig.action == "SKIP"

    def test_zero_fee_matches_legacy_behavior(self):
        """taker_fee_rate=0 produces identical results to pre-fee code."""
        det = self._make_detector(taker_fee_rate=0.0)
        sig = det.evaluate(
            model_prob=0.40,
            market_prob=0.30,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.28,
            market_ask=0.32,
        )
        assert abs(sig.edge - 0.08) < 1e-9
        assert sig.action == "TRADE"

    # --- skip_reason tests ---

    def test_skip_reason_trade_is_empty(self):
        """TRADE signals have empty skip_reason."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.75,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.action == "TRADE"
        assert sig.skip_reason == ""

    def test_skip_reason_low_confidence(self):
        """Edge=0.10 is plausible (<0.25) but LOW regime → skip."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.60,
            market_prob=0.50,
            regime=_make_regime(confidence="LOW"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.skip_reason == "low_confidence_regime"

    def test_skip_reason_foregone_conclusion(self):
        det = self._make_detector()
        # Both model and market agree on extreme probability → foregone conclusion
        sig = det.evaluate(
            model_prob=0.96,
            market_prob=0.95,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
            market_bid=0.94,
            market_ask=0.96,
        )
        assert sig.skip_reason == "foregone_conclusion"

    def test_skip_reason_low_volume(self):
        """Edge=0.15 is plausible, HIGH regime, but volume too low."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.65,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=100.0,
            hours_to_resolution=24.0,
        )
        assert sig.skip_reason == "low_volume"

    def test_skip_reason_implausible_edge(self):
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.005,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.skip_reason == "implausible_edge"

    def test_skip_reason_below_threshold(self):
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.63,
            market_prob=0.60,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=24.0,
        )
        assert sig.skip_reason == "below_threshold"

    def test_skip_reason_too_close(self):
        """Edge=0.15 is plausible, HIGH regime, good volume, but <2h to resolution."""
        det = self._make_detector()
        sig = det.evaluate(
            model_prob=0.65,
            market_prob=0.50,
            regime=_make_regime(confidence="HIGH"),
            volume_24h=5000.0,
            hours_to_resolution=1.5,
        )
        assert sig.skip_reason == "too_close_to_resolution"


# ===================================================================
# PositionSizer tests
# ===================================================================

class TestPositionSizer:
    """Test suite for PositionSizer.compute()."""

    def _make_sizer(self, **kwargs):
        from src.trading.position_sizer import PositionSizer
        # Default to zero min for legacy tests so hand-calculations remain valid
        kwargs.setdefault("min_trade_usd", 0.0)
        return PositionSizer(**kwargs)

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

    def test_buy_no_uses_market_prob_denominator(self):
        """BUY_NO: kelly = edge / market_prob (NOT 1 - market_prob).

        Hand calculation:
            edge=0.10, market_prob=0.80, bankroll=300
            BUY_NO denom = market_prob = 0.80
            kelly = 0.10 / 0.80 = 0.125
            raw = 0.08 * 0.125 * 300 = 3.0

        If bug used (1 - market_prob) = 0.20:
            kelly = 0.10 / 0.20 = 0.50
            raw = 0.08 * 0.50 * 300 = 12.0 (4x oversized!)
        """
        sizer = self._make_sizer()
        size = sizer.compute(
            edge=0.10, market_prob=0.80, bankroll=300.0, direction="BUY_NO"
        )
        # kelly = 0.10 / 0.80 = 0.125
        # raw = 0.08 * 0.125 * 300 = 3.0
        # capped at min(3.0, max_trade=3.0, 3%*300=9.0, 20%*300=60.0) = 3.0
        assert size == pytest.approx(3.0)

    def test_buy_no_small_position(self):
        """BUY_NO with small bankroll to verify formula without cap interference.

        Hand calculation:
            edge=0.10, market_prob=0.80, bankroll=30
            BUY_NO denom = market_prob = 0.80
            kelly = 0.10 / 0.80 = 0.125
            raw = 0.08 * 0.125 * 30 = 0.30
        """
        sizer = self._make_sizer()
        size = sizer.compute(
            edge=0.10, market_prob=0.80, bankroll=30.0, direction="BUY_NO"
        )
        assert size == pytest.approx(0.30)

    def test_buy_yes_vs_buy_no_different_sizes(self):
        """Same edge/market_prob but different direction -> different Kelly sizes."""
        sizer = self._make_sizer()
        # BUY_YES: kelly = 0.10 / (1 - 0.60) = 0.25, raw = 0.08 * 0.25 * 30 = 0.60
        yes_size = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, direction="BUY_YES"
        )
        # BUY_NO: kelly = 0.10 / 0.60 = 0.1667, raw = 0.08 * 0.1667 * 30 = 0.40
        no_size = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, direction="BUY_NO"
        )
        assert yes_size == pytest.approx(0.60)
        assert no_size == pytest.approx(0.40)
        assert yes_size > no_size  # different denominators give different sizes


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
        # BUY_YES uses BUY side on YES token
        call_kwargs = mock_clob.place_limit_order.call_args
        assert call_kwargs.kwargs["side"] == "BUY"
        assert call_kwargs.kwargs["token_id"] == "tok_abc"
        mock_clob.place_limit_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_buy_no_buys_no_token(self):
        """BUY_NO signal should BUY the NO token (not SELL the YES token).

        The caller is responsible for passing the NO token_id and its price.
        The executor always uses BUY side.
        """
        executor, mock_clob = self._make_executor()
        signal = TradingSignal(
            market_id="market_1",
            direction="BUY_NO",
            action="TRADE",
            edge=0.15,
            kelly_size=2.50,
            timestamp=datetime.now(timezone.utc),
        )
        no_price = _make_market_price(token_id="tok_no_abc", bid=0.38, ask=0.42)
        record = await executor.execute(
            signal, token_id="tok_no_abc", market_price=no_price
        )

        assert record is not None
        assert record.direction == "BUY_NO"
        # Must BUY the NO token at ask, not SELL the YES token
        call_kwargs = mock_clob.place_limit_order.call_args
        assert call_kwargs.kwargs["side"] == "BUY"
        assert call_kwargs.kwargs["token_id"] == "tok_no_abc"
        assert call_kwargs.kwargs["price"] == 0.42  # NO ask price
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
        """Model update >3h old and market moved >3% -> cancel affected orders only."""
        executor, mock_clob = self._make_executor()
        mock_clob.cancel_order = AsyncMock(return_value=True)
        old_time = datetime.now(timezone.utc) - timedelta(hours=4)
        current_prices = {"tok_abc": 0.65}
        previous_prices = {"tok_abc": 0.60}  # 8.3% move

        # Simulate open orders on affected and unaffected tokens
        now = datetime.now(timezone.utc)
        executor._open_orders = {"o1": now, "o2": now}
        executor._order_tokens = {"o1": "tok_abc", "o2": "tok_xyz"}

        cancelled = await executor.check_stale_quotes(
            last_model_update=old_time,
            current_prices=current_prices,
            previous_prices=previous_prices,
        )
        assert cancelled is True
        # Only tok_abc order cancelled, tok_xyz survives
        mock_clob.cancel_order.assert_called_once_with("o1")
        assert "o2" in executor._open_orders
        assert "o2" in executor._order_tokens

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
        mock_clob.cancel_order = AsyncMock(return_value=True)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
        current_prices = {"tok_abc": 0.65}
        previous_prices = {"tok_abc": 0.60}

        cancelled = await executor.check_stale_quotes(
            last_model_update=recent_time,
            current_prices=current_prices,
            previous_prices=previous_prices,
        )
        assert cancelled is False
        mock_clob.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_cancel_far_from_resolution(self):
        """Far from resolution -> no cancellation."""
        executor, mock_clob = self._make_executor()
        cancelled = await executor.check_resolution_proximity(hours_to_resolution=10.0)
        assert cancelled is False
        mock_clob.cancel_all_orders.assert_not_awaited()


# ===================================================================
# Duplicate order prevention
# ===================================================================

class TestDuplicateOrderPrevention:
    @pytest.mark.asyncio
    async def test_duplicate_order_rejected(self):
        """Second order on same token is rejected while first is open."""
        from src.trading.executor import OrderExecutor
        clob = AsyncMock()
        clob.paper_trading = True
        clob.place_limit_order = AsyncMock(side_effect=["order_1", "order_2"])
        executor = OrderExecutor(clob_client=clob, paper_trading=True)

        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=2.0, timestamp=datetime.now(timezone.utc),
        )
        price = MarketPrice(
            token_id="tok1", timestamp=datetime.now(timezone.utc),
            bid=0.48, ask=0.52, mid=0.50, volume_24h=5000.0,
        )

        result1 = await executor.execute(signal, "tok1", price)
        assert result1 is not None

        result2 = await executor.execute(signal, "tok1", price)
        assert result2 is None  # duplicate rejected
        assert clob.place_limit_order.call_count == 1

    @pytest.mark.asyncio
    async def test_different_tokens_allowed(self):
        """Orders on different tokens are not considered duplicates."""
        from src.trading.executor import OrderExecutor
        clob = AsyncMock()
        clob.paper_trading = True
        clob.place_limit_order = AsyncMock(side_effect=["order_1", "order_2"])
        executor = OrderExecutor(clob_client=clob, paper_trading=True)

        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=2.0, timestamp=datetime.now(timezone.utc),
        )
        price1 = MarketPrice(
            token_id="tok1", timestamp=datetime.now(timezone.utc),
            bid=0.48, ask=0.52, mid=0.50, volume_24h=5000.0,
        )
        price2 = MarketPrice(
            token_id="tok2", timestamp=datetime.now(timezone.utc),
            bid=0.48, ask=0.52, mid=0.50, volume_24h=5000.0,
        )

        result1 = await executor.execute(signal, "tok1", price1)
        result2 = await executor.execute(signal, "tok2", price2)
        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_ttl_clears_duplicate_guard(self):
        """After TTL sweep, previously-blocked token can be traded again."""
        from src.trading.executor import OrderExecutor
        clob = AsyncMock()
        clob.paper_trading = True
        clob.place_limit_order = AsyncMock(side_effect=["order_1", "order_2"])
        clob.cancel_order = AsyncMock(return_value=True)
        executor = OrderExecutor(clob_client=clob, paper_trading=True, order_ttl_seconds=0.0)

        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=2.0, timestamp=datetime.now(timezone.utc),
        )
        price = MarketPrice(
            token_id="tok1", timestamp=datetime.now(timezone.utc),
            bid=0.48, ask=0.52, mid=0.50, volume_24h=5000.0,
        )

        await executor.execute(signal, "tok1", price)
        assert executor.has_open_order("tok1") is True

        await executor.cancel_expired_orders()
        assert executor.has_open_order("tok1") is False


# ===================================================================
# Order TTL tests
# ===================================================================

class TestOrderTTL:
    """Test suite for order time-to-live expiry."""

    def _make_executor(self, ttl=60.0):
        from src.trading.executor import OrderExecutor
        mock_clob = AsyncMock()
        mock_clob.paper_trading = True
        mock_clob.place_limit_order = AsyncMock(return_value="order-123")
        mock_clob.cancel_order = AsyncMock(return_value=True)
        mock_clob.cancel_all_orders = AsyncMock(return_value=1)
        return OrderExecutor(
            clob_client=mock_clob, paper_trading=True, order_ttl_seconds=ttl
        ), mock_clob

    @pytest.mark.asyncio
    async def test_no_expired_orders_does_nothing(self):
        executor, mock_clob = self._make_executor(ttl=3600.0)
        cancelled = await executor.cancel_expired_orders()
        assert cancelled == 0
        mock_clob.cancel_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fresh_order_not_cancelled(self):
        executor, mock_clob = self._make_executor(ttl=3600.0)
        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=2.0, timestamp=datetime.now(timezone.utc),
        )
        price = _make_market_price()
        await executor.execute(signal, "tok_abc", price)
        # Order just placed — should not be expired
        cancelled = await executor.cancel_expired_orders()
        assert cancelled == 0
        mock_clob.cancel_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_expired_order_cancelled(self):
        executor, mock_clob = self._make_executor(ttl=60.0)
        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=2.0, timestamp=datetime.now(timezone.utc),
        )
        price = _make_market_price()
        await executor.execute(signal, "tok_abc", price)

        # Backdate the order to 2 minutes ago
        oid = list(executor._open_orders.keys())[0]
        executor._open_orders[oid] = datetime.now(timezone.utc) - timedelta(seconds=120)

        cancelled = await executor.cancel_expired_orders()
        assert cancelled == 1
        mock_clob.cancel_order.assert_awaited_once_with(oid)
        assert len(executor._open_orders) == 0

    @pytest.mark.asyncio
    async def test_mixed_fresh_and_expired_preserves_fresh(self):
        """Only expired orders are cancelled; fresh orders survive."""
        executor, mock_clob = self._make_executor(ttl=60.0)
        # Simulate two orders: one expired, one fresh
        executor._open_orders["old-order"] = datetime.now(timezone.utc) - timedelta(seconds=120)
        executor._order_tokens["old-order"] = "tok_old"
        executor._open_orders["new-order"] = datetime.now(timezone.utc)
        executor._order_tokens["new-order"] = "tok_new"

        cancelled = await executor.cancel_expired_orders()
        assert cancelled == 1  # only 1 expired
        mock_clob.cancel_order.assert_awaited_once_with("old-order")
        mock_clob.cancel_all_orders.assert_not_awaited()
        # Fresh order must survive
        assert "new-order" in executor._open_orders
        assert executor.has_open_order("tok_new")
        assert not executor.has_open_order("tok_old")


# ===================================================================
# Fill Rate Monitoring tests
# ===================================================================

class TestFillRateMonitoring:
    """Test suite for fill rate and adverse selection monitoring."""

    def _make_executor(self):
        from src.trading.executor import OrderExecutor
        mock_clob = AsyncMock()
        mock_clob.paper_trading = True
        return OrderExecutor(clob_client=mock_clob, paper_trading=True)

    def test_fill_rate_empty(self):
        executor = self._make_executor()
        assert executor.get_fill_rate() == 0.0

    def test_fill_rate_all_filled(self):
        executor = self._make_executor()
        executor.record_fill("t1", filled=True, pnl=1.0)
        executor.record_fill("t2", filled=True, pnl=-0.5)
        assert executor.get_fill_rate() == pytest.approx(1.0)

    def test_fill_rate_partial(self):
        executor = self._make_executor()
        executor.record_fill("t1", filled=True, pnl=1.0)
        executor.record_fill("t2", filled=False, pnl=0.5)
        executor.record_fill("t3", filled=True, pnl=-0.3)
        executor.record_fill("t4", filled=False, pnl=0.2)
        assert executor.get_fill_rate() == pytest.approx(0.5)

    def test_adverse_selection_none_when_no_data(self):
        executor = self._make_executor()
        assert executor.get_adverse_selection_ratio() is None

    def test_adverse_selection_none_when_only_filled(self):
        executor = self._make_executor()
        executor.record_fill("t1", filled=True, pnl=1.0)
        assert executor.get_adverse_selection_ratio() is None

    def test_adverse_selection_none_when_only_unfilled(self):
        executor = self._make_executor()
        executor.record_fill("t1", filled=False, pnl=0.5)
        assert executor.get_adverse_selection_ratio() is None

    def test_adverse_selection_positive(self):
        """Filled orders win more → positive ratio (good)."""
        executor = self._make_executor()
        # Filled: 2 wins, 0 losses → 100% win rate
        executor.record_fill("t1", filled=True, pnl=1.0)
        executor.record_fill("t2", filled=True, pnl=0.5)
        # Unfilled: 1 win, 1 loss → 50% win rate
        executor.record_fill("t3", filled=False, pnl=0.3)
        executor.record_fill("t4", filled=False, pnl=-0.2)
        ratio = executor.get_adverse_selection_ratio()
        assert ratio is not None
        assert ratio == pytest.approx(0.5)  # 1.0 - 0.5

    def test_adverse_selection_negative(self):
        """Filled orders lose more → negative ratio (being picked off)."""
        executor = self._make_executor()
        # Filled: 0 wins, 2 losses → 0% win rate
        executor.record_fill("t1", filled=True, pnl=-1.0)
        executor.record_fill("t2", filled=True, pnl=-0.5)
        # Unfilled: 2 wins, 0 losses → 100% win rate
        executor.record_fill("t3", filled=False, pnl=0.3)
        executor.record_fill("t4", filled=False, pnl=0.2)
        ratio = executor.get_adverse_selection_ratio()
        assert ratio is not None
        assert ratio == pytest.approx(-1.0)  # 0.0 - 1.0

    def test_is_being_picked_off_true(self):
        executor = self._make_executor()
        executor.record_fill("t1", filled=True, pnl=-1.0)
        executor.record_fill("t2", filled=True, pnl=-0.5)
        executor.record_fill("t3", filled=False, pnl=0.3)
        executor.record_fill("t4", filled=False, pnl=0.2)
        assert executor.is_being_picked_off() is True

    def test_is_being_picked_off_false_positive_ratio(self):
        executor = self._make_executor()
        executor.record_fill("t1", filled=True, pnl=1.0)
        executor.record_fill("t2", filled=True, pnl=0.5)
        executor.record_fill("t3", filled=False, pnl=-0.3)
        executor.record_fill("t4", filled=False, pnl=-0.2)
        assert executor.is_being_picked_off() is False

    def test_is_being_picked_off_false_no_data(self):
        executor = self._make_executor()
        assert executor.is_being_picked_off() is False

    def test_pnl_none_excluded(self):
        """Records with pnl=None are excluded from adverse selection calc."""
        executor = self._make_executor()
        executor.record_fill("t1", filled=True, pnl=None)
        executor.record_fill("t2", filled=False, pnl=None)
        assert executor.get_adverse_selection_ratio() is None


# ===================================================================
# Correlation-Aware Position Sizing tests
# ===================================================================

class TestCorrelationPositionSizing:
    """Test suite for correlation discount in PositionSizer."""

    def _make_sizer(self, **kwargs):
        from src.trading.position_sizer import PositionSizer
        kwargs.setdefault("min_trade_usd", 0.0)
        return PositionSizer(**kwargs)

    def test_single_station_no_discount(self):
        """1 station → 1.0x multiplier."""
        sizer = self._make_sizer()
        size_1 = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, active_station_count=1
        )
        # kelly = 0.10/0.40 = 0.25, raw = 0.08*0.25*30 = 0.60
        assert size_1 == pytest.approx(0.60)

    def test_two_stations_discount(self):
        """2 stations → 0.7x."""
        sizer = self._make_sizer()
        size_2 = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, active_station_count=2
        )
        assert size_2 == pytest.approx(0.60 * 0.7)

    def test_three_stations_discount(self):
        """3 stations → 0.5x."""
        sizer = self._make_sizer()
        size_3 = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, active_station_count=3
        )
        assert size_3 == pytest.approx(0.60 * 0.5)

    def test_four_stations_discount(self):
        """4 stations → 0.4x."""
        sizer = self._make_sizer()
        size_4 = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, active_station_count=4
        )
        assert size_4 == pytest.approx(0.60 * 0.4)

    def test_five_stations_same_as_four(self):
        """5 stations → still 0.4x."""
        sizer = self._make_sizer()
        size_5 = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, active_station_count=5
        )
        assert size_5 == pytest.approx(0.60 * 0.4)

    def test_discount_interacts_with_cap(self):
        """Discount applied before caps — large raw sizes still capped at $3."""
        sizer = self._make_sizer()
        size = sizer.compute(
            edge=0.50, market_prob=0.50, bankroll=1000.0, active_station_count=2
        )
        # raw = 0.08 * 1.0 * 1000 = 80.0, * 0.7 = 56.0, capped at 3.0
        assert size == pytest.approx(3.0)

    def test_backward_compatible_default(self):
        """Default active_station_count=1 produces same result as before."""
        sizer = self._make_sizer()
        size_default = sizer.compute(edge=0.10, market_prob=0.60, bankroll=30.0)
        size_explicit = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, active_station_count=1
        )
        assert size_default == pytest.approx(size_explicit)


# ===================================================================
# Minimum Trade Size tests
# ===================================================================

class TestMinimumTradeSize:
    """Test suite for minimum trade size guard in PositionSizer."""

    def test_below_minimum_returns_zero(self):
        """Trade sized below min_trade_usd is rejected (returns 0)."""
        from src.trading.position_sizer import PositionSizer
        sizer = PositionSizer(min_trade_usd=0.50)
        # kelly = 0.10/0.40 = 0.25, raw = 0.08 * 0.25 * 30 = 0.60
        # BUY_NO: kelly = 0.10/0.60 = 0.167, raw = 0.08 * 0.167 * 30 = 0.40
        # 0.40 < 0.50 min -> rejected
        size = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, direction="BUY_NO"
        )
        assert size == 0.0

    def test_above_minimum_passes(self):
        """Trade sized above min_trade_usd is accepted normally."""
        from src.trading.position_sizer import PositionSizer
        sizer = PositionSizer(min_trade_usd=0.50)
        # BUY_YES: kelly = 0.10/0.40 = 0.25, raw = 0.08 * 0.25 * 30 = 0.60
        # 0.60 >= 0.50 min -> accepted
        size = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, direction="BUY_YES"
        )
        assert size == pytest.approx(0.60)

    def test_exactly_at_minimum_passes(self):
        """Trade exactly at min_trade_usd is accepted."""
        from src.trading.position_sizer import PositionSizer
        sizer = PositionSizer(min_trade_usd=0.60)
        size = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, direction="BUY_YES"
        )
        assert size == pytest.approx(0.60)

    def test_zero_edge_still_zero(self):
        """Edge=0 still returns 0 (not rejected by min guard, just zero)."""
        from src.trading.position_sizer import PositionSizer
        sizer = PositionSizer(min_trade_usd=0.50)
        size = sizer.compute(edge=0.0, market_prob=0.60, bankroll=30.0)
        assert size == 0.0


class TestResolutionProximity:
    """Targeted cancellation near resolution time."""

    def _make_executor(self):
        from src.trading.executor import OrderExecutor

        clob = AsyncMock()
        clob.paper_trading = True
        clob.cancel_order = AsyncMock(return_value=True)
        clob.cancel_all_orders = AsyncMock()
        executor = OrderExecutor(clob, paper_trading=True)
        return executor, clob

    @pytest.mark.asyncio
    async def test_targeted_cancel_only_near_resolution_tokens(self):
        """Only orders on near-resolution tokens are cancelled."""
        executor, clob = self._make_executor()
        # Simulate two open orders on different tokens
        executor._open_orders = {"o1": datetime.now(timezone.utc), "o2": datetime.now(timezone.utc)}
        executor._order_tokens = {"o1": "token-A", "o2": "token-B"}

        cancelled = await executor.check_resolution_proximity(
            2.0, token_ids={"token-A"}
        )
        assert cancelled is True
        clob.cancel_order.assert_called_once_with("o1")
        # token-B order survives
        assert "o2" in executor._open_orders
        assert "o2" in executor._order_tokens

    @pytest.mark.asyncio
    async def test_no_cancel_when_far_from_resolution(self):
        """Nothing cancelled when hours >= threshold."""
        executor, clob = self._make_executor()
        executor._open_orders = {"o1": datetime.now(timezone.utc)}
        executor._order_tokens = {"o1": "token-A"}

        cancelled = await executor.check_resolution_proximity(
            5.0, token_ids={"token-A"}
        )
        assert cancelled is False
        clob.cancel_order.assert_not_called()
        clob.cancel_all_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_cancel_all_without_token_ids(self):
        """Legacy behavior: cancel all when no token_ids provided."""
        executor, clob = self._make_executor()
        executor._open_orders = {"o1": datetime.now(timezone.utc)}
        executor._order_tokens = {"o1": "token-A"}

        cancelled = await executor.check_resolution_proximity(2.0)
        assert cancelled is True
        clob.cancel_all_orders.assert_called_once()


class TestOrderPriceRounding:
    """Prices and sizes sent to the CLOB must be rounded to valid ticks."""

    @pytest.mark.asyncio
    async def test_price_rounded_to_cent(self):
        """Synthetic NO-token prices with 3+ decimals must be rounded to 0.01.

        Polymarket CLOB rejects prices like 0.029 or 0.626 — they must
        be exact multiples of 0.01.
        """
        from src.trading.executor import OrderExecutor

        clob = AsyncMock()
        clob.paper_trading = True
        clob.place_limit_order = AsyncMock(return_value="order-1")
        executor = OrderExecutor(clob_client=clob, paper_trading=True)

        signal = TradingSignal(
            market_id="m1", direction="BUY_NO", action="TRADE",
            edge=0.10, kelly_size=2.00, timestamp=datetime.now(timezone.utc),
        )
        # Synthetic NO-token price: ask=0.029 (from 1.0 - 0.971)
        bad_price = MarketPrice(
            token_id="tok_no", timestamp=datetime.now(timezone.utc),
            bid=0.019, ask=0.029, mid=0.024, volume_24h=5000.0,
        )

        await executor.execute(signal, "tok_no", bad_price)

        call_kwargs = clob.place_limit_order.call_args.kwargs
        assert call_kwargs["price"] == 0.03, (
            f"Price 0.029 should round to 0.03, got {call_kwargs['price']}"
        )

    @pytest.mark.asyncio
    async def test_size_rounded_to_two_decimals(self):
        """Share sizes must not have excessive decimal places."""
        from src.trading.executor import OrderExecutor

        clob = AsyncMock()
        clob.paper_trading = True
        clob.place_limit_order = AsyncMock(return_value="order-1")
        executor = OrderExecutor(clob_client=clob, paper_trading=True)

        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=2.00, timestamp=datetime.now(timezone.utc),
        )
        price = MarketPrice(
            token_id="tok1", timestamp=datetime.now(timezone.utc),
            bid=0.06, ask=0.07, mid=0.065, volume_24h=5000.0,
        )

        await executor.execute(signal, "tok1", price)

        call_kwargs = clob.place_limit_order.call_args.kwargs
        # 2.00 / 0.07 = 28.571428... should round to 28.57
        assert call_kwargs["size"] == pytest.approx(28.57, abs=0.01)

    @pytest.mark.asyncio
    async def test_floating_point_noise_cleaned(self):
        """Floating-point artifacts like 0.5900000000000001 must round to 0.59."""
        from src.trading.executor import OrderExecutor

        clob = AsyncMock()
        clob.paper_trading = True
        clob.place_limit_order = AsyncMock(return_value="order-1")
        executor = OrderExecutor(clob_client=clob, paper_trading=True)

        signal = TradingSignal(
            market_id="m1", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=3.00, timestamp=datetime.now(timezone.utc),
        )
        # Simulate floating-point noise
        noisy_price = MarketPrice(
            token_id="tok1", timestamp=datetime.now(timezone.utc),
            bid=0.57, ask=0.5900000000000001, mid=0.58, volume_24h=5000.0,
        )

        await executor.execute(signal, "tok1", noisy_price)

        call_kwargs = clob.place_limit_order.call_args.kwargs
        assert call_kwargs["price"] == 0.59

    def test_min_zero_disables_guard(self):
        """min_trade_usd=0 allows any positive size."""
        from src.trading.position_sizer import PositionSizer
        sizer = PositionSizer(min_trade_usd=0.0)
        # BUY_NO: raw = 0.40
        size = sizer.compute(
            edge=0.10, market_prob=0.60, bankroll=30.0, direction="BUY_NO"
        )
        assert size == pytest.approx(0.40)
