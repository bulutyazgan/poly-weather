"""Tests for the verification system: PredictionLog, PaperTrader, HypothesisTester."""

from datetime import date, datetime

import pytest

from src.data.models import (
    MarketContract,
    RegimeClassification,
    TradingSignal,
)
from src.verification.prediction_log import PredictionLog, SignalLogEntry
from src.verification.paper_trader import PaperTrader
from src.verification.hypothesis_tester import HypothesisTester


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_signal(
    market_id: str = "mkt-1",
    direction: str = "BUY_YES",
    action: str = "TRADE",
    edge: float = 0.10,
    kelly_size: float = 0.05,
) -> TradingSignal:
    return TradingSignal(
        market_id=market_id,
        direction=direction,
        action=action,
        edge=edge,
        kelly_size=kelly_size,
        timestamp=datetime(2026, 4, 15, 12, 0),
    )


def _make_regime(
    station_id: str = "KNYC",
    confidence: str = "HIGH",
) -> RegimeClassification:
    return RegimeClassification(
        station_id=station_id,
        valid_date=date(2026, 4, 16),
        regime="stable",
        confidence=confidence,
        confidence_score=0.85,
    )


def _make_contract(token_id: str = "tok-1") -> MarketContract:
    return MarketContract(
        token_id=token_id,
        condition_id="cond-1",
        question="Will NYC high be 72-73°F on Apr 16?",
        city="NYC",
        resolution_date=date(2026, 4, 16),
        temp_bucket_low=72.0,
        temp_bucket_high=73.0,
        outcome="Yes",
    )


# ── PredictionLog tests ──────────────────────────────────────────────────────


class TestPredictionLog:
    def test_log_signal(self):
        """Log a TradingSignal with metadata; verify it's stored and retrievable."""
        log = PredictionLog()
        entry = SignalLogEntry(
            signal=_make_signal(),
            station_id="KNYC",
            regime=_make_regime(),
            model_probability=0.55,
            market_probability=0.45,
        )
        log.log(entry)
        assert len(log.export()) == 1
        assert log.export()[0]["station_id"] == "KNYC"
        assert log.export()[0]["model_probability"] == 0.55

    def test_log_includes_skip_signals(self):
        """SKIP signals must be logged (survivorship bias prevention)."""
        log = PredictionLog()
        skip_signal = _make_signal(action="SKIP")
        entry = SignalLogEntry(
            signal=skip_signal,
            station_id="KNYC",
            regime=_make_regime(),
            model_probability=0.48,
            market_probability=0.45,
        )
        log.log(entry)
        exported = log.export()
        assert len(exported) == 1
        assert exported[0]["action"] == "SKIP"

    def test_get_signals_by_station(self):
        """Filter logged signals by station_id."""
        log = PredictionLog()
        for sid in ["KNYC", "KORD", "KNYC"]:
            log.log(SignalLogEntry(
                signal=_make_signal(),
                station_id=sid,
                regime=_make_regime(station_id=sid),
                model_probability=0.5,
                market_probability=0.4,
            ))
        nyc = log.get_by_station("KNYC")
        assert len(nyc) == 2
        assert all(e.station_id == "KNYC" for e in nyc)

    def test_get_signals_by_regime(self):
        """Filter by regime confidence level."""
        log = PredictionLog()
        for conf in ["HIGH", "MEDIUM", "LOW", "HIGH"]:
            log.log(SignalLogEntry(
                signal=_make_signal(),
                station_id="KNYC",
                regime=_make_regime(confidence=conf),
                model_probability=0.5,
                market_probability=0.4,
            ))
        high = log.get_by_regime("HIGH")
        assert len(high) == 2

    def test_export_to_records(self):
        """Export all logged signals as list of dicts."""
        log = PredictionLog()
        log.log(SignalLogEntry(
            signal=_make_signal(direction="BUY_NO"),
            station_id="KORD",
            regime=_make_regime(station_id="KORD"),
            model_probability=0.3,
            market_probability=0.5,
            contract=_make_contract(),
        ))
        records = log.export()
        assert isinstance(records, list)
        assert len(records) == 1
        rec = records[0]
        assert rec["direction"] == "BUY_NO"
        assert rec["station_id"] == "KORD"
        assert rec["market_id"] == "mkt-1"
        assert "contract" in rec


# ── PaperTrader tests ─────────────────────────────────────────────────────────


class TestPaperTrader:
    def test_record_paper_trade(self):
        """Record a paper trade with signal + contract info."""
        trader = PaperTrader(taker_fee_rate=0.0)
        signal = _make_signal()
        contract = _make_contract()
        trade_id = trader.record_trade(signal, contract, entry_price=0.40, amount_usd=3.0)
        assert isinstance(trade_id, str)
        assert len(trade_id) > 0

    def test_resolve_trade_win(self):
        """BUY_YES at 0.40, outcome True → profit.

        Shares = 3.0 / 0.40 = 7.5. Each pays $1, so payout = 7.5.
        PnL = 7.5 - 3.0 = 4.50.
        Equivalently: amount * (1/price - 1) = 3 * (2.5 - 1) = 3 * 1.5 = 4.50.
        """
        trader = PaperTrader(taker_fee_rate=0.0)
        signal = _make_signal(direction="BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.40, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=True)
        assert pnl == pytest.approx(4.50)

    def test_resolve_trade_loss(self):
        """BUY_YES at 0.40, outcome False → loss = -$3.00."""
        trader = PaperTrader(taker_fee_rate=0.0)
        signal = _make_signal(direction="BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.40, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=False)
        assert pnl == pytest.approx(-3.0)

    def test_resolve_buy_no_win(self):
        """BUY_NO at market_price=0.60 (YES price). NO token costs 1-0.60=0.40.

        Shares = 3.0 / 0.40 = 7.5. outcome=False means YES didn't happen → NO wins.
        Payout = 7.5 * 1.0 = 7.5. PnL = 7.5 - 3.0 = 4.50.
        Formula: amount * (1/(1-entry_price) - 1) = 3 * (1/0.4 - 1) = 3 * 1.5 = 4.50.
        """
        trader = PaperTrader(taker_fee_rate=0.0)
        signal = _make_signal(direction="BUY_NO")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.60, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=False)
        assert pnl == pytest.approx(4.50)

    def test_resolve_buy_no_loss(self):
        """BUY_NO at 0.60, outcome True → NO loses → loss = -$3.00."""
        trader = PaperTrader(taker_fee_rate=0.0)
        signal = _make_signal(direction="BUY_NO")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.60, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=True)
        assert pnl == pytest.approx(-3.0)

    def test_total_pnl(self):
        """After multiple resolved trades, total_pnl sums correctly."""
        trader = PaperTrader(taker_fee_rate=0.0)
        contract = _make_contract()

        # Trade 1: BUY_YES at 0.40, win → +4.50
        tid1 = trader.record_trade(_make_signal(direction="BUY_YES"), contract, 0.40, 3.0)
        trader.resolve(tid1, outcome=True)

        # Trade 2: BUY_YES at 0.50, loss → -2.00
        tid2 = trader.record_trade(_make_signal(direction="BUY_YES"), contract, 0.50, 2.0)
        trader.resolve(tid2, outcome=False)

        # Trade 3: BUY_NO at 0.70, win (outcome=False) → 1.0 * (1/0.3 - 1) = 1.0 * 2.333.. = 2.333..
        tid3 = trader.record_trade(_make_signal(direction="BUY_NO"), contract, 0.70, 1.0)
        trader.resolve(tid3, outcome=False)

        expected = 4.50 + (-2.0) + (1.0 * (1.0 / 0.3 - 1.0))
        assert trader.total_pnl() == pytest.approx(expected)

    def test_win_rate(self):
        """3 wins out of 5 resolved → 60%."""
        trader = PaperTrader(taker_fee_rate=0.0)
        contract = _make_contract()
        outcomes = [True, False, True, True, False]  # 3 wins for BUY_YES
        for outcome in outcomes:
            tid = trader.record_trade(_make_signal(direction="BUY_YES"), contract, 0.50, 1.0)
            trader.resolve(tid, outcome=outcome)
        assert trader.win_rate() == pytest.approx(0.60)

    def test_counterfactual_tracking(self):
        """For SKIP signals, track what would have happened."""
        trader = PaperTrader(taker_fee_rate=0.0)
        skip_signal = _make_signal(action="SKIP", direction="BUY_YES")
        contract = _make_contract()
        trader.record_counterfactual(skip_signal, contract, hypothetical_price=0.40, hypothetical_size=3.0)
        counterfactuals = trader.get_counterfactuals()
        assert len(counterfactuals) == 1
        assert counterfactuals[0]["signal"].action == "SKIP"

        # Resolve the counterfactual
        trader.resolve_counterfactual(0, outcome=True)
        cf = trader.get_counterfactuals()[0]
        assert cf["pnl"] == pytest.approx(4.50)


class TestPaperTraderFees:
    """Tests for taker fee deduction in PaperTrader P&L."""

    def test_fee_reduces_win_pnl_buy_yes(self):
        """BUY_YES at 0.40, win with 2% fee.

        Gross PnL = 3 * (1/0.40 - 1) = 4.50
        Shares = 3.0 / 0.40 = 7.5
        Fee = 0.02 * min(0.40, 0.60) * 7.5 = 0.02 * 0.40 * 7.5 = 0.06
        Net PnL = 4.50 - 0.06 = 4.44
        """
        trader = PaperTrader(taker_fee_rate=0.02)
        signal = _make_signal(direction="BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.40, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=True)
        assert pnl == pytest.approx(4.44)

    def test_fee_increases_loss_buy_yes(self):
        """BUY_YES at 0.40, loss with 2% fee.

        Gross PnL = -3.00
        Fee = 0.06 (same as above)
        Net PnL = -3.00 - 0.06 = -3.06
        """
        trader = PaperTrader(taker_fee_rate=0.02)
        signal = _make_signal(direction="BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.40, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=False)
        assert pnl == pytest.approx(-3.06)

    def test_fee_reduces_win_pnl_buy_no(self):
        """BUY_NO at YES_price=0.60, win (outcome=False) with 2% fee.

        NO price = 1 - 0.60 = 0.40
        Gross PnL = 3 * (1/0.40 - 1) = 4.50
        Shares = 3.0 / 0.40 = 7.5
        Fee = 0.02 * min(0.40, 0.60) * 7.5 = 0.06
        Net PnL = 4.50 - 0.06 = 4.44
        """
        trader = PaperTrader(taker_fee_rate=0.02)
        signal = _make_signal(direction="BUY_NO")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.60, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=False)
        assert pnl == pytest.approx(4.44)

    def test_fee_on_counterfactual(self):
        """Counterfactual P&L also deducts fees."""
        trader = PaperTrader(taker_fee_rate=0.02)
        skip_signal = _make_signal(action="SKIP", direction="BUY_YES")
        contract = _make_contract()
        trader.record_counterfactual(skip_signal, contract, hypothetical_price=0.40, hypothetical_size=3.0)
        trader.resolve_counterfactual(0, outcome=True)
        cf = trader.get_counterfactuals()[0]
        # Same as test_fee_reduces_win_pnl_buy_yes: 4.50 - 0.06 = 4.44
        assert cf["pnl"] == pytest.approx(4.44)

    def test_total_pnl_includes_fees(self):
        """Total P&L reflects fee-adjusted results."""
        trader = PaperTrader(taker_fee_rate=0.02)
        contract = _make_contract()

        # Win: gross 4.50, fee 0.06, net 4.44
        tid1 = trader.record_trade(_make_signal(direction="BUY_YES"), contract, 0.40, 3.0)
        trader.resolve(tid1, outcome=True)

        # Loss: gross -2.00, fee = 0.02 * min(0.50, 0.50) * (2/0.50) = 0.02 * 0.50 * 4 = 0.04
        tid2 = trader.record_trade(_make_signal(direction="BUY_YES"), contract, 0.50, 2.0)
        trader.resolve(tid2, outcome=False)

        expected = 4.44 + (-2.0 - 0.04)
        assert trader.total_pnl() == pytest.approx(expected)

    def test_zero_fee_matches_legacy(self):
        """taker_fee_rate=0 matches pre-fee behavior exactly."""
        trader = PaperTrader(taker_fee_rate=0.0)
        signal = _make_signal(direction="BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.40, amount_usd=3.0)
        pnl = trader.resolve(tid, outcome=True)
        assert pnl == pytest.approx(4.50)


# ── HypothesisTester tests ───────────────────────────────────────────────────


class TestHypothesisTester:
    def test_h0_1_climatology_beaten(self):
        """Good model forecasts + outcomes → BSS > 0, H0-1 rejected."""
        tester = HypothesisTester()
        # Model with skill: when outcome is True, forecast ~0.8; when False, ~0.2
        forecasts = [0.8, 0.2, 0.85, 0.15, 0.9, 0.1, 0.75, 0.25, 0.8, 0.2,
                     0.8, 0.2, 0.85, 0.15, 0.9, 0.1, 0.75, 0.25, 0.8, 0.2,
                     0.8, 0.2, 0.85, 0.15, 0.9, 0.1, 0.75, 0.25, 0.8, 0.2]
        outcomes = [True, False, True, False, True, False, True, False, True, False,
                    True, False, True, False, True, False, True, False, True, False,
                    True, False, True, False, True, False, True, False, True, False]
        result = tester.test_h0_1(forecasts, outcomes)
        assert result["rejected"] is True
        assert result["bss"] > 0
        assert result["p_value"] < 0.05
        assert result["n"] == 30

    def test_h0_1_climatology_not_beaten(self):
        """Poor model → BSS near/below 0, H0-1 not rejected."""
        tester = HypothesisTester()
        # Model that just predicts base rate (0.5) for everything
        forecasts = [0.5] * 20
        outcomes = [True, False] * 10
        result = tester.test_h0_1(forecasts, outcomes)
        assert result["rejected"] is False
        assert result["bss"] <= 0.0 + 1e-9

    def test_h0_4_market_edge_significant(self):
        """Model significantly better than market → H0-4 rejected."""
        tester = HypothesisTester()
        # Model is well-calibrated, market is not
        outcomes = [True, False] * 25
        model_forecasts = [0.9, 0.1] * 25  # good model
        market_forecasts = [0.6, 0.4] * 25  # market is mediocre
        result = tester.test_h0_4(model_forecasts, market_forecasts, outcomes)
        assert result["rejected"] is True
        assert result["p_value"] < 0.05
        assert result["model_bs"] < result["market_bs"]

    def test_h0_4_market_edge_insignificant(self):
        """Small sample, no real edge → H0-4 not rejected."""
        tester = HypothesisTester()
        outcomes = [True, False, True]
        model_forecasts = [0.6, 0.4, 0.55]
        market_forecasts = [0.55, 0.45, 0.50]
        result = tester.test_h0_4(model_forecasts, market_forecasts, outcomes)
        assert result["rejected"] is False

    def test_multiple_testing_correction(self):
        """With 5 city tests, apply Benjamini-Hochberg correction."""
        # p-values: 3 are significant raw, but after BH only some survive
        p_values = [0.001, 0.01, 0.03, 0.20, 0.80]
        rejected = HypothesisTester.benjamini_hochberg(p_values, alpha=0.05)
        # Sorted p-values: 0.001, 0.01, 0.03, 0.20, 0.80
        # Thresholds: 1/5*0.05=0.01, 2/5*0.05=0.02, 3/5*0.05=0.03, 4/5*0.05=0.04, 5/5*0.05=0.05
        # 0.001 < 0.01 ✓, 0.01 <= 0.02 ✓, 0.03 <= 0.03 ✓, 0.20 > 0.04 ✗, 0.80 > 0.05 ✗
        assert rejected == [True, True, True, False, False]
