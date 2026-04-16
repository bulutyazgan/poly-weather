"""End-to-end test of the resolution system.

Verifies the full flow: paper trades are placed, resolution checker
resolves them against mock Gamma outcomes, PnL is computed correctly,
exposure is released, and the dashboard API reflects the changes.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app, set_state
from src.data.models import MarketContract, TradingSignal
from src.orchestrator.scheduler import PipelineScheduler
from src.trading.exposure_tracker import ExposureTracker
from src.verification.paper_trader import PaperTrader
from src.verification.prediction_log import PredictionLog
from src.verification.resolution_checker import ResolutionChecker


def _contract(city: str, bucket_low: float, bucket_high: float, cond_id: str) -> MarketContract:
    return MarketContract(
        token_id=f"tok_{cond_id}",
        no_token_id=f"tok_{cond_id}_no",
        condition_id=cond_id,
        question=f"Will {city} high be {bucket_low}-{bucket_high}F?",
        city=city,
        resolution_date=date(2026, 4, 16),
        temp_bucket_low=bucket_low,
        temp_bucket_high=bucket_high,
        outcome="Yes",
        volume_24h=5000.0,
    )


def _signal(direction: str, market_id: str) -> TradingSignal:
    return TradingSignal(
        market_id=market_id,
        direction=direction,
        action="TRADE",
        edge=0.12,
        kelly_size=3.0,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.anyio
async def test_full_resolution_flow():
    """Place 4 trades, resolve them with mixed outcomes, verify PnL + exposure + API."""

    # --- Setup ---
    trader = PaperTrader(taker_fee_rate=0.02)
    tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
    prediction_log = PredictionLog()
    mock_pipeline = MagicMock()
    scheduler = PipelineScheduler(pipeline=mock_pipeline)

    # --- Place 4 paper trades ---
    trades = [
        # (city, bucket, cond_id, direction, entry_price, amount, expected_outcome)
        ("NYC", 72.0, 74.0, "cond_nyc", "BUY_YES", 0.50, 3.00, True),   # WIN
        ("LA", 68.0, 70.0, "cond_la", "BUY_NO", 0.70, 3.00, False),     # WIN (BUY_NO wins when NO)
        ("Denver", 74.0, 76.0, "cond_den", "BUY_YES", 0.40, 2.50, False),  # LOSS
        ("Miami", 80.0, 82.0, "cond_mia", "BUY_YES", 0.25, 1.50, True),   # WIN
    ]

    trade_ids = []
    for city, lo, hi, cond, direction, price, amount, _ in trades:
        contract = _contract(city, lo, hi, cond)
        signal = _signal(direction, f"tok_{cond}")
        tid = trader.record_trade(signal, contract, entry_price=price, amount_usd=amount)
        tracker.add(amount)
        trade_ids.append(tid)

    # Verify pre-resolution state
    assert len(trader._trades) == 4
    assert trader.total_pnl() == 0.0
    assert tracker.current == pytest.approx(10.0)  # 3+3+2.5+1.5

    # --- Mock Gamma resolution responses ---
    outcomes = {
        "cond_nyc": True,   # YES won → BUY_YES wins
        "cond_la": False,    # NO won → BUY_NO wins
        "cond_den": False,   # NO won → BUY_YES loses
        "cond_mia": True,    # YES won → BUY_YES wins
    }

    gamma = AsyncMock()

    async def mock_fetch(condition_id: str):
        return outcomes.get(condition_id)

    gamma.fetch_market_resolution = AsyncMock(side_effect=mock_fetch)

    # --- Run resolution checker ---
    checker = ResolutionChecker(gamma=gamma, paper_trader=trader, exposure_tracker=tracker)
    result = await checker.check_resolutions()

    # Verify resolution ran
    assert result["checked"] == 4
    assert result["resolved"] == 4
    assert result["errors"] == 0

    # Verify all trades are resolved
    for tid in trade_ids:
        assert trader._trades[tid]["resolved"] is True
        assert trader._trades[tid]["pnl"] is not None

    # Verify PnL computations (manual check)
    # Trade 1: BUY_YES@0.50, $3.00, outcome=True → gross = 3.0*(1/0.5-1) = 3.0
    #   fee = 0.02 * min(0.50, 0.50) * (3.0/0.50) = 0.02 * 0.50 * 6.0 = 0.06
    #   net = 3.0 - 0.06 = 2.94
    t1 = trader._trades[trade_ids[0]]
    assert t1["pnl"] == pytest.approx(2.94, abs=0.01)
    assert t1["outcome"] is True

    # Trade 2: BUY_NO, entry_price(YES)=0.70, $3.00, outcome=False → win
    #   NO costs 0.30, shares = 3.00/0.30 = 10, gross = 3.0*(1/0.30-1) = 7.0
    #   fee = 0.02 * min(0.30, 0.70) * 10 = 0.02 * 0.30 * 10 = 0.06
    #   net = 7.0 - 0.06 = 6.94
    t2 = trader._trades[trade_ids[1]]
    assert t2["pnl"] == pytest.approx(6.94, abs=0.01)

    # Trade 3: BUY_YES@0.40, $2.50, outcome=False → loss
    #   gross = -2.50, fee = 0.02 * min(0.40, 0.60) * (2.50/0.40) = 0.02 * 0.40 * 6.25 = 0.05
    #   net = -2.50 - 0.05 = -2.55
    t3 = trader._trades[trade_ids[2]]
    assert t3["pnl"] == pytest.approx(-2.55, abs=0.01)

    # Trade 4: BUY_YES@0.25, $1.50, outcome=True → win
    #   gross = 1.50*(1/0.25-1) = 4.50, fee = 0.02 * min(0.25, 0.75) * (1.50/0.25) = 0.02*0.25*6 = 0.03
    #   net = 4.50 - 0.03 = 4.47
    t4 = trader._trades[trade_ids[3]]
    assert t4["pnl"] == pytest.approx(4.47, abs=0.01)

    # Total PnL: 2.94 + 6.94 - 2.55 + 4.47 = 11.80
    assert trader.total_pnl() == pytest.approx(11.80, abs=0.05)
    assert trader.win_rate() == pytest.approx(0.75)  # 3 wins / 4 trades

    # Verify exposure was released
    assert tracker.current == pytest.approx(0.0, abs=0.01)

    # --- Now verify the dashboard API reflects the resolved state ---
    set_state(prediction_log, trader, scheduler)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Performance endpoint
            perf = (await client.get("/api/performance")).json()
            assert perf["trade_count"] == 4
            assert perf["win_rate"] == pytest.approx(0.75)
            assert perf["total_pnl"] == pytest.approx(11.80, abs=0.05)

            # Trades endpoint
            api_trades = (await client.get("/api/trades")).json()
            assert len(api_trades) == 4
            assert all(t["resolved"] for t in api_trades)
            assert all(t["pnl"] is not None for t in api_trades)

            # Filter by resolved
            resolved = (await client.get("/api/trades?status=resolved")).json()
            assert len(resolved) == 4

            pending = (await client.get("/api/trades?status=pending")).json()
            assert len(pending) == 0

    finally:
        set_state(None, None, None)


@pytest.mark.anyio
async def test_partial_resolution():
    """Only some markets resolve; unresolved trades remain pending."""
    trader = PaperTrader(taker_fee_rate=0.0)
    tracker = ExposureTracker(bankroll=300.0)

    # Two trades
    c1 = _contract("NYC", 72, 74, "cond_1")
    c2 = _contract("LA", 68, 70, "cond_2")
    t1 = trader.record_trade(_signal("BUY_YES", "tok_1"), c1, entry_price=0.50, amount_usd=2.0)
    t2 = trader.record_trade(_signal("BUY_YES", "tok_2"), c2, entry_price=0.50, amount_usd=2.0)
    tracker.add(4.0)

    # Only cond_1 resolves; cond_2 returns None (not yet settled)
    gamma = AsyncMock()
    gamma.fetch_market_resolution = AsyncMock(
        side_effect=lambda cid: True if cid == "cond_1" else None
    )

    checker = ResolutionChecker(gamma=gamma, paper_trader=trader, exposure_tracker=tracker)
    result = await checker.check_resolutions()

    assert result["resolved"] == 1
    assert trader._trades[t1]["resolved"] is True
    assert trader._trades[t2]["resolved"] is False
    # Exposure released for t1 only
    assert tracker.current == pytest.approx(2.0)


@pytest.mark.anyio
async def test_resolution_api_error_doesnt_block_others():
    """An API error on one trade doesn't prevent others from resolving."""
    trader = PaperTrader(taker_fee_rate=0.0)

    c1 = _contract("NYC", 72, 74, "cond_ok")
    c2 = _contract("LA", 68, 70, "cond_fail")
    trader.record_trade(_signal("BUY_YES", "tok_1"), c1, entry_price=0.50, amount_usd=2.0)
    trader.record_trade(_signal("BUY_YES", "tok_2"), c2, entry_price=0.50, amount_usd=2.0)

    async def flaky_fetch(cid):
        if cid == "cond_fail":
            raise ConnectionError("Gamma API timeout")
        return True

    gamma = AsyncMock()
    gamma.fetch_market_resolution = AsyncMock(side_effect=flaky_fetch)

    checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
    result = await checker.check_resolutions()

    assert result["resolved"] == 1
    assert result["errors"] == 1
    # The successful trade still resolved
    assert trader.total_pnl() == pytest.approx(2.0)


@pytest.mark.anyio
async def test_second_resolution_pass_skips_already_resolved():
    """Running resolution check twice doesn't double-count or re-resolve."""
    trader = PaperTrader(taker_fee_rate=0.0)
    c1 = _contract("NYC", 72, 74, "cond_1")
    trader.record_trade(_signal("BUY_YES", "tok_1"), c1, entry_price=0.50, amount_usd=2.0)

    gamma = AsyncMock()
    gamma.fetch_market_resolution = AsyncMock(return_value=True)

    checker = ResolutionChecker(gamma=gamma, paper_trader=trader)

    r1 = await checker.check_resolutions()
    assert r1["resolved"] == 1

    # Second pass — already resolved, nothing to do
    r2 = await checker.check_resolutions()
    assert r2["checked"] == 0
    assert r2["resolved"] == 0
    gamma.fetch_market_resolution.assert_called_once()  # Only called in first pass
    assert trader.total_pnl() == pytest.approx(2.0)  # PnL unchanged
