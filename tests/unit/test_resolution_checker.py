"""Tests for the resolution checker."""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.models import MarketContract, TradingSignal
from src.verification.paper_trader import PaperTrader
from src.verification.resolution_checker import ResolutionChecker


def _make_contract(condition_id: str = "cond_1") -> MarketContract:
    return MarketContract(
        token_id="tok_1",
        no_token_id="tok_2",
        condition_id=condition_id,
        question="Will NYC high be 72-74F?",
        city="NYC",
        resolution_date=date(2026, 4, 15),
        temp_bucket_low=72.0,
        temp_bucket_high=74.0,
        outcome="Yes",
    )


def _make_signal(direction: str = "BUY_YES") -> TradingSignal:
    return TradingSignal(
        market_id="tok_1",
        direction=direction,
        action="TRADE",
        edge=0.15,
        kelly_size=2.0,
        timestamp=datetime.now(timezone.utc),
    )


class TestResolutionChecker:

    @pytest.mark.asyncio
    async def test_resolves_winning_trade(self):
        """BUY_YES trade resolved with outcome=True computes positive PnL."""
        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(return_value=True)

        trader = PaperTrader()
        signal = _make_signal("BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.50, amount_usd=2.0)

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["checked"] == 1
        assert result["resolved"] == 1
        assert result["errors"] == 0
        assert trader.total_pnl() == 2.0  # 2.0 * (1/0.5 - 1) = 2.0

    @pytest.mark.asyncio
    async def test_resolves_losing_trade(self):
        """BUY_YES trade resolved with outcome=False computes negative PnL."""
        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(return_value=False)

        trader = PaperTrader()
        signal = _make_signal("BUY_YES")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.50, amount_usd=2.0)

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["resolved"] == 1
        assert trader.total_pnl() == -2.0

    @pytest.mark.asyncio
    async def test_buy_no_winning_trade(self):
        """BUY_NO trade wins when outcome=False."""
        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(return_value=False)

        trader = PaperTrader()
        signal = _make_signal("BUY_NO")
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.70, amount_usd=1.50)

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["resolved"] == 1
        # BUY_NO at YES=0.70 (NO costs 0.30): pnl = 1.50 * (1/0.30 - 1) ≈ 3.50
        assert abs(trader.total_pnl() - 3.50) < 0.01

    @pytest.mark.asyncio
    async def test_skips_unresolved_market(self):
        """Markets that haven't resolved yet return outcome=None and are skipped."""
        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(return_value=None)

        trader = PaperTrader()
        signal = _make_signal()
        contract = _make_contract()
        trader.record_trade(signal, contract, entry_price=0.50, amount_usd=2.0)

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["checked"] == 1
        assert result["resolved"] == 0
        assert trader.total_pnl() == 0.0

    @pytest.mark.asyncio
    async def test_no_unresolved_trades(self):
        """When all trades are already resolved, check does nothing."""
        gamma = AsyncMock()
        trader = PaperTrader()

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result == {"checked": 0, "resolved": 0, "errors": 0}
        gamma.fetch_market_resolution.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(self):
        """API errors are counted but don't crash the checker."""
        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(side_effect=Exception("API down"))

        trader = PaperTrader()
        signal = _make_signal()
        contract = _make_contract()
        trader.record_trade(signal, contract, entry_price=0.50, amount_usd=2.0)

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["checked"] == 1
        assert result["resolved"] == 0
        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_multiple_trades_different_outcomes(self):
        """Multiple trades with mixed resolutions are handled correctly."""
        outcomes = {"cond_1": True, "cond_2": False, "cond_3": None}

        async def mock_resolution(condition_id):
            return outcomes.get(condition_id)

        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(side_effect=mock_resolution)

        trader = PaperTrader()
        for i, cid in enumerate(["cond_1", "cond_2", "cond_3"]):
            sig = _make_signal()
            contract = _make_contract(condition_id=cid)
            trader.record_trade(sig, contract, entry_price=0.50, amount_usd=2.0)

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["checked"] == 3
        assert result["resolved"] == 2  # cond_1 and cond_2
        assert result["errors"] == 0
        # cond_1: BUY_YES, outcome=True, pnl=+2.0
        # cond_2: BUY_YES, outcome=False, pnl=-2.0
        assert abs(trader.total_pnl() - 0.0) < 0.01  # net zero

    @pytest.mark.asyncio
    async def test_already_resolved_trades_not_rechecked(self):
        """Trades that were already resolved are not sent to the API again."""
        gamma = AsyncMock()
        gamma.fetch_market_resolution = AsyncMock(return_value=True)

        trader = PaperTrader()
        signal = _make_signal()
        contract = _make_contract()
        tid = trader.record_trade(signal, contract, entry_price=0.50, amount_usd=2.0)
        trader.resolve(tid, True)  # Already resolved

        checker = ResolutionChecker(gamma=gamma, paper_trader=trader)
        result = await checker.check_resolutions()

        assert result["checked"] == 0
        gamma.fetch_market_resolution.assert_not_called()
