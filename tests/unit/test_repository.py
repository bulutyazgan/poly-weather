"""Tests for the database repository (mocked asyncpg pool)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.repository import TradeRepository, get_repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pool() -> MagicMock:
    """Create a mock asyncpg pool with acquire() context manager.

    asyncpg's pool.acquire() returns a sync context-manager-like object
    that supports ``async with``.  We model this with a MagicMock whose
    acquire() returns a plain object with async __aenter__/__aexit__.
    """
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    pool.close = AsyncMock()

    return pool


# ===================================================================
# Pool lifecycle
# ===================================================================

class TestPoolLifecycle:
    def test_not_connected_initially(self):
        repo = TradeRepository()
        assert repo.is_connected is False

    @pytest.mark.asyncio
    async def test_connected_after_create(self):
        repo = TradeRepository()
        pool = _mock_pool()
        with patch("src.db.repository.asyncpg") as mock_pg:
            mock_pg.create_pool = AsyncMock(return_value=pool)
            await repo.create_pool("postgresql://localhost/test")
        assert repo.is_connected is True

    @pytest.mark.asyncio
    async def test_close_pool(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.close_pool()
        assert repo.is_connected is False

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        repo = TradeRepository()
        await repo.close_pool()  # should not raise
        assert repo.is_connected is False


# ===================================================================
# Write operations (save_*)
# ===================================================================

class TestSaveSignal:
    @pytest.mark.asyncio
    async def test_save_signal(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.save_signal(
            trade_id="t1",
            market_id="m1",
            station_id="KNYC",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=2.5,
            entry_price=0.60,
        )
        # Should have called execute on the connection
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_signal_no_pool(self):
        repo = TradeRepository()
        # Should not raise when pool is None
        await repo.save_signal(
            trade_id=None,
            market_id="m1",
            station_id="KNYC",
            direction="BUY_YES",
            action="SKIP",
            edge=0.02,
        )


class TestSaveTrade:
    @pytest.mark.asyncio
    async def test_save_trade_delegates_to_save_signal(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.save_trade(
            trade_id="t1",
            market_id="m1",
            station_id="KNYC",
            direction="BUY_YES",
            edge=0.10,
            kelly_size=2.5,
            entry_price=0.60,
        )
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()


class TestSaveForecast:
    @pytest.mark.asyncio
    async def test_save_forecast(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.save_forecast(
            station_id="KNYC",
            model_name="gfs",
            run_time=datetime.now(timezone.utc),
            valid_date=date(2026, 4, 16),
            ensemble_mean=72.0,
            ensemble_std=3.5,
        )
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()


class TestSaveObservation:
    @pytest.mark.asyncio
    async def test_save_observation(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.save_observation(
            station_id="KNYC",
            observed_time=datetime.now(timezone.utc),
            temp_f=68.0,
        )
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()


class TestSaveMarketPrice:
    @pytest.mark.asyncio
    async def test_save_market_price(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.save_market_price(
            token_id="tok_abc",
            ts=datetime.now(timezone.utc),
            bid=0.58,
            ask=0.62,
            mid=0.60,
            volume_24h=5000.0,
        )
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()


class TestSaveRegime:
    @pytest.mark.asyncio
    async def test_save_regime(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.save_regime(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            regime="zonal",
            confidence="HIGH",
            confidence_score=0.85,
            spread_percentile=25.0,
            active_flags=["post_frontal_clear"],
        )
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()


# ===================================================================
# Read operations
# ===================================================================

class TestQueryOperations:
    @pytest.mark.asyncio
    async def test_get_signals_returns_list(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        result = await repo.get_signals()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_signals_by_station(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        result = await repo.get_signals(station_id="KNYC")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_resolved_trades(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        result = await repo.get_resolved_trades()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_calibration_data(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        result = await repo.get_calibration_data()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_forecasts(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        result = await repo.get_forecasts("KNYC", date(2026, 4, 16))
        assert result == []

    @pytest.mark.asyncio
    async def test_query_no_pool(self):
        repo = TradeRepository()
        result = await repo.get_signals()
        assert result == []


class TestUpsertCalibration:
    @pytest.mark.asyncio
    async def test_upsert_calibration(self):
        repo = TradeRepository()
        repo._pool = _mock_pool()
        await repo.upsert_calibration(
            station_id="KNYC",
            regime="HIGH",
            brier_score=0.18,
            sample_count=50,
        )
        ctx = repo._pool.acquire.return_value
        conn = await ctx.__aenter__()
        conn.execute.assert_awaited_once()


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_repository_returns_same_instance(self):
        import src.db.repository as mod
        mod._repository = None  # reset
        r1 = get_repository()
        r2 = get_repository()
        assert r1 is r2
        mod._repository = None  # cleanup

    def test_get_repository_creates_instance(self):
        import src.db.repository as mod
        mod._repository = None
        r = get_repository()
        assert isinstance(r, TradeRepository)
        mod._repository = None


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_execute_error_logged_not_raised(self):
        repo = TradeRepository()
        pool = _mock_pool()
        ctx = pool.acquire.return_value
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("DB down"))
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)
        repo._pool = pool

        # Should not raise
        await repo.save_signal(
            trade_id=None,
            market_id="m1",
            station_id="KNYC",
            direction="BUY_YES",
            action="SKIP",
            edge=0.01,
        )

    @pytest.mark.asyncio
    async def test_query_error_returns_empty(self):
        repo = TradeRepository()
        pool = _mock_pool()
        ctx = pool.acquire.return_value
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=RuntimeError("DB down"))
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)
        repo._pool = pool

        result = await repo.get_signals()
        assert result == []
