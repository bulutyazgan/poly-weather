"""Async database repository for TradeBot.

Uses asyncpg connection pool directly for simplicity and performance.
All public methods are exception-safe — they log errors and return
None/[] rather than crashing the pipeline.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing asyncpg; fall back gracefully if not installed
# ---------------------------------------------------------------------------

try:
    import asyncpg  # type: ignore[import-untyped]
    _HAS_ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    _HAS_ASYNCPG = False


class TradeRepository:
    """Async PostgreSQL repository backed by an asyncpg connection pool."""

    def __init__(self) -> None:
        self._pool: Any | None = None

    # -- lifecycle -----------------------------------------------------------

    async def create_pool(self, database_url: str) -> None:
        """Create the connection pool.

        Accepts postgresql:// or postgresql+asyncpg:// URLs — the
        +asyncpg dialect suffix is stripped automatically.
        """
        if not _HAS_ASYNCPG:
            logger.warning("asyncpg not installed — database features disabled")
            return

        # Normalise SQLAlchemy-style URLs
        url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        self._pool = await asyncpg.create_pool(url, min_size=2, max_size=10)
        logger.info("Database pool created")

    async def close_pool(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    # -- internal helpers ----------------------------------------------------

    async def _execute(self, sql: str, *args: Any) -> None:
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(sql, *args)
        except Exception:
            logger.exception("DB execute failed: %s", sql[:120])

    async def _query(self, sql: str, *args: Any) -> list[dict]:
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *args)
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("DB query failed: %s", sql[:120])
            return []

    # -- write operations ----------------------------------------------------

    async def save_signal(
        self,
        *,
        trade_id: str | None,
        market_id: str,
        station_id: str,
        direction: str,
        action: str,
        edge: float,
        kelly_size: float = 0.0,
        entry_price: float | None = None,
    ) -> None:
        await self._execute(
            """INSERT INTO trades
               (trade_id, market_id, station_id, direction, action, edge, kelly_size, entry_price)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            trade_id, market_id, station_id, direction, action, edge, kelly_size, entry_price,
        )

    async def save_trade(
        self,
        *,
        trade_id: str,
        market_id: str,
        station_id: str,
        direction: str,
        edge: float,
        kelly_size: float,
        entry_price: float,
    ) -> None:
        await self.save_signal(
            trade_id=trade_id,
            market_id=market_id,
            station_id=station_id,
            direction=direction,
            action="TRADE",
            edge=edge,
            kelly_size=kelly_size,
            entry_price=entry_price,
        )

    async def save_forecast(
        self,
        *,
        station_id: str,
        model_name: str,
        run_time: datetime,
        valid_date: date,
        ensemble_mean: float | None = None,
        ensemble_std: float | None = None,
        mos_high: float | None = None,
        mos_low: float | None = None,
        hrrr_temp_f: float | None = None,
        hrrr_dewpoint_f: float | None = None,
        hrrr_wind_kt: float | None = None,
    ) -> None:
        await self._execute(
            """INSERT INTO forecasts
               (station_id, model_name, run_time, valid_date,
                ensemble_mean, ensemble_std, mos_high, mos_low,
                hrrr_temp_f, hrrr_dewpoint_f, hrrr_wind_kt)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
            station_id, model_name, run_time, valid_date,
            ensemble_mean, ensemble_std, mos_high, mos_low,
            hrrr_temp_f, hrrr_dewpoint_f, hrrr_wind_kt,
        )

    async def save_observation(
        self,
        *,
        station_id: str,
        observed_time: datetime,
        temp_f: float,
        dewpoint_f: float | None = None,
        wind_speed_kt: float | None = None,
        wind_dir: float | None = None,
        pressure_mb: float | None = None,
        cloud_cover: str | None = None,
    ) -> None:
        await self._execute(
            """INSERT INTO observations
               (station_id, observed_time, temp_f, dewpoint_f,
                wind_speed_kt, wind_dir, pressure_mb, cloud_cover)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
            station_id, observed_time, temp_f, dewpoint_f,
            wind_speed_kt, wind_dir, pressure_mb, cloud_cover,
        )

    async def save_market_price(
        self,
        *,
        token_id: str,
        ts: datetime,
        bid: float,
        ask: float,
        mid: float,
        volume_24h: float | None = None,
    ) -> None:
        await self._execute(
            """INSERT INTO market_prices
               (token_id, ts, bid, ask, mid, volume_24h)
               VALUES ($1,$2,$3,$4,$5,$6)""",
            token_id, ts, bid, ask, mid, volume_24h,
        )

    async def save_regime(
        self,
        *,
        station_id: str,
        valid_date: date,
        regime: str,
        confidence: str,
        confidence_score: float = 0.5,
        spread_percentile: float | None = None,
        active_flags: list[str] | None = None,
    ) -> None:
        await self._execute(
            """INSERT INTO regimes
               (station_id, valid_date, regime, confidence,
                confidence_score, spread_percentile, active_flags)
               VALUES ($1,$2,$3,$4,$5,$6,$7)
               ON CONFLICT (station_id, valid_date) DO UPDATE SET
                 regime = EXCLUDED.regime,
                 confidence = EXCLUDED.confidence,
                 confidence_score = EXCLUDED.confidence_score,
                 spread_percentile = EXCLUDED.spread_percentile,
                 active_flags = EXCLUDED.active_flags""",
            station_id, valid_date, regime, confidence,
            confidence_score, spread_percentile, active_flags or [],
        )

    # -- read operations -----------------------------------------------------

    async def get_signals(
        self, station_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        if station_id:
            return await self._query(
                "SELECT * FROM trades WHERE station_id = $1 ORDER BY created_at DESC LIMIT $2",
                station_id, limit,
            )
        return await self._query(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT $1", limit
        )

    async def get_resolved_trades(self) -> list[dict]:
        return await self._query(
            "SELECT * FROM trades WHERE resolved = TRUE ORDER BY created_at DESC"
        )

    async def get_calibration_data(
        self, station_id: str | None = None
    ) -> list[dict]:
        if station_id:
            return await self._query(
                "SELECT * FROM calibration WHERE station_id = $1", station_id
            )
        return await self._query("SELECT * FROM calibration")

    async def get_forecasts(
        self, station_id: str, valid_date: date
    ) -> list[dict]:
        return await self._query(
            "SELECT * FROM forecasts WHERE station_id = $1 AND valid_date = $2",
            station_id, valid_date,
        )

    async def upsert_calibration(
        self,
        *,
        station_id: str,
        regime: str,
        brier_score: float,
        sample_count: int,
    ) -> None:
        await self._execute(
            """INSERT INTO calibration (station_id, regime, brier_score, sample_count)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (station_id, regime) DO UPDATE SET
                 brier_score = EXCLUDED.brier_score,
                 sample_count = EXCLUDED.sample_count,
                 updated_at = NOW()""",
            station_id, regime, brier_score, sample_count,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_repository: TradeRepository | None = None


def get_repository() -> TradeRepository:
    """Return the module-level singleton, creating it if needed."""
    global _repository
    if _repository is None:
        _repository = TradeRepository()
    return _repository
