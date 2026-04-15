"""Data collection and storage for forward validation."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.config.stations import get_stations, Station
from src.data.models import (
    EnsembleForecast,
    HRRRForecast,
    MarketContract,
    MarketPrice,
    Observation,
)
from src.data.weather_client import OpenMeteoClient, MesonetClient
from src.data.polymarket_client import GammaClient, CLOBClient

logger = logging.getLogger(__name__)


@dataclass
class DataSnapshot:
    """A point-in-time snapshot of all data for one station."""

    station_id: str
    timestamp: datetime
    gfs_ensemble: EnsembleForecast | None = None
    ecmwf_ensemble: EnsembleForecast | None = None
    hrrr: list[HRRRForecast] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    market_contracts: list[MarketContract] = field(default_factory=list)
    market_prices: dict[str, MarketPrice] = field(default_factory=dict)


class DataCollector:
    """Collects and stores matched (forecast, market, outcome) data."""

    def __init__(
        self,
        weather: OpenMeteoClient,
        mesonet: MesonetClient,
        gamma: GammaClient,
        clob: CLOBClient,
    ) -> None:
        self._weather = weather
        self._mesonet = mesonet
        self._gamma = gamma
        self._clob = clob
        self._snapshots: list[DataSnapshot] = []
        self._max_snapshots = 5000  # ~5 stations × 7 runs/day × ~140 days
        self._outcomes: dict[str, bool] = {}  # token_id -> resolved outcome

    async def collect_snapshot(self) -> list[DataSnapshot]:
        """Collect current data for all stations. Returns snapshots."""
        now = datetime.now(tz=timezone.utc)
        stations = get_stations()
        snapshots: list[DataSnapshot] = []

        # Fetch markets once (shared across stations)
        try:
            all_contracts = await self._gamma.fetch_weather_markets()
        except Exception:
            logger.exception("Failed to fetch weather markets")
            all_contracts = []

        async def _collect_station(city: str, station: Station) -> DataSnapshot:
            snap = DataSnapshot(station_id=station.station_id, timestamp=now)

            # Fetch weather data concurrently per station
            async def _fetch_gfs():
                try:
                    gfs = await self._weather.fetch_ensemble(station, model="gfs")
                    snap.gfs_ensemble = _pick_daily_forecast(gfs, now) if gfs else None
                except Exception:
                    logger.exception("GFS ensemble failed for %s", station.station_id)

            async def _fetch_ecmwf():
                try:
                    ecmwf = await self._weather.fetch_ensemble(station, model="ecmwf")
                    snap.ecmwf_ensemble = _pick_daily_forecast(ecmwf, now) if ecmwf else None
                except Exception:
                    logger.exception("ECMWF ensemble failed for %s", station.station_id)

            async def _fetch_hrrr():
                try:
                    snap.hrrr = await self._weather.fetch_hrrr(station)
                except Exception:
                    logger.exception("HRRR failed for %s", station.station_id)

            async def _fetch_obs():
                try:
                    snap.observations = await self._mesonet.fetch_observations(
                        station.station_id,
                        start=now - timedelta(hours=24),
                        end=now,
                    )
                except Exception:
                    logger.exception("Observations failed for %s", station.station_id)

            await asyncio.gather(_fetch_gfs(), _fetch_ecmwf(), _fetch_hrrr(), _fetch_obs())

            # Filter contracts for this city
            city_contracts = [c for c in all_contracts if c.city == city]
            snap.market_contracts = city_contracts

            # Fetch prices concurrently
            async def _fetch_price(contract):
                try:
                    price = await self._clob.get_market_price(contract.token_id)
                    if price is not None:
                        snap.market_prices[contract.token_id] = price
                except Exception:
                    logger.exception("Price fetch failed for %s", contract.token_id)

            if city_contracts:
                await asyncio.gather(*[_fetch_price(c) for c in city_contracts])

            return snap

        # Collect all stations concurrently
        snapshots = await asyncio.gather(
            *[_collect_station(city, station) for city, station in stations.items()]
        )

        self._snapshots.extend(snapshots)
        # Trim to rolling window to prevent unbounded memory growth
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]
        return snapshots

    def record_outcome(self, token_id: str, outcome: bool) -> None:
        """Record a resolved market outcome."""
        self._outcomes[token_id] = outcome
        logger.info("Recorded outcome for %s: %s", token_id, outcome)

    def get_matched_records(self) -> list[dict]:
        """Return matched (forecast, market, outcome) records for analysis.

        Only returns records where we have a snapshot with a market price
        AND a resolved outcome for that token.
        """
        records: list[dict] = []
        for snap in self._snapshots:
            for token_id, price in snap.market_prices.items():
                if token_id not in self._outcomes:
                    continue
                records.append({
                    "station_id": snap.station_id,
                    "forecast_time": snap.timestamp.isoformat(),
                    "model_prob": None,  # filled in by pipeline during analysis
                    "market_prob": price.mid,
                    "actual_outcome": self._outcomes[token_id],
                })
        return records


def _pick_daily_forecast(
    forecasts: list[EnsembleForecast],
    now: datetime | None = None,
) -> EnsembleForecast | None:
    """Pick the ensemble forecast closest to +24h valid time."""
    if not forecasts:
        return None
    if now is None:
        now = datetime.now(tz=timezone.utc)
    target = now + timedelta(hours=24)
    return min(forecasts, key=lambda f: abs((f.valid_time - target).total_seconds()))
