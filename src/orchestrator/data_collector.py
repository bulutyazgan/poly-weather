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

_WIDE_SPREAD_THRESHOLD = 0.50  # spreads wider than this are unusable
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
    # Full hourly ensemble series (for per-contract date matching)
    gfs_ensemble_all: list[EnsembleForecast] = field(default_factory=list)
    ecmwf_ensemble_all: list[EnsembleForecast] = field(default_factory=list)
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
                    snap.gfs_ensemble_all = gfs or []
                    snap.gfs_ensemble = _pick_daily_forecast(gfs, now) if gfs else None
                except Exception:
                    logger.exception("GFS ensemble failed for %s", station.station_id)

            async def _fetch_ecmwf():
                try:
                    ecmwf = await self._weather.fetch_ensemble(station, model="ecmwf")
                    snap.ecmwf_ensemble_all = ecmwf or []
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

            # Build prices from Gamma API data first (reliable aggregated prices).
            # Fall back to CLOB order book only for tokens where Gamma has no price.
            clob_needed: list[str] = []
            for c in city_contracts:
                gamma_price = _gamma_market_price(c, now)
                if gamma_price is not None:
                    snap.market_prices[c.token_id] = gamma_price
                    # Build a synthetic NO-token price from the YES price
                    if c.no_token_id:
                        snap.market_prices[c.no_token_id] = MarketPrice(
                            token_id=c.no_token_id,
                            timestamp=now,
                            bid=max(0.0, 1.0 - gamma_price.ask),
                            ask=min(1.0, 1.0 - gamma_price.bid),
                            mid=1.0 - gamma_price.mid,
                            volume_24h=gamma_price.volume_24h,
                        )
                else:
                    clob_needed.append(c.token_id)
                    if c.no_token_id:
                        clob_needed.append(c.no_token_id)

            # CLOB fallback for any tokens missing Gamma prices
            async def _fetch_price(token_id: str):
                if not token_id:
                    return
                try:
                    price = await self._clob.get_market_price(token_id)
                    if price is not None:
                        snap.market_prices[token_id] = price
                except Exception:
                    logger.exception("Price fetch failed for %s", token_id)

            if clob_needed:
                await asyncio.gather(*[_fetch_price(tid) for tid in clob_needed])

            return snap

        # Collect stations sequentially to respect API rate limits
        # (per-station weather fetches are still concurrent)
        snapshots = []
        for city, station in stations.items():
            snap = await _collect_station(city, station)
            snapshots.append(snap)

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


def _gamma_market_price(contract: MarketContract, now: datetime) -> MarketPrice | None:
    """Build a MarketPrice from Gamma API fields embedded in the contract.

    Gamma provides aggregated ``bestBid``, ``bestAsk``, and ``outcomePrices``
    which are far more reliable than the raw CLOB order book (often nearly
    empty for weather markets).  Returns None only if Gamma had no price at
    all, in which case the caller should fall back to the CLOB book.
    """
    bid = contract.gamma_best_bid
    ask = contract.gamma_best_ask
    outcome_price = contract.gamma_outcome_price

    # Need at least *some* price signal
    if bid is None and ask is None and outcome_price is None:
        return None

    # Derive best available bid/ask/mid
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    elif outcome_price is not None:
        mid = outcome_price
        # Approximate bid/ask from outcome price when only one side exists
        bid = bid if bid is not None else max(0.0, mid - 0.01)
        ask = ask if ask is not None else min(1.0, mid + 0.01)
    else:
        # Only one side available
        mid = bid if bid is not None else ask  # type: ignore[assignment]
        bid = bid if bid is not None else max(0.0, mid - 0.01)
        ask = ask if ask is not None else min(1.0, mid + 0.01)

    return MarketPrice(
        token_id=contract.token_id,
        timestamp=now,
        bid=bid,
        ask=ask,
        mid=mid,
        volume_24h=contract.volume_24h,
    )


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
