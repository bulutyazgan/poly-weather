"""In-memory cache of per-contract model probabilities.

Bridges the forecast pipeline (writes) and the price monitor (reads).
Updated after each NWP forecast cycle with a full replacement.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from src.data.models import MarketContract, RegimeClassification


@dataclass
class CachedSignal:
    """Cached model output for a single market contract."""

    model_prob: float
    regime: RegimeClassification
    contract: MarketContract
    station_id: str
    forecast_time: datetime


class SignalCache:
    """Thread-safe (asyncio-safe) cache of per-contract model probabilities.

    The ``updated`` event is set on every ``update()`` call so that watchers
    (e.g., PriceMonitor) can react to new forecasts without polling.
    """

    def __init__(self) -> None:
        self._signals: dict[str, CachedSignal] = {}
        self._last_update: datetime | None = None
        self.updated: asyncio.Event = asyncio.Event()

    def update(self, signals: dict[str, CachedSignal]) -> None:
        """Full replacement of cache. Sets the ``updated`` event."""
        self._signals = dict(signals)
        self._last_update = datetime.now(tz=timezone.utc)
        self.updated.set()

    def get(self, token_id: str) -> CachedSignal | None:
        """Return cached signal for a token, or None."""
        return self._signals.get(token_id)

    def get_all(self) -> dict[str, CachedSignal]:
        """Return a shallow copy of all cached signals."""
        return dict(self._signals)

    @property
    def forecast_age_seconds(self) -> float:
        """Seconds since the last ``update()`` call. Inf if never updated."""
        if self._last_update is None:
            return float("inf")
        delta = datetime.now(tz=timezone.utc) - self._last_update
        return delta.total_seconds()
