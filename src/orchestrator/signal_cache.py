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

    Maintains a reverse lookup from NO token IDs to their parent YES token's
    cache entry so that NO token price updates can trigger BUY_NO evaluations.
    """

    def __init__(self) -> None:
        self._signals: dict[str, CachedSignal] = {}
        self._no_to_yes: dict[str, str] = {}  # no_token_id → yes_token_id
        self._last_update: datetime | None = None
        self.updated: asyncio.Event = asyncio.Event()

    def update(self, signals: dict[str, CachedSignal]) -> None:
        """Full replacement of cache. Sets the ``updated`` event."""
        self._signals = dict(signals)
        self._no_to_yes = {}
        for yes_tid, cached in self._signals.items():
            if cached.contract.no_token_id:
                self._no_to_yes[cached.contract.no_token_id] = yes_tid
        self._last_update = datetime.now(tz=timezone.utc)
        self.updated.set()

    def get(self, token_id: str) -> CachedSignal | None:
        """Return cached signal for a token, or None.

        Looks up by YES token_id first, then checks if token_id is a
        NO token and returns the parent YES token's cache entry.
        """
        cached = self._signals.get(token_id)
        if cached is not None:
            return cached
        # Reverse lookup: NO token → parent YES token
        yes_tid = self._no_to_yes.get(token_id)
        if yes_tid is not None:
            return self._signals.get(yes_tid)
        return None

    def yes_token_for(self, token_id: str) -> str | None:
        """Return the YES token_id for a given token.

        Returns token_id itself if it's a YES token in the cache,
        or the parent YES token_id if it's a NO token.
        Returns None if not found.
        """
        if token_id in self._signals:
            return token_id
        return self._no_to_yes.get(token_id)

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
