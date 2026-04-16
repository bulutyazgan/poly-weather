"""In-memory log of all prediction signals, including SKIPs.

Logging every signal (not just trades) prevents survivorship bias and enables
rigorous backtesting analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.data.models import MarketContract, RegimeClassification, TradingSignal


@dataclass
class SignalLogEntry:
    """A single logged prediction signal with context."""

    signal: TradingSignal
    station_id: str
    regime: RegimeClassification
    model_probability: float
    market_probability: float
    contract: MarketContract | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PredictionLog:
    """In-memory log of all prediction signals (including SKIPs)."""

    def __init__(self, max_entries: int = 50_000) -> None:
        self._entries: list[SignalLogEntry] = []
        self._max_entries = max_entries

    def log(self, entry: SignalLogEntry) -> None:
        """Append a signal log entry, trimming oldest if at capacity."""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def get_all(self) -> list[SignalLogEntry]:
        """Return all entries."""
        return list(self._entries)

    def count(self) -> int:
        """Return total number of logged entries."""
        return len(self._entries)

    def get_by_station(self, station_id: str) -> list[SignalLogEntry]:
        """Return entries matching the given station_id."""
        return [e for e in self._entries if e.station_id == station_id]

    def get_by_regime(self, confidence: str) -> list[SignalLogEntry]:
        """Return entries matching the given regime confidence level."""
        return [e for e in self._entries if e.regime.confidence == confidence]

    def export(self) -> list[dict]:
        """Export all logged signals as a list of flat dicts for analysis."""
        records: list[dict] = []
        for e in self._entries:
            rec = {
                "market_id": e.signal.market_id,
                "direction": e.signal.direction,
                "action": e.signal.action,
                "skip_reason": e.signal.skip_reason,
                "edge": e.signal.edge,
                "kelly_size": e.signal.kelly_size,
                "signal_timestamp": e.signal.timestamp.isoformat(),
                "station_id": e.station_id,
                "regime": e.regime.regime,
                "regime_confidence": e.regime.confidence,
                "model_probability": e.model_probability,
                "market_probability": e.market_probability,
                "contract": e.contract.model_dump() if e.contract else None,
                "logged_at": e.timestamp.isoformat(),
            }
            records.append(rec)
        return records
