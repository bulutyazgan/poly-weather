"""Pydantic data models for the weather prediction trading system."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, computed_field


class EnsembleForecast(BaseModel):
    """Ensemble weather model forecast (GFS or ECMWF)."""

    model_name: Literal["gfs", "ecmwf"]
    run_time: datetime
    valid_time: datetime
    station_id: str
    members: list[float]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean(self) -> float:
        return float(np.mean(self.members))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def std(self) -> float:
        return float(np.std(self.members, ddof=0)) if len(self.members) > 1 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def member_count(self) -> int:
        return len(self.members)


class MOSForecast(BaseModel):
    """Model Output Statistics forecast."""

    station_id: str
    run_time: datetime
    valid_date: date
    high_f: float
    low_f: float


class HRRRForecast(BaseModel):
    """High-Resolution Rapid Refresh forecast."""

    station_id: str
    run_time: datetime
    valid_time: datetime
    temp_f: float
    dewpoint_f: float | None = None
    wind_speed_kt: float | None = None


class Observation(BaseModel):
    """Weather observation from a station."""

    station_id: str
    observed_time: datetime
    temp_f: float
    dewpoint_f: float | None = None
    wind_speed_kt: float | None = None
    cloud_cover: str | None = None


class MarketContract(BaseModel):
    """A Polymarket temperature prediction contract."""

    token_id: str  # YES outcome token
    no_token_id: str = ""  # NO outcome token
    condition_id: str
    question: str  # e.g., "Will NYC high be 72-73°F on Apr 16?"
    city: str
    resolution_date: date
    temp_bucket_low: float  # lower bound (Fahrenheit)
    temp_bucket_high: float  # upper bound (Fahrenheit)
    outcome: Literal["Yes", "No"]


class MarketPrice(BaseModel):
    """Snapshot of market prices."""

    token_id: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    volume_24h: float


class RegimeClassification(BaseModel):
    """Weather regime classification for a station/date."""

    station_id: str
    valid_date: date
    regime: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    confidence_score: float = 0.5
    ensemble_spread_percentile: float = 50.0
    active_flags: list[str] = []
    regime_description: str = ""


class TradingSignal(BaseModel):
    """A signal indicating whether and how to trade."""

    market_id: str
    direction: Literal["BUY_YES", "BUY_NO"]
    action: Literal["TRADE", "SKIP"]
    edge: float
    kelly_size: float
    timestamp: datetime


class TradeRecord(BaseModel):
    """Record of an executed (or paper) trade."""

    trade_id: str
    market_id: str
    direction: Literal["BUY_YES", "BUY_NO"]
    amount_usd: float
    price: float
    timestamp: datetime
    resolution_outcome: str | None = None


class CalibrationRecord(BaseModel):
    """Historical calibration metrics for a model/station pair."""

    station_id: str
    model_name: str
    date_range_start: date
    date_range_end: date
    bias_f: float
    rmse_f: float
    sample_count: int
