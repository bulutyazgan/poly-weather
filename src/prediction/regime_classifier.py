"""Weather regime classification from ensemble spread and physical flags."""
from __future__ import annotations

from datetime import date

import numpy as np
from scipy.stats import percentileofscore

from src.data.models import RegimeClassification

# Default spread thresholds when no history is provided
_DEFAULT_LOW_THRESHOLD = 1.5   # spreads below this → ~30th pctile
_DEFAULT_HIGH_THRESHOLD = 3.0  # spreads above this → ~60th pctile


def _is_bimodal(members: list[float]) -> bool:
    """Simple bimodality test: check if largest gap in sorted members exceeds
    1.5 times the median gap, indicating two separated clusters."""
    if len(members) < 6:
        return False
    s = sorted(members)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    if not gaps:
        return False
    median_gap = float(np.median(gaps))
    max_gap = max(gaps)
    # Also check that max gap exceeds 2x overall std as an absolute guard
    overall_std = float(np.std(s, ddof=1))
    if overall_std <= 0:
        return False
    return max_gap > 1.5 * median_gap and max_gap > overall_std


def _is_santa_ana_wind_direction(direction: float) -> bool:
    """Santa Ana winds come from NE quadrant: 315-360 or 0-90 degrees."""
    return direction >= 315 or direction <= 90


class RegimeClassifier:
    """Classify weather regime from ensemble spread and physical flags."""

    def __init__(self, spread_history: list[float] | None = None) -> None:
        self.spread_history = spread_history or []

    def _spread_percentile(self, spread: float) -> float:
        """Compute percentile of current spread within historical distribution."""
        if self.spread_history:
            return float(percentileofscore(self.spread_history, spread, kind="rank"))
        # Fallback to default thresholds
        if spread < _DEFAULT_LOW_THRESHOLD:
            return 20.0
        if spread < _DEFAULT_HIGH_THRESHOLD:
            return 45.0
        return 70.0

    def classify(
        self,
        station_id: str,
        valid_date: date,
        ensemble_spread: float,
        pressure_tendency_3h: float = 0.0,
        wind_direction_change: float = 0.0,
        cape: float = 0.0,
        precip_forecast: float = 0.0,
        wind_700mb_speed: float = 0.0,
        wind_700mb_direction: float = 0.0,
        surface_rh: float = 50.0,
        cloud_cover_trend: float = 0.0,
        ensemble_members: list[float] | None = None,
        station_flags: list[str] | None = None,
    ) -> RegimeClassification:
        station_flags = station_flags or []
        active_flags: list[str] = []
        low_flags: list[str] = []
        high_override_flags: list[str] = []

        # --- Compute spread percentile ---
        spread_pct = self._spread_percentile(ensemble_spread)

        # --- Check physical flags ---

        # Frontal passage
        if abs(pressure_tendency_3h) > 3 and wind_direction_change > 90:
            low_flags.append("frontal_passage")

        # Convective
        if cape > 1000 and precip_forecast > 0.1:
            low_flags.append("convective")

        # Bimodal ensemble
        if ensemble_members is not None and _is_bimodal(ensemble_members):
            low_flags.append("bimodal_ensemble")

        # Santa Ana (HIGH override)
        if (
            "santa_ana" in station_flags
            and wind_700mb_speed > 25
            and _is_santa_ana_wind_direction(wind_700mb_direction)
            and surface_rh < 20
        ):
            high_override_flags.append("santa_ana")

        # Post-frontal clear (HIGH override)
        if pressure_tendency_3h > 2 and cloud_cover_trend < -20:
            high_override_flags.append("post_frontal_clear")

        # Lake breeze risk (LOW)
        if "lake_effect" in station_flags and 45 < wind_direction_change < 135:
            low_flags.append("lake_breeze_risk")

        # Chinook (HIGH override)
        if (
            "chinook" in station_flags
            and wind_700mb_speed > 30
            and 250 < wind_700mb_direction < 310
        ):
            high_override_flags.append("chinook")

        active_flags = low_flags + high_override_flags

        # --- Determine confidence ---
        if high_override_flags and not low_flags:
            confidence = "HIGH"
        elif low_flags:
            confidence = "LOW"
        elif spread_pct < 30:
            confidence = "HIGH"
        elif spread_pct < 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # --- Confidence score ---
        base_score = 1.0 - (spread_pct / 100.0)
        if low_flags:
            base_score = min(base_score, 0.3)
        if high_override_flags and not low_flags:
            base_score = max(base_score, 0.8)
        confidence_score = max(0.0, min(1.0, base_score))

        # --- Build description ---
        if active_flags:
            regime_desc = f"Flags: {', '.join(active_flags)}. Spread pctile: {spread_pct:.0f}."
        else:
            regime_desc = f"Spread pctile: {spread_pct:.0f}. Confidence: {confidence}."

        regime_name = confidence.lower() + "_confidence"

        return RegimeClassification(
            station_id=station_id,
            valid_date=valid_date,
            regime=regime_name,
            confidence=confidence,
            confidence_score=confidence_score,
            ensemble_spread_percentile=spread_pct,
            active_flags=active_flags,
            regime_description=regime_desc,
        )
