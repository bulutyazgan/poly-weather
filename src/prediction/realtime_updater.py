"""Same-day morning refinement for Tmax probability estimates.

Uses three signals available on the morning of resolution:
1. Morning observation anchoring (12Z obs vs model prediction)
2. Cloud cover discrepancy (satellite/obs vs model forecast)
3. Latest HRRR run (captures actual boundary layer state)
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MORNING_CORRELATION = 0.6   # How much a morning temp error propagates to Tmax
_CLOUD_ADJUSTMENT_F = 3.0    # Max °F adjustment for cloud cover mismatch
_HRRR_WEIGHT = 0.4           # Blend weight for latest HRRR Tmax
_MIN_STD_F = 1.5             # Floor for standard deviation (°F)
_MAX_CONFIDENCE_BOOST = 0.30  # Cap on confidence boost from refinement


# ---------------------------------------------------------------------------
# Cloud cover helpers
# ---------------------------------------------------------------------------

_CLOUD_CATEGORIES = {
    "CLR": 0.0,
    "FEW": 0.15,
    "SCT": 0.40,
    "BKN": 0.70,
    "OVC": 1.0,
}


def _parse_cloud_category(code: str | None) -> float | None:
    """Convert METAR sky condition code to fractional cloud cover.

    Returns None if code is unrecognised.
    """
    if code is None:
        return None
    return _CLOUD_CATEGORIES.get(code.upper())


def _cloud_delta_f(
    model_cloud_cover: float,
    obs_cloud_cover: float,
) -> float:
    """Compute Tmax adjustment (°F) from cloud cover mismatch.

    Clear when model said cloudy  →  positive (warmer)
    Cloudy when model said clear  →  negative (cooler)
    """
    diff = model_cloud_cover - obs_cloud_cover  # positive = model over-predicted clouds
    return diff * _CLOUD_ADJUSTMENT_F


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinedEstimate:
    """Result of morning refinement."""

    refined_tmax_f: float
    refined_std_f: float
    confidence_boost: float   # 0.0–0.30 improvement in regime confidence
    refinement_source: str    # human-readable summary of signals used


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class RealtimeUpdater:
    """Refine overnight Tmax estimate using morning observations.

    Designed to run once on the morning of resolution, after the 12Z
    HRRR becomes available (~9-10 AM Eastern).
    """

    def __init__(
        self,
        morning_correlation: float = _MORNING_CORRELATION,
        cloud_adjustment_f: float = _CLOUD_ADJUSTMENT_F,
        hrrr_weight: float = _HRRR_WEIGHT,
    ) -> None:
        self.morning_correlation = morning_correlation
        self.cloud_adjustment_f = cloud_adjustment_f
        self.hrrr_weight = hrrr_weight

    def refine(
        self,
        overnight_tmax_f: float,
        overnight_std_f: float,
        *,
        model_12z_predicted_temp_f: float | None = None,
        obs_12z_temp_f: float | None = None,
        model_cloud_cover: float | None = None,
        obs_cloud_cover: str | float | None = None,
        hrrr_tmax_f: float | None = None,
    ) -> RefinedEstimate:
        """Produce a refined Tmax estimate from available morning data.

        Parameters
        ----------
        overnight_tmax_f : float
            Tmax estimate from the overnight model run.
        overnight_std_f : float
            Standard deviation of the overnight estimate.
        model_12z_predicted_temp_f : float, optional
            What the model predicted the 12Z temperature would be.
        obs_12z_temp_f : float, optional
            Observed 12Z temperature.
        model_cloud_cover : float, optional
            Model-forecast cloud fraction (0-1).
        obs_cloud_cover : str or float, optional
            Observed cloud cover — METAR code (e.g. "CLR") or fraction.
        hrrr_tmax_f : float, optional
            Latest HRRR Tmax forecast.

        Returns
        -------
        RefinedEstimate
        """
        tmax = overnight_tmax_f
        std = overnight_std_f
        boost = 0.0
        sources: list[str] = []

        # --- Signal 1: Morning observation anchoring ---
        if model_12z_predicted_temp_f is not None and obs_12z_temp_f is not None:
            error_12z = obs_12z_temp_f - model_12z_predicted_temp_f
            tmax += self.morning_correlation * error_12z
            std *= 0.90  # tighter after observational update
            boost += 0.10
            sources.append(f"obs_anchor({error_12z:+.1f}F)")

        # --- Signal 2: Cloud cover discrepancy ---
        if model_cloud_cover is not None and obs_cloud_cover is not None:
            if isinstance(obs_cloud_cover, str):
                obs_frac = _parse_cloud_category(obs_cloud_cover)
            else:
                obs_frac = float(obs_cloud_cover)

            if obs_frac is not None:
                delta = _cloud_delta_f(model_cloud_cover, obs_frac)
                tmax += delta
                std *= 0.95
                boost += 0.05
                sources.append(f"cloud({delta:+.1f}F)")

        # --- Signal 3: Latest HRRR blend ---
        if hrrr_tmax_f is not None:
            tmax = (1.0 - self.hrrr_weight) * tmax + self.hrrr_weight * hrrr_tmax_f
            std *= 0.85  # HRRR captures boundary layer → much tighter
            boost += 0.10
            sources.append(f"hrrr({hrrr_tmax_f:.1f}F)")

        # Floor std and cap boost
        std = max(std, _MIN_STD_F)
        boost = min(boost, _MAX_CONFIDENCE_BOOST)

        if not sources:
            sources.append("no_refinement")

        return RefinedEstimate(
            refined_tmax_f=tmax,
            refined_std_f=std,
            confidence_boost=boost,
            refinement_source=", ".join(sources),
        )
