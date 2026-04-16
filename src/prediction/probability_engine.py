"""MOS-anchored ensemble probability estimation."""
from __future__ import annotations

import math
from datetime import date

from scipy import stats

from src.data.models import EnsembleForecast, MOSForecast
from src.config.stations import Station

CLIMATOLOGICAL_STD = 4.0  # fallback when no ensemble data available

# Monthly average high temperatures (°F) — NOAA 30-year normals (approximate)
CLIMO_MONTHLY_HIGH: dict[str, list[float]] = {
    "KNYC": [39, 42, 50, 62, 72, 81, 85, 84, 76, 65, 54, 43],
    "KORD": [31, 35, 46, 59, 70, 80, 84, 82, 75, 62, 48, 35],
    "KLAX": [68, 68, 69, 71, 72, 75, 81, 82, 81, 77, 72, 67],
    "KDEN": [45, 48, 55, 62, 71, 82, 90, 88, 79, 66, 53, 44],
    "KMIA": [76, 78, 80, 83, 87, 89, 91, 91, 89, 86, 82, 78],
}
CLIMO_MONTHLY_STD = 8.0  # typical monthly Tmax variability

# Minimum spread floor based on NWP Tmax forecast skill.
# Ensemble spread at a single valid time measures inter-model disagreement
# at that hour, NOT the full uncertainty about the daily high.  The true
# Tmax uncertainty includes: (a) which hour the peak occurs, (b) boundary
# layer mixing, (c) cloud timing, (d) instrument exposure.
# Published RMSE values for GFS/ECMWF Tmax: day 0 ~2-3°F, day 1 ~3-4°F.
MIN_SPREAD_STD = 2.5


class ProbabilityEngine:
    """Build probability distributions from MOS + ensemble forecasts."""

    def __init__(
        self,
        ecmwf_weight: float = 0.6,
        gfs_weight: float = 0.4,
        min_spread: float = MIN_SPREAD_STD,
    ) -> None:
        self.ecmwf_weight = ecmwf_weight
        self.gfs_weight = gfs_weight
        self.min_spread = min_spread

    def compute_distribution(
        self,
        mos: MOSForecast,
        gfs_ensemble: EnsembleForecast | None,
        ecmwf_ensemble: EnsembleForecast | None,
        station: Station,
        valid_date: date | None = None,
    ) -> stats.rv_continuous:
        """Build a normal distribution anchored on MOS with ensemble-derived spread.

        Centre = MOS high_f + station lapse rate correction.
        Spread = weighted combination of ensemble stds, floored at min_spread
        to prevent overconfident distributions from narrow ensemble agreement.

        Climatological sanity check: when the forecast centre deviates far
        from NOAA 30-year monthly normals, the spread is widened to reflect
        additional uncertainty.  This prevents the model from being
        overconfident on extreme outlier forecasts (e.g. 88°F in April NYC).
        """
        center = mos.high_f + station.lapse_rate_correction_f

        # Determine spread from ensembles.
        # Two sources of uncertainty:
        #   1. Within-model spread (member disagreement at peak hour)
        #   2. Between-model spread (GFS vs ECMWF Tmax disagreement)
        # Combine via quadrature: sqrt(within² + between²)
        if gfs_ensemble is not None and ecmwf_ensemble is not None:
            within_std = (
                self.gfs_weight * gfs_ensemble.std
                + self.ecmwf_weight * ecmwf_ensemble.std
            )
            between_std = abs(gfs_ensemble.mean - ecmwf_ensemble.mean) / 2.0
            combined_std = math.sqrt(within_std**2 + between_std**2)
        elif ecmwf_ensemble is not None:
            combined_std = ecmwf_ensemble.std
        elif gfs_ensemble is not None:
            combined_std = gfs_ensemble.std
        else:
            combined_std = CLIMATOLOGICAL_STD

        # Floor: ensemble spread at one hour underestimates Tmax uncertainty
        combined_std = max(combined_std, self.min_spread)

        # Climatological sanity check — widen spread for outlier forecasts
        if valid_date is not None and station.station_id in CLIMO_MONTHLY_HIGH:
            month_idx = valid_date.month - 1  # 0-indexed
            climo_normal = CLIMO_MONTHLY_HIGH[station.station_id][month_idx]
            deviation = abs(center - climo_normal)
            if deviation > 2 * CLIMO_MONTHLY_STD:
                combined_std *= 1.5
            elif deviation > 1.5 * CLIMO_MONTHLY_STD:
                combined_std *= 1.25

        return stats.norm(loc=center, scale=combined_std)

    def compute_bucket_probability(
        self,
        distribution: stats.rv_continuous,
        bucket_low: float,
        bucket_high: float,
        prob_floor: float = 0.0005,
        prob_ceil: float = 0.9995,
    ) -> float:
        """P(bucket_low <= T < bucket_high) via CDF difference.

        Clamps to [prob_floor, prob_ceil] to prevent overconfident 0%/100%
        predictions that create phantom edges against any market price.

        The floor was 0.005 (0.5%), which inflated far-tail buckets by up
        to 500x (true prob 0.001% → 0.5%).  Reduced to 0.05% to stay above
        numerical noise while not materially distorting tail probabilities.
        """
        raw = float(distribution.cdf(bucket_high) - distribution.cdf(bucket_low))
        return max(prob_floor, min(prob_ceil, raw))

    def compute_all_bucket_probabilities(
        self,
        distribution: stats.rv_continuous,
        buckets: list[tuple[float, float]],
    ) -> dict[tuple[float, float], float]:
        """Compute probability for each bucket."""
        return {
            (lo, hi): self.compute_bucket_probability(distribution, lo, hi)
            for lo, hi in buckets
        }
