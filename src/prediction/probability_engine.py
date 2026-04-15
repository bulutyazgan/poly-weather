"""MOS-anchored ensemble probability estimation."""
from __future__ import annotations

from scipy import stats

from src.data.models import EnsembleForecast, MOSForecast
from src.config.stations import Station

CLIMATOLOGICAL_STD = 4.0  # fallback when no ensemble data available

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
    ) -> stats.rv_continuous:
        """Build a normal distribution anchored on MOS with ensemble-derived spread.

        Centre = MOS high_f + station lapse rate correction.
        Spread = weighted combination of ensemble stds, floored at min_spread
        to prevent overconfident distributions from narrow ensemble agreement.
        """
        center = mos.high_f + station.lapse_rate_correction_f

        # Determine spread from ensembles
        if gfs_ensemble is not None and ecmwf_ensemble is not None:
            combined_std = (
                self.gfs_weight * gfs_ensemble.std
                + self.ecmwf_weight * ecmwf_ensemble.std
            )
        elif ecmwf_ensemble is not None:
            combined_std = ecmwf_ensemble.std
        elif gfs_ensemble is not None:
            combined_std = gfs_ensemble.std
        else:
            combined_std = CLIMATOLOGICAL_STD

        # Floor: ensemble spread at one hour underestimates Tmax uncertainty
        combined_std = max(combined_std, self.min_spread)

        return stats.norm(loc=center, scale=combined_std)

    def compute_bucket_probability(
        self,
        distribution: stats.rv_continuous,
        bucket_low: float,
        bucket_high: float,
        prob_floor: float = 0.005,
        prob_ceil: float = 0.995,
    ) -> float:
        """P(bucket_low <= T < bucket_high) via CDF difference.

        Clamps to [prob_floor, prob_ceil] to prevent overconfident 0%/100%
        predictions that create phantom edges against any market price.
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
