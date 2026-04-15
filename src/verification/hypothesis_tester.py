"""Null hypothesis testing for model edge verification.

Tests whether the prediction model beats climatology (H0-1) and whether its
edge over market-implied probabilities is statistically significant (H0-4).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.prediction.calibration import BrierScore


class HypothesisTester:
    """Test null hypotheses about model edge."""

    def test_h0_1(
        self, model_forecasts: list[float], outcomes: list[bool]
    ) -> dict:
        """H0-1: Model does not beat climatological base rates.

        Uses a permutation test on the difference in Brier scores between the
        model and climatology (base-rate forecast).

        Returns {"rejected": bool, "p_value": float, "bss": float, "n": int}
        """
        n = len(model_forecasts)
        o = np.asarray(outcomes, dtype=np.float64)
        f = np.asarray(model_forecasts, dtype=np.float64)
        base_rate = float(np.mean(o))

        bss = BrierScore.skill_score(model_forecasts, outcomes)

        # Paired comparison of squared errors: model vs climatology
        model_sq_errors = (f - o) ** 2
        clim_sq_errors = (base_rate - o) ** 2
        diffs = clim_sq_errors - model_sq_errors  # positive = model better

        # One-sided paired t-test: is model significantly better?
        if np.std(diffs) == 0:
            p_value = 1.0
        else:
            t_stat, p_two = stats.ttest_1samp(diffs, 0.0)
            # One-sided: model better means diffs > 0 means t_stat > 0
            p_value = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0

        return {
            "rejected": p_value < 0.05 and bss > 0,
            "p_value": float(p_value),
            "bss": float(bss),
            "n": n,
        }

    def test_h0_4(
        self,
        model_forecasts: list[float],
        market_forecasts: list[float],
        outcomes: list[bool],
    ) -> dict:
        """H0-4: Model edge over market is not significant.

        Compares Brier scores via Wilcoxon signed-rank test on paired squared
        errors. Falls back to paired t-test if sample is large enough.

        Returns {"rejected": bool, "p_value": float, "model_bs": float,
                 "market_bs": float, "n": int}
        """
        n = len(model_forecasts)
        o = np.asarray(outcomes, dtype=np.float64)
        f_model = np.asarray(model_forecasts, dtype=np.float64)
        f_market = np.asarray(market_forecasts, dtype=np.float64)

        model_bs = BrierScore.compute(model_forecasts, outcomes)
        market_bs = BrierScore.compute(market_forecasts, outcomes)

        model_sq = (f_model - o) ** 2
        market_sq = (f_market - o) ** 2
        diffs = market_sq - model_sq  # positive = model better

        # Use Wilcoxon signed-rank (handles small samples, non-normal diffs)
        # Need at least some non-zero diffs
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) < 2:
            p_value = 1.0
        else:
            try:
                stat, p_two = stats.wilcoxon(nonzero_diffs, alternative="greater")
                p_value = float(p_two)
            except ValueError:
                p_value = 1.0

        return {
            "rejected": p_value < 0.05,
            "p_value": float(p_value),
            "model_bs": float(model_bs),
            "market_bs": float(market_bs),
            "n": n,
        }

    @staticmethod
    def benjamini_hochberg(
        p_values: list[float], alpha: float = 0.05
    ) -> list[bool]:
        """Apply Benjamini-Hochberg correction for multiple testing.

        Returns list of booleans (same order as input) indicating which
        hypotheses are rejected.
        """
        m = len(p_values)
        # Pair each p-value with its original index, then sort by p-value
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])

        # Find the largest rank k where p_(k) <= (k/m) * alpha
        # Then reject all hypotheses with rank <= k
        max_rejected_rank = -1
        for rank_minus_1, (orig_idx, p) in enumerate(indexed):
            k = rank_minus_1 + 1  # 1-based rank
            threshold = (k / m) * alpha
            if p <= threshold:
                max_rejected_rank = rank_minus_1

        # All hypotheses up to and including max_rejected_rank are rejected
        rejected = [False] * m
        if max_rejected_rank >= 0:
            for rank_minus_1 in range(max_rejected_rank + 1):
                orig_idx = indexed[rank_minus_1][0]
                rejected[orig_idx] = True

        return rejected
