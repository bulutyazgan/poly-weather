"""Calibration and verification statistics for probability forecasts.

Provides Brier score computation, reliability diagrams, isotonic regression
calibration, and CUSUM sequential monitoring for model degradation detection.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _validate_inputs(forecasts: list[float], outcomes: list[bool]) -> None:
    """Validate forecast/outcome inputs."""
    if len(forecasts) == 0 or len(outcomes) == 0:
        raise ValueError("forecasts and outcomes must be non-empty")
    if len(forecasts) != len(outcomes):
        raise ValueError(
            f"forecasts ({len(forecasts)}) and outcomes ({len(outcomes)}) must have the same length"
        )


class BrierScore:
    """Compute Brier score and decomposition."""

    @staticmethod
    def compute(forecasts: list[float], outcomes: list[bool]) -> float:
        """BS = mean((forecast - outcome)^2)"""
        _validate_inputs(forecasts, outcomes)
        f = np.asarray(forecasts, dtype=np.float64)
        o = np.asarray(outcomes, dtype=np.float64)
        return float(np.mean((f - o) ** 2))

    @staticmethod
    def skill_score(forecasts: list[float], outcomes: list[bool]) -> float:
        """BSS = 1 - BS / BS_climatology, where BS_climatology uses base rate."""
        _validate_inputs(forecasts, outcomes)
        bs = BrierScore.compute(forecasts, outcomes)
        base_rate = np.mean(np.asarray(outcomes, dtype=np.float64))
        bs_clim = float(base_rate * (1.0 - base_rate))
        if bs_clim == 0.0:
            return 0.0
        return 1.0 - bs / bs_clim

    @staticmethod
    def decompose(
        forecasts: list[float], outcomes: list[bool], n_bins: int = 10
    ) -> dict:
        """Decompose Brier score into reliability, resolution, uncertainty.

        reliability = (1/N) * sum(n_k * (f_k - o_k)^2)  [lower is better]
        resolution = (1/N) * sum(n_k * (o_k - base_rate)^2)  [higher is better]
        uncertainty = base_rate * (1 - base_rate)
        BS = reliability - resolution + uncertainty
        """
        _validate_inputs(forecasts, outcomes)
        f = np.asarray(forecasts, dtype=np.float64)
        o = np.asarray(outcomes, dtype=np.float64)
        n = len(f)
        base_rate = float(np.mean(o))

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Use digitize: values in [0, edge1) → bin 1, etc.
        bin_indices = np.digitize(f, bin_edges[1:-1])  # 0-indexed bins 0..n_bins-1

        reliability = 0.0
        resolution = 0.0

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = int(np.sum(mask))
            if n_k == 0:
                continue
            f_k = float(np.mean(f[mask]))
            o_k = float(np.mean(o[mask]))
            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2

        reliability /= n
        resolution /= n
        uncertainty = base_rate * (1.0 - base_rate)
        total = reliability - resolution + uncertainty

        return {
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "total": total,
        }


class ReliabilityDiagram:
    """Build reliability diagram data for plotting."""

    @staticmethod
    def compute(
        forecasts: list[float], outcomes: list[bool], n_bins: int = 10
    ) -> dict:
        """Return binned calibration data, excluding empty bins.

        Bins are [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
        """
        _validate_inputs(forecasts, outcomes)
        f = np.asarray(forecasts, dtype=np.float64)
        o = np.asarray(outcomes, dtype=np.float64)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(f, bin_edges[1:-1])  # 0-indexed bins

        bin_centers: list[float] = []
        observed_freq: list[float] = []
        forecast_mean: list[float] = []
        bin_counts: list[int] = []

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = int(np.sum(mask))
            if n_k == 0:
                continue
            bin_centers.append(float((bin_edges[k] + bin_edges[k + 1]) / 2.0))
            observed_freq.append(float(np.mean(o[mask])))
            forecast_mean.append(float(np.mean(f[mask])))
            bin_counts.append(n_k)

        return {
            "bin_centers": bin_centers,
            "observed_freq": observed_freq,
            "forecast_mean": forecast_mean,
            "bin_counts": bin_counts,
        }


class IsotonicCalibrator:
    """Isotonic regression calibration for probability forecasts."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw_probabilities: list[float], outcomes: list[bool]) -> None:
        """Fit isotonic regression on historical forecasts and outcomes."""
        _validate_inputs(raw_probabilities, outcomes)
        x = np.asarray(raw_probabilities, dtype=np.float64)
        y = np.asarray(outcomes, dtype=np.float64)
        self._model.fit(x, y)
        self._fitted = True

    def transform(self, raw_probabilities: list[float]) -> list[float]:
        """Apply calibration to new forecasts. Raises RuntimeError if not fitted."""
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before calling transform")
        x = np.asarray(raw_probabilities, dtype=np.float64)
        result = self._model.predict(x)
        return result.tolist()

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class CUSUMMonitor:
    """CUSUM (Cumulative Sum) sequential monitor for detecting model degradation.

    Tracks cumulative sum of residuals. When the cumulative sum exceeds a
    threshold, an alarm is triggered indicating the model's edge has degraded.
    """

    def __init__(self, threshold: float = 2.0, drift: float = 0.0) -> None:
        """
        threshold: alarm triggers when CUSUM exceeds this value
        drift: allowable drift before counting (similar to ARL adjustment)
        """
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.alarm: bool = False

    def update(self, residual: float) -> bool:
        """Add new observation (residual = expected - actual performance).

        Positive residual means worse than expected.
        Returns True if alarm is triggered.

        cusum_pos = max(0, cusum_pos + residual - drift)
        cusum_neg = max(0, cusum_neg - residual - drift)
        alarm if cusum_pos > threshold or cusum_neg > threshold

        Once alarm triggers it stays True (sticky) until the caller
        calls ``reset()``.  This lets the pipeline block an entire
        cycle then reset to allow recovery.
        """
        self.cusum_pos = max(0.0, self.cusum_pos + residual - self.drift)
        self.cusum_neg = max(0.0, self.cusum_neg - residual - self.drift)
        if self.cusum_pos > self.threshold or self.cusum_neg > self.threshold:
            self.alarm = True
        return self.alarm

    def reset(self) -> None:
        """Reset CUSUM counters and alarm."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alarm = False
