"""Tests for the calibration and verification statistics system."""

import math

import numpy as np
import pytest

from src.prediction.calibration import (
    BrierScore,
    CUSUMMonitor,
    IsotonicCalibrator,
    ReliabilityDiagram,
)


# ── BrierScore tests ───────────────────────────────────────────────────────


class TestBrierScore:
    def test_brier_score_perfect(self):
        """All forecasts correct with 100% confidence → score = 0.0."""
        forecasts = [1.0, 1.0, 0.0, 0.0]
        outcomes = [True, True, False, False]
        assert BrierScore.compute(forecasts, outcomes) == pytest.approx(0.0)

    def test_brier_score_worst(self):
        """All forecasts wrong with 100% confidence → score = 1.0."""
        forecasts = [1.0, 1.0, 0.0, 0.0]
        outcomes = [False, False, True, True]
        assert BrierScore.compute(forecasts, outcomes) == pytest.approx(1.0)

    def test_brier_score_climatology(self):
        """50% forecasts for 50% base rate → score = 0.25."""
        forecasts = [0.5] * 100
        outcomes = [True] * 50 + [False] * 50
        assert BrierScore.compute(forecasts, outcomes) == pytest.approx(0.25)

    def test_brier_skill_score_positive(self):
        """Model better than climatology → BSS > 0."""
        # Decent forecasts: 0.8 for True events, 0.2 for False events
        forecasts = [0.8] * 50 + [0.2] * 50
        outcomes = [True] * 50 + [False] * 50
        bss = BrierScore.skill_score(forecasts, outcomes)
        assert bss > 0.0

    def test_brier_skill_score_zero(self):
        """Model equals climatology → BSS = 0."""
        forecasts = [0.5] * 100
        outcomes = [True] * 50 + [False] * 50
        bss = BrierScore.skill_score(forecasts, outcomes)
        assert bss == pytest.approx(0.0, abs=1e-10)

    def test_brier_decomposition(self):
        """Decomposition components reconstruct to total BS."""
        rng = np.random.default_rng(42)
        forecasts = rng.uniform(0, 1, 200).tolist()
        outcomes = [bool(rng.random() < f) for f in forecasts]

        result = BrierScore.decompose(forecasts, outcomes, n_bins=10)

        assert "reliability" in result
        assert "resolution" in result
        assert "uncertainty" in result
        assert "total" in result

        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert result["total"] == pytest.approx(reconstructed, abs=1e-10)

        # Total from decomposition approximates direct compute (difference is
        # within-bin variance, which shrinks with more bins or more data)
        direct = BrierScore.compute(forecasts, outcomes)
        assert result["total"] == pytest.approx(direct, abs=0.02)

    def test_brier_score_empty_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            BrierScore.compute([], [])

    def test_brier_score_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            BrierScore.compute([0.5, 0.5], [True])


# ── ReliabilityDiagram tests ──────────────────────────────────────────────


class TestReliabilityDiagram:
    def test_reliability_perfect_calibration(self):
        """Forecasts of 0.3 verify 30% of the time, 0.7 verify 70% → near diagonal."""
        rng = np.random.default_rng(123)
        n = 2000
        forecasts_low = [0.3] * n
        outcomes_low = [bool(rng.random() < 0.3) for _ in range(n)]
        forecasts_high = [0.7] * n
        outcomes_high = [bool(rng.random() < 0.7) for _ in range(n)]

        forecasts = forecasts_low + forecasts_high
        outcomes = outcomes_low + outcomes_high

        result = ReliabilityDiagram.compute(forecasts, outcomes, n_bins=10)

        # Find the bins containing our forecasts
        for center, obs_freq, fc_mean in zip(
            result["bin_centers"], result["observed_freq"], result["forecast_mean"]
        ):
            # The observed frequency should be close to the forecast mean
            assert obs_freq == pytest.approx(fc_mean, abs=0.05)

    def test_reliability_overconfident(self):
        """Forecasts of 0.9 only verify 60% → overconfident."""
        rng = np.random.default_rng(999)
        n = 1000
        forecasts = [0.9] * n
        outcomes = [bool(rng.random() < 0.6) for _ in range(n)]

        result = ReliabilityDiagram.compute(forecasts, outcomes, n_bins=10)

        # The bin containing 0.9 should have observed_freq ≈ 0.6, well below 0.9
        for center, obs_freq, fc_mean in zip(
            result["bin_centers"], result["observed_freq"], result["forecast_mean"]
        ):
            if abs(fc_mean - 0.9) < 0.05:
                assert obs_freq < 0.75  # clearly overconfident

    def test_reliability_bin_counts(self):
        """Each bin has correct sample count."""
        forecasts = [0.15] * 30 + [0.85] * 70
        outcomes = [True] * 15 + [False] * 15 + [True] * 50 + [False] * 20

        result = ReliabilityDiagram.compute(forecasts, outcomes, n_bins=10)

        total = sum(result["bin_counts"])
        assert total == 100

        # Should have exactly 2 non-empty bins
        assert len(result["bin_counts"]) == 2

    def test_reliability_empty_bins_excluded(self):
        """Bins with no forecasts are not included."""
        # All forecasts in one bin
        forecasts = [0.55] * 50
        outcomes = [True] * 25 + [False] * 25

        result = ReliabilityDiagram.compute(forecasts, outcomes, n_bins=10)

        assert len(result["bin_centers"]) == 1
        assert result["bin_counts"] == [50]


# ── IsotonicCalibrator tests ─────────────────────────────────────────────


class TestIsotonicCalibrator:
    def test_isotonic_fit_and_transform(self):
        """Fit on overconfident data, transform should improve calibration."""
        rng = np.random.default_rng(77)
        n = 500
        # Overconfident: forecast 0.9 but only 60% verify
        raw = rng.uniform(0.6, 1.0, n).tolist()
        outcomes = [bool(rng.random() < 0.5) for _ in range(n)]

        cal = IsotonicCalibrator()
        cal.fit(raw, outcomes)

        calibrated = cal.transform(raw)

        # Calibrated values should be pulled toward the true base rate (~0.5)
        assert np.mean(calibrated) < np.mean(raw)

    def test_isotonic_preserves_ordering(self):
        """Higher raw probabilities should map to higher (or equal) calibrated probabilities."""
        rng = np.random.default_rng(55)
        n = 300
        raw = sorted(rng.uniform(0, 1, n).tolist())
        outcomes = [bool(rng.random() < p) for p in raw]

        cal = IsotonicCalibrator()
        cal.fit(raw, outcomes)
        calibrated = cal.transform(raw)

        # Isotonic regression guarantees monotonicity
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1] + 1e-10

    def test_isotonic_bounds(self):
        """Output always in [0, 1]."""
        rng = np.random.default_rng(33)
        n = 200
        raw = rng.uniform(0, 1, n).tolist()
        outcomes = [bool(rng.random() < 0.5) for _ in range(n)]

        cal = IsotonicCalibrator()
        cal.fit(raw, outcomes)
        calibrated = cal.transform(raw)

        assert all(0.0 <= v <= 1.0 for v in calibrated)

    def test_isotonic_unfitted_raises(self):
        """Calling transform before fit raises RuntimeError."""
        cal = IsotonicCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform([0.5])


# ── CUSUMMonitor tests ───────────────────────────────────────────────────


class TestCUSUMMonitor:
    def test_cusum_no_alarm_stable(self):
        """Stable performance → no alarm triggered."""
        monitor = CUSUMMonitor(threshold=5.0, drift=0.5)
        # Small residuals around zero
        for _ in range(50):
            alarm = monitor.update(0.1)
        assert not alarm
        assert not monitor.alarm

    def test_cusum_alarm_on_degradation(self):
        """After sustained negative performance shift → alarm triggered."""
        monitor = CUSUMMonitor(threshold=3.0, drift=0.0)
        # Large positive residuals (model worse than expected)
        alarm_triggered = False
        for _ in range(20):
            if monitor.update(1.0):
                alarm_triggered = True
                break
        assert alarm_triggered
        assert monitor.alarm

    def test_cusum_reset_after_alarm(self):
        """After alarm and reset, counter starts fresh."""
        monitor = CUSUMMonitor(threshold=2.0, drift=0.0)
        # Trigger alarm
        for _ in range(10):
            monitor.update(1.0)
        assert monitor.alarm

        monitor.reset()
        assert not monitor.alarm
        assert monitor.cusum_pos == 0.0
        assert monitor.cusum_neg == 0.0

        # Small residuals should not trigger alarm
        for _ in range(5):
            monitor.update(0.1)
        assert not monitor.alarm

    def test_cusum_threshold_sensitivity(self):
        """Lower threshold → earlier alarm."""
        residuals = [0.5] * 20

        # High threshold
        high = CUSUMMonitor(threshold=8.0, drift=0.0)
        high_alarm_step = None
        for i, r in enumerate(residuals):
            if high.update(r):
                high_alarm_step = i
                break

        # Low threshold
        low = CUSUMMonitor(threshold=2.0, drift=0.0)
        low_alarm_step = None
        for i, r in enumerate(residuals):
            if low.update(r):
                low_alarm_step = i
                break

        assert low_alarm_step is not None
        assert low_alarm_step < (high_alarm_step if high_alarm_step is not None else len(residuals))
