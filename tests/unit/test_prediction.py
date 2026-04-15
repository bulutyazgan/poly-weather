"""Tests for prediction engine and regime classifier."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest
from scipy import stats

from src.config.stations import Station
from src.data.models import EnsembleForecast, MOSForecast, RegimeClassification
from src.prediction.probability_engine import ProbabilityEngine
from src.prediction.regime_classifier import RegimeClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mos(tmax: float = 72.0, tmin: float = 55.0) -> MOSForecast:
    return MOSForecast(
        station_id="KNYC",
        run_time=datetime(2026, 4, 15, 6),
        valid_date=date(2026, 4, 16),
        high_f=tmax,
        low_f=tmin,
    )


def _make_ensemble(
    model: str = "gfs",
    mean: float = 72.0,
    std: float = 2.0,
    n: int = 30,
) -> EnsembleForecast:
    rng = np.random.default_rng(42)
    members = rng.normal(mean, std, n).tolist()
    return EnsembleForecast(
        model_name=model,
        run_time=datetime(2026, 4, 15, 0),
        valid_time=datetime(2026, 4, 16, 18),
        station_id="KNYC",
        members=members,
    )


def _make_station(
    lapse_correction: float = 0.0,
    flags: list[str] | None = None,
) -> Station:
    """Create a station with a controlled lapse rate correction.

    To get the desired correction we set elevation_ft and model_grid_elevation_ft
    so that -(elev - grid_elev) * 3.57/1000 == lapse_correction.
    Solving: elev - grid_elev = -lapse_correction * 1000 / 3.57
    """
    diff = -lapse_correction * 1000 / 3.57
    return Station(
        station_id="KNYC",
        city="NYC",
        lat=40.78,
        lon=-73.97,
        elevation_ft=100 + int(round(diff)),
        model_grid_elevation_ft=100,
        flags=flags or [],
    )


# ===========================================================================
# ProbabilityEngine tests
# ===========================================================================

class TestProbabilityEngine:
    """Tests for ProbabilityEngine."""

    def setup_method(self) -> None:
        self.engine = ProbabilityEngine()

    # --- bucket probability tests ---

    def test_compute_bucket_probability_centered(self) -> None:
        """MOS=72, std=2 → P(71 <= T < 73) should be ~40% (centered bucket)."""
        mos = _make_mos(tmax=72.0)
        station = _make_station()
        dist = stats.norm(loc=72.0, scale=2.0)
        prob = self.engine.compute_bucket_probability(dist, 71.0, 73.0)

        # scipy reference
        expected = stats.norm.cdf(73.0, 72.0, 2.0) - stats.norm.cdf(71.0, 72.0, 2.0)
        assert abs(prob - expected) < 1e-6
        # Rough sanity: ~38%
        assert 0.30 < prob < 0.45

    def test_compute_bucket_probability_tail(self) -> None:
        """MOS=72, std=2 → P(78 <= T < 79) should be very small (<5%)."""
        dist = stats.norm(loc=72.0, scale=2.0)
        prob = self.engine.compute_bucket_probability(dist, 78.0, 79.0)
        assert prob < 0.05

    def test_compute_bucket_probability_wide_spread(self) -> None:
        """Large std=5 → probability more spread, centred bucket smaller."""
        dist_narrow = stats.norm(loc=72.0, scale=2.0)
        dist_wide = stats.norm(loc=72.0, scale=5.0)
        prob_narrow = self.engine.compute_bucket_probability(dist_narrow, 71.0, 73.0)
        prob_wide = self.engine.compute_bucket_probability(dist_wide, 71.0, 73.0)
        # Wider spread → lower probability in the centre bucket
        assert prob_wide < prob_narrow

    def test_compute_all_bucket_probabilities_sum_to_one(self) -> None:
        """Probabilities across realistic range sum to ~1.

        Uses 15 buckets (60-75F) centred on the distribution — this
        captures >99% of the mass while keeping tail-clamping (prob_floor=0.005)
        from inflating the sum significantly.
        """
        dist = stats.norm(loc=68.0, scale=2.0)
        buckets = [(float(t), float(t + 1)) for t in range(60, 75)]
        probs = self.engine.compute_all_bucket_probabilities(dist, buckets)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.05

    def test_probability_floor_prevents_zero(self) -> None:
        """Far-away bucket gets clamped to prob_floor, never exactly 0.0."""
        dist = stats.norm(loc=55.0, scale=2.0)
        # 85-90F is ~15 std deviations away — raw prob is 0.0
        prob = self.engine.compute_bucket_probability(dist, 85.0, 90.0)
        assert prob == 0.005  # default floor
        assert prob > 0.0

    def test_probability_ceil_prevents_one(self) -> None:
        """Very wide bucket around mean gets clamped below 1.0."""
        dist = stats.norm(loc=70.0, scale=0.1)
        # Bucket covers almost all mass
        prob = self.engine.compute_bucket_probability(dist, 50.0, 90.0)
        assert prob == 0.995  # default ceil
        assert prob < 1.0

    def test_mos_anchored_distribution(self) -> None:
        """Combined distribution centred on MOS, spread between the two ensembles."""
        mos = _make_mos(tmax=75.0)
        gfs = _make_ensemble(model="gfs", mean=74.0, std=2.5, n=30)
        ecmwf = _make_ensemble(model="ecmwf", mean=76.0, std=2.0, n=50)
        station = _make_station()

        dist = self.engine.compute_distribution(mos, gfs, ecmwf, station)

        # Centre should be MOS (75), not ensemble means
        assert abs(dist.mean() - 75.0) < 0.1
        # Combined std = 0.4*gfs.std + 0.6*ecmwf.std (weighted by engine weights)
        expected_std = 0.4 * gfs.std + 0.6 * ecmwf.std
        assert abs(dist.std() - expected_std) < 0.01
        # Sanity: combined std should be between the smaller and larger ensemble stds
        assert min(gfs.std, ecmwf.std) <= dist.std() <= max(gfs.std, ecmwf.std)

    def test_lapse_rate_correction_applied(self) -> None:
        """Station lapse_rate_correction_f should shift distribution centre."""
        mos = _make_mos(tmax=72.0)
        station = _make_station(lapse_correction=-0.8)

        dist = self.engine.compute_distribution(mos, None, None, station)

        expected_center = 72.0 + (-0.8)
        assert abs(dist.mean() - expected_center) < 0.1

    def test_fallback_std_when_no_ensembles(self) -> None:
        """Without ensembles, climatological std of 4.0F used."""
        mos = _make_mos(tmax=72.0)
        station = _make_station()
        dist = self.engine.compute_distribution(mos, None, None, station)
        assert abs(dist.std() - 4.0) < 0.01

    def test_single_ensemble_used_alone(self) -> None:
        """When only one ensemble is available, use its std directly."""
        mos = _make_mos(tmax=72.0)
        ecmwf = _make_ensemble(model="ecmwf", mean=73.0, std=2.0, n=50)
        station = _make_station()

        dist = self.engine.compute_distribution(mos, None, ecmwf, station)
        # std should be close to ecmwf's std
        assert abs(dist.std() - ecmwf.std) < 0.3


# ===========================================================================
# RegimeClassifier tests
# ===========================================================================

class TestRegimeClassifier:
    """Tests for RegimeClassifier."""

    def setup_method(self) -> None:
        # Build a spread history where 1.0 is 20th pctile, 2.0 is ~45th, 3.5 is ~70th
        rng = np.random.default_rng(0)
        self.spread_history = sorted(rng.exponential(2.0, 100).tolist())
        self.classifier = RegimeClassifier(spread_history=self.spread_history)

    def test_high_confidence_low_spread(self) -> None:
        """Spread at ~20th percentile, no flags → HIGH."""
        # pick a spread at the 20th percentile of our history
        spread = float(np.percentile(self.spread_history, 20))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
        )
        assert isinstance(result, RegimeClassification)
        assert result.confidence == "HIGH"

    def test_medium_confidence_moderate_spread(self) -> None:
        """Spread at ~45th percentile → MEDIUM."""
        spread = float(np.percentile(self.spread_history, 45))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
        )
        assert result.confidence == "MEDIUM"

    def test_low_confidence_high_spread(self) -> None:
        """Spread at ~70th percentile → LOW."""
        spread = float(np.percentile(self.spread_history, 70))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
        )
        assert result.confidence == "LOW"

    def test_frontal_passage_forces_low(self) -> None:
        """Large pressure tendency + wind shift → LOW with frontal_passage flag."""
        spread = float(np.percentile(self.spread_history, 10))  # low spread
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            pressure_tendency_3h=4.0,
            wind_direction_change=100.0,
        )
        assert result.confidence == "LOW"
        assert "frontal_passage" in result.active_flags

    def test_convective_forces_low(self) -> None:
        """CAPE > 1000 and precip → LOW with convective flag."""
        spread = float(np.percentile(self.spread_history, 10))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            cape=1500.0,
            precip_forecast=0.2,
        )
        assert result.confidence == "LOW"
        assert "convective" in result.active_flags

    def test_bimodal_ensemble_forces_low(self) -> None:
        """Bimodal members → LOW with bimodal_ensemble flag."""
        # Create obviously bimodal distribution: two well-separated clusters
        members = [60.0] * 15 + [80.0] * 15
        spread = float(np.std(members, ddof=1))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            ensemble_members=members,
        )
        assert result.confidence == "LOW"
        assert "bimodal_ensemble" in result.active_flags

    def test_santa_ana_overrides_to_high(self) -> None:
        """LA station with Santa Ana conditions → HIGH with santa_ana flag."""
        # Even moderate spread, santa_ana override should push to HIGH
        spread = float(np.percentile(self.spread_history, 45))
        result = self.classifier.classify(
            station_id="KLAX",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            wind_700mb_speed=30.0,
            wind_700mb_direction=45.0,  # NE (between 315-360 or 0-90)
            surface_rh=15.0,
            station_flags=["santa_ana"],
        )
        assert result.confidence == "HIGH"
        assert "santa_ana" in result.active_flags

    def test_post_frontal_clear_high(self) -> None:
        """Rising pressure + clearing skies → HIGH with post_frontal_clear flag."""
        spread = float(np.percentile(self.spread_history, 25))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            pressure_tendency_3h=3.0,
            cloud_cover_trend=-30.0,
        )
        assert result.confidence == "HIGH"
        assert "post_frontal_clear" in result.active_flags

    def test_classify_returns_regime_classification(self) -> None:
        """Return type is RegimeClassification model."""
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=1.5,
        )
        assert isinstance(result, RegimeClassification)
        assert result.station_id == "KNYC"
        assert result.valid_date == date(2026, 4, 16)
