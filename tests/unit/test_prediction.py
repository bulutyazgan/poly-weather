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
        captures >99% of the mass while keeping tail-clamping (prob_floor=0.0005)
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
        assert prob == 0.0005  # default floor
        assert prob > 0.0

    def test_probability_ceil_prevents_one(self) -> None:
        """Very wide bucket around mean gets clamped below 1.0."""
        dist = stats.norm(loc=70.0, scale=0.1)
        # Bucket covers almost all mass
        prob = self.engine.compute_bucket_probability(dist, 50.0, 90.0)
        assert prob == 0.9995  # default ceil
        assert prob < 1.0

    def test_mos_anchored_distribution(self) -> None:
        """Combined distribution centred on MOS, spread between the two ensembles."""
        mos = _make_mos(tmax=75.0)
        gfs = _make_ensemble(model="gfs", mean=74.0, std=4.0, n=30)
        ecmwf = _make_ensemble(model="ecmwf", mean=76.0, std=3.5, n=50)
        station = _make_station()

        dist = self.engine.compute_distribution(mos, gfs, ecmwf, station)

        # Centre should be MOS (75), not ensemble means
        assert abs(dist.mean() - 75.0) < 0.1
        # Combined std = sqrt(within² + between²)
        # within = 0.4*4.0 + 0.6*3.5 = 3.7
        # between = |74.0 - 76.0| / 2 = 1.0
        # combined = sqrt(3.7² + 1.0²) = sqrt(13.69 + 1.0) = sqrt(14.69) ≈ 3.83
        import math
        within = 0.4 * gfs.std + 0.6 * ecmwf.std
        between = abs(gfs.mean - ecmwf.mean) / 2.0
        expected_std = math.sqrt(within**2 + between**2)
        assert abs(dist.std() - expected_std) < 0.01

    def test_min_spread_floor(self) -> None:
        """Ensemble spread below min_spread (2.5°F) is floored."""
        mos = _make_mos(tmax=75.0)
        gfs = _make_ensemble(model="gfs", mean=74.0, std=0.7, n=30)
        station = _make_station()

        dist = self.engine.compute_distribution(mos, gfs, None, station)

        # GFS std=0.7 is below min_spread=2.5, so floor applies
        from src.prediction.probability_engine import MIN_SPREAD_STD
        assert abs(dist.std() - MIN_SPREAD_STD) < 0.01

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

    def test_between_model_disagreement_widens_spread(self) -> None:
        """When GFS and ECMWF disagree, std should be much wider than within-model.

        Hand calculation:
            GFS mean=68, ECMWF mean=77, both have std=1.0
            within = 0.4*1.0 + 0.6*1.0 = 1.0
            between = |68 - 77| / 2 = 4.5
            combined = sqrt(1.0² + 4.5²) = sqrt(1 + 20.25) = sqrt(21.25) ≈ 4.61
        Without this fix, combined would be max(1.0, 2.5) = 2.5 — overconfident.
        """
        import math
        mos = _make_mos(tmax=72.0)
        gfs = _make_ensemble(model="gfs", mean=68.0, std=3.0, n=30)
        ecmwf = _make_ensemble(model="ecmwf", mean=77.0, std=3.0, n=50)
        station = _make_station()

        dist = self.engine.compute_distribution(mos, gfs, ecmwf, station)

        within = 0.4 * gfs.std + 0.6 * ecmwf.std
        between = abs(gfs.mean - ecmwf.mean) / 2.0
        expected = math.sqrt(within**2 + between**2)
        assert abs(dist.std() - expected) < 0.01
        # Key: combined should be well above the floor
        assert dist.std() > 4.0

    def test_single_ensemble_used_alone(self) -> None:
        """When only one ensemble is available, use its std (floored at min_spread)."""
        mos = _make_mos(tmax=72.0)
        ecmwf = _make_ensemble(model="ecmwf", mean=73.0, std=3.5, n=50)
        station = _make_station()

        dist = self.engine.compute_distribution(mos, None, ecmwf, station)
        # std should be ecmwf's std (3.5 > min_spread 2.5)
        assert abs(dist.std() - ecmwf.std) < 0.01


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

    def test_quantized_ensemble_not_bimodal(self) -> None:
        """Tightly clustered quantized data (Open-Meteo 0.2°F steps) must NOT
        trigger false bimodal detection.

        Real example: 30 GFS members at 75-76°F with 0.2° quantization.
        median_gap ≈ 0, max_gap < 0.5°F — this is noise, not bimodality.
        The 3°F minimum gap threshold prevents this false positive.
        """
        members = [75.2, 75.2, 75.4, 75.4, 75.6, 75.6, 75.6, 75.7, 75.7,
                    75.7, 75.9, 76.1, 76.1, 76.1, 76.1, 76.1, 76.1, 76.3,
                    76.3, 76.3, 76.3, 76.3, 76.3, 76.5, 76.5, 76.5, 76.5,
                    76.5, 76.6, 76.6]
        spread = float(np.std(members, ddof=1))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            ensemble_members=members,
        )
        assert "bimodal_ensemble" not in result.active_flags
        # Low spread + no flags → HIGH confidence
        assert result.confidence == "HIGH"

    def test_genuine_bimodal_with_large_gap(self) -> None:
        """Two clusters separated by 5°F+ should still trigger bimodal.

        This represents a genuine regime split (e.g. chinook vs no-chinook).
        """
        members = [60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5,
                    70.0, 70.5, 71.0, 71.5, 72.0, 72.5, 73.0, 73.5]
        spread = float(np.std(members, ddof=1))
        result = self.classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=spread,
            ensemble_members=members,
        )
        assert "bimodal_ensemble" in result.active_flags
        assert result.confidence == "LOW"

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

    def test_default_thresholds_benign_weather_high(self) -> None:
        """Without spread_history, a 2.5°F spread (typical benign weather)
        should yield HIGH confidence, enabling the 0.08 edge threshold.

        Blended GFS/ECMWF Tmax spread of 2-3°F is normal for routine
        forecasts without frontal passages or convection.
        """
        classifier_no_history = RegimeClassifier()
        result = classifier_no_history.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=2.5,
        )
        assert result.confidence == "HIGH"

    def test_default_thresholds_midwest_spread(self) -> None:
        """Without spread_history, a 4°F spread (typical Midwest spring)
        should yield MEDIUM, not LOW.

        Bug: the old _DEFAULT_HIGH_THRESHOLD of 3.0°F made Chicago
        permanently LOW-confidence, blocking all trades for the city.
        4°F ensemble spread is normal for Midwest spring weather.
        """
        classifier_no_history = RegimeClassifier()
        result = classifier_no_history.classify(
            station_id="KORD",
            valid_date=date(2026, 4, 16),
            ensemble_spread=4.0,
        )
        assert result.confidence == "MEDIUM"

    def test_default_thresholds_genuinely_wide_spread(self) -> None:
        """Without spread_history, a 6°F spread should yield LOW."""
        classifier_no_history = RegimeClassifier()
        result = classifier_no_history.classify(
            station_id="KORD",
            valid_date=date(2026, 4, 16),
            ensemble_spread=6.0,
        )
        assert result.confidence == "LOW"

    def test_between_model_moderate_does_not_force_low(self) -> None:
        """A MEDIUM-confidence station with 4-8°F between-model spread
        should stay MEDIUM, not get pushed to LOW.

        Rationale: ensemble spread already set MEDIUM (higher edge
        threshold).  Double-penalizing with a LOW downgrade blocks all
        trading for normal GFS/ECMWF disagreement (4-7°F is routine).
        """
        classifier = RegimeClassifier()
        result = classifier.classify(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            ensemble_spread=4.0,  # → MEDIUM (45th pctile)
            between_model_spread=6.5,  # moderate disagreement
        )
        assert result.confidence == "MEDIUM", (
            f"MEDIUM + 6.5°F between-model should stay MEDIUM, got {result.confidence}"
        )
        assert "model_disagreement" in result.active_flags

    def test_between_model_high_downgrades_to_medium(self) -> None:
        """A HIGH-confidence station with moderate between-model spread
        should downgrade to MEDIUM (one step), not LOW."""
        classifier = RegimeClassifier()
        result = classifier.classify(
            station_id="KDEN",
            valid_date=date(2026, 4, 16),
            ensemble_spread=1.0,  # → HIGH (20th pctile)
            between_model_spread=5.0,  # moderate disagreement
        )
        assert result.confidence == "MEDIUM"
        assert "model_disagreement" in result.active_flags

    def test_between_model_severe_forces_low(self) -> None:
        """Severe between-model disagreement (>8°F) should force LOW
        regardless of starting confidence."""
        classifier = RegimeClassifier()
        result = classifier.classify(
            station_id="KDEN",
            valid_date=date(2026, 4, 16),
            ensemble_spread=1.0,  # → HIGH normally
            between_model_spread=9.0,  # severe
        )
        assert result.confidence == "LOW"
        assert "model_disagreement" in result.active_flags
