"""Tests for the morning refinement module (RealtimeUpdater)."""
from __future__ import annotations

import pytest

from src.prediction.realtime_updater import (
    RealtimeUpdater,
    RefinedEstimate,
    _cloud_delta_f,
    _parse_cloud_category,
)


# ===================================================================
# _parse_cloud_category
# ===================================================================

class TestParseCloudCategory:
    def test_clr(self):
        assert _parse_cloud_category("CLR") == 0.0

    def test_few(self):
        assert _parse_cloud_category("FEW") == 0.15

    def test_sct(self):
        assert _parse_cloud_category("SCT") == 0.40

    def test_bkn(self):
        assert _parse_cloud_category("BKN") == 0.70

    def test_ovc(self):
        assert _parse_cloud_category("OVC") == 1.0

    def test_case_insensitive(self):
        assert _parse_cloud_category("clr") == 0.0
        assert _parse_cloud_category("Bkn") == 0.70

    def test_none_input(self):
        assert _parse_cloud_category(None) is None

    def test_unknown_code(self):
        assert _parse_cloud_category("XYZ") is None


# ===================================================================
# _cloud_delta_f
# ===================================================================

class TestCloudDeltaF:
    def test_model_overforecast_clouds(self):
        """Model said 0.8 clouds, obs shows 0.2 → warmer (+1.8F)."""
        delta = _cloud_delta_f(model_cloud_cover=0.8, obs_cloud_cover=0.2)
        assert delta == pytest.approx(1.8)

    def test_model_underforecast_clouds(self):
        """Model said 0.1 clouds, obs shows 0.9 → cooler (-2.4F)."""
        delta = _cloud_delta_f(model_cloud_cover=0.1, obs_cloud_cover=0.9)
        assert delta == pytest.approx(-2.4)

    def test_no_mismatch(self):
        delta = _cloud_delta_f(model_cloud_cover=0.5, obs_cloud_cover=0.5)
        assert delta == pytest.approx(0.0)

    def test_full_mismatch_clear(self):
        """Model said full overcast, obs is clear → +3.0F."""
        delta = _cloud_delta_f(model_cloud_cover=1.0, obs_cloud_cover=0.0)
        assert delta == pytest.approx(3.0)


# ===================================================================
# RealtimeUpdater constructor
# ===================================================================

class TestRealtimeUpdaterConstructor:
    def test_defaults(self):
        u = RealtimeUpdater()
        assert u.morning_correlation == 0.6
        assert u.cloud_adjustment_f == 3.0
        assert u.hrrr_weight == 0.4

    def test_custom_params(self):
        u = RealtimeUpdater(morning_correlation=0.5, cloud_adjustment_f=2.0, hrrr_weight=0.3)
        assert u.morning_correlation == 0.5
        assert u.cloud_adjustment_f == 2.0
        assert u.hrrr_weight == 0.3


# ===================================================================
# Morning observation signal
# ===================================================================

class TestMorningObsSignal:
    def test_warm_bias_adjustment(self):
        """Obs 5F warmer than model predicted → Tmax increases by 0.6*5=3.0F."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=65.0,
        )
        assert r.refined_tmax_f == pytest.approx(78.0)  # 75 + 0.6*5
        assert r.refined_std_f == pytest.approx(4.0 * 0.90)
        assert r.confidence_boost >= 0.10
        assert "obs_anchor" in r.refinement_source

    def test_cold_bias_adjustment(self):
        """Obs 4F colder → Tmax decreases by 0.6*(-4)=-2.4F."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=56.0,
        )
        assert r.refined_tmax_f == pytest.approx(72.6)  # 75 + 0.6*(-4)

    def test_no_obs_no_change(self):
        """Without obs, tmax is unchanged."""
        u = RealtimeUpdater()
        r = u.refine(overnight_tmax_f=75.0, overnight_std_f=4.0)
        assert r.refined_tmax_f == pytest.approx(75.0)
        assert r.refined_std_f == pytest.approx(4.0)
        assert r.confidence_boost == 0.0


# ===================================================================
# Cloud cover signal
# ===================================================================

class TestCloudCoverSignal:
    def test_cloud_string_code(self):
        """METAR code 'CLR' with model forecasting 0.7 → warmer."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_cloud_cover=0.7,
            obs_cloud_cover="CLR",
        )
        expected_delta = (0.7 - 0.0) * 3.0  # +2.1F
        assert r.refined_tmax_f == pytest.approx(75.0 + expected_delta)
        assert "cloud" in r.refinement_source

    def test_cloud_float_value(self):
        """Numeric obs cloud cover works too."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_cloud_cover=0.3,
            obs_cloud_cover=0.8,
        )
        expected_delta = (0.3 - 0.8) * 3.0  # -1.5F
        assert r.refined_tmax_f == pytest.approx(73.5)

    def test_unknown_cloud_code_ignored(self):
        """Unrecognised METAR code → no cloud adjustment."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_cloud_cover=0.5,
            obs_cloud_cover="XYZ",
        )
        assert r.refined_tmax_f == pytest.approx(75.0)


# ===================================================================
# HRRR signal
# ===================================================================

class TestHRRRSignal:
    def test_hrrr_blend(self):
        """HRRR=80F, overnight=70F → blended = 0.6*70 + 0.4*80 = 74.0F."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=70.0,
            overnight_std_f=4.0,
            hrrr_tmax_f=80.0,
        )
        assert r.refined_tmax_f == pytest.approx(74.0)
        assert r.refined_std_f == pytest.approx(4.0 * 0.85)
        assert "hrrr" in r.refinement_source

    def test_hrrr_same_as_overnight(self):
        """HRRR agrees → no change to tmax."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            hrrr_tmax_f=75.0,
        )
        assert r.refined_tmax_f == pytest.approx(75.0)


# ===================================================================
# Combined refinement
# ===================================================================

class TestCombinedRefinement:
    def test_all_three_signals(self):
        """All signals present: obs + cloud + HRRR."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=5.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=63.0,           # +3F error → +1.8F
            model_cloud_cover=0.5,
            obs_cloud_cover="CLR",         # +1.5F
            hrrr_tmax_f=80.0,
        )
        # Step 1: obs anchor: 75 + 0.6*3 = 76.8, std *= 0.90
        # Step 2: cloud: 76.8 + (0.5-0.0)*3.0 = 78.3, std *= 0.95
        # Step 3: hrrr blend: 0.6*78.3 + 0.4*80.0 = 78.98, std *= 0.85
        assert r.refined_tmax_f == pytest.approx(78.98)
        assert r.confidence_boost == pytest.approx(0.25)  # 0.10+0.05+0.10
        assert "obs_anchor" in r.refinement_source
        assert "cloud" in r.refinement_source
        assert "hrrr" in r.refinement_source

    def test_confidence_boost_capped(self):
        """Boost never exceeds 0.30."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=5.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=63.0,
            model_cloud_cover=0.5,
            obs_cloud_cover="CLR",
            hrrr_tmax_f=80.0,
        )
        assert r.confidence_boost <= 0.30


# ===================================================================
# Missing data degradation
# ===================================================================

class TestMissingDataDegradation:
    def test_only_obs(self):
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=62.0,
        )
        assert "obs_anchor" in r.refinement_source
        assert "cloud" not in r.refinement_source
        assert "hrrr" not in r.refinement_source

    def test_only_hrrr(self):
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            hrrr_tmax_f=78.0,
        )
        assert "hrrr" in r.refinement_source
        assert "obs_anchor" not in r.refinement_source

    def test_no_signals_passthrough(self):
        u = RealtimeUpdater()
        r = u.refine(overnight_tmax_f=75.0, overnight_std_f=4.0)
        assert r.refined_tmax_f == 75.0
        assert r.refined_std_f == 4.0
        assert r.confidence_boost == 0.0
        assert "no_refinement" in r.refinement_source


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_std_floor(self):
        """Std never goes below 1.5F even with all signals tightening it."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=2.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=60.0,
            model_cloud_cover=0.5,
            obs_cloud_cover=0.5,
            hrrr_tmax_f=75.0,
        )
        assert r.refined_std_f >= 1.5

    def test_zero_obs_error(self):
        """Model perfectly predicted 12Z → no tmax shift from obs."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            model_12z_predicted_temp_f=60.0,
            obs_12z_temp_f=60.0,
        )
        assert r.refined_tmax_f == pytest.approx(75.0)

    def test_refined_estimate_is_frozen(self):
        """RefinedEstimate should be immutable."""
        r = RefinedEstimate(
            refined_tmax_f=75.0,
            refined_std_f=3.0,
            confidence_boost=0.10,
            refinement_source="test",
        )
        with pytest.raises(AttributeError):
            r.refined_tmax_f = 80.0  # type: ignore[misc]

    def test_partial_obs_missing_model_prediction(self):
        """obs_12z present but model_12z absent → no obs anchoring."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            obs_12z_temp_f=63.0,
        )
        assert r.refined_tmax_f == pytest.approx(75.0)

    def test_partial_cloud_missing_model(self):
        """obs_cloud present but model_cloud absent → no cloud adjustment."""
        u = RealtimeUpdater()
        r = u.refine(
            overnight_tmax_f=75.0,
            overnight_std_f=4.0,
            obs_cloud_cover="CLR",
        )
        assert r.refined_tmax_f == pytest.approx(75.0)
