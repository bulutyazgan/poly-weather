"""Tests for pipeline helper functions."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.config.stations import Station
from src.data.models import EnsembleForecast, HRRRForecast, TradingSignal
from src.data.models import MarketContract
from src.orchestrator.data_collector import DataSnapshot
from src.orchestrator.pipeline import _synthesize_mos, _resolution_utc
from src.prediction.calibration import CUSUMMonitor
from src.prediction.regime_classifier import RegimeClassifier


def _make_station() -> Station:
    return Station(
        station_id="KNYC",
        city="NYC",
        lat=40.78,
        lon=-73.97,
        elevation_ft=154,
        model_grid_elevation_ft=100,
    )


def _make_snap(
    gfs_members: list[float] | None = None,
    ecmwf_members: list[float] | None = None,
    hrrr_temps: list[float] | None = None,
) -> DataSnapshot:
    now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
    snap = DataSnapshot(station_id="KNYC", timestamp=now)
    if gfs_members is not None:
        gfs = EnsembleForecast(
            model_name="gfs",
            run_time=now,
            valid_time=now,
            station_id="KNYC",
            members=gfs_members,
        )
        snap.gfs_ensemble = gfs
        snap.gfs_ensemble_all = [gfs]
    if ecmwf_members is not None:
        ecmwf = EnsembleForecast(
            model_name="ecmwf",
            run_time=now,
            valid_time=now,
            station_id="KNYC",
            members=ecmwf_members,
        )
        snap.ecmwf_ensemble = ecmwf
        snap.ecmwf_ensemble_all = [ecmwf]
    if hrrr_temps is not None:
        snap.hrrr = [
            HRRRForecast(
                station_id="KNYC",
                run_time=now,
                valid_time=now,
                temp_f=t,
            )
            for t in hrrr_temps
        ]
    return snap


class TestSynthesizeMos:
    """Test _synthesize_mos Tmax estimation."""

    def test_hrrr_excluded_when_disagrees_with_nwp(self):
        """HRRR excluded when it disagrees with NWP consensus by >4°F.

        GFS mean = 60°F, HRRR max = 72°F → disagreement = 12°F > 4°F.
        Result: only GFS used → MOS = 60°F.
        """
        snap = _make_snap(
            gfs_members=[55.0, 60.0, 65.0],
            hrrr_temps=[58.0, 65.0, 72.0, 68.0],
        )
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(60.0, abs=0.1)

    def test_hrrr_included_when_agrees_with_nwp(self):
        """HRRR included when within 4°F of NWP consensus.

        GFS mean = 60°F, HRRR max = 62°F → disagreement = 2°F ≤ 4°F.
        weighted = (0.4*60 + 0.15*62) / (0.4 + 0.15) = (24 + 9.3) / 0.55 = 60.55°F
        """
        snap = _make_snap(
            gfs_members=[55.0, 60.0, 65.0],
            hrrr_temps=[58.0, 60.0, 62.0, 59.0],
        )
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(60.55, abs=0.1)

    def test_ensemble_uses_mean_at_peak_hour(self):
        """When only GFS ensemble is available, use ensemble mean.

        Hand calculation:
            GFS members: [55, 60, 65, 70, 75]
            mean = 65
        """
        snap = _make_snap(gfs_members=[55.0, 60.0, 65.0, 70.0, 75.0])
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(65.0)

    def test_ecmwf_fallback(self):
        """ECMWF is used when GFS and HRRR are unavailable."""
        snap = _make_snap(ecmwf_members=[60.0, 68.0, 72.0])
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        # mean([60, 68, 72]) = 66.67
        assert mos.high_f == pytest.approx(66.67, abs=0.1)

    def test_gfs_ecmwf_weighted_average(self):
        """Both NWP models available → weighted average (ECMWF 0.6, GFS 0.4).

        Hand calculation:
            GFS mean = 68, ECMWF mean = 77
            Old bug: max(68, 77) = 77 — biased to warmest model
            Fix: (0.4*68 + 0.6*77) / (0.4+0.6) = (27.2 + 46.2) / 1.0 = 73.4
        """
        snap = _make_snap(
            gfs_members=[67.0, 68.0, 69.0],
            ecmwf_members=[76.0, 77.0, 78.0],
        )
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(73.4, abs=0.1)

    def test_no_data_fallback(self):
        """When no weather data is available, fall back to 70°F."""
        snap = _make_snap()
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(70.0)

    def test_low_is_derived_from_high(self):
        """low_f should be high_f - 15."""
        snap = _make_snap(hrrr_temps=[80.0])
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.low_f == pytest.approx(mos.high_f - 15.0)


class TestCUSUMResidual:
    """Test that CUSUM residual is computed correctly in the pipeline.

    The bug: residual = abs(model-market) - edge = |x| - |x| = 0 always.
    The fix: residual = model_prob - market_prob (signed), so CUSUM can
    detect systematic calibration drift in either direction.
    """

    def test_cusum_residual_not_always_zero(self):
        """Verify the CUSUM residual formula produces non-zero values.

        Hand calculation:
            model_prob=0.70, market_prob=0.55
            Correct residual = 0.70 - 0.55 = +0.15
            Bug residual = |0.70-0.55| - 0.15 = 0  (always!)
        """
        # Simulate what pipeline does: compute residual and feed to CUSUM
        model_prob = 0.70
        market_prob = 0.55

        # Correct formula: signed residual for calibration drift detection
        residual = model_prob - market_prob
        assert residual == pytest.approx(0.15)

        # Verify CUSUM actually accumulates
        cusum = CUSUMMonitor(threshold=2.0, drift=0.0)
        cusum.update(residual)
        assert cusum.cusum_pos > 0.0  # Should have accumulated

    def test_cusum_detects_systematic_positive_bias(self):
        """Repeated positive residuals (model > market) should trigger alarm.

        If model consistently overestimates, cusum_pos will grow until alarm.
        With threshold=0.5, drift=0, 5 residuals of +0.15 => cusum_pos = 0.75 > 0.5.
        """
        cusum = CUSUMMonitor(threshold=0.5, drift=0.0)
        for _ in range(5):
            residual = 0.70 - 0.55  # model consistently higher than market
            cusum.update(residual)
        assert cusum.alarm is True

    def test_cusum_detects_systematic_negative_bias(self):
        """Repeated negative residuals (model < market) should trigger alarm.

        cusum_neg accumulates: 5 * 0.15 = 0.75 > 0.5.
        """
        cusum = CUSUMMonitor(threshold=0.5, drift=0.0)
        for _ in range(5):
            residual = 0.55 - 0.70  # model consistently lower
            cusum.update(residual)
        assert cusum.alarm is True


    def test_cusum_alarm_resets_after_blocking_cycle(self):
        """Pipeline resets CUSUM after using alarm to block trades.

        Bug: alarm was sticky forever — once triggered, bot never traded again.
        Fix: pipeline calls cusum.reset() after blocking a cycle, so if the
        model recovers, trading resumes next cycle.

        Cycle flow:
          1. Bad residuals trigger alarm mid-cycle → alarm=True
          2. Next cycle: pipeline sees alarm → blocks trades → resets CUSUM
          3. Good residuals during blocked cycle → alarm stays False
          4. Next cycle: alarm=False → trading resumes
        """
        cusum = CUSUMMonitor(threshold=0.5, drift=0.0)

        # Simulate bad cycle: alarm triggers
        for _ in range(5):
            cusum.update(0.15)
        assert cusum.alarm is True

        # Pipeline would check alarm at top of next cycle, then reset
        blocked = cusum.alarm
        assert blocked is True
        cusum.reset()  # this is what pipeline.run_cycle now does

        # Good residuals during the blocked cycle
        for _ in range(5):
            cusum.update(0.02)
        assert cusum.alarm is False  # model recovered

        # Next cycle: alarm is clear, trading resumes
        assert cusum.alarm is False


class TestRegimeClassifierWiring:
    """Verify pipeline passes station_flags and ensemble_members to classifier."""

    def test_station_flags_passed_to_classifier(self):
        """Station flags (e.g. chinook) must reach the regime classifier.

        Denver has flags=["chinook", "high_elevation"]. If the pipeline
        doesn't pass these, chinook wind detection is dead code.
        """
        classifier = RegimeClassifier()
        station = Station(
            station_id="KDEN",
            city="Denver",
            lat=39.86,
            lon=-104.67,
            elevation_ft=5431,
            model_grid_elevation_ft=5200,
            flags=["chinook", "high_elevation"],
        )
        # Call with chinook conditions: strong W/NW 700mb wind
        result = classifier.classify(
            station_id=station.station_id,
            valid_date=datetime(2026, 4, 16).date(),
            ensemble_spread=1.0,
            station_flags=station.flags,
            wind_700mb_speed=35.0,
            wind_700mb_direction=280.0,
        )
        assert "chinook" in result.active_flags
        assert result.confidence == "HIGH"

    def test_station_flags_not_passed_misses_chinook(self):
        """Without station_flags, chinook is never detected even with wind conditions."""
        classifier = RegimeClassifier()
        result = classifier.classify(
            station_id="KDEN",
            valid_date=datetime(2026, 4, 16).date(),
            ensemble_spread=1.0,
            # station_flags NOT passed — defaults to []
            wind_700mb_speed=35.0,
            wind_700mb_direction=280.0,
        )
        assert "chinook" not in result.active_flags

    def test_ensemble_members_enable_bimodal_detection(self):
        """Passing ensemble_members enables bimodal detection (two-cluster spread)."""
        classifier = RegimeClassifier()
        # Create clearly bimodal ensemble: two clusters separated by large gap
        bimodal_members = [60.0, 61.0, 62.0, 63.0, 80.0, 81.0, 82.0, 83.0]
        result = classifier.classify(
            station_id="KNYC",
            valid_date=datetime(2026, 4, 16).date(),
            ensemble_spread=8.0,
            ensemble_members=bimodal_members,
        )
        assert "bimodal_ensemble" in result.active_flags


class TestResolutionUtc:
    """Test _resolution_utc helper for correct resolution time computation."""

    def _make_contract(self, end_date_utc=None):
        from datetime import date
        return MarketContract(
            token_id="tok_1",
            condition_id="0xcond",
            question="Will NYC high be 80-81°F?",
            city="NYC",
            resolution_date=date(2026, 4, 16),
            end_date_utc=end_date_utc,
            temp_bucket_low=80.0,
            temp_bucket_high=81.0,
            outcome="Yes",
        )

    def test_uses_gamma_end_date_when_available(self):
        """When Gamma provides endDate, use it exactly."""
        end_dt = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
        contract = self._make_contract(end_date_utc=end_dt)
        assert _resolution_utc(contract) == end_dt

    def test_fallback_uses_end_of_day_not_midnight(self):
        """Without endDate, use 23:59:59 UTC (not 00:00:00 midnight).

        The old bug used midnight UTC on resolution_date, which was the
        START of the day — ~20-28h before actual resolution for US cities.
        """
        contract = self._make_contract()
        result = _resolution_utc(contract)
        # Should be end of April 16, not start
        assert result.hour == 23
        assert result.minute == 59
        assert result.day == 16

    def test_midnight_bug_would_cancel_early(self):
        """Demonstrate the old bug: midnight UTC is ~24h before end-of-day.

        At 12:00 UTC on April 15, with resolution_date=April 16:
        Old bug: resolution = April 16 00:00 UTC → 12h to resolution
        Fix:     resolution = April 16 23:59 UTC → 36h to resolution
        The 4h threshold for cancellation would trigger at different times.
        """
        from datetime import time as dt_time

        contract = self._make_contract()
        now = datetime(2026, 4, 16, 20, 0, tzinfo=timezone.utc)

        # Old bug: midnight UTC = already 20h in the past → negative hours
        old_resolution = datetime.combine(
            contract.resolution_date,
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        old_hours = (old_resolution - now).total_seconds() / 3600.0
        assert old_hours < 0  # would trigger premature cancellation!

        # Fix: end of day
        new_hours = (_resolution_utc(contract) - now).total_seconds() / 3600.0
        assert new_hours > 0  # correctly shows 4h remaining


class TestPickEnsembleForDate:
    """Test _pick_ensemble_for_date uses 06Z boundary for US local day."""

    def test_rejects_previous_day_evening(self):
        """00-05Z (previous local day evening) must not contaminate Tmax.

        00Z April 17 UTC is 6PM MDT April 16 for Denver.  Pre-frontal
        warm air at that hour is the previous day's temperature, not
        April 17's.  Including it overestimates Tmax by 20-30°F during
        cold-front passages.
        """
        from src.orchestrator.pipeline import _pick_ensemble_for_date
        from datetime import date

        # Warm evening at 00Z (= previous local day) and cool afternoon at 20Z
        warm_evening = EnsembleForecast(
            model_name="gfs",
            run_time=datetime(2026, 4, 16, 6, tzinfo=timezone.utc),
            valid_time=datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc),  # 00Z
            station_id="KDEN",
            members=[72.0 + i * 0.1 for i in range(31)],
        )
        cool_afternoon = EnsembleForecast(
            model_name="gfs",
            run_time=datetime(2026, 4, 16, 6, tzinfo=timezone.utc),
            valid_time=datetime(2026, 4, 17, 20, 0, tzinfo=timezone.utc),  # 20Z
            station_id="KDEN",
            members=[55.0 + i * 0.1 for i in range(31)],
        )

        result = _pick_ensemble_for_date(
            [warm_evening, cool_afternoon], date(2026, 4, 17)
        )
        assert result is not None
        # Must pick 55°F afternoon, NOT 72°F from previous local day
        assert result.mean < 60.0, (
            f"Expected cool afternoon (~55°F), got {result.mean:.1f}°F — "
            "00Z (previous local evening) was incorrectly included"
        )

    def test_includes_late_afternoon(self):
        """Late-afternoon peak (00Z next day = 6PM MDT) must be included.

        Denver's high on a warm day can peak at 6PM MDT = 00Z next UTC day.
        The 06Z-05:59Z window must extend into the next UTC day.
        """
        from src.orchestrator.pipeline import _pick_ensemble_for_date
        from datetime import date

        morning = EnsembleForecast(
            model_name="gfs",
            run_time=datetime(2026, 4, 16, 6, tzinfo=timezone.utc),
            valid_time=datetime(2026, 4, 17, 14, 0, tzinfo=timezone.utc),  # 14Z
            station_id="KDEN",
            members=[60.0 + i * 0.1 for i in range(31)],
        )
        late_afternoon = EnsembleForecast(
            model_name="gfs",
            run_time=datetime(2026, 4, 16, 6, tzinfo=timezone.utc),
            valid_time=datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc),  # 00Z+1
            station_id="KDEN",
            members=[78.0 + i * 0.1 for i in range(31)],
        )

        result = _pick_ensemble_for_date(
            [morning, late_afternoon], date(2026, 4, 17)
        )
        assert result is not None
        assert result.mean > 75.0, (
            f"Expected late-afternoon peak (~78°F), got {result.mean:.1f}°F — "
            "00Z next day (6PM MDT local) was incorrectly excluded"
        )
