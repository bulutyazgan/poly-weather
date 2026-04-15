"""Tests for pipeline helper functions."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.config.stations import Station
from src.data.models import EnsembleForecast, HRRRForecast
from src.orchestrator.data_collector import DataSnapshot
from src.orchestrator.pipeline import _synthesize_mos


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
        snap.gfs_ensemble = EnsembleForecast(
            model_name="gfs",
            run_time=now,
            valid_time=now,
            station_id="KNYC",
            members=gfs_members,
        )
    if ecmwf_members is not None:
        snap.ecmwf_ensemble = EnsembleForecast(
            model_name="ecmwf",
            run_time=now,
            valid_time=now,
            station_id="KNYC",
            members=ecmwf_members,
        )
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

    def test_hrrr_preferred_over_ensemble(self):
        """HRRR max across hours is the best Tmax — should be used even when
        ensemble data is available.

        Hand calculation:
            HRRR temps: [58, 65, 72, 68] -> max = 72°F
            GFS ensemble mean: (55+60+65)/3 = 60°F
            We want 72, not 60.
        """
        snap = _make_snap(
            gfs_members=[55.0, 60.0, 65.0],
            hrrr_temps=[58.0, 65.0, 72.0, 68.0],
        )
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(72.0)

    def test_ensemble_uses_max_member_not_mean(self):
        """When only ensemble is available, use max of members (not mean)
        as a conservative upper-bound Tmax estimate.

        Hand calculation:
            GFS members: [55, 60, 65, 70, 75]
            mean = 65, max = 75
            We want 75 (conservative upper bound for Tmax).
        """
        snap = _make_snap(gfs_members=[55.0, 60.0, 65.0, 70.0, 75.0])
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        # Must NOT be the mean (65) — should be max member (75)
        assert mos.high_f == pytest.approx(75.0)

    def test_ecmwf_fallback(self):
        """ECMWF is used when GFS and HRRR are unavailable."""
        snap = _make_snap(ecmwf_members=[60.0, 68.0, 72.0])
        station = _make_station()
        now = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)

        mos = _synthesize_mos(snap, station, now)
        assert mos.high_f == pytest.approx(72.0)

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
        assert mos.low_f == pytest.approx(65.0)
