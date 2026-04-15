"""Tests for weather data clients — OpenMeteo, Mesonet, MOS."""
from __future__ import annotations

from datetime import date, datetime, timezone

import httpx
import pytest
import respx

from src.config.stations import Station, celsius_to_fahrenheit
from src.data.models import EnsembleForecast, HRRRForecast, MOSForecast, Observation
from src.data.weather_client import MesonetClient, MOSClient, OpenMeteoClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nyc_station() -> Station:
    return Station(
        station_id="KNYC",
        city="NYC",
        lat=40.7789,
        lon=-73.9692,
        elevation_ft=154,
        model_grid_elevation_ft=100,
    )


OPEN_METEO_FORECAST_BASE = "https://api.open-meteo.com/v1"
OPEN_METEO_ENSEMBLE_BASE = "https://ensemble-api.open-meteo.com/v1"
MESONET_BASE = "https://mesonet.agron.iastate.edu/api/1"


# ---------------------------------------------------------------------------
# OpenMeteoClient — ensemble GFS
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_ensemble_gfs(nyc_station: Station) -> None:
    # Build a mock response with 31 members, 2 timesteps
    hourly: dict[str, list] = {"time": ["2026-04-15T00:00", "2026-04-15T01:00"]}
    for i in range(31):
        # Temperatures in Celsius
        hourly[f"temperature_2m_member{i}"] = [10.0 + i * 0.1, 11.0 + i * 0.1]

    payload = {"hourly": hourly}

    respx.get(f"{OPEN_METEO_ENSEMBLE_BASE}/ensemble").mock(
        return_value=httpx.Response(200, json=payload)
    )

    client = OpenMeteoClient(forecast_url=OPEN_METEO_FORECAST_BASE, ensemble_url=OPEN_METEO_ENSEMBLE_BASE)
    async with client:
        results = await client.fetch_ensemble(nyc_station, model="gfs")

    assert len(results) == 2
    for fc in results:
        assert isinstance(fc, EnsembleForecast)
        assert fc.model_name == "gfs"
        assert fc.station_id == "KNYC"
        assert fc.member_count == 31
        # Check Fahrenheit conversion for first member of first timestep
        for member_val in fc.members:
            assert member_val > 40  # 10°C ≈ 50°F, all should be > 40


# ---------------------------------------------------------------------------
# OpenMeteoClient — ensemble ECMWF
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_ensemble_ecmwf(nyc_station: Station) -> None:
    hourly: dict[str, list] = {"time": ["2026-04-15T12:00"]}
    for i in range(51):
        hourly[f"temperature_2m_member{i}"] = [20.0 + i * 0.05]

    payload = {"hourly": hourly}

    respx.get(f"{OPEN_METEO_ENSEMBLE_BASE}/ensemble").mock(
        return_value=httpx.Response(200, json=payload)
    )

    client = OpenMeteoClient(forecast_url=OPEN_METEO_FORECAST_BASE, ensemble_url=OPEN_METEO_ENSEMBLE_BASE)
    async with client:
        results = await client.fetch_ensemble(nyc_station, model="ecmwf")

    assert len(results) == 1
    fc = results[0]
    assert fc.model_name == "ecmwf"
    assert fc.member_count == 51
    # 20°C = 68°F
    assert abs(fc.members[0] - celsius_to_fahrenheit(20.0)) < 0.01


# ---------------------------------------------------------------------------
# OpenMeteoClient — fetch HRRR
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_hrrr(nyc_station: Station) -> None:
    payload = {
        "hourly": {
            "time": ["2026-04-15T00:00", "2026-04-15T01:00"],
            "temperature_2m": [15.0, 16.0],  # Celsius
            "dewpoint_2m": [10.0, 11.0],
            "wind_speed_10m": [20.0, 25.0],  # km/h
        }
    }

    respx.get(f"{OPEN_METEO_FORECAST_BASE}/gfs").mock(
        return_value=httpx.Response(200, json=payload)
    )

    client = OpenMeteoClient(forecast_url=OPEN_METEO_FORECAST_BASE, ensemble_url=OPEN_METEO_ENSEMBLE_BASE)
    async with client:
        results = await client.fetch_hrrr(nyc_station)

    assert len(results) == 2
    for fc in results:
        assert isinstance(fc, HRRRForecast)
        assert fc.station_id == "KNYC"
    # Check conversions: 15°C → 59°F
    assert abs(results[0].temp_f - 59.0) < 0.01
    # Wind: 20 km/h → ~10.79 knots (km/h / 1.852)
    assert results[0].wind_speed_kt is not None
    assert abs(results[0].wind_speed_kt - 20.0 / 1.852) < 0.1


# ---------------------------------------------------------------------------
# OpenMeteoClient — empty / error response
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_ensemble_empty_response(nyc_station: Station) -> None:
    respx.get(f"{OPEN_METEO_ENSEMBLE_BASE}/ensemble").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    client = OpenMeteoClient(forecast_url=OPEN_METEO_FORECAST_BASE, ensemble_url=OPEN_METEO_ENSEMBLE_BASE)
    async with client:
        results = await client.fetch_ensemble(nyc_station, model="gfs")

    assert results == []


# ---------------------------------------------------------------------------
# MesonetClient — fetch observations
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_observations(nyc_station: Station) -> None:
    payload = {
        "data": [
            {
                "station": "NYC",
                "utc_valid": "2026-04-15T00:00:00Z",
                "tmpf": 65.0,
                "dwpf": 50.0,
                "sknt": 10.0,
                "drct": 180.0,
                "mslp": 1013.0,
                "skyc1": "FEW",
                "p01i": 0.0,
            },
            {
                "station": "NYC",
                "utc_valid": "2026-04-15T01:00:00Z",
                "tmpf": 63.0,
                "dwpf": 48.0,
                "sknt": 8.0,
                "drct": 200.0,
                "mslp": 1012.5,
                "skyc1": "CLR",
                "p01i": 0.01,
            },
        ]
    }

    respx.get(f"{MESONET_BASE}/obhistory.json").mock(
        return_value=httpx.Response(200, json=payload)
    )

    client = MesonetClient(base_url=MESONET_BASE)
    async with client:
        results = await client.fetch_observations(
            "KNYC",
            start=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 15, 2, 0, tzinfo=timezone.utc),
        )

    assert len(results) == 2
    for obs in results:
        assert isinstance(obs, Observation)
        assert obs.station_id == "KNYC"
    assert results[0].temp_f == 65.0
    assert results[0].cloud_cover == "FEW"


# ---------------------------------------------------------------------------
# MesonetClient — missing data fields
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_fetch_observations_handles_missing_data(nyc_station: Station) -> None:
    payload = {
        "data": [
            {
                "station": "NYC",
                "utc_valid": "2026-04-15T03:00:00Z",
                "tmpf": 60.0,
                "dwpf": None,
                "sknt": None,
                "drct": None,
                "mslp": None,
                "skyc1": None,
                "p01i": None,
            },
        ]
    }

    respx.get(f"{MESONET_BASE}/obhistory.json").mock(
        return_value=httpx.Response(200, json=payload)
    )

    client = MesonetClient(base_url=MESONET_BASE)
    async with client:
        results = await client.fetch_observations(
            "KNYC",
            start=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 15, 6, 0, tzinfo=timezone.utc),
        )

    assert len(results) == 1
    obs = results[0]
    assert obs.temp_f == 60.0
    assert obs.dewpoint_f is None
    assert obs.wind_speed_kt is None
    assert obs.cloud_cover is None


# ---------------------------------------------------------------------------
# MOSClient — parse bulletin
# ---------------------------------------------------------------------------

SAMPLE_MOS_BULLETIN = """\
KNYC   GFS MOS GUIDANCE   4/15/2026  0000 UTC
DT /APR  15                /APR  16               /
HR   06 09 12 15 18 21 00 03 06 09 12 15 18 21 00
X/N              72          55       73
TMP  58 55 54 64 70 68 62 58 56 55 58 66 71 69 63
DPT  45 44 43 42 40 39 41 42 43 44 43 42 41 40 42
"""


def test_parse_mos_bulletin() -> None:
    client = MOSClient()
    result = client.parse_mos_bulletin(
        SAMPLE_MOS_BULLETIN,
        station_id="KNYC",
        run_time=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
    )
    assert result is not None
    assert isinstance(result, MOSForecast)
    assert result.station_id == "KNYC"
    assert result.high_f == 72.0
    assert result.low_f == 55.0


def test_parse_mos_station_not_found() -> None:
    client = MOSClient()
    result = client.parse_mos_bulletin(
        SAMPLE_MOS_BULLETIN,
        station_id="KORD",
        run_time=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
    )
    assert result is None
