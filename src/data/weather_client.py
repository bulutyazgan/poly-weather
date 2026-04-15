"""Weather data clients for Open-Meteo, Iowa State Mesonet, and NOAA MOS."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from types import TracebackType

import httpx

from src.config.stations import Station, celsius_to_fahrenheit
from src.data.models import EnsembleForecast, HRRRForecast, MOSForecast, Observation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit-conversion helpers
# ---------------------------------------------------------------------------

KMH_TO_KNOTS = 1.0 / 1.852


# ---------------------------------------------------------------------------
# OpenMeteoClient
# ---------------------------------------------------------------------------


class OpenMeteoClient:
    """Async client for the Open-Meteo API (ensemble + HRRR forecasts)."""

    def __init__(
        self,
        forecast_url: str = "https://api.open-meteo.com/v1",
        ensemble_url: str = "https://ensemble-api.open-meteo.com/v1",
    ) -> None:
        self.client = httpx.AsyncClient(base_url=forecast_url, timeout=30.0)
        self._ensemble_client = httpx.AsyncClient(base_url=ensemble_url, timeout=30.0)

    async def __aenter__(self) -> OpenMeteoClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.client.aclose()
        await self._ensemble_client.aclose()

    # -- Ensemble (GFS / ECMWF) ------------------------------------------

    async def fetch_ensemble(
        self, station: Station, model: str = "gfs"
    ) -> list[EnsembleForecast]:
        """Fetch ensemble temperature forecast from Open-Meteo."""
        model_map = {
            "gfs": ("gfs_seamless", 30),
            "ecmwf": ("ecmwf_ifs025", 50),
        }
        if model not in model_map:
            logger.warning("Unknown ensemble model: %s", model)
            return []

        api_model, member_count = model_map[model]

        try:
            resp = await self._ensemble_client.get(
                "/ensemble",
                params={
                    "latitude": station.lat,
                    "longitude": station.lon,
                    "hourly": "temperature_2m",
                    "models": api_model,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            logger.warning("Ensemble HTTP error for %s/%s", station.station_id, model)
            return []
        except httpx.HTTPError as exc:
            logger.warning("Ensemble request failed: %s", exc)
            return []

        data = resp.json()
        hourly = data.get("hourly", {})
        times: list[str] = hourly.get("time", [])
        if not times:
            return []

        # Collect member arrays — Open-Meteo uses zero-padded keys
        # starting from 01 (e.g. temperature_2m_member01 .. member30)
        member_arrays: list[list[float]] = []
        for i in range(1, member_count + 1):
            key = f"temperature_2m_member{i:02d}"
            arr = hourly.get(key)
            if arr is not None:
                member_arrays.append(arr)

        actual_members = len(member_arrays)
        if actual_members == 0:
            return []

        now = datetime.now(tz=timezone.utc)
        results: list[EnsembleForecast] = []
        for t_idx, time_str in enumerate(times):
            members_f = [
                celsius_to_fahrenheit(member_arrays[m][t_idx])
                for m in range(actual_members)
            ]
            # Open-Meteo returns naive ISO timestamps in UTC — make aware
            vt = datetime.fromisoformat(time_str)
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)
            results.append(
                EnsembleForecast(
                    model_name=model,  # type: ignore[arg-type]
                    run_time=now,
                    valid_time=vt,
                    station_id=station.station_id,
                    members=members_f,
                )
            )

        return results

    # -- HRRR ---------------------------------------------------------------

    async def fetch_hrrr(self, station: Station) -> list[HRRRForecast]:
        """Fetch HRRR hourly forecast from Open-Meteo."""
        try:
            resp = await self.client.get(
                "/forecast",
                params={
                    "latitude": station.lat,
                    "longitude": station.lon,
                    "hourly": "temperature_2m,dewpoint_2m,wind_speed_10m",
                    "forecast_hours": 18,
                    "model": "hrrr_conus",
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            logger.warning("HRRR HTTP error for %s", station.station_id)
            return []
        except httpx.HTTPError as exc:
            logger.warning("HRRR request failed: %s", exc)
            return []

        data = resp.json()
        hourly = data.get("hourly", {})
        times: list[str] = hourly.get("time", [])
        temps: list[float | None] = hourly.get("temperature_2m", [])
        dewpoints: list[float | None] = hourly.get("dewpoint_2m", [])
        winds_kmh: list[float | None] = hourly.get("wind_speed_10m", [])

        now = datetime.now(tz=timezone.utc)
        results: list[HRRRForecast] = []
        for i, time_str in enumerate(times):
            temp_c = temps[i] if i < len(temps) else None
            dp_c = dewpoints[i] if i < len(dewpoints) else None
            wind = winds_kmh[i] if i < len(winds_kmh) else None

            if temp_c is None:
                continue  # Skip records with missing temperature

            vt = datetime.fromisoformat(time_str)
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)

            results.append(
                HRRRForecast(
                    station_id=station.station_id,
                    run_time=now,
                    valid_time=vt,
                    temp_f=celsius_to_fahrenheit(temp_c),
                    dewpoint_f=celsius_to_fahrenheit(dp_c) if dp_c is not None else None,
                    wind_speed_kt=wind * KMH_TO_KNOTS if wind is not None else None,
                )
            )

        return results


# ---------------------------------------------------------------------------
# MesonetClient
# ---------------------------------------------------------------------------


# ICAO station ID → IEM network mapping
_STATION_NETWORK: dict[str, str] = {
    "KNYC": "NY_ASOS",
    "KORD": "IL_ASOS",
    "KLAX": "CA_ASOS",
    "KDEN": "CO_ASOS",
    "KMIA": "FL_ASOS",
}


class MesonetClient:
    """Async client for the Iowa State Mesonet observation API."""

    def __init__(
        self, base_url: str = "https://mesonet.agron.iastate.edu/api/1"
    ) -> None:
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def __aenter__(self) -> MesonetClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.client.aclose()

    async def fetch_observations(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
    ) -> list[Observation]:
        """Fetch METAR observations from Mesonet. station_id uses ICAO (KNYC)."""
        # Mesonet uses 3-letter IDs without K prefix
        short_id = station_id.lstrip("K") if station_id.startswith("K") else station_id

        network = _STATION_NETWORK.get(station_id, "")

        try:
            params: dict[str, str] = {
                "station": short_id,
                "date": start.strftime("%Y-%m-%d"),
            }
            if network:
                params["network"] = network

            resp = await self.client.get("/obhistory.json", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            logger.warning("Mesonet HTTP error for %s", station_id)
            return []
        except httpx.HTTPError as exc:
            logger.warning("Mesonet request failed: %s", exc)
            return []

        data = resp.json()
        records = data.get("data", [])

        results: list[Observation] = []
        for rec in records:
            tmpf = rec.get("tmpf")
            if tmpf is None:
                continue  # Temperature is required

            observed = datetime.fromisoformat(rec["utc_valid"].replace("Z", "+00:00"))

            results.append(
                Observation(
                    station_id=station_id,
                    observed_time=observed,
                    temp_f=float(tmpf),
                    dewpoint_f=float(rec["dwpf"]) if rec.get("dwpf") is not None else None,
                    wind_speed_kt=float(rec["sknt"]) if rec.get("sknt") is not None else None,
                    cloud_cover=rec.get("skyc1"),
                )
            )

        return results


# ---------------------------------------------------------------------------
# MOSClient
# ---------------------------------------------------------------------------


class MOSClient:
    """Client for NOAA GFS-MOS bulletin text parsing."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self) -> MOSClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    def parse_mos_bulletin(
        self,
        bulletin_text: str,
        station_id: str,
        run_time: datetime,
    ) -> MOSForecast | None:
        """Parse a MOS MAV bulletin and extract Tmax/Tmin from the X/N row.

        Returns None if the station is not found in the bulletin.
        """
        # Check that the bulletin is for the requested station
        if station_id not in bulletin_text:
            return None

        # Find the X/N row
        xn_match = re.search(r"^X/N\s+(.+)$", bulletin_text, re.MULTILINE)
        if xn_match is None:
            return None

        # Parse all numeric values from the X/N row
        values = [float(v) for v in xn_match.group(1).split() if v.strip()]
        if not values:
            return None

        # MOS X/N row alternates between max and min temps. For 00Z runs the
        # first value is next-day Tmax; for 12Z runs the first value is
        # tonight's Tmin. Filter out 999 (MOS sentinel for "not applicable").
        filtered = [v for v in values if abs(v) < 200]
        if not filtered:
            return None

        if run_time.hour >= 12:
            # 12Z: first value is tonight's low, second is tomorrow's high
            low = filtered[0]
            high = filtered[1] if len(filtered) > 1 else filtered[0]
        else:
            # 00Z: first value is today's/tomorrow's high, second is low
            high = filtered[0]
            low = filtered[1] if len(filtered) > 1 else filtered[0]

        # Determine valid date from run_time (the next forecast day)
        valid = run_time.date() if run_time.hour < 12 else (
            run_time + timedelta(days=1)
        ).date()

        return MOSForecast(
            station_id=station_id,
            run_time=run_time,
            valid_date=valid,
            high_f=high,
            low_f=low,
        )
