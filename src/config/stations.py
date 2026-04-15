"""Weather station definitions and utility functions."""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class Station(BaseModel):
    """A weather observation station."""

    station_id: str
    city: str
    lat: float
    lon: float
    elevation_ft: int
    model_grid_elevation_ft: int
    flags: list[str] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lapse_rate_correction_f(self) -> float:
        """Temperature correction for elevation difference from model grid (°F)."""
        return -(self.elevation_ft - self.model_grid_elevation_ft) * 3.57 / 1000


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


_STATIONS: dict[str, Station] = {
    "NYC": Station(
        station_id="KNYC",
        city="NYC",
        lat=40.7789,
        lon=-73.9692,
        elevation_ft=154,
        model_grid_elevation_ft=100,
    ),
    "Chicago": Station(
        station_id="KORD",
        city="Chicago",
        lat=41.9742,
        lon=-87.9073,
        elevation_ft=672,
        model_grid_elevation_ft=650,
    ),
    "LA": Station(
        station_id="KLAX",
        city="LA",
        lat=33.9425,
        lon=-118.4081,
        elevation_ft=126,
        model_grid_elevation_ft=50,
    ),
    "Denver": Station(
        station_id="KDEN",
        city="Denver",
        lat=39.8561,
        lon=-104.6737,
        elevation_ft=5431,
        model_grid_elevation_ft=5200,
        flags=["chinook", "high_elevation"],
    ),
    "Miami": Station(
        station_id="KMIA",
        city="Miami",
        lat=25.7959,
        lon=-80.2870,
        elevation_ft=9,
        model_grid_elevation_ft=5,
    ),
}


def get_stations() -> dict[str, Station]:
    """Return all configured stations."""
    return dict(_STATIONS)


def get_station(city: str) -> Station:
    """Return a station by city name. Raises KeyError if not found."""
    return _STATIONS[city]
