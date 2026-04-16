"""Tests for config settings and stations."""
import pytest


class TestSettingsDefaults:
    def test_settings_defaults(self, monkeypatch):
        monkeypatch.setenv("POLYGON_WALLET_PRIVATE_KEY", "test_key_123")
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")

        from src.config.settings import Settings

        s = Settings()
        assert s.KELLY_FRACTION == 0.04
        assert s.MAX_TRADE_USD == 1.50
        assert s.MAX_BANKROLL_PCT == 0.01
        assert s.MAX_PORTFOLIO_EXPOSURE == 0.35
        assert s.HIGH_REGIME_EDGE_THRESHOLD == 0.05
        assert s.MEDIUM_REGIME_EDGE_THRESHOLD == 0.08
        assert s.MIN_HOURS_TO_RESOLUTION == 2
        assert s.MIN_MARKET_VOLUME == 200
        assert s.PAPER_TRADING is True

    def test_settings_env_override(self, monkeypatch):
        monkeypatch.setenv("POLYGON_WALLET_PRIVATE_KEY", "test_key_123")
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
        monkeypatch.setenv("KELLY_FRACTION", "0.15")

        from src.config.settings import Settings

        s = Settings()
        assert s.KELLY_FRACTION == 0.15


class TestStations:
    def test_get_stations_returns_five(self):
        from src.config.stations import get_stations

        stations = get_stations()
        assert len(stations) == 5

    def test_get_station_nyc(self):
        from src.config.stations import get_station

        station = get_station("NYC")
        assert station.station_id == "KNYC"
        assert abs(station.lat - 40.7789) < 0.01
        assert station.elevation_ft == 154

    def test_get_station_denver(self):
        from src.config.stations import get_station

        station = get_station("Denver")
        assert station.elevation_ft == 5431
        assert "chinook" in station.flags

    def test_get_station_invalid(self):
        from src.config.stations import get_station

        with pytest.raises(KeyError):
            get_station("Atlantis")


class TestConversions:
    def test_celsius_fahrenheit_conversion(self):
        from src.config.stations import celsius_to_fahrenheit, fahrenheit_to_celsius

        assert celsius_to_fahrenheit(0) == 32.0
        assert celsius_to_fahrenheit(100) == 212.0
        # Round-trip
        assert abs(fahrenheit_to_celsius(celsius_to_fahrenheit(37.5)) - 37.5) < 1e-9

    def test_lapse_rate_correction(self):
        from src.config.stations import get_station

        denver = get_station("Denver")
        # correction = -(5431 - 5200) * 3.57 / 1000 = -231 * 3.57 / 1000 ≈ -0.82467
        correction = denver.lapse_rate_correction_f
        assert abs(correction - (-0.82467)) < 0.01
