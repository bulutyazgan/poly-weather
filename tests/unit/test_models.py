"""Tests for data models."""
from datetime import datetime, date
import pytest
from pydantic import ValidationError


class TestEnsembleForecast:
    def test_valid_construction(self):
        from src.data.models import EnsembleForecast

        ef = EnsembleForecast(
            model_name="gfs",
            run_time=datetime(2026, 4, 15, 0, 0),
            valid_time=datetime(2026, 4, 16, 12, 0),
            station_id="KNYC",
            members=[70.0, 72.0, 74.0],
        )
        assert ef.model_name == "gfs"
        assert ef.station_id == "KNYC"

    def test_computed_fields(self):
        from src.data.models import EnsembleForecast

        ef = EnsembleForecast(
            model_name="gfs",
            run_time=datetime(2026, 4, 15, 0, 0),
            valid_time=datetime(2026, 4, 16, 12, 0),
            station_id="KNYC",
            members=[70.0, 72.0, 74.0],
        )
        assert abs(ef.mean - 72.0) < 0.01
        assert ef.std > 0
        assert ef.member_count == 3

    def test_invalid_model_name(self):
        from src.data.models import EnsembleForecast

        with pytest.raises(ValidationError):
            EnsembleForecast(
                model_name="nam",
                run_time=datetime(2026, 4, 15, 0, 0),
                valid_time=datetime(2026, 4, 16, 12, 0),
                station_id="KNYC",
                members=[70.0],
            )


class TestMOSForecast:
    def test_valid_construction(self):
        from src.data.models import MOSForecast

        mf = MOSForecast(
            station_id="KNYC",
            run_time=datetime(2026, 4, 15, 0, 0),
            valid_date=date(2026, 4, 16),
            high_f=75.0,
            low_f=55.0,
        )
        assert mf.high_f == 75.0
        assert mf.low_f == 55.0


class TestHRRRForecast:
    def test_valid_construction(self):
        from src.data.models import HRRRForecast

        hf = HRRRForecast(
            station_id="KNYC",
            run_time=datetime(2026, 4, 15, 0, 0),
            valid_time=datetime(2026, 4, 15, 6, 0),
            temp_f=68.5,
            dewpoint_f=55.0,
            wind_speed_kt=10.0,
        )
        assert hf.temp_f == 68.5


class TestObservation:
    def test_valid_construction(self):
        from src.data.models import Observation

        obs = Observation(
            station_id="KNYC",
            observed_time=datetime(2026, 4, 15, 12, 0),
            temp_f=72.0,
        )
        assert obs.temp_f == 72.0

    def test_optional_none_fields(self):
        from src.data.models import Observation

        obs = Observation(
            station_id="KNYC",
            observed_time=datetime(2026, 4, 15, 12, 0),
            temp_f=72.0,
            cloud_cover=None,
        )
        assert obs.cloud_cover is None


class TestMarketContract:
    def test_valid_construction(self):
        from src.data.models import MarketContract

        mc = MarketContract(
            token_id="tok_123",
            condition_id="cond_abc",
            question="Will NYC high be 72-73F?",
            city="NYC",
            resolution_date=date(2026, 4, 16),
            temp_bucket_low=72.0,
            temp_bucket_high=73.0,
            outcome="Yes",
        )
        assert mc.outcome == "Yes"

    def test_invalid_outcome(self):
        from src.data.models import MarketContract

        with pytest.raises(ValidationError):
            MarketContract(
                token_id="tok_123",
                condition_id="cond_abc",
                question="Will NYC high exceed 80F?",
                city="NYC",
                resolution_date=date(2026, 4, 16),
                temp_bucket_low=80.0,
                temp_bucket_high=float("inf"),
                outcome="Maybe",
            )


class TestMarketPrice:
    def test_valid_construction(self):
        from src.data.models import MarketPrice

        mp = MarketPrice(
            token_id="tok_123",
            timestamp=datetime(2026, 4, 15, 12, 0),
            bid=0.55,
            ask=0.57,
            mid=0.56,
            volume_24h=5000.0,
        )
        assert mp.bid == 0.55


class TestRegimeClassification:
    def test_valid_construction(self):
        from src.data.models import RegimeClassification

        rc = RegimeClassification(
            station_id="KNYC",
            valid_date=date(2026, 4, 16),
            regime="high_variance",
            confidence="HIGH",
        )
        assert rc.confidence == "HIGH"

    def test_invalid_confidence(self):
        from src.data.models import RegimeClassification

        with pytest.raises(ValidationError):
            RegimeClassification(
                station_id="KNYC",
                valid_date=date(2026, 4, 16),
                regime="high_variance",
                confidence="VERY_HIGH",
            )


class TestTradingSignal:
    def test_valid_construction(self):
        from src.data.models import TradingSignal

        ts = TradingSignal(
            market_id="0x123",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=0.05,
            timestamp=datetime(2026, 4, 15, 12, 0),
        )
        assert ts.direction == "BUY_YES"
        assert ts.action == "TRADE"

    def test_invalid_direction(self):
        from src.data.models import TradingSignal

        with pytest.raises(ValidationError):
            TradingSignal(
                market_id="0x123",
                direction="SELL",
                action="TRADE",
                edge=0.10,
                kelly_size=0.05,
                timestamp=datetime(2026, 4, 15, 12, 0),
            )


class TestTradeRecord:
    def test_valid_construction(self):
        from src.data.models import TradeRecord

        tr = TradeRecord(
            trade_id="trade_001",
            market_id="0x123",
            direction="BUY_YES",
            amount_usd=2.50,
            price=0.65,
            timestamp=datetime(2026, 4, 15, 12, 0),
        )
        assert tr.amount_usd == 2.50

    def test_optional_resolution_outcome(self):
        from src.data.models import TradeRecord

        tr = TradeRecord(
            trade_id="trade_001",
            market_id="0x123",
            direction="BUY_YES",
            amount_usd=2.50,
            price=0.65,
            timestamp=datetime(2026, 4, 15, 12, 0),
            resolution_outcome=None,
        )
        assert tr.resolution_outcome is None


class TestCalibrationRecord:
    def test_valid_construction(self):
        from src.data.models import CalibrationRecord

        cr = CalibrationRecord(
            station_id="KNYC",
            model_name="gfs",
            date_range_start=date(2026, 1, 1),
            date_range_end=date(2026, 3, 31),
            bias_f=1.2,
            rmse_f=3.5,
            sample_count=90,
        )
        assert cr.bias_f == 1.2
        assert cr.sample_count == 90
