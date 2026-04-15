"""Tests for the orchestrator: DataCollector, TradingPipeline, PipelineScheduler."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, time as dt_time, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.stations import Station, get_stations
from src.data.models import (
    EnsembleForecast,
    HRRRForecast,
    MarketContract,
    MarketPrice,
    Observation,
    RegimeClassification,
    TradingSignal,
)
from src.orchestrator.data_collector import DataCollector, DataSnapshot
from src.orchestrator.pipeline import TradingPipeline
from src.orchestrator.scheduler import PipelineScheduler


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_station() -> Station:
    return Station(
        station_id="KNYC",
        city="NYC",
        lat=40.78,
        lon=-73.97,
        elevation_ft=154,
        model_grid_elevation_ft=100,
    )


def _make_ensemble(station_id: str = "KNYC", model: str = "gfs") -> EnsembleForecast:
    now = datetime.now(tz=timezone.utc)
    return EnsembleForecast(
        model_name=model,
        run_time=now,
        valid_time=now + timedelta(hours=24),
        station_id=station_id,
        members=[70.0 + i * 0.5 for i in range(31)],
    )


def _make_hrrr(station_id: str = "KNYC") -> HRRRForecast:
    now = datetime.now(tz=timezone.utc)
    return HRRRForecast(
        station_id=station_id,
        run_time=now,
        valid_time=now + timedelta(hours=6),
        temp_f=72.0,
        dewpoint_f=55.0,
        wind_speed_kt=10.0,
    )


def _make_observation(station_id: str = "KNYC") -> Observation:
    return Observation(
        station_id=station_id,
        observed_time=datetime.now(tz=timezone.utc),
        temp_f=73.0,
        dewpoint_f=56.0,
        wind_speed_kt=8.0,
    )


def _make_contract(token_id: str = "tok_1", city: str = "NYC") -> MarketContract:
    return MarketContract(
        token_id=token_id,
        condition_id="cond_1",
        question="Will the high temperature in New York City on April 16, 2026 be between 70°F and 75°F?",
        city=city,
        resolution_date=date(2026, 4, 16),
        temp_bucket_low=70.0,
        temp_bucket_high=75.0,
        outcome="Yes",
    )


def _make_market_price(token_id: str = "tok_1") -> MarketPrice:
    return MarketPrice(
        token_id=token_id,
        timestamp=datetime.now(tz=timezone.utc),
        bid=0.30,
        ask=0.35,
        mid=0.325,
        volume_24h=5000.0,
    )


def _make_regime(station_id: str = "KNYC", confidence: str = "HIGH") -> RegimeClassification:
    return RegimeClassification(
        station_id=station_id,
        valid_date=date(2026, 4, 16),
        regime="high_confidence",
        confidence=confidence,
        confidence_score=0.85,
        ensemble_spread_percentile=20.0,
    )


def _make_signal(action: str = "TRADE", edge: float = 0.10, market_id: str = "tok_1") -> TradingSignal:
    return TradingSignal(
        market_id=market_id,
        direction="BUY_YES",
        action=action,
        edge=edge,
        kelly_size=2.0,
        timestamp=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# DataCollector tests
# ---------------------------------------------------------------------------


class TestDataCollector:

    @pytest.mark.asyncio
    async def test_collect_snapshot_stores_data(self):
        """Mock all clients. collect_snapshot() stores forecasts, market prices, observations."""
        weather = AsyncMock()
        mesonet = AsyncMock()
        gamma = AsyncMock()
        clob = AsyncMock()

        station = _make_station()
        gfs_ens = _make_ensemble("KNYC", "gfs")
        ecmwf_ens = _make_ensemble("KNYC", "ecmwf")
        hrrr_list = [_make_hrrr("KNYC")]
        obs_list = [_make_observation("KNYC")]
        contracts = [_make_contract("tok_1", "NYC")]
        price = _make_market_price("tok_1")

        weather.fetch_ensemble = AsyncMock(side_effect=lambda s, model="gfs": [gfs_ens] if model == "gfs" else [ecmwf_ens])
        weather.fetch_hrrr = AsyncMock(return_value=hrrr_list)
        mesonet.fetch_observations = AsyncMock(return_value=obs_list)
        gamma.fetch_weather_markets = AsyncMock(return_value=contracts)
        clob.get_market_price = AsyncMock(return_value=price)

        collector = DataCollector(weather=weather, mesonet=mesonet, gamma=gamma, clob=clob)

        with patch("src.orchestrator.data_collector.get_stations", return_value={"NYC": station}):
            snapshots = await collector.collect_snapshot()

        assert len(snapshots) >= 1
        snap = snapshots[0]
        assert snap.station_id == "KNYC"
        assert snap.gfs_ensemble is not None
        assert snap.market_contracts == contracts
        assert len(snap.market_prices) > 0

    @pytest.mark.asyncio
    async def test_collect_snapshot_handles_api_failure(self):
        """If one weather client fails, other data still collected. No crash."""
        weather = AsyncMock()
        mesonet = AsyncMock()
        gamma = AsyncMock()
        clob = AsyncMock()

        station = _make_station()
        contracts = [_make_contract()]
        price = _make_market_price()

        # Make ensemble fail, but HRRR and obs succeed
        weather.fetch_ensemble = AsyncMock(side_effect=Exception("API down"))
        weather.fetch_hrrr = AsyncMock(return_value=[_make_hrrr()])
        mesonet.fetch_observations = AsyncMock(return_value=[_make_observation()])
        gamma.fetch_weather_markets = AsyncMock(return_value=contracts)
        clob.get_market_price = AsyncMock(return_value=price)

        collector = DataCollector(weather=weather, mesonet=mesonet, gamma=gamma, clob=clob)

        with patch("src.orchestrator.data_collector.get_stations", return_value={"NYC": station}):
            snapshots = await collector.collect_snapshot()

        # Should not crash, and should still have HRRR and observations
        assert len(snapshots) >= 1
        snap = snapshots[0]
        assert snap.gfs_ensemble is None
        assert snap.ecmwf_ensemble is None
        assert len(snap.hrrr) == 1
        assert len(snap.observations) == 1

    @pytest.mark.asyncio
    async def test_get_matched_records(self):
        """After collecting snapshots and resolving outcomes, get_matched_records() returns matched data."""
        weather = AsyncMock()
        mesonet = AsyncMock()
        gamma = AsyncMock()
        clob = AsyncMock()

        station = _make_station()
        contracts = [_make_contract("tok_1")]
        price = _make_market_price("tok_1")

        weather.fetch_ensemble = AsyncMock(return_value=[_make_ensemble()])
        weather.fetch_hrrr = AsyncMock(return_value=[])
        mesonet.fetch_observations = AsyncMock(return_value=[])
        gamma.fetch_weather_markets = AsyncMock(return_value=contracts)
        clob.get_market_price = AsyncMock(return_value=price)

        collector = DataCollector(weather=weather, mesonet=mesonet, gamma=gamma, clob=clob)

        with patch("src.orchestrator.data_collector.get_stations", return_value={"NYC": station}):
            await collector.collect_snapshot()

        # Record outcome
        collector.record_outcome("tok_1", True)

        records = collector.get_matched_records()
        assert len(records) >= 1
        rec = records[0]
        assert rec["station_id"] == "KNYC"
        assert "market_prob" in rec
        assert rec["actual_outcome"] is True


# ---------------------------------------------------------------------------
# TradingPipeline tests
# ---------------------------------------------------------------------------


class TestTradingPipeline:

    def _build_pipeline(
        self,
        snapshots: list[DataSnapshot] | None = None,
        signal_action: str = "TRADE",
        signal_edge: float = 0.10,
    ) -> tuple[TradingPipeline, dict]:
        """Build a pipeline with all mocked dependencies. Returns (pipeline, mocks_dict)."""
        collector = AsyncMock()
        prob_engine = MagicMock()
        regime_classifier = MagicMock()
        edge_detector = MagicMock()
        position_sizer = MagicMock()
        executor = AsyncMock()
        prediction_log = MagicMock()
        paper_trader = MagicMock()

        station = _make_station()
        contract = _make_contract("tok_1", "NYC")
        price = _make_market_price("tok_1")
        regime = _make_regime("KNYC", "HIGH")
        signal = _make_signal(signal_action, signal_edge, "tok_1")

        now = datetime.now(tz=timezone.utc)

        snap = DataSnapshot(
            station_id="KNYC",
            timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=[contract],
            market_prices={"tok_1": price},
        )

        if snapshots is None:
            snapshots = [snap]

        collector.collect_snapshot = AsyncMock(return_value=snapshots)
        regime_classifier.classify = MagicMock(return_value=regime)

        # prob_engine returns a mock distribution
        mock_dist = MagicMock()
        prob_engine.compute_distribution = MagicMock(return_value=mock_dist)
        prob_engine.compute_bucket_probability = MagicMock(return_value=0.45)

        edge_detector.evaluate = MagicMock(return_value=signal)
        position_sizer.compute = MagicMock(return_value=2.0)

        executor.execute = AsyncMock(return_value=MagicMock(trade_id="t1") if signal_action == "TRADE" else None)
        executor.check_stale_quotes = AsyncMock(return_value=False)
        executor.check_resolution_proximity = AsyncMock(return_value=False)

        pipeline = TradingPipeline(
            collector=collector,
            prob_engine=prob_engine,
            regime_classifier=regime_classifier,
            edge_detector=edge_detector,
            position_sizer=position_sizer,
            executor=executor,
            prediction_log=prediction_log,
            paper_trader=paper_trader,
        )

        mocks = {
            "collector": collector,
            "prob_engine": prob_engine,
            "regime_classifier": regime_classifier,
            "edge_detector": edge_detector,
            "position_sizer": position_sizer,
            "executor": executor,
            "prediction_log": prediction_log,
            "paper_trader": paper_trader,
            "station": station,
            "contract": contract,
            "price": price,
            "regime": regime,
            "signal": signal,
        }
        return pipeline, mocks

    @pytest.mark.asyncio
    async def test_pipeline_run_single_cycle(self):
        """One cycle: fetches data, classifies regimes, computes probabilities, evaluates edges, sizes, logs."""
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            result = await pipeline.run_cycle(bankroll=300.0, current_exposure=0.0)

        # Verify the cycle ran
        mocks["collector"].collect_snapshot.assert_awaited_once()
        mocks["regime_classifier"].classify.assert_called()
        mocks["prob_engine"].compute_distribution.assert_called()
        mocks["edge_detector"].evaluate.assert_called()
        mocks["prediction_log"].log.assert_called()

        assert "signals_generated" in result
        assert "trades_placed" in result
        assert "skips" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_pipeline_paper_trade_executed(self):
        """When edge detected and paper_trading=True, paper trade recorded via PaperTrader."""
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["paper_trader"].record_trade.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_skip_logged(self):
        """When edge insufficient, SKIP signal logged to PredictionLog."""
        pipeline, mocks = self._build_pipeline(signal_action="SKIP", signal_edge=0.02)

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["prediction_log"].log.assert_called()
        # Verify the logged entry contains a SKIP signal
        logged_entry = mocks["prediction_log"].log.call_args[0][0]
        assert logged_entry.signal.action == "SKIP"

    @pytest.mark.asyncio
    async def test_pipeline_stale_quote_check(self):
        """Pipeline checks for stale quotes before placing new orders."""
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["executor"].check_stale_quotes.assert_awaited()

    @pytest.mark.asyncio
    async def test_pipeline_resolution_check(self):
        """Pipeline cancels orders near resolution."""
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["executor"].check_resolution_proximity.assert_awaited()


# ---------------------------------------------------------------------------
# PipelineScheduler tests
# ---------------------------------------------------------------------------


class TestPipelineScheduler:

    def test_scheduler_registers_model_run_events(self):
        """Scheduler creates events for GFS (00,06,12,18 UTC) and ECMWF (00,12 UTC) model runs."""
        pipeline = MagicMock()
        scheduler = PipelineScheduler(pipeline=pipeline)
        events = scheduler.get_scheduled_events()

        # GFS: 4 runs -> 4 events
        gfs_events = [e for e in events if e["event_type"] == "gfs_update"]
        assert len(gfs_events) == 4

        # ECMWF: 2 runs -> 2 events
        ecmwf_events = [e for e in events if e["event_type"] == "ecmwf_update"]
        assert len(ecmwf_events) == 2

    def test_scheduler_morning_refinement_event(self):
        """Scheduler creates event at 14:30 UTC for same-day refinement."""
        pipeline = MagicMock()
        scheduler = PipelineScheduler(pipeline=pipeline)
        events = scheduler.get_scheduled_events()

        refinement_events = [e for e in events if e["event_type"] == "morning_refinement"]
        assert len(refinement_events) == 1
        assert refinement_events[0]["time"] == dt_time(14, 30)
