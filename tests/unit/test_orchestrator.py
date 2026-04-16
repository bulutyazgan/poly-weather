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
    async def test_pipeline_paper_trade_uses_ask_not_mid(self):
        """Paper trade entry_price must be the ask (execution) price, not mid.

        Mock MarketPrice has bid=0.30, ask=0.35, mid=0.325.
        The paper trade must record entry_price=0.35 (ask), not 0.325 (mid).
        Using mid overstates paper P&L by ~8% per trade.
        """
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        call_kwargs = mocks["paper_trader"].record_trade.call_args
        entry_price = call_kwargs.kwargs.get("entry_price") or call_kwargs[1].get("entry_price")
        if entry_price is None:
            # Might be passed as positional
            entry_price = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
        assert entry_price == pytest.approx(0.35), (
            f"Paper trade should use ask price (0.35), not mid (0.325). Got {entry_price}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_kelly_uses_ask_not_mid(self):
        """Kelly denominator must use ask (BUY_YES) not mid for consistent sizing.

        Mock MarketPrice: bid=0.30, ask=0.35, mid=0.325.
        BUY_YES Kelly denominator = (1 - ask) = 0.65, not (1 - mid) = 0.675.
        Position sizer should receive market_prob=0.35 (ask).
        """
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        call_kwargs = mocks["position_sizer"].compute.call_args[1]
        # BUY_YES signal → market_prob should be ask=0.35, not mid=0.325
        assert call_kwargs["market_prob"] == pytest.approx(0.35), (
            f"Kelly should use ask price (0.35), not mid (0.325). Got {call_kwargs['market_prob']}"
        )

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
    async def test_expired_orders_cancelled_before_new_trades(self):
        """TTL sweep must run BEFORE new orders are placed.

        Bug: cancel_expired_orders() was called after pass 2, so
        cancel_all_orders() would nuke orders just placed in the
        current cycle alongside the expired ones.
        """
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")
        call_order = []
        mocks["executor"].cancel_expired_orders = AsyncMock(
            side_effect=lambda: call_order.append("cancel_expired")
        )
        orig_execute = mocks["executor"].execute
        async def track_execute(*a, **kw):
            call_order.append("execute")
            return await orig_execute(*a, **kw)
        mocks["executor"].execute = AsyncMock(side_effect=track_execute)

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        assert "cancel_expired" in call_order, "cancel_expired_orders was not called"
        assert "execute" in call_order, "execute was not called"
        assert call_order.index("cancel_expired") < call_order.index("execute"), (
            f"cancel_expired_orders must run BEFORE execute, but order was: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_resolved_contracts_skipped(self):
        """Contracts whose resolution time has passed must not be evaluated.

        Bug: already-resolved contracts (e.g. April 15 when running on
        April 16) create huge model-vs-market residuals that pollute
        the CUSUM monitor, potentially triggering a false alarm that
        blocks ALL trades.
        """
        from src.prediction.calibration import CUSUMMonitor

        now = datetime.now(tz=timezone.utc)
        # Past contract: resolved yesterday
        past_contract = MarketContract(
            token_id="tok_past",
            condition_id="cond_past",
            question="Already resolved?",
            city="NYC",
            resolution_date=(now - timedelta(days=1)).date(),
            temp_bucket_low=70.0,
            temp_bucket_high=75.0,
            outcome="Yes",
        )
        # Future contract: resolves tomorrow
        future_contract = _make_contract("tok_future", "NYC")
        future_contract = future_contract.model_copy(
            update={"resolution_date": (now + timedelta(days=1)).date()}
        )

        past_price = MarketPrice(
            token_id="tok_past",
            timestamp=now,
            bid=0.90, ask=0.95, mid=0.925, volume_24h=5000.0,
        )
        future_price = _make_market_price("tok_future")

        snap = DataSnapshot(
            station_id="KNYC",
            timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=[past_contract, future_contract],
            market_prices={
                "tok_past": past_price,
                "tok_future": future_price,
            },
        )

        pipeline, mocks = self._build_pipeline(snapshots=[snap], signal_action="SKIP")
        cusum = CUSUMMonitor(threshold=2.0)
        pipeline._cusum = cusum

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            result = await pipeline.run_cycle(bankroll=300.0)

        # Edge detector should only be called for the future contract
        assert mocks["edge_detector"].evaluate.call_count == 1
        # The past contract should not be in the signal count
        assert result["signals_generated"] == 1

    @pytest.mark.asyncio
    async def test_cusum_excludes_implausible_and_foregone(self):
        """CUSUM must not update on implausible_edge or foregone_conclusion.

        Bug: implausible_edge contracts feed huge residuals (e.g. -0.73)
        into CUSUM.  A handful of these across 1-2 cycles can trigger a
        false alarm that blocks ALL trades system-wide.
        """
        from src.prediction.calibration import CUSUMMonitor

        # Create signals with different skip reasons
        implausible_signal = TradingSignal(
            market_id="tok_impl", direction="BUY_NO", action="SKIP",
            edge=0.0, kelly_size=0.0, timestamp=datetime.now(tz=timezone.utc),
            skip_reason="implausible_edge",
        )
        foregone_signal = TradingSignal(
            market_id="tok_fore", direction="BUY_NO", action="SKIP",
            edge=0.0, kelly_size=0.0, timestamp=datetime.now(tz=timezone.utc),
            skip_reason="foregone_conclusion",
        )
        normal_signal = TradingSignal(
            market_id="tok_norm", direction="BUY_YES", action="SKIP",
            edge=0.05, kelly_size=0.0, timestamp=datetime.now(tz=timezone.utc),
            skip_reason="below_threshold",
        )

        pipeline, mocks = self._build_pipeline(signal_action="SKIP")
        cusum = CUSUMMonitor(threshold=2.0)
        pipeline._cusum = cusum

        # Return different signals for each call
        mocks["edge_detector"].evaluate = MagicMock(
            side_effect=[implausible_signal, foregone_signal, normal_signal]
        )

        # 3 contracts, each with a price
        now = datetime.now(tz=timezone.utc)
        contracts = [
            _make_contract("tok_impl", "NYC"),
            _make_contract("tok_fore", "NYC"),
            _make_contract("tok_norm", "NYC"),
        ]
        prices = {
            c.token_id: _make_market_price(c.token_id) for c in contracts
        }
        snap = DataSnapshot(
            station_id="KNYC", timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=contracts, market_prices=prices,
        )
        mocks["collector"].collect_snapshot = AsyncMock(return_value=[snap])

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        # Only the normal (below_threshold) signal should update CUSUM.
        # CUSUM starts at 0; one small residual should leave both sums near 0.
        assert cusum.cusum_pos < 0.5
        assert cusum.cusum_neg < 0.5

    @pytest.mark.asyncio
    async def test_pipeline_stale_quote_check(self):
        """Pipeline checks for stale quotes before placing new orders."""
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["executor"].check_stale_quotes.assert_awaited()

    @pytest.mark.asyncio
    async def test_stale_quote_check_runs_before_execution(self):
        """Stale quote check must run BEFORE placing new orders.

        If it runs after, cancel_all_orders() would nuke fresh orders
        placed in the current cycle.
        """
        call_order = []

        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        original_execute = mocks["executor"].execute
        original_stale = mocks["executor"].check_stale_quotes

        async def track_execute(*args, **kwargs):
            call_order.append("execute")
            return await original_execute(*args, **kwargs)

        async def track_stale(*args, **kwargs):
            call_order.append("check_stale_quotes")
            return await original_stale(*args, **kwargs)

        mocks["executor"].execute = AsyncMock(side_effect=track_execute)
        mocks["executor"].check_stale_quotes = AsyncMock(side_effect=track_stale)

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        assert "check_stale_quotes" in call_order
        assert "execute" in call_order
        stale_idx = call_order.index("check_stale_quotes")
        execute_idx = call_order.index("execute")
        assert stale_idx < execute_idx, (
            f"Stale quote check ran at index {stale_idx} but execute ran at {execute_idx}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_resolution_check(self):
        """Pipeline cancels orders near resolution."""
        pipeline, mocks = self._build_pipeline(signal_action="TRADE")

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["executor"].check_resolution_proximity.assert_awaited()

    @pytest.mark.asyncio
    async def test_resolution_check_ignores_untraded_contracts(self):
        """Past-resolution contracts that weren't traded must not cancel future-date orders.

        Bug: the old code scanned ALL contracts for min hours_to_resolution.
        If today's contract was past resolution (-4h), it cancelled ALL orders
        including valid trades on tomorrow's contracts.
        """
        now = datetime.now(tz=timezone.utc)

        # Contract 1: already past resolution (today, negative hours)
        past_contract = MarketContract(
            token_id="tok_past",
            condition_id="cond_past",
            question="Will NYC high be 70-75°F today?",
            city="NYC",
            resolution_date=now.date(),
            temp_bucket_low=70.0,
            temp_bucket_high=75.0,
            outcome="Yes",
            end_date_utc=now - timedelta(hours=4),
        )

        # Contract 2: future date, valid for trading
        future_contract = _make_contract("tok_future", "NYC")

        past_price = _make_market_price("tok_past")
        future_price = _make_market_price("tok_future")

        snap = DataSnapshot(
            station_id="KNYC",
            timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=[past_contract, future_contract],
            market_prices={
                "tok_past": past_price,
                "tok_future": future_price,
            },
        )

        # Edge detector: SKIP the past contract (too close), TRADE the future one
        def edge_side_effect(*, model_prob, market_prob, regime, volume_24h,
                             hours_to_resolution, market_id, market_bid, market_ask, **kwargs):
            if market_id == "tok_past":
                return _make_signal("SKIP", 0.0, "tok_past")
            return _make_signal("TRADE", 0.10, "tok_future")

        pipeline, mocks = self._build_pipeline(snapshots=[snap])
        mocks["edge_detector"].evaluate = MagicMock(side_effect=edge_side_effect)

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            result = await pipeline.run_cycle(bankroll=300.0)

        # The future contract should be traded
        assert result["trades_placed"] == 1

        # Resolution proximity check should use the future contract's hours,
        # NOT the past contract's negative hours
        call_args = mocks["executor"].check_resolution_proximity.call_args
        hours_checked = call_args[0][0]
        assert hours_checked > 0, (
            f"Resolution check got {hours_checked}h — past-resolution contract "
            f"leaked into the check and would cancel valid future-date trades"
        )

    @pytest.mark.asyncio
    async def test_resolution_check_skipped_when_no_trades(self):
        """When all signals are SKIP, resolution proximity check is not called."""
        pipeline, mocks = self._build_pipeline(signal_action="SKIP", signal_edge=0.02)

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        mocks["executor"].check_resolution_proximity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_buy_no_uses_no_token_price_not_yes(self):
        """BUY_NO must pass the NO token's MarketPrice to executor, not the YES token's.

        Bug: pipeline stored `snap` during Pass 1 and referenced it in Pass 2,
        but `snap` was stale (last station from the loop). Even if `snap` was
        correct, the fallback `snap.market_prices.get(no_token_id, price)` used
        the YES token's MarketPrice when the NO token was missing.

        Fix: NO token price is looked up during Pass 1 and stored in `pending`.
        """
        collector = AsyncMock()
        prob_engine = MagicMock()
        regime_classifier = MagicMock()
        edge_detector = MagicMock()
        position_sizer = MagicMock()
        executor = AsyncMock()
        prediction_log = MagicMock()
        paper_trader = MagicMock()

        now = datetime.now(tz=timezone.utc)

        # Contract with both YES and NO token IDs
        contract = MarketContract(
            token_id="tok_yes",
            no_token_id="tok_no",
            condition_id="cond_1",
            question="Will NYC high be 70-75°F?",
            city="NYC",
            resolution_date=date(2026, 4, 16),
            temp_bucket_low=70.0,
            temp_bucket_high=75.0,
            outcome="Yes",
            volume_24h=5000.0,
        )

        yes_price = MarketPrice(
            token_id="tok_yes", timestamp=now,
            bid=0.58, ask=0.62, mid=0.60, volume_24h=5000.0,
        )
        no_price = MarketPrice(
            token_id="tok_no", timestamp=now,
            bid=0.38, ask=0.42, mid=0.40, volume_24h=5000.0,
        )

        snap = DataSnapshot(
            station_id="KNYC",
            timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=[contract],
            market_prices={"tok_yes": yes_price, "tok_no": no_price},
        )

        collector.collect_snapshot = AsyncMock(return_value=[snap])
        regime_classifier.classify = MagicMock(return_value=_make_regime("KNYC", "HIGH"))

        mock_dist = MagicMock()
        prob_engine.compute_distribution = MagicMock(return_value=mock_dist)
        # model_prob=0.30 < market_prob=0.60 → BUY_NO direction
        prob_engine.compute_bucket_probability = MagicMock(return_value=0.30)

        buy_no_signal = TradingSignal(
            market_id="tok_yes",
            direction="BUY_NO",
            action="TRADE",
            edge=0.30,
            kelly_size=0.0,
            timestamp=now,
        )
        edge_detector.evaluate = MagicMock(return_value=buy_no_signal)
        position_sizer.compute = MagicMock(return_value=2.0)
        executor.execute = AsyncMock(return_value=MagicMock(trade_id="t1"))
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

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": _make_station()}):
            await pipeline.run_cycle(bankroll=300.0)

        # The executor must have been called with the NO token ID and NO price
        executor.execute.assert_awaited_once()
        call_kwargs = executor.execute.call_args
        assert call_kwargs.kwargs["token_id"] == "tok_no", (
            "BUY_NO should use the NO token, not the YES token"
        )
        exec_price = call_kwargs.kwargs["market_price"]
        assert exec_price.token_id == "tok_no", (
            f"BUY_NO should use NO token price (ask={no_price.ask}), "
            f"got {exec_price.token_id} price (ask={exec_price.ask})"
        )


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

    def test_get_sorted_times_returns_chronological_order(self):
        """Sorted times should be in ascending chronological order."""
        pipeline = MagicMock()
        scheduler = PipelineScheduler(pipeline=pipeline)
        sorted_times = scheduler._get_sorted_times()

        times_only = [t for t, _ in sorted_times]
        assert times_only == sorted(times_only)
        # Should have 7 events total (4 GFS + 2 ECMWF + 1 refinement)
        assert len(sorted_times) == 7

    def test_seconds_until_future_time(self):
        """_seconds_until returns positive seconds for a future time today."""
        pipeline = MagicMock()
        scheduler = PipelineScheduler(pipeline=pipeline)

        # Mock "now" to 10:00 UTC, target 14:30 UTC -> 4.5 hours = 16200s
        mock_now = datetime(2026, 4, 15, 10, 0, 0, tzinfo=timezone.utc)
        with patch("src.orchestrator.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.combine = datetime.combine
            seconds = scheduler._seconds_until(dt_time(14, 30))

        assert abs(seconds - 16200.0) < 1.0

    def test_seconds_until_past_time_wraps_to_tomorrow(self):
        """_seconds_until wraps to next day for a time that already passed today."""
        pipeline = MagicMock()
        scheduler = PipelineScheduler(pipeline=pipeline)

        # Mock "now" to 15:00 UTC, target 14:30 UTC -> should be ~23.5h until tomorrow
        mock_now = datetime(2026, 4, 15, 15, 0, 0, tzinfo=timezone.utc)
        with patch("src.orchestrator.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.combine = datetime.combine
            seconds = scheduler._seconds_until(dt_time(14, 30))

        expected = 23.5 * 3600  # 23.5 hours
        assert abs(seconds - expected) < 1.0

    async def test_run_event_delegates_to_pipeline(self):
        """run_event calls pipeline.run_cycle and returns its result."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 5, "trades_placed": 1, "skips": 4, "errors": 0
        })
        scheduler = PipelineScheduler(pipeline=pipeline)

        result = await scheduler.run_event("gfs_update", bankroll=300.0)

        pipeline.run_cycle.assert_called_once_with(bankroll=300.0)
        assert result["signals_generated"] == 5
        assert result["trades_placed"] == 1

    async def test_run_event_checks_resolutions_before_pipeline(self):
        """Resolution checker runs before every pipeline cycle to free exposure."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 5, "trades_placed": 2, "skips": 3, "errors": 0
        })
        resolution_checker = MagicMock()
        resolution_checker.check_resolutions = AsyncMock(return_value={
            "checked": 3, "resolved": 2, "errors": 0
        })
        scheduler = PipelineScheduler(
            pipeline=pipeline, resolution_checker=resolution_checker
        )

        # Non-resolution event (e.g. GFS update) should ALSO check resolutions
        result = await scheduler.run_event("gfs_update", bankroll=300.0)

        # Resolution checker must have been called before the pipeline
        resolution_checker.check_resolutions.assert_awaited_once()
        pipeline.run_cycle.assert_awaited_once_with(bankroll=300.0)
        assert result["trades_placed"] == 2

    async def test_run_event_resolution_failure_doesnt_block_pipeline(self):
        """If resolution check fails, pipeline still runs."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 3, "trades_placed": 1, "skips": 2, "errors": 0
        })
        resolution_checker = MagicMock()
        resolution_checker.check_resolutions = AsyncMock(
            side_effect=Exception("Gamma API timeout")
        )
        scheduler = PipelineScheduler(
            pipeline=pipeline, resolution_checker=resolution_checker
        )

        result = await scheduler.run_event("market_refresh", bankroll=300.0)

        # Pipeline must still run despite resolution check failure
        pipeline.run_cycle.assert_awaited_once()
        assert result["trades_placed"] == 1

    async def test_start_and_stop(self):
        """start() creates a background task, stop() cancels it."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 0, "trades_placed": 0, "skips": 0, "errors": 0
        })
        scheduler = PipelineScheduler(pipeline=pipeline)

        await scheduler.start(bankroll=300.0)
        assert scheduler._running is True
        assert scheduler._task is not None

        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._task is None

    async def test_loop_runs_event_on_schedule(self):
        """_loop sleeps until next event then runs pipeline."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 1, "trades_placed": 0, "skips": 1, "errors": 0
        })
        scheduler = PipelineScheduler(pipeline=pipeline)

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                scheduler._running = False

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler._running = True
            await scheduler._loop(bankroll=300.0)

        # Pipeline should have been called at least once
        assert pipeline.run_cycle.call_count >= 1

    async def test_loop_handles_pipeline_exception(self):
        """_loop logs but doesn't crash when pipeline raises."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(side_effect=RuntimeError("API down"))
        scheduler = PipelineScheduler(pipeline=pipeline)

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                scheduler._running = False

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler._running = True
            # Should not raise
            await scheduler._loop(bankroll=300.0)

    def test_status_property_initial(self):
        """status property returns correct shape before any run."""
        pipeline = MagicMock()
        scheduler = PipelineScheduler(pipeline=pipeline)
        s = scheduler.status
        assert s["running"] is False
        assert s["last_run_time"] is None
        assert s["last_run_result"] is None
        assert s["last_error"] is None
        assert s["next_event_type"] == ""

    async def test_loop_caps_sleep_at_market_refresh(self):
        """_loop should cap sleep at MARKET_REFRESH_MINUTES and fire market_refresh."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 5, "trades_placed": 0, "skips": 5, "errors": 0
        })
        scheduler = PipelineScheduler(pipeline=pipeline)
        scheduler.MARKET_REFRESH_MINUTES = 1  # 1 minute cap for test

        recorded_sleeps: list[float] = []

        async def fake_sleep(seconds):
            recorded_sleeps.append(seconds)
            scheduler._running = False  # stop after one iteration

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler._running = True
            await scheduler._loop(bankroll=300.0)

        # Sleep should be capped to 60s (1 minute), not the next NWP time
        assert recorded_sleeps[0] <= 60.0
        assert scheduler._next_event_type == "market_refresh"

    async def test_loop_tracks_last_run(self):
        """After a successful event, status shows last_run_time and result."""
        pipeline = MagicMock()
        result = {"signals_generated": 3, "trades_placed": 1, "skips": 2, "errors": 0}
        pipeline.run_cycle = AsyncMock(return_value=result)
        scheduler = PipelineScheduler(pipeline=pipeline)

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                scheduler._running = False

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler._running = True
            await scheduler._loop(bankroll=300.0)

        s = scheduler.status
        assert s["last_run_time"] is not None
        assert s["last_run_result"] == result
        assert s["last_error"] is None

    async def test_loop_tracks_errors(self):
        """After a failed event, status shows last_error."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(side_effect=RuntimeError("timeout"))
        scheduler = PipelineScheduler(pipeline=pipeline)

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                scheduler._running = False

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler._running = True
            await scheduler._loop(bankroll=300.0)

        s = scheduler.status
        assert s["last_run_time"] is not None
        assert "timeout" in s["last_error"]

    async def test_loop_passes_bankroll_to_run_cycle(self):
        """Scheduler must forward bankroll kwarg to pipeline.run_cycle()."""
        pipeline = MagicMock()
        pipeline.run_cycle = AsyncMock(return_value={
            "signals_generated": 0, "trades_placed": 0, "skips": 0, "errors": 0
        })
        scheduler = PipelineScheduler(pipeline=pipeline)

        call_count = 0

        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                scheduler._running = False

        with patch("asyncio.sleep", side_effect=fake_sleep):
            scheduler._running = True
            await scheduler._loop(bankroll=500.0)

        # Verify bankroll was forwarded, not the default 300
        call_kwargs = pipeline.run_cycle.call_args[1]
        assert call_kwargs["bankroll"] == 500.0


class TestMixedDirectionTrades:
    """BUY_YES on one bucket + BUY_NO on another for the same station/date
    are complementary positions (not conflicting) and must all execute."""

    @pytest.mark.asyncio
    async def test_mixed_directions_all_trade(self):
        """BUY_YES on 70-72°F + BUY_NO on 76-78°F should both execute.

        The model predicts temp ~70-72°F.  BUY_YES on that bucket and
        BUY_NO on a far-away bucket are perfectly consistent — the old
        conflicting_adjacent_trade logic incorrectly demoted one of them.
        """
        now = datetime.now(tz=timezone.utc)
        station = _make_station()

        contract_yes = MarketContract(
            token_id="tok_yes",
            no_token_id="tok_yes_no",
            condition_id="c1",
            question="70-72°F",
            city="NYC",
            resolution_date=date(2026, 4, 18),
            temp_bucket_low=70.0,
            temp_bucket_high=72.0,
            outcome="Yes",
        )
        contract_no = MarketContract(
            token_id="tok_no",
            no_token_id="tok_no_no",
            condition_id="c2",
            question="76-78°F",
            city="NYC",
            resolution_date=date(2026, 4, 18),
            temp_bucket_low=76.0,
            temp_bucket_high=78.0,
            outcome="Yes",
        )

        price_yes = MarketPrice(
            token_id="tok_yes", timestamp=now,
            bid=0.25, ask=0.30, mid=0.275, volume_24h=5000.0,
        )
        price_no = MarketPrice(
            token_id="tok_no", timestamp=now,
            bid=0.20, ask=0.25, mid=0.225, volume_24h=5000.0,
        )
        # Synthetic NO token prices
        price_no_token = MarketPrice(
            token_id="tok_no_no", timestamp=now,
            bid=0.75, ask=0.80, mid=0.775, volume_24h=5000.0,
        )

        snap = DataSnapshot(
            station_id="KNYC",
            timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            gfs_ensemble_all=[_make_ensemble("KNYC", "gfs")],
            ecmwf_ensemble_all=[_make_ensemble("KNYC", "ecmwf")],
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=[contract_yes, contract_no],
            market_prices={
                "tok_yes": price_yes,
                "tok_no": price_no,
                "tok_no_no": price_no_token,
            },
        )

        collector = AsyncMock()
        collector.collect_snapshot = AsyncMock(return_value=[snap])

        prob_engine = MagicMock()
        prob_engine.compute_distribution = MagicMock(return_value=MagicMock())
        # Model: 40% on 70-72, 10% on 76-78
        prob_engine.compute_bucket_probability = MagicMock(side_effect=[0.40, 0.10])

        regime = _make_regime("KNYC", "HIGH")
        regime_classifier = MagicMock()
        regime_classifier.classify = MagicMock(return_value=regime)

        # Edge detector returns TRADE for both — different buckets, not conflicting
        signal_yes = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.12, kelly_size=0.0, timestamp=now,
        )
        signal_no = TradingSignal(
            market_id="tok_no", direction="BUY_NO", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=now,
        )
        edge_detector = MagicMock()
        edge_detector.evaluate = MagicMock(side_effect=[signal_yes, signal_no])

        position_sizer = MagicMock()
        position_sizer.compute = MagicMock(return_value=2.0)

        executor = AsyncMock()
        executor.execute = AsyncMock(return_value=MagicMock(trade_id="t1"))
        executor.cancel_expired_orders = AsyncMock(return_value=0)
        executor.check_stale_quotes = AsyncMock(return_value=False)
        executor.check_resolution_proximity = AsyncMock(return_value=False)

        prediction_log = MagicMock()
        paper_trader = MagicMock()

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

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": station}):
            result = await pipeline.run_cycle(bankroll=300.0)

        # BUY_YES on 70-72°F and BUY_NO on 76-78°F are different buckets —
        # both should execute since they're complementary, not conflicting.
        assert result["trades_placed"] == 2, (
            f"Cross-bucket BUY_YES + BUY_NO should both execute, "
            f"got {result['trades_placed']}"
        )
        assert executor.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_buy_no_without_no_token_skips(self):
        """BUY_NO signal must NOT fall through to buying the YES token.

        If no_token_id is empty or no_price is unavailable, the trade
        must be skipped rather than executing on the wrong token.
        """
        now = datetime.now(tz=timezone.utc)
        station = _make_station()

        # Contract with NO no_token_id
        contract = MarketContract(
            token_id="tok_yes",
            no_token_id="",  # missing NO token
            condition_id="c1",
            question="70-72°F",
            city="NYC",
            resolution_date=date(2026, 4, 18),
            temp_bucket_low=70.0,
            temp_bucket_high=72.0,
            outcome="Yes",
        )
        price = MarketPrice(
            token_id="tok_yes", timestamp=now,
            bid=0.25, ask=0.30, mid=0.275, volume_24h=5000.0,
        )
        snap = DataSnapshot(
            station_id="KNYC",
            timestamp=now,
            gfs_ensemble=_make_ensemble("KNYC", "gfs"),
            ecmwf_ensemble=_make_ensemble("KNYC", "ecmwf"),
            gfs_ensemble_all=[_make_ensemble("KNYC", "gfs")],
            ecmwf_ensemble_all=[_make_ensemble("KNYC", "ecmwf")],
            hrrr=[_make_hrrr("KNYC")],
            observations=[_make_observation("KNYC")],
            market_contracts=[contract],
            market_prices={"tok_yes": price},
        )

        collector = AsyncMock()
        collector.collect_snapshot = AsyncMock(return_value=[snap])
        prob_engine = MagicMock()
        prob_engine.compute_distribution = MagicMock(return_value=MagicMock())
        prob_engine.compute_bucket_probability = MagicMock(return_value=0.10)

        regime = _make_regime("KNYC", "HIGH")
        regime_classifier = MagicMock()
        regime_classifier.classify = MagicMock(return_value=regime)

        # Edge detector returns BUY_NO
        signal = TradingSignal(
            market_id="tok_yes", direction="BUY_NO", action="TRADE",
            edge=0.15, kelly_size=0.0, timestamp=now,
        )
        edge_detector = MagicMock()
        edge_detector.evaluate = MagicMock(return_value=signal)

        position_sizer = MagicMock()
        position_sizer.compute = MagicMock(return_value=2.0)

        executor = AsyncMock()
        executor.execute = AsyncMock(return_value=MagicMock(trade_id="t1"))
        executor.cancel_expired_orders = AsyncMock(return_value=0)
        executor.check_stale_quotes = AsyncMock(return_value=False)
        executor.check_resolution_proximity = AsyncMock(return_value=False)

        pipeline = TradingPipeline(
            collector=collector,
            prob_engine=prob_engine,
            regime_classifier=regime_classifier,
            edge_detector=edge_detector,
            position_sizer=position_sizer,
            executor=executor,
            prediction_log=MagicMock(),
            paper_trader=MagicMock(),
        )

        with patch("src.orchestrator.pipeline.get_stations", return_value={"NYC": station}):
            result = await pipeline.run_cycle(bankroll=300.0)

        # Must NOT execute — buying YES token for a BUY_NO signal reverses the position
        assert result["trades_placed"] == 0, (
            "BUY_NO without NO token must be skipped, not executed on YES token"
        )
        executor.execute.assert_not_called()
