"""Main trading pipeline -- runs one cycle of the full system."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.config.stations import get_stations, Station
from src.data.models import (
    MarketContract,
    MarketPrice,
    MOSForecast,
    RegimeClassification,
    TradingSignal,
)
from src.orchestrator.data_collector import DataCollector, DataSnapshot
from src.prediction.probability_engine import ProbabilityEngine
from src.prediction.regime_classifier import RegimeClassifier
from src.prediction.calibration import IsotonicCalibrator, CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.position_sizer import PositionSizer
from src.trading.executor import OrderExecutor
from src.verification.prediction_log import PredictionLog, SignalLogEntry
from src.verification.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class TradingPipeline:
    """Main trading pipeline -- runs one cycle of the full system."""

    def __init__(
        self,
        collector: DataCollector,
        prob_engine: ProbabilityEngine,
        regime_classifier: RegimeClassifier,
        edge_detector: EdgeDetector,
        position_sizer: PositionSizer,
        executor: OrderExecutor,
        prediction_log: PredictionLog,
        paper_trader: PaperTrader,
        calibrator: IsotonicCalibrator | None = None,
        cusum: CUSUMMonitor | None = None,
    ) -> None:
        self._collector = collector
        self._prob_engine = prob_engine
        self._regime_classifier = regime_classifier
        self._edge_detector = edge_detector
        self._position_sizer = position_sizer
        self._executor = executor
        self._prediction_log = prediction_log
        self._paper_trader = paper_trader
        self._calibrator = calibrator
        self._cusum = cusum
        self._last_model_update: datetime = datetime.now(tz=timezone.utc)
        self._previous_prices: dict[str, float] = {}

    async def run_cycle(
        self,
        bankroll: float = 300.0,
        current_exposure: float = 0.0,
    ) -> dict:
        """Run one full pipeline cycle.

        For each station:
        1. Get latest data snapshot
        2. Classify regime
        3. Compute MOS-anchored probability distribution
        4. For each active market contract:
           a. Compute bucket probability (model_prob)
           b. Get market price (market_prob)
           c. Evaluate edge
           d. If TRADE: size position, execute (paper or live)
           e. Log signal to PredictionLog (TRADE or SKIP)
        5. Check stale quotes
        6. Check resolution proximity

        Returns summary dict.
        """
        signals_generated = 0
        trades_placed = 0
        skips = 0
        errors = 0

        now = datetime.now(tz=timezone.utc)
        stations = get_stations()

        # 1. Collect latest data
        try:
            snapshots = await self._collector.collect_snapshot()
        except Exception:
            logger.exception("Failed to collect snapshots")
            return {"signals_generated": 0, "trades_placed": 0, "skips": 0, "errors": 1}

        # Build lookup by station_id
        snap_by_station: dict[str, DataSnapshot] = {s.station_id: s for s in snapshots}

        # Track current prices for stale-quote checking
        current_prices: dict[str, float] = {}

        for city, station in stations.items():
            snap = snap_by_station.get(station.station_id)
            if snap is None:
                logger.warning("No snapshot for station %s", station.station_id)
                continue

            # 2. Classify regime
            try:
                ensemble_spread = snap.gfs_ensemble.std if snap.gfs_ensemble else 4.0
                regime = self._regime_classifier.classify(
                    station_id=station.station_id,
                    valid_date=now.date(),
                    ensemble_spread=ensemble_spread,
                )
            except Exception:
                logger.exception("Regime classification failed for %s", station.station_id)
                errors += 1
                continue

            # 3. Build probability distribution
            #    We need a MOS forecast; synthesize one from ensemble if unavailable
            try:
                mos = _synthesize_mos(snap, station, now)
                distribution = self._prob_engine.compute_distribution(
                    mos=mos,
                    gfs_ensemble=snap.gfs_ensemble,
                    ecmwf_ensemble=snap.ecmwf_ensemble,
                    station=station,
                )
            except Exception:
                logger.exception("Probability distribution failed for %s", station.station_id)
                errors += 1
                continue

            # 4. Evaluate each market contract
            for contract in snap.market_contracts:
                try:
                    # a. Model probability
                    model_prob = self._prob_engine.compute_bucket_probability(
                        distribution,
                        contract.temp_bucket_low,
                        contract.temp_bucket_high,
                    )

                    # Apply calibration if available
                    if self._calibrator and self._calibrator.is_fitted:
                        calibrated = self._calibrator.transform([model_prob])
                        model_prob = calibrated[0]

                    # b. Market price
                    price = snap.market_prices.get(contract.token_id)
                    if price is None:
                        logger.warning("No price for %s, skipping", contract.token_id)
                        continue

                    market_prob = price.mid
                    current_prices[contract.token_id] = market_prob

                    # Hours to resolution
                    resolution_dt = datetime.combine(
                        contract.resolution_date,
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                    hours_to_resolution = max(
                        0.0,
                        (resolution_dt - now).total_seconds() / 3600.0,
                    )

                    # c. Evaluate edge
                    signal = self._edge_detector.evaluate(
                        model_prob=model_prob,
                        market_prob=market_prob,
                        regime=regime,
                        volume_24h=price.volume_24h,
                        hours_to_resolution=hours_to_resolution,
                        market_id=contract.token_id,
                    )

                    signals_generated += 1

                    # d. Size and execute if TRADE
                    if signal.action == "TRADE":
                        size_usd = self._position_sizer.compute(
                            edge=signal.edge,
                            market_prob=market_prob,
                            bankroll=bankroll,
                            current_exposure=current_exposure,
                            ensemble_spread_pctile=regime.ensemble_spread_percentile,
                        )

                        # Update signal with sized kelly
                        signal = TradingSignal(
                            market_id=signal.market_id,
                            direction=signal.direction,
                            action=signal.action,
                            edge=signal.edge,
                            kelly_size=size_usd,
                            timestamp=signal.timestamp,
                        )

                        # Execute
                        trade_record = await self._executor.execute(
                            signal=signal,
                            token_id=contract.token_id,
                            market_price=price,
                        )

                        # Record paper trade
                        if trade_record is not None:
                            self._paper_trader.record_trade(
                                signal=signal,
                                contract=contract,
                                entry_price=price.mid,
                                amount_usd=size_usd,
                            )
                            current_exposure += size_usd
                            trades_placed += 1
                    else:
                        skips += 1

                    # e. Log signal (TRADE or SKIP)
                    log_entry = SignalLogEntry(
                        signal=signal,
                        station_id=station.station_id,
                        regime=regime,
                        model_probability=model_prob,
                        market_probability=market_prob,
                        contract=contract,
                    )
                    self._prediction_log.log(log_entry)

                    # CUSUM monitoring
                    if self._cusum is not None:
                        residual = abs(model_prob - market_prob) - signal.edge
                        self._cusum.update(residual)
                        if self._cusum.alarm:
                            logger.warning("CUSUM alarm triggered -- model may be degrading")

                except Exception:
                    logger.exception(
                        "Error processing contract %s for station %s",
                        contract.token_id,
                        station.station_id,
                    )
                    errors += 1

        # 5. Check stale quotes
        try:
            await self._executor.check_stale_quotes(
                last_model_update=self._last_model_update,
                current_prices=current_prices,
                previous_prices=self._previous_prices,
            )
        except Exception:
            logger.exception("Stale quote check failed")
            errors += 1

        # 6. Check resolution proximity for earliest-resolving contract
        try:
            min_hours = float("inf")
            for snap in snapshots:
                for contract in snap.market_contracts:
                    resolution_dt = datetime.combine(
                        contract.resolution_date,
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                    h = (resolution_dt - now).total_seconds() / 3600.0
                    min_hours = min(min_hours, h)
            if min_hours < float("inf"):
                await self._executor.check_resolution_proximity(min_hours)
        except Exception:
            logger.exception("Resolution proximity check failed")
            errors += 1

        # Update previous prices for next cycle
        self._previous_prices = current_prices
        self._last_model_update = now

        summary = {
            "signals_generated": signals_generated,
            "trades_placed": trades_placed,
            "skips": skips,
            "errors": errors,
        }
        logger.info("Pipeline cycle complete: %s", summary)
        return summary

    async def check_and_cancel_stale(self, last_model_update: datetime) -> None:
        """Check for stale quotes and cancel if needed."""
        await self._executor.check_stale_quotes(
            last_model_update=last_model_update,
            current_prices={},
            previous_prices=self._previous_prices,
        )


def _synthesize_mos(snap: DataSnapshot, station: Station, now: datetime) -> MOSForecast:
    """Create a synthetic MOS forecast from ensemble data when real MOS is unavailable."""
    if snap.gfs_ensemble is not None:
        high = snap.gfs_ensemble.mean
    elif snap.ecmwf_ensemble is not None:
        high = snap.ecmwf_ensemble.mean
    elif snap.hrrr:
        high = max(h.temp_f for h in snap.hrrr)
    else:
        high = 70.0  # fallback

    return MOSForecast(
        station_id=station.station_id,
        run_time=now,
        valid_date=now.date(),
        high_f=high,
        low_f=high - 15.0,  # rough estimate
    )
