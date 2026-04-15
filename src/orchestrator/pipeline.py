"""Main trading pipeline -- runs one cycle of the full system."""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone

from src.config.stations import get_stations, Station
from src.data.models import (
    EnsembleForecast,
    MarketContract,
    MOSForecast,
    TradingSignal,
)
from src.orchestrator.data_collector import DataCollector, DataSnapshot
from src.prediction.probability_engine import ProbabilityEngine
from src.prediction.regime_classifier import RegimeClassifier
from src.orchestrator.signal_cache import CachedSignal, SignalCache
from src.prediction.calibration import IsotonicCalibrator, CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.exposure_tracker import ExposureTracker
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
        signal_cache: SignalCache | None = None,
        exposure_tracker: ExposureTracker | None = None,
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
        self._signal_cache = signal_cache
        self._exposure_tracker = exposure_tracker
        self._last_model_update: datetime = datetime.now(tz=timezone.utc)
        self._previous_prices: dict[str, float] = {}

    async def run_cycle(
        self,
        bankroll: float = 300.0,
        current_exposure: float = 0.0,
    ) -> dict:
        """Run one full pipeline cycle using two-pass architecture.

        Pass 1 — Signal generation:
          For each station, classify regime, compute probabilities, evaluate
          edge for every market contract.  Collect all signals into a pending
          list.  No sizing or execution yet.

        Pass 2 — Sizing & execution:
          Count how many stations have TRADE signals (for correlation
          penalty).  If CUSUM alarm is active, convert ALL TRADE → SKIP.
          Otherwise, size each TRADE with the correlation-aware sizer and
          execute.

        Returns summary dict.
        """
        signals_generated = 0
        trades_placed = 0
        skips = 0
        errors = 0

        now = datetime.now(tz=timezone.utc)
        stations = get_stations()

        # -- Check CUSUM alarm at the TOP of the cycle -----------------------
        # If alarm fired during a previous cycle, block this cycle's trades
        # then reset so CUSUM starts fresh.  If degradation continues,
        # alarm re-triggers quickly; if model recovered, trading resumes.
        cusum_blocked = self._cusum is not None and self._cusum.alarm
        if cusum_blocked:
            logger.warning("CUSUM alarm active — all trades blocked this cycle")
            self._cusum.reset()

        # -- 1. Collect latest data ------------------------------------------
        try:
            snapshots = await self._collector.collect_snapshot()
        except Exception:
            logger.exception("Failed to collect snapshots")
            return {"signals_generated": 0, "trades_placed": 0, "skips": 0, "errors": 1}

        snap_by_station: dict[str, DataSnapshot] = {s.station_id: s for s in snapshots}
        current_prices: dict[str, float] = {}

        # ================================================================
        # PASS 1 — Signal generation (no sizing, no execution)
        # ================================================================
        pending: list[dict] = []  # each entry holds everything needed for pass 2

        for city, station in stations.items():
            snap = snap_by_station.get(station.station_id)
            if snap is None:
                logger.warning("No snapshot for station %s", station.station_id)
                continue

            # Classify regime — pass station flags and ensemble members
            # so physical flag detectors (chinook, bimodal, etc.) are active
            try:
                ensemble_spread = snap.gfs_ensemble.std if snap.gfs_ensemble else 4.0
                ensemble_members = (
                    snap.gfs_ensemble.members if snap.gfs_ensemble else None
                )
                regime = self._regime_classifier.classify(
                    station_id=station.station_id,
                    valid_date=now.date(),
                    ensemble_spread=ensemble_spread,
                    ensemble_members=ensemble_members,
                    station_flags=station.flags,
                )
            except Exception:
                logger.exception("Regime classification failed for %s", station.station_id)
                errors += 1
                continue

            # Evaluate each market contract with a date-matched distribution
            for contract in snap.market_contracts:
                try:
                    # Build distribution matched to this contract's resolution date
                    mos = _synthesize_mos(snap, station, now, contract.resolution_date)
                    gfs_for_date = _pick_ensemble_for_date(
                        snap.gfs_ensemble_all, contract.resolution_date
                    )
                    ecmwf_for_date = _pick_ensemble_for_date(
                        snap.ecmwf_ensemble_all, contract.resolution_date
                    )
                    distribution = self._prob_engine.compute_distribution(
                        mos=mos,
                        gfs_ensemble=gfs_for_date,
                        ecmwf_ensemble=ecmwf_for_date,
                        station=station,
                    )

                    model_prob = self._prob_engine.compute_bucket_probability(
                        distribution,
                        contract.temp_bucket_low,
                        contract.temp_bucket_high,
                    )

                    if self._calibrator and self._calibrator.is_fitted:
                        model_prob = self._calibrator.transform([model_prob])[0]

                    price = snap.market_prices.get(contract.token_id)
                    if price is None:
                        continue

                    market_prob = price.mid
                    current_prices[contract.token_id] = market_prob

                    resolution_dt = _resolution_utc(contract)
                    hours_to_resolution = max(
                        0.0,
                        (resolution_dt - now).total_seconds() / 3600.0,
                    )

                    signal = self._edge_detector.evaluate(
                        model_prob=model_prob,
                        market_prob=market_prob,
                        regime=regime,
                        volume_24h=contract.volume_24h,  # real traded volume from Gamma
                        hours_to_resolution=hours_to_resolution,
                        market_id=contract.token_id,
                        market_bid=price.bid,
                        market_ask=price.ask,
                    )

                    signals_generated += 1

                    # If CUSUM alarm is active, force all TRADE → SKIP
                    if cusum_blocked and signal.action == "TRADE":
                        signal = TradingSignal(
                            market_id=signal.market_id,
                            direction=signal.direction,
                            action="SKIP",
                            edge=signal.edge,
                            kelly_size=0.0,
                            timestamp=signal.timestamp,
                        )

                    # Look up NO token price now while we have the correct snapshot
                    no_price = None
                    if contract.no_token_id:
                        no_price = snap.market_prices.get(contract.no_token_id)

                    pending.append({
                        "signal": signal,
                        "station": station,
                        "contract": contract,
                        "price": price,
                        "no_price": no_price,
                        "regime": regime,
                        "model_prob": model_prob,
                        "market_prob": market_prob,
                    })

                except Exception:
                    logger.exception(
                        "Error processing contract %s for station %s",
                        contract.token_id,
                        station.station_id,
                    )
                    errors += 1

        # Write model probabilities to signal cache for the price monitor
        if self._signal_cache is not None:
            cache_entries: dict[str, CachedSignal] = {}
            for p in pending:
                token_id = p["contract"].token_id
                cache_entries[token_id] = CachedSignal(
                    model_prob=p["model_prob"],
                    regime=p["regime"],
                    contract=p["contract"],
                    station_id=p["station"].station_id,
                    forecast_time=now,
                )
            self._signal_cache.update(cache_entries)

        # ================================================================
        # PASS 2 — Sizing & execution (correlation-aware)
        # ================================================================

        # Count distinct stations with TRADE signals for correlation penalty
        trade_station_ids = {
            p["station"].station_id
            for p in pending
            if p["signal"].action == "TRADE"
        }
        active_station_count = max(len(trade_station_ids), 1)

        for p in pending:
            signal = p["signal"]
            station = p["station"]
            contract = p["contract"]
            price = p["price"]
            regime = p["regime"]
            model_prob = p["model_prob"]
            market_prob = p["market_prob"]

            if signal.action == "TRADE":
                effective_exposure = (
                    self._exposure_tracker.current
                    if self._exposure_tracker is not None
                    else current_exposure
                )
                size_usd = self._position_sizer.compute(
                    edge=signal.edge,
                    market_prob=market_prob,
                    bankroll=bankroll,
                    current_exposure=effective_exposure,
                    ensemble_spread_pctile=regime.ensemble_spread_percentile,
                    direction=signal.direction,
                    active_station_count=active_station_count,
                )

                signal = TradingSignal(
                    market_id=signal.market_id,
                    direction=signal.direction,
                    action=signal.action,
                    edge=signal.edge,
                    kelly_size=size_usd,
                    timestamp=signal.timestamp,
                )

                # Select correct token and price for the trade direction
                no_price = p["no_price"]
                if signal.direction == "BUY_NO" and contract.no_token_id and no_price is not None:
                    exec_token = contract.no_token_id
                    exec_price = no_price
                else:
                    exec_token = contract.token_id
                    exec_price = price

                trade_record = await self._executor.execute(
                    signal=signal,
                    token_id=exec_token,
                    market_price=exec_price,
                )

                if trade_record is not None:
                    self._paper_trader.record_trade(
                        signal=signal,
                        contract=contract,
                        entry_price=price.mid,
                        amount_usd=size_usd,
                        model_probability=model_prob,
                    )
                    current_exposure += size_usd
                    if self._exposure_tracker is not None:
                        self._exposure_tracker.add(size_usd)
                    trades_placed += 1
            else:
                skips += 1

            # Log signal (TRADE or SKIP)
            log_entry = SignalLogEntry(
                signal=signal,
                station_id=station.station_id,
                regime=regime,
                model_probability=model_prob,
                market_probability=market_prob,
                contract=contract,
            )
            self._prediction_log.log(log_entry)

            # CUSUM monitoring (update even when blocked — tracks ongoing state)
            # Signed residual: positive = model overestimates vs market,
            # negative = model underestimates. CUSUM tracks both directions
            # via cusum_pos and cusum_neg to detect systematic calibration drift.
            if self._cusum is not None:
                residual = model_prob - market_prob
                self._cusum.update(residual)

        # -- Post-trade checks -----------------------------------------------

        # Check stale quotes
        try:
            await self._executor.check_stale_quotes(
                last_model_update=self._last_model_update,
                current_prices=current_prices,
                previous_prices=self._previous_prices,
            )
        except Exception:
            logger.exception("Stale quote check failed")
            errors += 1

        # Check resolution proximity for earliest-resolving contract
        try:
            min_hours = float("inf")
            for snap in snapshots:
                for contract in snap.market_contracts:
                    resolution_dt = _resolution_utc(contract)
                    h = (resolution_dt - now).total_seconds() / 3600.0
                    min_hours = min(min_hours, h)
            if min_hours < float("inf"):
                await self._executor.check_resolution_proximity(min_hours)
        except Exception:
            logger.exception("Resolution proximity check failed")
            errors += 1

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


def _pick_ensemble_for_date(
    forecasts: list[EnsembleForecast],
    target_date: date,
) -> EnsembleForecast | None:
    """Pick the ensemble forecast at the hour of peak ensemble-mean temperature.

    Scans daytime hours (12-00 UTC ≈ 8am-8pm EDT) on the target date
    and returns the hour where the ensemble mean is highest.  This gives
    the consensus Tmax estimate — using the mean (not max member) avoids
    a systematic warm bias of 1-3°F.
    """
    if not forecasts:
        return None

    # Daytime window: 12-00 UTC covers US afternoon heating
    # (8am-8pm EDT, 6am-6pm PDT, 7am-7pm CDT, etc.)
    day_start = datetime.combine(target_date, time(12, 0), tzinfo=timezone.utc)
    day_end = datetime.combine(target_date, time(23, 59), tzinfo=timezone.utc)

    candidates = [
        f for f in forecasts
        if day_start <= f.valid_time <= day_end
        and f.members
    ]
    if not candidates:
        # Broader fallback: any hour within ±18h of noon
        noon_utc = datetime.combine(target_date, time(12, 0), tzinfo=timezone.utc)
        candidates = [
            f for f in forecasts
            if abs((f.valid_time - noon_utc).total_seconds()) <= 18 * 3600
            and f.members
        ]

    if not candidates:
        return None

    # Return the hour with the highest ensemble mean (= consensus Tmax)
    return max(candidates, key=lambda f: f.mean)


def _synthesize_mos(
    snap: DataSnapshot,
    station: Station,
    now: datetime,
    target_date: date | None = None,
) -> MOSForecast:
    """Create a synthetic MOS forecast from available weather data.

    Matches forecast data to the contract's resolution date:
      1. GFS/ECMWF hourly max across the target date's daytime hours
         (12-00 UTC ≈ 8am-8pm EDT) gives a direct Tmax estimate.
      2. HRRR max (only if target_date is today — HRRR is ≤18h ahead).
      3. Hard fallback: 70°F.
    """
    if target_date is None:
        target_date = now.date()

    high: float | None = None

    # Use ensemble hourly data matched to the target date's daytime window
    gfs_day = _pick_ensemble_for_date(snap.gfs_ensemble_all, target_date)
    ecmwf_day = _pick_ensemble_for_date(snap.ecmwf_ensemble_all, target_date)

    candidates: list[float] = []

    # Ensemble Tmax from date-matched peak-heating hour.
    # Use ensemble mean (consensus), not max member — max member
    # systematically overestimates Tmax by 1-3°F.
    if gfs_day is not None:
        candidates.append(gfs_day.mean)
    if ecmwf_day is not None:
        candidates.append(ecmwf_day.mean)

    # HRRR: only valid for today (≤18h ahead), but has actual hourly temps
    # spanning daytime heating — often captures Tmax better than a single
    # ensemble time step
    if snap.hrrr and target_date == now.date():
        candidates.append(max(h.temp_f for h in snap.hrrr))

    high = max(candidates) if candidates else 70.0

    return MOSForecast(
        station_id=station.station_id,
        run_time=now,
        valid_date=target_date,
        high_f=high,
        low_f=high - 15.0,
    )


def _resolution_utc(contract: MarketContract) -> datetime:
    """Return the UTC resolution time for a contract.

    Uses the Gamma API's endDate if available. Falls back to 23:59 UTC
    on the resolution date (end-of-day), NOT midnight (start-of-day).
    The old code used midnight UTC which was ~20-28h early for US cities.
    """
    if contract.end_date_utc is not None:
        return contract.end_date_utc
    # Fallback: end of resolution day in UTC
    return datetime.combine(
        contract.resolution_date,
        time(23, 59, 59),
        tzinfo=timezone.utc,
    )
