"""Main trading pipeline -- runs one cycle of the full system."""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone

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


def _resolve_conflicting_trades(pending: list[dict]) -> None:
    """Resolve same-bucket conflicts (BUY_YES + BUY_NO on identical bucket).

    Cross-bucket conflicts are NOT real conflicts: BUY_YES on 72-73°F
    and BUY_NO on 80-81°F are perfectly consistent — both profit when
    temp is ~72°F.  Only same-bucket conflicts (which shouldn't happen
    since the edge detector picks one direction per contract) are resolved.

    Mutates *pending* in-place.
    """
    from collections import defaultdict

    # Group by (station, date, bucket) — only same-bucket matters
    groups: dict[tuple[str, date, float, float], list[dict]] = defaultdict(list)
    for p in pending:
        if p["signal"].action != "TRADE":
            continue
        c = p["contract"]
        key = (p["station"].station_id, c.resolution_date, c.temp_bucket_low, c.temp_bucket_high)
        groups[key].append(p)

    for key, items in groups.items():
        directions = {p["signal"].direction for p in items}
        if len(directions) < 2:
            continue  # all same direction — no conflict

        # Same bucket, both directions — keep the higher edge
        edge_by_dir: dict[str, float] = defaultdict(float)
        for p in items:
            edge_by_dir[p["signal"].direction] += p["signal"].edge
        keep_dir = max(edge_by_dir, key=edge_by_dir.get)  # type: ignore[arg-type]

        for p in items:
            if p["signal"].direction != keep_dir:
                p["signal"] = TradingSignal(
                    market_id=p["signal"].market_id,
                    direction=p["signal"].direction,
                    action="SKIP",
                    edge=p["signal"].edge,
                    kelly_size=0.0,
                    timestamp=p["signal"].timestamp,
                    skip_reason="same_bucket_conflict",
                )
        logger.info(
            "Same-bucket conflict on %s/%s [%.0f-%.0f°F] — keeping %s",
            key[0], key[1], key[2], key[3], keep_dir,
        )


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
        event_bus=None,
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
        self._event_bus = event_bus
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
        if self._event_bus:
            self._event_bus.publish("pipeline_start", {"event_type": "pipeline_cycle"})
        stations = get_stations()

        # -- Drawdown circuit breaker ------------------------------------------
        if self._exposure_tracker is not None and self._exposure_tracker.is_halted:
            logger.warning(
                "Drawdown circuit breaker active (realized P&L $%.2f) — "
                "skipping entire cycle",
                self._exposure_tracker.realized_pnl,
            )
            return {"signals_generated": 0, "trades_placed": 0, "skips": 0, "errors": 0}

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
                # Blend GFS and ECMWF spreads using the same weights as
                # _synthesize_mos (GFS 0.4, ECMWF 0.6).  Using GFS alone
                # caused Chicago (KORD) to be permanently LOW-confidence
                # because GFS runs wider ensembles than ECMWF at that site.
                gfs_std = snap.gfs_ensemble.std if snap.gfs_ensemble else None
                ecmwf_std = snap.ecmwf_ensemble.std if snap.ecmwf_ensemble else None
                if gfs_std is not None and ecmwf_std is not None:
                    ensemble_spread = 0.4 * gfs_std + 0.6 * ecmwf_std
                elif gfs_std is not None:
                    ensemble_spread = gfs_std
                elif ecmwf_std is not None:
                    ensemble_spread = ecmwf_std
                else:
                    ensemble_spread = 4.0
                ensemble_members = (
                    snap.gfs_ensemble.members if snap.gfs_ensemble else None
                )
                # Between-model spread: compare GFS vs ECMWF at peak
                # heating hour.  Use the WORST-CASE spread across all
                # contract dates so that a station with high disagreement
                # on tomorrow's forecast doesn't slip through with today's
                # low-disagreement regime.
                between_spread = 0.0
                contract_dates = {c.resolution_date for c in snap.market_contracts}
                for cdate in contract_dates:
                    gfs_d = _pick_ensemble_for_date(snap.gfs_ensemble_all, cdate)
                    ecmwf_d = _pick_ensemble_for_date(snap.ecmwf_ensemble_all, cdate)
                    if gfs_d is not None and ecmwf_d is not None:
                        between_spread = max(
                            between_spread, abs(gfs_d.mean - ecmwf_d.mean)
                        )
                regime = self._regime_classifier.classify(
                    station_id=station.station_id,
                    valid_date=now.date(),
                    ensemble_spread=ensemble_spread,
                    ensemble_members=ensemble_members,
                    station_flags=station.flags,
                    between_model_spread=between_spread,
                )
            except Exception:
                logger.exception("Regime classification failed for %s", station.station_id)
                errors += 1
                continue

            # Evaluate each market contract with a date-matched distribution
            for contract in snap.market_contracts:
                # Skip already-resolved contracts — their settled market
                # prices would create huge model-vs-market residuals that
                # pollute CUSUM and waste edge computation.
                resolution_dt = _resolution_utc(contract)
                if resolution_dt <= now:
                    continue

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
                        valid_date=contract.resolution_date,
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

                    hours_to_resolution = (
                        (resolution_dt - now).total_seconds() / 3600.0
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
                        dist_std=distribution.std(),
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
                            skip_reason="cusum_alarm",
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

        # -- Pre-execution cleanup: cancel stale/expired orders BEFORE
        # placing new ones so cancel_all_orders() doesn't nuke orders
        # placed in the current cycle.
        try:
            await self._executor.cancel_expired_orders()
        except Exception:
            logger.exception("Order TTL sweep failed")
            errors += 1

        try:
            await self._executor.check_stale_quotes(
                last_model_update=self._last_model_update,
                current_prices=current_prices,
                previous_prices=self._previous_prices,
            )
        except Exception:
            logger.exception("Stale quote check failed")
            errors += 1

        # ================================================================
        # PASS 1.5 — Resolve conflicting adjacent trades
        # ================================================================
        # When a station has TRADE signals in both BUY_YES and BUY_NO
        # directions for the same resolution date, only one bucket can
        # win.  Keep the direction with the highest total edge; demote
        # the other direction to SKIP.  This prevents guaranteed partial
        # losses from offsetting positions.
        _resolve_conflicting_trades(pending)

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
        traded_contracts: list[MarketContract] = []

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
                # Kelly denominator must match the execution price, not mid.
                # BUY_YES pays ask, wins (1 - ask): denom = 1 - ask → pass ask
                # BUY_NO sells YES at bid, wins bid: denom = bid → pass bid
                if signal.direction == "BUY_YES" and price.ask is not None:
                    kelly_prob = price.ask
                elif signal.direction == "BUY_NO" and price.bid is not None:
                    kelly_prob = price.bid
                else:
                    kelly_prob = market_prob  # fallback to mid

                size_usd = self._position_sizer.compute(
                    edge=signal.edge,
                    market_prob=kelly_prob,
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
                if signal.direction == "BUY_NO":
                    if contract.no_token_id and no_price is not None:
                        exec_token = contract.no_token_id
                        exec_price = no_price
                    else:
                        # Cannot buy YES token for a BUY_NO signal — that
                        # would reverse the trade direction.  Skip instead.
                        logger.warning(
                            "BUY_NO signal for %s but no NO token/price — skipping",
                            contract.token_id,
                        )
                        skips += 1
                        continue
                else:
                    exec_token = contract.token_id
                    exec_price = price

                trade_record = await self._executor.execute(
                    signal=signal,
                    token_id=exec_token,
                    market_price=exec_price,
                )

                if trade_record is not None:
                    # Use actual execution price, not mid, for realistic P&L
                    if signal.direction == "BUY_NO" and no_price is not None:
                        # Paper trader expects YES price; convert NO ask back
                        paper_entry_price = 1.0 - no_price.ask
                    else:
                        paper_entry_price = exec_price.ask

                    self._paper_trader.record_trade(
                        signal=signal,
                        contract=contract,
                        entry_price=paper_entry_price,
                        amount_usd=size_usd,
                        model_probability=model_prob,
                    )
                    current_exposure += size_usd
                    if self._exposure_tracker is not None:
                        self._exposure_tracker.add(size_usd)
                    traded_contracts.append(contract)
                    trades_placed += 1
                    if self._event_bus:
                        self._event_bus.publish("trade_executed", {
                            "trade_id": getattr(trade_record, "trade_id", ""),
                            "direction": signal.direction,
                            "amount_usd": round(size_usd, 2),
                            "price": round(exec_price.ask, 4),
                            "city": contract.city,
                            "question": contract.question,
                        })
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

            # CUSUM monitoring — track model-vs-market residuals to detect
            # systematic calibration drift.  Exclude signals where the
            # residual is uninformative:
            #   implausible_edge: model failure, not calibration drift
            #   foregone_conclusion: both agree, residual ≈ 0
            if self._cusum is not None and signal.skip_reason not in (
                "implausible_edge",
                "foregone_conclusion",
            ):
                residual = model_prob - market_prob
                self._cusum.update(residual)
                if self._event_bus:
                    peak = max(self._cusum.cusum_pos, self._cusum.cusum_neg)
                    pct = (peak / self._cusum.threshold * 100) if self._cusum.threshold > 0 else 0.0
                    self._event_bus.publish("cusum_update", {
                        "alarm": self._cusum.alarm,
                        "cusum_pos": round(self._cusum.cusum_pos, 4),
                        "cusum_neg": round(self._cusum.cusum_neg, 4),
                        "pct_of_threshold": round(pct, 1),
                    })

        # -- Post-trade checks -----------------------------------------------
        # (Stale quote check moved to pre-execution cleanup above.)

        # Check resolution proximity only for contracts we actually traded.
        # Pass the specific token_ids of near-resolution contracts so the
        # executor cancels only those orders — orders on tomorrow's
        # contracts survive.
        try:
            near_resolution_tokens: set[str] = set()
            min_hours = float("inf")
            for contract in traded_contracts:
                resolution_dt = _resolution_utc(contract)
                h = (resolution_dt - now).total_seconds() / 3600.0
                if h < 4.0:
                    near_resolution_tokens.add(contract.token_id)
                    if contract.no_token_id:
                        near_resolution_tokens.add(contract.no_token_id)
                min_hours = min(min_hours, h)
            if min_hours < float("inf"):
                await self._executor.check_resolution_proximity(
                    min_hours, token_ids=near_resolution_tokens or None,
                )
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
        if self._event_bus:
            self._event_bus.publish("pipeline_complete", summary)
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

    Scans 06Z on target_date through 05:59Z the next day (the approximate
    local calendar day for all US time zones).  Returns the hour where
    the ensemble mean is highest.  Using the mean (not max member) avoids
    a systematic warm bias of 1-3°F.

    Why 06Z, not 00Z?  All US cities are UTC-5 to UTC-8.  Using 00Z as
    the day start includes 00-05Z, which is the PREVIOUS local day's
    evening (e.g. 00Z April 17 = 6PM MDT April 16 for Denver).  Pre-
    frontal warm air from the previous evening would be picked as the
    current day's Tmax — a 20-30°F overestimation during cold-front
    passages.  The 06Z boundary (midnight EST / 10PM PST) safely captures
    all US daytime hours without contamination from the prior day.
    """
    if not forecasts:
        return None

    day_start = datetime.combine(target_date, time(6, 0), tzinfo=timezone.utc)
    day_end = datetime.combine(
        target_date + timedelta(days=1), time(5, 59), tzinfo=timezone.utc
    )

    candidates = [
        f for f in forecasts
        if day_start <= f.valid_time <= day_end
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
    *,
    gfs_weight: float = 0.4,
    ecmwf_weight: float = 0.6,
    hrrr_weight: float = 0.15,
    hrrr_max_disagreement: float = 4.0,
) -> MOSForecast:
    """Create a synthetic MOS forecast from available weather data.

    Uses a *weighted average* of available model Tmax estimates, not
    max().  max() systematically biases warm when models disagree —
    e.g. GFS=68°F vs ECMWF=77°F → max picks 77°F, creating phantom
    edge against a market that correctly prices near 68°F.

    Weights are consistent with ProbabilityEngine (ECMWF 0.6, GFS 0.4).
    HRRR gets low weight (0.15) for same-day forecasts: it resolves the
    diurnal cycle at 3km but overnight HRRR runs often have stale
    boundary conditions, causing systematic cool bias of 2-3°F when the
    HRRR max reflects early-morning temps rather than the true afternoon
    peak.  Additionally, HRRR is excluded entirely when it disagrees
    with the GFS/ECMWF consensus by more than 4°F — a sign that the
    HRRR initialization is too stale to be useful.

    Matches forecast data to the contract's resolution date:
      1. GFS/ECMWF ensemble mean at peak heating hour for target date.
      2. HRRR max (only if target_date is today and within 4°F of NWP consensus).
      3. Hard fallback: 70°F.
    """
    if target_date is None:
        target_date = now.date()

    # Use ensemble hourly data matched to the target date's daytime window
    gfs_day = _pick_ensemble_for_date(snap.gfs_ensemble_all, target_date)
    ecmwf_day = _pick_ensemble_for_date(snap.ecmwf_ensemble_all, target_date)

    # Weighted average of available model Tmax estimates
    weights_and_values: list[tuple[float, float]] = []

    if gfs_day is not None:
        weights_and_values.append((gfs_weight, gfs_day.mean))
    if ecmwf_day is not None:
        weights_and_values.append((ecmwf_weight, ecmwf_day.mean))

    # HRRR: only valid for today (≤18h ahead).  Guard against stale
    # overnight runs: if HRRR max disagrees with the GFS/ECMWF weighted
    # consensus by more than hrrr_max_disagreement, exclude it entirely.
    if snap.hrrr and target_date == now.date():
        hrrr_max = max(h.temp_f for h in snap.hrrr)
        # Compute NWP consensus to check HRRR agreement
        if weights_and_values:
            nwp_total_w = sum(w for w, _ in weights_and_values)
            nwp_consensus = sum(w * v for w, v in weights_and_values) / nwp_total_w
            if abs(hrrr_max - nwp_consensus) <= hrrr_max_disagreement:
                weights_and_values.append((hrrr_weight, hrrr_max))
            else:
                logger.info(
                    "HRRR excluded: max=%.1f disagrees with NWP consensus=%.1f by %.1f°F",
                    hrrr_max, nwp_consensus, abs(hrrr_max - nwp_consensus),
                )
        else:
            weights_and_values.append((hrrr_weight, hrrr_max))

    if weights_and_values:
        total_w = sum(w for w, _ in weights_and_values)
        high = sum(w * v for w, v in weights_and_values) / total_w
    else:
        high = 70.0

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
