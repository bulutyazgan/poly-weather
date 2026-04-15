"""Continuous price monitor — evaluates edges against cached model probs.

Subscribes to the WebSocket feed and checks for tradeable edges whenever
market prices update, using model probabilities cached by the forecast
pipeline. Includes debounce (edge must persist) and cooldown (per-contract
post-trade lockout) to prevent overtrading.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone

from src.data.models import TradingSignal
from src.data.ws_feed import WebSocketFeed
from src.orchestrator.signal_cache import SignalCache
from src.prediction.calibration import CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.exposure_tracker import ExposureTracker
from src.trading.position_sizer import PositionSizer
from src.trading.executor import OrderExecutor
from src.verification.prediction_log import PredictionLog, SignalLogEntry
from src.verification.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class PriceMonitor:
    """WebSocket-driven price monitor for continuous edge detection."""

    def __init__(
        self,
        ws_feed: WebSocketFeed,
        signal_cache: SignalCache,
        edge_detector: EdgeDetector,
        position_sizer: PositionSizer,
        exposure_tracker: ExposureTracker,
        executor: OrderExecutor,
        prediction_log: PredictionLog,
        paper_trader: PaperTrader,
        cusum: CUSUMMonitor | None = None,
        debounce_seconds: float = 10.0,
        cooldown_seconds: float = 900.0,
        max_forecast_age_s: float = 28800.0,
        bankroll: float = 300.0,
    ) -> None:
        self._ws_feed = ws_feed
        self._signal_cache = signal_cache
        self._edge_detector = edge_detector
        self._position_sizer = position_sizer
        self._exposure_tracker = exposure_tracker
        self._executor = executor
        self._prediction_log = prediction_log
        self._paper_trader = paper_trader
        self._cusum = cusum
        self._debounce_seconds = debounce_seconds
        self._cooldown_seconds = cooldown_seconds
        self._max_forecast_age_s = max_forecast_age_s
        self._bankroll = bankroll

        self._pending_edges: dict[str, datetime] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start the price monitor as background tasks."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._watch_prices()),
            asyncio.create_task(self._watch_resubscription()),
        ]
        logger.info("PriceMonitor started")

    async def stop(self) -> None:
        """Stop the price monitor."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks = []
        logger.info("PriceMonitor stopped")

    async def _watch_prices(self) -> None:
        """Main loop: listen for WS updates and evaluate edges."""
        async for update in self._ws_feed.listen():
            if not self._running:
                break
            token_id = update.get("asset_id", "")
            if token_id:
                await self._handle_price_update(token_id)

    async def _watch_resubscription(self) -> None:
        """Resubscribe to WS when the signal cache is refreshed."""
        while self._running:
            await self._signal_cache.updated.wait()
            self._signal_cache.updated.clear()
            all_signals = self._signal_cache.get_all()
            token_ids = list(all_signals.keys())
            for cached in all_signals.values():
                if cached.contract.no_token_id:
                    token_ids.append(cached.contract.no_token_id)
            logger.info(
                "Signal cache updated — resubscribing to %d tokens", len(token_ids)
            )
            await self._ws_feed.resubscribe(token_ids)

    async def _handle_price_update(self, token_id: str):
        """Evaluate a single token's edge and trade if conditions are met.

        Returns the trade record if a trade was executed, else None.
        """
        now = datetime.now(tz=timezone.utc)

        cached = self._signal_cache.get(token_id)
        if cached is None:
            return None

        if self._signal_cache.forecast_age_seconds > self._max_forecast_age_s:
            return None

        live_price = self._ws_feed.get_latest_price(token_id)
        if live_price is None:
            return None

        # Compute hours to resolution live (not cached)
        resolution_dt = cached.contract.end_date_utc
        if resolution_dt is None:
            resolution_dt = datetime.combine(
                cached.contract.resolution_date,
                time(23, 59, 59),
                tzinfo=timezone.utc,
            )
        hours_to_resolution = max(
            0.0, (resolution_dt - now).total_seconds() / 3600.0
        )

        # Use contract's volume_24h, NOT WS price's (WS returns 0.0)
        signal = self._edge_detector.evaluate(
            model_prob=cached.model_prob,
            market_prob=live_price.mid,
            regime=cached.regime,
            volume_24h=cached.contract.volume_24h,
            hours_to_resolution=hours_to_resolution,
            market_id=token_id,
            market_bid=live_price.bid,
            market_ask=live_price.ask,
        )

        if signal.action != "TRADE":
            self._pending_edges.pop(token_id, None)
            return None

        # Debounce: edge must persist for debounce_seconds
        if self._debounce_seconds > 0:
            if token_id not in self._pending_edges:
                self._pending_edges[token_id] = now
                return None
            elapsed = (now - self._pending_edges[token_id]).total_seconds()
            if elapsed < self._debounce_seconds:
                return None

        # Cooldown: skip if recently traded
        if token_id in self._cooldowns and now < self._cooldowns[token_id]:
            return None

        # CUSUM check (alarm only reset by scheduled pipeline)
        if self._cusum is not None and self._cusum.alarm:
            return None

        # Size position
        size_usd = self._position_sizer.compute(
            edge=signal.edge,
            market_prob=live_price.mid,
            bankroll=self._bankroll,
            current_exposure=self._exposure_tracker.current,
            ensemble_spread_pctile=cached.regime.ensemble_spread_percentile,
            direction=signal.direction,
            active_station_count=1,
        )

        if size_usd <= 0:
            return None

        # Build sized signal
        sized_signal = TradingSignal(
            market_id=signal.market_id,
            direction=signal.direction,
            action=signal.action,
            edge=signal.edge,
            kelly_size=size_usd,
            timestamp=signal.timestamp,
        )

        # BUY_NO token routing
        if signal.direction == "BUY_NO" and cached.contract.no_token_id:
            no_price = self._ws_feed.get_latest_price(cached.contract.no_token_id)
            if no_price is None:
                return None
            exec_token = cached.contract.no_token_id
            exec_price = no_price
        else:
            exec_token = token_id
            exec_price = live_price

        trade_record = await self._executor.execute(
            sized_signal, exec_token, exec_price
        )

        if trade_record is not None:
            self._exposure_tracker.add(size_usd)

            self._paper_trader.record_trade(
                signal=sized_signal,
                contract=cached.contract,
                entry_price=live_price.mid,
                amount_usd=size_usd,
                model_probability=cached.model_prob,
            )

            log_entry = SignalLogEntry(
                signal=sized_signal,
                station_id=cached.station_id,
                regime=cached.regime,
                model_probability=cached.model_prob,
                market_probability=live_price.mid,
                contract=cached.contract,
            )
            self._prediction_log.log(log_entry)

            self._cooldowns[token_id] = now + timedelta(
                seconds=self._cooldown_seconds
            )
            self._pending_edges.pop(token_id, None)

            logger.info(
                "Price monitor trade: %s %s edge=%.3f size=$%.2f",
                signal.direction,
                token_id,
                signal.edge,
                size_usd,
            )

        return trade_record
