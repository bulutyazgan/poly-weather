"""Main entry point for the trading bot."""
import asyncio
import logging
import sys
from datetime import datetime, timezone

from src.config.settings import Settings
from src.config.stations import get_stations
from src.data.weather_client import OpenMeteoClient, MesonetClient
from src.data.polymarket_client import GammaClient, CLOBClient
from src.data.ws_feed import WebSocketFeed
from src.orchestrator.data_collector import DataCollector
from src.orchestrator.pipeline import TradingPipeline
from src.orchestrator.price_monitor import PriceMonitor
from src.orchestrator.scheduler import PipelineScheduler
from src.orchestrator.signal_cache import SignalCache
from src.prediction.probability_engine import ProbabilityEngine
from src.prediction.regime_classifier import RegimeClassifier
from src.prediction.calibration import IsotonicCalibrator, CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.exposure_tracker import ExposureTracker
from src.trading.position_sizer import PositionSizer
from src.trading.executor import OrderExecutor
from src.verification.prediction_log import PredictionLog
from src.verification.paper_trader import PaperTrader
from src.verification.resolution_checker import ResolutionChecker
from src.api.event_bus import EventBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tradebot")


async def main():
    settings = Settings()

    event_bus = EventBus()

    logger.info("Starting TradeBot in %s mode", "PAPER" if settings.PAPER_TRADING else "LIVE")
    logger.info("Monitoring %d stations", len(get_stations()))

    # Initialize clients
    weather = OpenMeteoClient()
    mesonet = MesonetClient()
    gamma = GammaClient()
    clob = CLOBClient(
        api_url=settings.POLYMARKET_API_URL,
        private_key=settings.POLYGON_WALLET_PRIVATE_KEY.get_secret_value(),
        paper_trading=settings.PAPER_TRADING,
    )

    # Initialize components
    collector = DataCollector(weather=weather, mesonet=mesonet, gamma=gamma, clob=clob)
    prob_engine = ProbabilityEngine()
    regime_classifier = RegimeClassifier()
    edge_detector = EdgeDetector(
        high_threshold=settings.HIGH_REGIME_EDGE_THRESHOLD,
        medium_threshold=settings.MEDIUM_REGIME_EDGE_THRESHOLD,
        min_volume=settings.MIN_MARKET_VOLUME,
        min_hours=settings.MIN_HOURS_TO_RESOLUTION,
        max_market_certainty=settings.MAX_MARKET_CERTAINTY,
        max_edge=settings.MAX_EDGE,
        taker_fee_rate=settings.TAKER_FEE_RATE,
        max_spread=settings.MAX_SPREAD,
    )
    position_sizer = PositionSizer(
        kelly_fraction=settings.KELLY_FRACTION,
        max_trade_usd=settings.MAX_TRADE_USD,
        max_bankroll_pct=settings.MAX_BANKROLL_PCT,
        max_portfolio_exposure=settings.MAX_PORTFOLIO_EXPOSURE,
        min_trade_usd=settings.MIN_TRADE_USD,
    )
    executor = OrderExecutor(
        clob_client=clob,
        paper_trading=settings.PAPER_TRADING,
        order_ttl_seconds=settings.ORDER_TTL_SECONDS,
    )
    prediction_log = PredictionLog()
    paper_trader = PaperTrader(taker_fee_rate=settings.TAKER_FEE_RATE)
    cusum = CUSUMMonitor(threshold=2.0, drift=0.05)
    signal_cache = SignalCache()
    exposure_tracker = ExposureTracker(
        bankroll=settings.BANKROLL,
        max_drawdown_pct=settings.MAX_DRAWDOWN_PCT,
        event_bus=event_bus,
    )

    # Build pipeline
    pipeline = TradingPipeline(
        collector=collector,
        prob_engine=prob_engine,
        regime_classifier=regime_classifier,
        edge_detector=edge_detector,
        position_sizer=position_sizer,
        executor=executor,
        prediction_log=prediction_log,
        paper_trader=paper_trader,
        cusum=cusum,
        signal_cache=signal_cache,
        exposure_tracker=exposure_tracker,
        event_bus=event_bus,
    )

    resolution_checker = ResolutionChecker(
        gamma=gamma, paper_trader=paper_trader, exposure_tracker=exposure_tracker,
        event_bus=event_bus,
    )
    scheduler = PipelineScheduler(pipeline=pipeline, resolution_checker=resolution_checker)

    # Start API server in background
    import uvicorn
    config = uvicorn.Config(
        app="src.api.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
    server = uvicorn.Server(config)

    logger.info("Starting API server on http://127.0.0.1:8000")
    logger.info("Dashboard: http://localhost:8000/health")
    logger.info("Schedule: %d events registered", len(scheduler.get_scheduled_events()))

    # Run initial data collection cycle
    logger.info("Running initial data collection...")
    result = await pipeline.run_cycle(bankroll=settings.BANKROLL)
    logger.info("Initial cycle: %s", result)

    # Seed scheduler status so /api/status shows the initial cycle
    scheduler._last_run_time = datetime.now(tz=timezone.utc)
    scheduler._last_run_result = result

    # Start continuous price monitor
    ws_feed = None
    price_monitor = None
    if settings.PRICE_MONITOR_ENABLED:
        ws_feed = WebSocketFeed(event_bus=event_bus)

        # Bootstrap WS feed with tokens from initial cycle so the
        # PriceMonitor doesn't wait for the next scheduler event.
        # (signal_cache.updated was already set() before anyone was
        # listening — _watch_resubscription would block until next cycle.)
        initial_signals = signal_cache.get_all()
        token_ids = list(initial_signals.keys())
        for cached in initial_signals.values():
            if cached.contract.no_token_id:
                token_ids.append(cached.contract.no_token_id)
        if token_ids:
            await ws_feed.subscribe(token_ids)
            logger.info("WS feed bootstrapped with %d tokens from initial cycle", len(token_ids))

        price_monitor = PriceMonitor(
            ws_feed=ws_feed,
            signal_cache=signal_cache,
            edge_detector=edge_detector,
            position_sizer=position_sizer,
            exposure_tracker=exposure_tracker,
            executor=executor,
            prediction_log=prediction_log,
            paper_trader=paper_trader,
            cusum=cusum,
            event_bus=event_bus,
            debounce_seconds=settings.PRICE_MONITOR_DEBOUNCE_S,
            cooldown_seconds=settings.PRICE_MONITOR_COOLDOWN_S,
            max_forecast_age_s=settings.PRICE_MONITOR_MAX_FORECAST_AGE_S,
            max_price_age_s=settings.PRICE_MONITOR_MAX_PRICE_AGE_S,
            bankroll=settings.BANKROLL,
        )
        await price_monitor.start()
        logger.info("Price monitor started (debounce=%.0fs, cooldown=%.0fs)",
                     settings.PRICE_MONITOR_DEBOUNCE_S,
                     settings.PRICE_MONITOR_COOLDOWN_S)

    # Set API state (after all components are initialized)
    from src.api.main import set_state
    set_state(
        prediction_log, paper_trader, scheduler,
        executor=executor, cusum=cusum, signal_cache=signal_cache,
        exposure_tracker=exposure_tracker,
        ws_feed=ws_feed,
        price_monitor=price_monitor,
        event_bus=event_bus,
    )

    # Start scheduler in background
    await scheduler.start(bankroll=settings.BANKROLL)

    # Start server (blocks until shutdown)
    try:
        await server.serve()
    finally:
        if price_monitor is not None:
            await price_monitor.stop()
        if ws_feed is not None:
            await ws_feed.close()
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
