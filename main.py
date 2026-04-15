"""Main entry point for the trading bot."""
import asyncio
import logging
import sys

from src.config.settings import Settings
from src.config.stations import get_stations
from src.data.weather_client import OpenMeteoClient, MesonetClient
from src.data.polymarket_client import GammaClient, CLOBClient
from src.orchestrator.data_collector import DataCollector
from src.orchestrator.pipeline import TradingPipeline
from src.orchestrator.scheduler import PipelineScheduler
from src.prediction.probability_engine import ProbabilityEngine
from src.prediction.regime_classifier import RegimeClassifier
from src.prediction.calibration import IsotonicCalibrator, CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.position_sizer import PositionSizer
from src.trading.executor import OrderExecutor
from src.verification.prediction_log import PredictionLog
from src.verification.paper_trader import PaperTrader
from src.verification.resolution_checker import ResolutionChecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tradebot")


async def main():
    settings = Settings()

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
    )
    position_sizer = PositionSizer(
        kelly_fraction=settings.KELLY_FRACTION,
        max_trade_usd=settings.MAX_TRADE_USD,
        max_bankroll_pct=settings.MAX_BANKROLL_PCT,
        max_portfolio_exposure=settings.MAX_PORTFOLIO_EXPOSURE,
    )
    executor = OrderExecutor(clob_client=clob, paper_trading=settings.PAPER_TRADING)
    prediction_log = PredictionLog()
    paper_trader = PaperTrader()
    cusum = CUSUMMonitor(threshold=2.0)

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
    )

    resolution_checker = ResolutionChecker(gamma=gamma, paper_trader=paper_trader)
    scheduler = PipelineScheduler(pipeline=pipeline, resolution_checker=resolution_checker)

    # Set API state
    from src.api.main import set_state
    set_state(prediction_log, paper_trader, scheduler, cusum=cusum)

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
    result = await pipeline.run_cycle()
    logger.info("Initial cycle: %s", result)

    # Start scheduler in background
    await scheduler.start()

    # Start server (blocks until shutdown)
    try:
        await server.serve()
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
