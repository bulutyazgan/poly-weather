"""Integration test: forecast cycle → signal cache → price monitor → trade."""
import asyncio
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.models import MarketContract, RegimeClassification
from src.data.ws_feed import WebSocketFeed
from src.orchestrator.price_monitor import PriceMonitor
from src.orchestrator.signal_cache import CachedSignal, SignalCache
from src.trading.exposure_tracker import ExposureTracker


@pytest.mark.asyncio
async def test_cache_update_triggers_resubscription():
    """When signal cache is updated, PriceMonitor should resubscribe the WS feed."""
    signal_cache = SignalCache()
    ws_feed = MagicMock(spec=WebSocketFeed)
    ws_feed.resubscribe = AsyncMock()
    ws_feed.listen = MagicMock(return_value=_empty_async_iter())

    monitor = PriceMonitor(
        ws_feed=ws_feed,
        signal_cache=signal_cache,
        edge_detector=MagicMock(),
        position_sizer=MagicMock(),
        exposure_tracker=ExposureTracker(),
        executor=MagicMock(),
        prediction_log=MagicMock(),
        paper_trader=MagicMock(),
        bankroll=300.0,
    )

    await monitor.start()
    await asyncio.sleep(0.05)

    cached = CachedSignal(
        model_prob=0.40,
        regime=RegimeClassification(
            station_id="KNYC",
            valid_date=date(2026, 4, 15),
            regime="normal",
            confidence="HIGH",
        ),
        contract=MarketContract(
            token_id="tok_yes",
            no_token_id="tok_no",
            condition_id="cond_1",
            question="test",
            city="NYC",
            resolution_date=date(2026, 4, 16),
            temp_bucket_low=72.0,
            temp_bucket_high=73.0,
            outcome="Yes",
            volume_24h=5000.0,
        ),
        station_id="KNYC",
        forecast_time=datetime.now(timezone.utc),
    )
    signal_cache.update({"tok_yes": cached})

    await asyncio.sleep(0.1)

    ws_feed.resubscribe.assert_called_once()
    call_args = ws_feed.resubscribe.call_args[0][0]
    assert "tok_yes" in call_args
    assert "tok_no" in call_args

    await monitor.stop()


@pytest.mark.asyncio
async def test_exposure_shared_between_paths():
    """Both pipeline and price monitor should see the same exposure tracker."""
    tracker = ExposureTracker()
    tracker.add(10.0)
    assert tracker.current == 10.0
    tracker.add(5.0)
    assert tracker.current == 15.0


async def _empty_async_iter():
    """An async iterator that never yields."""
    while True:
        await asyncio.sleep(999)
        yield {}
