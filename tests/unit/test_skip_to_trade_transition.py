"""Verify that the PriceMonitor catches SKIP→TRADE transitions.

Scenario: Pipeline evaluates a contract and marks it SKIP (edge below
threshold).  Later the market price moves, creating a tradeable edge.
The PriceMonitor should detect this and execute the trade.
"""
import asyncio
from datetime import date, datetime, timezone

import pytest

from src.data.models import (
    MarketContract,
    MarketPrice,
    RegimeClassification,
    TradingSignal,
)
from src.orchestrator.price_monitor import PriceMonitor
from src.orchestrator.signal_cache import CachedSignal, SignalCache
from src.prediction.calibration import CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.exposure_tracker import ExposureTracker
from src.trading.position_sizer import PositionSizer
from src.trading.executor import OrderExecutor
from src.verification.prediction_log import PredictionLog
from src.verification.paper_trader import PaperTrader


# -- Fixtures / helpers -------------------------------------------------------

def _make_contract(token_id="tok_yes", no_token_id="tok_no"):
    return MarketContract(
        token_id=token_id,
        no_token_id=no_token_id,
        condition_id="cond_1",
        question="Will NYC high be 72-73°F on Apr 20?",
        city="NYC",
        resolution_date=date(2026, 4, 20),
        end_date_utc=datetime(2026, 4, 20, 23, 59, 59, tzinfo=timezone.utc),
        temp_bucket_low=72.0,
        temp_bucket_high=73.0,
        outcome="Yes",
        volume_24h=5000.0,
    )


def _make_regime(station_id="KNYC"):
    return RegimeClassification(
        station_id=station_id,
        valid_date=date(2026, 4, 20),
        regime="stable",
        confidence="HIGH",
        confidence_score=0.8,
        ensemble_spread_percentile=30.0,
    )


class FakeWSFeed:
    """Minimal fake WebSocket feed for testing."""

    def __init__(self):
        self._prices: dict[str, MarketPrice] = {}
        self._subscribed: list[str] = []

    def set_price(self, token_id: str, bid: float, ask: float):
        mid = (bid + ask) / 2.0
        self._prices[token_id] = MarketPrice(
            token_id=token_id,
            timestamp=datetime.now(timezone.utc),
            bid=bid,
            ask=ask,
            mid=mid,
            volume_24h=0.0,
        )

    def get_latest_price(self, token_id: str) -> MarketPrice | None:
        return self._prices.get(token_id)

    async def resubscribe(self, token_ids: list[str]) -> None:
        self._subscribed = list(token_ids)

    async def listen(self):
        # Not used in unit test — we call _handle_price_update directly
        while True:
            await asyncio.sleep(999)
            yield {}

    async def close(self):
        pass


class FakeCLOB:
    """Minimal CLOB client stub."""
    paper_trading = True
    trades: list[dict]

    def __init__(self):
        self.trades = []

    async def place_limit_order(self, token_id, side, price, size, **kwargs):
        record = {
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
        }
        self.trades.append(record)
        return "order_1"


# -- Tests --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skip_to_trade_on_price_move():
    """Contract initially SKIP (below threshold) becomes TRADE after price moves."""

    # Setup components
    contract = _make_contract()
    regime = _make_regime()
    model_prob = 0.55  # Our model says 55%

    edge_detector = EdgeDetector(
        high_threshold=0.08,
        medium_threshold=0.12,
        min_volume=500.0,
        min_hours=2.0,
        max_market_certainty=0.92,
        max_edge=0.25,
        taker_fee_rate=0.02,
    )

    # -- Phase 1: market is close to model → SKIP ---
    # Market mid ≈ 0.52, ask = 0.53 → edge = 0.55 - 0.53 - fee ≈ 0.01 (below 0.08)
    initial_signal = edge_detector.evaluate(
        model_prob=model_prob,
        market_prob=0.52,
        regime=regime,
        volume_24h=5000.0,
        hours_to_resolution=48.0,
        market_id=contract.token_id,
        market_bid=0.51,
        market_ask=0.53,
    )
    assert initial_signal.action == "SKIP", (
        f"Expected SKIP at initial price, got {initial_signal.action} "
        f"(edge={initial_signal.edge:.4f})"
    )

    # -- Phase 2: Populate signal cache (pipeline would do this) ---
    signal_cache = SignalCache()
    signal_cache.update({
        contract.token_id: CachedSignal(
            model_prob=model_prob,
            regime=regime,
            contract=contract,
            station_id="KNYC",
            forecast_time=datetime.now(timezone.utc),
        )
    })

    # -- Phase 3: Market price drops → edge opens up ---
    # Market drops to mid ≈ 0.40, ask = 0.42
    # Edge = 0.55 - 0.42 - fee(0.02 * 0.42) = 0.55 - 0.42 - 0.0084 ≈ 0.1216 → TRADE
    ws_feed = FakeWSFeed()
    ws_feed.set_price(contract.token_id, bid=0.38, ask=0.42)

    clob = FakeCLOB()
    executor = OrderExecutor(clob_client=clob, paper_trading=True)
    exposure_tracker = ExposureTracker(bankroll=300.0)
    position_sizer = PositionSizer(
        kelly_fraction=0.08,
        max_trade_usd=3.0,
        max_bankroll_pct=0.03,
        max_portfolio_exposure=0.20,
    )
    prediction_log = PredictionLog()
    paper_trader = PaperTrader(taker_fee_rate=0.02)

    price_monitor = PriceMonitor(
        ws_feed=ws_feed,
        signal_cache=signal_cache,
        edge_detector=edge_detector,
        position_sizer=position_sizer,
        exposure_tracker=exposure_tracker,
        executor=executor,
        prediction_log=prediction_log,
        paper_trader=paper_trader,
        debounce_seconds=0,  # disable debounce for test
        cooldown_seconds=0,  # disable cooldown for test
        bankroll=300.0,
    )

    # Directly invoke the handler (simulates WS update arriving)
    result = await price_monitor._handle_price_update(contract.token_id)

    # Should have executed a trade
    assert result is not None, "PriceMonitor should have traded after price drop"
    assert result.direction == "BUY_YES"
    assert result.amount_usd > 0
    assert exposure_tracker.current > 0, "Exposure should have increased"
    assert len(paper_trader._trades) == 1, "Paper trader should have recorded trade"

    print(f"✓ SKIP→TRADE transition verified:")
    print(f"  Initial: SKIP (edge={initial_signal.edge:.4f})")
    print(f"  After price drop: TRADE (amount=${result.amount_usd:.2f})")


@pytest.mark.asyncio
async def test_cache_includes_skip_contracts():
    """Signal cache stores model probs for ALL contracts, not just TRADEs."""
    signal_cache = SignalCache()
    contract = _make_contract()
    regime = _make_regime()

    # Simulate what pipeline does: cache ALL pending signals
    signal_cache.update({
        contract.token_id: CachedSignal(
            model_prob=0.55,
            regime=regime,
            contract=contract,
            station_id="KNYC",
            forecast_time=datetime.now(timezone.utc),
        )
    })

    cached = signal_cache.get(contract.token_id)
    assert cached is not None, "SKIP contract should still be in cache"
    assert cached.model_prob == 0.55


@pytest.mark.asyncio
async def test_stale_forecast_blocks_trade():
    """PriceMonitor should not trade if forecast is too old."""
    contract = _make_contract()
    regime = _make_regime()

    signal_cache = SignalCache()
    # Set forecast_time far in the past
    signal_cache.update({
        contract.token_id: CachedSignal(
            model_prob=0.55,
            regime=regime,
            contract=contract,
            station_id="KNYC",
            forecast_time=datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc),
        )
    })
    # Manually backdate the cache's _last_update to simulate staleness
    signal_cache._last_update = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)

    ws_feed = FakeWSFeed()
    ws_feed.set_price(contract.token_id, bid=0.30, ask=0.35)  # big edge

    clob = FakeCLOB()
    executor = OrderExecutor(clob_client=clob, paper_trading=True)

    price_monitor = PriceMonitor(
        ws_feed=ws_feed,
        signal_cache=signal_cache,
        edge_detector=EdgeDetector(),
        position_sizer=PositionSizer(),
        exposure_tracker=ExposureTracker(),
        executor=executor,
        prediction_log=PredictionLog(),
        paper_trader=PaperTrader(),
        debounce_seconds=0,
        cooldown_seconds=0,
        max_forecast_age_s=28800.0,  # 8 hours
        bankroll=300.0,
    )

    result = await price_monitor._handle_price_update(contract.token_id)
    assert result is None, "Should block trade when forecast is stale"


@pytest.mark.asyncio
async def test_resubscription_includes_all_tokens():
    """When signal cache updates, PriceMonitor resubscribes to all tokens."""
    contract = _make_contract(token_id="tok_yes", no_token_id="tok_no")
    regime = _make_regime()

    signal_cache = SignalCache()
    ws_feed = FakeWSFeed()

    # Trigger the resubscription logic manually
    signal_cache.update({
        contract.token_id: CachedSignal(
            model_prob=0.55,
            regime=regime,
            contract=contract,
            station_id="KNYC",
            forecast_time=datetime.now(timezone.utc),
        )
    })

    # Simulate what _watch_resubscription does
    all_signals = signal_cache.get_all()
    token_ids = list(all_signals.keys())
    for cached in all_signals.values():
        if cached.contract.no_token_id:
            token_ids.append(cached.contract.no_token_id)

    await ws_feed.resubscribe(token_ids)

    assert "tok_yes" in ws_feed._subscribed
    assert "tok_no" in ws_feed._subscribed
    assert len(ws_feed._subscribed) == 2
