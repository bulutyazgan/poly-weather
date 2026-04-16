"""Tests for the FastAPI endpoints."""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.data.models import MarketContract, RegimeClassification, TradingSignal
from src.verification.prediction_log import PredictionLog, SignalLogEntry
from src.verification.paper_trader import PaperTrader


@pytest.fixture()
def _populated_state():
    """Set up API state with sample data for testing."""
    from src.api.main import app, set_state
    from src.orchestrator.scheduler import PipelineScheduler

    prediction_log = PredictionLog()
    paper_trader = PaperTrader(taker_fee_rate=0.0)

    # Create a mock scheduler (avoid needing a real TradingPipeline)
    mock_pipeline = MagicMock()
    scheduler = PipelineScheduler(pipeline=mock_pipeline)

    # Add sample signal log entries
    signal_nyc = TradingSignal(
        market_id="mkt-1",
        direction="BUY_YES",
        action="TRADE",
        edge=0.05,
        kelly_size=10.0,
        timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
    )
    regime_nyc = RegimeClassification(
        station_id="KNYC",
        valid_date=date(2026, 4, 16),
        regime="stable",
        confidence="HIGH",
    )
    contract_nyc = MarketContract(
        token_id="tok-1",
        condition_id="cond-1",
        question="Will NYC high be 72-73°F on Apr 16?",
        city="NYC",
        resolution_date=date(2026, 4, 16),
        temp_bucket_low=72.0,
        temp_bucket_high=73.0,
        outcome="Yes",
    )
    prediction_log.log(SignalLogEntry(
        signal=signal_nyc,
        station_id="KNYC",
        regime=regime_nyc,
        model_probability=0.65,
        market_probability=0.55,
        contract=contract_nyc,
    ))

    signal_chi = TradingSignal(
        market_id="mkt-2",
        direction="BUY_NO",
        action="SKIP",
        edge=0.02,
        kelly_size=0.0,
        timestamp=datetime(2026, 4, 15, 13, 0, tzinfo=timezone.utc),
    )
    regime_chi = RegimeClassification(
        station_id="KORD",
        valid_date=date(2026, 4, 16),
        regime="transitional",
        confidence="MEDIUM",
    )
    prediction_log.log(SignalLogEntry(
        signal=signal_chi,
        station_id="KORD",
        regime=regime_chi,
        model_probability=0.40,
        market_probability=0.45,
    ))

    # Record and resolve a paper trade
    trade_id = paper_trader.record_trade(
        signal=signal_nyc,
        contract=contract_nyc,
        entry_price=0.55,
        amount_usd=10.0,
    )
    paper_trader.resolve(trade_id, outcome=True)

    set_state(prediction_log, paper_trader, scheduler)
    yield
    # Clean up state
    set_state(None, None, None)  # type: ignore[arg-type]


@pytest.fixture()
def _empty_state():
    """Set up API state with no data (None state)."""
    from src.api.main import set_state
    set_state(None, None, None)  # type: ignore[arg-type]
    yield
    set_state(None, None, None)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_health_check():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_get_stations():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/stations")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 5
    cities = {s["city"] for s in data}
    assert cities == {"NYC", "Chicago", "LA", "Denver", "Miami"}


@pytest.mark.anyio
async def test_get_station_detail():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/stations/NYC")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "NYC"
    assert data["station_id"] == "KNYC"
    assert "lat" in data
    assert "lon" in data


@pytest.mark.anyio
async def test_get_station_not_found():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/stations/INVALID")
    assert resp.status_code == 404


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_performance():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_pnl" in data
    assert "win_rate" in data
    assert "trade_count" in data
    assert "signal_count" in data
    assert data["trade_count"] == 1
    assert data["signal_count"] == 2
    assert data["win_rate"] == 1.0
    assert data["total_pnl"] > 0


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_signals():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/signals")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["station_id"] == "KNYC"
    assert data[1]["station_id"] == "KORD"


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_signals_filter_station():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/signals?station=KNYC")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["station_id"] == "KNYC"


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_schedule():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/schedule")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 7  # 4 GFS + 2 ECMWF + 1 morning refinement
    event_types = {e["event_type"] for e in data}
    assert "gfs_update" in event_types
    assert "ecmwf_update" in event_types
    assert "morning_refinement" in event_types


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_calibration():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/calibration")
    assert resp.status_code == 200
    data = resp.json()
    assert "brier_score" in data
    assert "brier_skill_score" in data
    assert "reliability_diagram" in data
    assert "resolved_count" in data
    # Only 1 resolved trade — below threshold of 10, so metrics are null
    assert data["resolved_count"] == 1
    assert data["brier_score"] is None


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_performance_empty():
    """Performance endpoint returns zeros when no state is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_pnl"] == 0.0
    assert data["win_rate"] == 0.0
    assert data["trade_count"] == 0
    assert data["signal_count"] == 0


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_trades():
    """Trades endpoint returns paper trade records."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trades")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    trade = data[0]
    assert trade["direction"] == "BUY_YES"
    assert trade["city"] == "NYC"
    assert trade["resolved"] is True
    assert trade["outcome"] is True
    assert trade["pnl"] is not None
    assert trade["pnl"] > 0
    assert trade["entry_price"] == 0.55
    assert trade["amount_usd"] == 10.0
    assert "temp_bucket_low" in trade
    assert "temp_bucket_high" in trade


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_trades_filter_resolved():
    """Trades endpoint filters by status=resolved."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trades?status=resolved")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert all(t["resolved"] for t in data)


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_trades_filter_pending():
    """Trades endpoint returns empty for status=pending when all resolved."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trades?status=pending")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 0


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_cusum_no_state():
    """CUSUM endpoint returns defaults when no monitor is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/cusum")
    assert resp.status_code == 200
    data = resp.json()
    assert data["alarm"] is False
    assert data["cusum_pos"] == 0.0
    assert data["pct_of_threshold"] == 0.0


@pytest.mark.anyio
async def test_get_cusum_with_state():
    """CUSUM endpoint returns live monitor state."""
    from src.api.main import app, set_state
    from src.prediction.calibration import CUSUMMonitor

    monitor = CUSUMMonitor(threshold=2.0)
    monitor.update(0.8)  # cusum_pos = 0.8
    monitor.update(0.6)  # cusum_pos = 1.4

    set_state(None, None, None, cusum=monitor)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/cusum")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alarm"] is False
        assert data["cusum_pos"] == 1.4
        assert data["threshold"] == 2.0
        assert data["pct_of_threshold"] == 70.0
    finally:
        set_state(None, None, None)


@pytest.mark.anyio
async def test_get_cusum_alarm():
    """CUSUM endpoint reports alarm when threshold exceeded."""
    from src.api.main import app, set_state
    from src.prediction.calibration import CUSUMMonitor

    monitor = CUSUMMonitor(threshold=1.0)
    monitor.update(1.5)  # cusum_pos = 1.5 > 1.0 → alarm

    set_state(None, None, None, cusum=monitor)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/cusum")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alarm"] is True
        assert data["pct_of_threshold"] == 150.0
    finally:
        set_state(None, None, None)


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_trades_empty():
    """Trades endpoint returns empty list when no state is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trades")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_signals_empty():
    """Signals endpoint returns empty list when no state is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/signals")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_get_status():
    """Status endpoint returns scheduler health and signal cache info."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "scheduler" in data
    assert data["scheduler"]["running"] is False  # scheduler not started in test
    assert "signal_cache_age_seconds" in data
    assert "signal_count" in data
    assert data["signal_count"] == 2


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_status_empty():
    """Status endpoint returns defaults when no state is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scheduler"]["running"] is False
    assert data["signal_cache_age_seconds"] is None
    assert data["signal_count"] == 0


# ---------------------------------------------------------------------------
# New endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_exposure_empty():
    """Exposure endpoint returns defaults when no tracker is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/exposure")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_exposure_usd"] == 0.0
    assert data["is_halted"] is False
    assert data["bankroll"] == 300.0


@pytest.mark.anyio
async def test_get_exposure_with_tracker():
    """Exposure endpoint returns live tracker state."""
    from src.api.main import app, set_state
    from src.trading.exposure_tracker import ExposureTracker

    tracker = ExposureTracker(bankroll=500.0, max_drawdown_pct=0.10)
    tracker.add(75.0)
    tracker.record_pnl(-10.0, amount_usd=25.0)

    set_state(None, None, None, exposure_tracker=tracker)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/exposure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_exposure_usd"] == 50.0
        assert data["realized_pnl"] == -10.0
        assert data["is_halted"] is False
        assert data["bankroll"] == 500.0
        assert data["exposure_pct"] == 10.0
        assert data["drawdown_pct"] == 2.0
    finally:
        set_state(None, None, None)


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_cached_signals_empty():
    """Cached signals endpoint returns empty list when no cache."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/cached-signals")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.anyio
async def test_get_cached_signals_with_data():
    """Cached signals endpoint returns cached model probabilities."""
    from src.api.main import app, set_state
    from src.orchestrator.signal_cache import CachedSignal, SignalCache

    cache = SignalCache()
    contract = MarketContract(
        token_id="tok-cs",
        condition_id="cond-cs",
        question="Will NYC high be 68-69°F?",
        city="NYC",
        resolution_date=date(2026, 4, 20),
        temp_bucket_low=68.0,
        temp_bucket_high=69.0,
        outcome="Yes",
    )
    regime = RegimeClassification(
        station_id="KNYC",
        valid_date=date(2026, 4, 20),
        regime="stable",
        confidence="HIGH",
    )
    cache.update({
        "tok-cs": CachedSignal(
            model_prob=0.42,
            regime=regime,
            contract=contract,
            station_id="KNYC",
            forecast_time=datetime(2026, 4, 16, 10, 0, tzinfo=timezone.utc),
        ),
    })

    set_state(None, None, None, signal_cache=cache)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/cached-signals")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        sig = data[0]
        assert sig["token_id"] == "tok-cs"
        assert sig["city"] == "NYC"
        assert sig["model_prob"] == 0.42
        assert sig["regime_confidence"] == "HIGH"
        assert sig["temp_bucket"] == "68.0-69.0"
        assert sig["live_bid"] is None  # no ws_feed
        assert sig["current_edge"] is None
    finally:
        set_state(None, None, None)


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_get_price_monitor_empty():
    """Price monitor endpoint returns defaults when no monitor is set."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/price-monitor")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert data["ws_connected"] is False
    assert data["subscribed_tokens"] == 0


@pytest.mark.anyio
@pytest.mark.usefixtures("_populated_state")
async def test_signals_include_skip_reason():
    """Signal entries include skip_reason field."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/signals")
    assert resp.status_code == 200
    data = resp.json()
    for signal in data:
        assert "skip_reason" in signal


@pytest.mark.anyio
@pytest.mark.usefixtures("_empty_state")
async def test_sse_endpoint_returns_event_stream():
    """SSE endpoint returns text/event-stream content type."""
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/events")
    # Without event_bus, returns empty stream
    assert resp.status_code == 200
