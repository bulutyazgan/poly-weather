# Continuous Price Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WebSocket-driven price monitor that continuously evaluates edges against cached model probabilities between NWP forecast cycles, catching market dislocations within seconds instead of hours.

**Architecture:** Three new components (SignalCache, ExposureTracker, PriceMonitor) wire into the existing pipeline. The forecast scheduler writes per-contract model probs into a SignalCache after each cycle. The PriceMonitor reads from that cache and compares against live WebSocket prices, trading when edges persist past a debounce window. A shared ExposureTracker prevents over-allocation between the two paths.

**Tech Stack:** Python 3.12, asyncio, pydantic, pytest, websockets

**Spec:** `docs/superpowers/specs/2026-04-15-continuous-price-monitor-design.md`

---

### Task 1: ExposureTracker

**Files:**
- Create: `src/trading/exposure_tracker.py`
- Test: `tests/unit/test_exposure_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_exposure_tracker.py
"""Tests for ExposureTracker."""
from src.trading.exposure_tracker import ExposureTracker


class TestExposureTracker:
    def test_initial_exposure_defaults_to_zero(self):
        tracker = ExposureTracker()
        assert tracker.current == 0.0

    def test_initial_exposure_custom(self):
        tracker = ExposureTracker(initial=50.0)
        assert tracker.current == 50.0

    def test_add_increases_exposure(self):
        tracker = ExposureTracker()
        tracker.add(10.0)
        tracker.add(5.0)
        assert tracker.current == 15.0

    def test_reset_sets_to_value(self):
        tracker = ExposureTracker(initial=100.0)
        tracker.reset(25.0)
        assert tracker.current == 25.0

    def test_add_after_reset(self):
        tracker = ExposureTracker(initial=100.0)
        tracker.reset(0.0)
        tracker.add(7.0)
        assert tracker.current == 7.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_exposure_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.trading.exposure_tracker'`

- [ ] **Step 3: Write the implementation**

```python
# src/trading/exposure_tracker.py
"""Shared portfolio exposure tracking for concurrent trading paths."""
from __future__ import annotations


class ExposureTracker:
    """Running tally of portfolio exposure, shared between pipeline and price monitor.

    Safe for asyncio (single-threaded event loop). Both trading paths read
    ``current`` before sizing and call ``add()`` after successful execution.
    """

    def __init__(self, initial: float = 0.0) -> None:
        self._exposure = initial

    @property
    def current(self) -> float:
        """Current total exposure in USD."""
        return self._exposure

    def add(self, amount: float) -> None:
        """Increment exposure after a trade is executed."""
        self._exposure += amount

    def reset(self, value: float) -> None:
        """Reset to a known value (e.g., after reconciliation)."""
        self._exposure = value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_exposure_tracker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading/exposure_tracker.py tests/unit/test_exposure_tracker.py
git commit -m "feat: add ExposureTracker for shared portfolio exposure"
```

---

### Task 2: SignalCache

**Files:**
- Create: `src/orchestrator/signal_cache.py`
- Test: `tests/unit/test_signal_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_signal_cache.py
"""Tests for SignalCache."""
import asyncio
from datetime import date, datetime, timezone

from src.data.models import MarketContract, RegimeClassification
from src.orchestrator.signal_cache import CachedSignal, SignalCache


def _make_cached_signal(
    token_id: str = "token_abc",
    model_prob: float = 0.40,
    station_id: str = "KNYC",
) -> CachedSignal:
    return CachedSignal(
        model_prob=model_prob,
        regime=RegimeClassification(
            station_id=station_id,
            valid_date=date(2026, 4, 15),
            regime="normal",
            confidence="HIGH",
            ensemble_spread_percentile=30.0,
        ),
        contract=MarketContract(
            token_id=token_id,
            no_token_id=f"no_{token_id}",
            condition_id="cond_1",
            question="Will NYC high be 72-73°F?",
            city="NYC",
            resolution_date=date(2026, 4, 16),
            end_date_utc=datetime(2026, 4, 16, 23, 59, 59, tzinfo=timezone.utc),
            temp_bucket_low=72.0,
            temp_bucket_high=73.0,
            outcome="Yes",
            volume_24h=5000.0,
        ),
        station_id=station_id,
        forecast_time=datetime(2026, 4, 15, 10, 30, tzinfo=timezone.utc),
    )


class TestSignalCache:
    def test_empty_cache_returns_none(self):
        cache = SignalCache()
        assert cache.get("nonexistent") is None

    def test_empty_cache_get_all_empty(self):
        cache = SignalCache()
        assert cache.get_all() == {}

    def test_update_and_get(self):
        cache = SignalCache()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        assert cache.get("tok_1") is sig

    def test_update_replaces_entirely(self):
        cache = SignalCache()
        sig1 = _make_cached_signal("tok_1")
        sig2 = _make_cached_signal("tok_2")
        cache.update({"tok_1": sig1})
        cache.update({"tok_2": sig2})
        # tok_1 should be gone — full replacement, not merge
        assert cache.get("tok_1") is None
        assert cache.get("tok_2") is sig2

    def test_get_all_returns_copy(self):
        cache = SignalCache()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        all_signals = cache.get_all()
        all_signals["tok_99"] = sig  # mutate the returned dict
        assert cache.get("tok_99") is None  # original unaffected

    def test_forecast_age_seconds(self):
        cache = SignalCache()
        sig = _make_cached_signal()
        cache.update({"tok_1": sig})
        age = cache.forecast_age_seconds
        # Should be very small — just updated
        assert 0.0 <= age < 2.0

    def test_forecast_age_before_any_update(self):
        cache = SignalCache()
        # No updates yet — age should be infinite
        assert cache.forecast_age_seconds == float("inf")

    def test_updated_event_is_set_on_update(self):
        cache = SignalCache()
        assert not cache.updated.is_set()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        assert cache.updated.is_set()

    def test_updated_event_can_be_cleared_and_reset(self):
        cache = SignalCache()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        cache.updated.clear()
        assert not cache.updated.is_set()
        cache.update({"tok_2": _make_cached_signal("tok_2")})
        assert cache.updated.is_set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_signal_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.orchestrator.signal_cache'`

- [ ] **Step 3: Write the implementation**

```python
# src/orchestrator/signal_cache.py
"""In-memory cache of per-contract model probabilities.

Bridges the forecast pipeline (writes) and the price monitor (reads).
Updated after each NWP forecast cycle with a full replacement.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from src.data.models import MarketContract, RegimeClassification


@dataclass
class CachedSignal:
    """Cached model output for a single market contract."""

    model_prob: float
    regime: RegimeClassification
    contract: MarketContract
    station_id: str
    forecast_time: datetime


class SignalCache:
    """Thread-safe (asyncio-safe) cache of per-contract model probabilities.

    The ``updated`` event is set on every ``update()`` call so that watchers
    (e.g., PriceMonitor) can react to new forecasts without polling.
    """

    def __init__(self) -> None:
        self._signals: dict[str, CachedSignal] = {}
        self._last_update: datetime | None = None
        self.updated: asyncio.Event = asyncio.Event()

    def update(self, signals: dict[str, CachedSignal]) -> None:
        """Full replacement of cache. Sets the ``updated`` event."""
        self._signals = dict(signals)
        self._last_update = datetime.now(tz=timezone.utc)
        self.updated.set()

    def get(self, token_id: str) -> CachedSignal | None:
        """Return cached signal for a token, or None."""
        return self._signals.get(token_id)

    def get_all(self) -> dict[str, CachedSignal]:
        """Return a shallow copy of all cached signals."""
        return dict(self._signals)

    @property
    def forecast_age_seconds(self) -> float:
        """Seconds since the last ``update()`` call. Inf if never updated."""
        if self._last_update is None:
            return float("inf")
        delta = datetime.now(tz=timezone.utc) - self._last_update
        return delta.total_seconds()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_signal_cache.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/signal_cache.py tests/unit/test_signal_cache.py
git commit -m "feat: add SignalCache for bridging forecasts to price monitor"
```

---

### Task 3: WebSocketFeed `resubscribe()` method

**Files:**
- Modify: `src/data/ws_feed.py`
- Test: `tests/unit/test_ws_feed.py` (existing)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_ws_feed.py`:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.data.ws_feed import WebSocketFeed


class TestResubscribe:
    @pytest.mark.asyncio
    async def test_resubscribe_updates_token_ids(self):
        feed = WebSocketFeed()
        feed._token_ids = ["old_token"]
        # Mock the reconnect task so it doesn't actually connect
        feed._task = asyncio.create_task(asyncio.sleep(999))
        await feed.resubscribe(["new_token_1", "new_token_2"])
        assert feed._token_ids == ["new_token_1", "new_token_2"]
        # Clean up
        await feed.close()

    @pytest.mark.asyncio
    async def test_resubscribe_cancels_old_task(self):
        feed = WebSocketFeed()
        old_task = asyncio.create_task(asyncio.sleep(999))
        feed._task = old_task
        # Patch _run_with_reconnect so it doesn't actually connect
        with patch.object(feed, "_run_with_reconnect", new_callable=AsyncMock):
            await feed.resubscribe(["tok_1"])
        assert old_task.cancelled() or old_task.done()
        await feed.close()

    @pytest.mark.asyncio
    async def test_resubscribe_starts_new_task(self):
        feed = WebSocketFeed()
        assert feed._task is None
        with patch.object(feed, "_run_with_reconnect", new_callable=AsyncMock):
            await feed.resubscribe(["tok_1"])
        assert feed._task is not None
        await feed.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_ws_feed.py::TestResubscribe -v`
Expected: FAIL with `AttributeError: 'WebSocketFeed' object has no attribute 'resubscribe'`

- [ ] **Step 3: Add `resubscribe()` to WebSocketFeed**

Add this method to the `WebSocketFeed` class in `src/data/ws_feed.py`, after the `close()` method (after line 87):

```python
    async def resubscribe(self, token_ids: list[str]) -> None:
        """Update subscriptions by restarting the connection.

        Cancels the current connection task and starts a new one with the
        updated token list.  This is the simplest correct approach given
        that ``_send_subscription`` only runs at connection time.
        """
        self._token_ids = list(token_ids)
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = asyncio.create_task(self._run_with_reconnect())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_ws_feed.py -v`
Expected: All tests PASS (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/data/ws_feed.py tests/unit/test_ws_feed.py
git commit -m "feat: add resubscribe() to WebSocketFeed"
```

---

### Task 4: Settings — add price monitor config

**Files:**
- Modify: `src/config/settings.py`

- [ ] **Step 1: Add the three new settings**

Add these fields to the `Settings` class in `src/config/settings.py`, after the `PAPER_TRADING` field (line 26):

```python
    # Price monitor
    PRICE_MONITOR_ENABLED: bool = True
    PRICE_MONITOR_DEBOUNCE_S: float = 10.0
    PRICE_MONITOR_COOLDOWN_S: float = 900.0  # 15 min per contract
    PRICE_MONITOR_MAX_FORECAST_AGE_S: float = 28800.0  # 8 hours
    BANKROLL: float = 300.0  # total bankroll in USD
```

- [ ] **Step 2: Verify settings load**

Run: `python -c "from src.config.settings import Settings; s = Settings(); print(s.PRICE_MONITOR_DEBOUNCE_S)"`
Expected: `10.0`

- [ ] **Step 3: Commit**

```bash
git add src/config/settings.py
git commit -m "feat: add price monitor settings"
```

---

### Task 5: Pipeline writes to SignalCache

**Files:**
- Modify: `src/orchestrator/pipeline.py`
- Test: `tests/unit/test_pipeline_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pipeline_cache.py
"""Test that TradingPipeline writes to SignalCache after Pass 1."""
import asyncio
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import (
    EnsembleForecast,
    MarketContract,
    MarketPrice,
    RegimeClassification,
)
from src.orchestrator.pipeline import TradingPipeline
from src.orchestrator.signal_cache import SignalCache
from src.trading.exposure_tracker import ExposureTracker


def _build_pipeline_with_cache():
    """Build a pipeline with mocked dependencies and a real SignalCache."""
    cache = SignalCache()
    tracker = ExposureTracker()

    # Minimal mocks for components
    collector = MagicMock()
    prob_engine = MagicMock()
    regime_classifier = MagicMock()
    edge_detector = MagicMock()
    position_sizer = MagicMock()
    executor = MagicMock()
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
        signal_cache=cache,
        exposure_tracker=tracker,
    )
    return pipeline, cache, collector, prob_engine, regime_classifier, edge_detector


@pytest.mark.asyncio
async def test_run_cycle_populates_signal_cache():
    pipeline, cache, collector, prob_engine, regime_cls, edge_det = _build_pipeline_with_cache()

    # Set up one station with one contract
    contract = MarketContract(
        token_id="tok_yes",
        no_token_id="tok_no",
        condition_id="cond_1",
        question="Will NYC high be 72-73°F?",
        city="NYC",
        resolution_date=date(2026, 4, 16),
        end_date_utc=datetime(2026, 4, 16, 23, 59, 59, tzinfo=timezone.utc),
        temp_bucket_low=72.0,
        temp_bucket_high=73.0,
        outcome="Yes",
        volume_24h=5000.0,
    )
    price = MarketPrice(
        token_id="tok_yes",
        timestamp=datetime.now(timezone.utc),
        bid=0.30,
        ask=0.35,
        mid=0.325,
        volume_24h=5000.0,
    )
    ensemble = EnsembleForecast(
        model_name="gfs",
        run_time=datetime.now(timezone.utc),
        valid_time=datetime.now(timezone.utc),
        station_id="KNYC",
        members=[70.0, 72.0, 74.0],
    )

    # Mock data collector to return one snapshot
    snapshot = MagicMock()
    snapshot.gfs_ensemble = ensemble
    snapshot.ecmwf_ensemble = None
    snapshot.hrrr = None
    snapshot.market_contracts = [contract]
    snapshot.market_prices = {"tok_yes": price}
    snapshot.metar_obs = []
    collector.collect_snapshot = AsyncMock(return_value=[snapshot])

    # Mock regime classifier
    regime = RegimeClassification(
        station_id="KNYC",
        valid_date=date(2026, 4, 15),
        regime="normal",
        confidence="HIGH",
        ensemble_spread_percentile=30.0,
    )
    regime_cls.classify.return_value = regime

    # Mock probability engine
    prob_engine.compute_distribution.return_value = MagicMock()
    prob_engine.compute_bucket_probability.return_value = 0.40

    # Mock edge detector to return SKIP (we just care about cache population)
    from src.data.models import TradingSignal
    edge_det.evaluate.return_value = TradingSignal(
        market_id="tok_yes",
        direction="BUY_YES",
        action="SKIP",
        edge=0.05,
        kelly_size=0.0,
        timestamp=datetime.now(timezone.utc),
    )

    # Mock stations
    from src.config.stations import Station
    test_stations = {
        "NYC": Station(
            station_id="KNYC",
            city="NYC",
            state="NY",
            lat=40.7128,
            lon=-74.006,
            timezone_str="America/New_York",
            flags=[],
        ),
    }
    with patch("src.orchestrator.pipeline.get_stations", return_value=test_stations):
        await pipeline.run_cycle()

    # Cache should now have one entry
    cached = cache.get("tok_yes")
    assert cached is not None
    assert cached.model_prob == 0.40
    assert cached.station_id == "KNYC"
    assert cached.regime is regime
    assert cached.contract is contract
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pipeline_cache.py -v`
Expected: FAIL — `TradingPipeline.__init__()` does not accept `signal_cache` or `exposure_tracker`

- [ ] **Step 3: Modify TradingPipeline to accept and write to SignalCache + ExposureTracker**

In `src/orchestrator/pipeline.py`:

**3a. Add imports** (top of file, after existing imports):

```python
from src.orchestrator.signal_cache import CachedSignal, SignalCache
from src.trading.exposure_tracker import ExposureTracker
```

**3b. Update `__init__`** — add two optional parameters after `cusum`:

```python
        signal_cache: SignalCache | None = None,
        exposure_tracker: ExposureTracker | None = None,
```

And store them:

```python
        self._signal_cache = signal_cache
        self._exposure_tracker = exposure_tracker
```

**3c. Write to cache at end of Pass 1** — after the `for city, station in stations.items()` loop ends (after line 217, before the Pass 2 comment), insert:

```python
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
```

**3d. Use ExposureTracker in Pass 2** — in the Pass 2 loop, where `current_exposure` is referenced in `self._position_sizer.compute(...)` (around line 241-249), read from the tracker:

Replace:
```python
                size_usd = self._position_sizer.compute(
                    edge=signal.edge,
                    market_prob=market_prob,
                    bankroll=bankroll,
                    current_exposure=current_exposure,
                    ensemble_spread_pctile=regime.ensemble_spread_percentile,
                    direction=signal.direction,
                    active_station_count=active_station_count,
                )
```

With:
```python
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
```

And after the successful trade execution (after `trades_placed += 1`, around line 284), add:

```python
                    if self._exposure_tracker is not None:
                        self._exposure_tracker.add(size_usd)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_pipeline_cache.py -v`
Expected: PASS

Also run existing pipeline tests to check for regressions:
Run: `python -m pytest tests/ -v -k "pipeline" --tb=short`

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/pipeline.py tests/unit/test_pipeline_cache.py
git commit -m "feat: pipeline writes to SignalCache + uses ExposureTracker"
```

---

### Task 6: PriceMonitor

**Files:**
- Create: `src/orchestrator/price_monitor.py`
- Test: `tests/unit/test_price_monitor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_price_monitor.py
"""Tests for PriceMonitor."""
import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import (
    MarketContract,
    MarketPrice,
    RegimeClassification,
    TradingSignal,
)
from src.orchestrator.price_monitor import PriceMonitor
from src.orchestrator.signal_cache import CachedSignal, SignalCache
from src.trading.exposure_tracker import ExposureTracker


def _make_contract(token_id="tok_yes", no_token_id="tok_no"):
    return MarketContract(
        token_id=token_id,
        no_token_id=no_token_id,
        condition_id="cond_1",
        question="Will NYC high be 72-73°F?",
        city="NYC",
        resolution_date=date(2026, 4, 16),
        end_date_utc=datetime(2026, 4, 16, 23, 59, 59, tzinfo=timezone.utc),
        temp_bucket_low=72.0,
        temp_bucket_high=73.0,
        outcome="Yes",
        volume_24h=5000.0,
    )


def _make_cached_signal(token_id="tok_yes", model_prob=0.40):
    return CachedSignal(
        model_prob=model_prob,
        regime=RegimeClassification(
            station_id="KNYC",
            valid_date=date(2026, 4, 15),
            regime="normal",
            confidence="HIGH",
            ensemble_spread_percentile=30.0,
        ),
        contract=_make_contract(token_id),
        station_id="KNYC",
        forecast_time=datetime.now(timezone.utc),
    )


def _make_price(token_id="tok_yes", bid=0.30, ask=0.35, mid=0.325):
    return MarketPrice(
        token_id=token_id,
        timestamp=datetime.now(timezone.utc),
        bid=bid,
        ask=ask,
        mid=mid,
        volume_24h=0.0,  # WS always returns 0 — PriceMonitor uses contract.volume_24h
    )


def _build_monitor(**overrides):
    """Build a PriceMonitor with mocked dependencies."""
    ws_feed = MagicMock()
    signal_cache = SignalCache()
    edge_detector = MagicMock()
    position_sizer = MagicMock()
    exposure_tracker = ExposureTracker()
    executor = MagicMock()
    prediction_log = MagicMock()
    paper_trader = MagicMock()

    defaults = dict(
        ws_feed=ws_feed,
        signal_cache=signal_cache,
        edge_detector=edge_detector,
        position_sizer=position_sizer,
        exposure_tracker=exposure_tracker,
        executor=executor,
        prediction_log=prediction_log,
        paper_trader=paper_trader,
        debounce_seconds=0.0,  # disable debounce for most tests
        cooldown_seconds=900.0,
        bankroll=300.0,
    )
    defaults.update(overrides)

    monitor = PriceMonitor(**defaults)
    return monitor, defaults


class TestHandlePriceUpdate:
    """Test the core _handle_price_update method directly."""

    @pytest.mark.asyncio
    async def test_skip_when_no_cached_signal(self):
        monitor, deps = _build_monitor()
        # No signals in cache — should return without trading
        result = await monitor._handle_price_update("unknown_token")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_forecast_too_old(self):
        monitor, deps = _build_monitor(max_forecast_age_s=100.0)
        sig = _make_cached_signal()
        sig.forecast_time = datetime.now(timezone.utc) - timedelta(hours=3)
        deps["signal_cache"].update({"tok_yes": sig})
        # Manually set _last_update to 3 hours ago to make forecast_age_seconds > 100
        deps["signal_cache"]._last_update = datetime.now(timezone.utc) - timedelta(hours=3)

        deps["ws_feed"].get_latest_price.return_value = _make_price()
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_edge_detector_says_skip(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="SKIP",
            edge=0.02,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_trade_executes_on_edge(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        result = await monitor._handle_price_update("tok_yes")
        assert result is not None
        deps["executor"].execute.assert_called_once()
        # Check kelly_size was set on the signal passed to executor
        call_args = deps["executor"].execute.call_args
        assert call_args[0][0].kelly_size == 2.50

    @pytest.mark.asyncio
    async def test_exposure_tracker_updated_after_trade(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")
        assert deps["exposure_tracker"].current == 2.50

    @pytest.mark.asyncio
    async def test_buy_no_routes_to_no_token(self):
        monitor, deps = _build_monitor()
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.side_effect = lambda tid: {
            "tok_yes": _make_price("tok_yes"),
            "tok_no": _make_price("tok_no", bid=0.65, ask=0.70, mid=0.675),
        }.get(tid)
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_NO",
            action="TRADE",
            edge=0.10,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.00
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        await monitor._handle_price_update("tok_yes")
        # Executor should have been called with the NO token
        call_args = deps["executor"].execute.call_args
        assert call_args[0][1] == "tok_no"  # token_id arg

    @pytest.mark.asyncio
    async def test_cusum_alarm_blocks_trade(self):
        cusum = MagicMock()
        cusum.alarm = True
        monitor, deps = _build_monitor(cusum=cusum)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_volume_uses_contract_not_ws(self):
        """EdgeDetector should receive volume_24h from cached contract, not WS price."""
        monitor, deps = _build_monitor()
        cached = _make_cached_signal()
        deps["signal_cache"].update({"tok_yes": cached})
        deps["ws_feed"].get_latest_price.return_value = _make_price()  # volume_24h=0.0
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="SKIP",
            edge=0.0,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        await monitor._handle_price_update("tok_yes")
        # Check that evaluate was called with contract's volume, not WS's 0.0
        call_kwargs = deps["edge_detector"].evaluate.call_args[1]
        assert call_kwargs["volume_24h"] == 5000.0


class TestDebounce:
    @pytest.mark.asyncio
    async def test_debounce_blocks_immediate_trade(self):
        monitor, deps = _build_monitor(debounce_seconds=5.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        # First call should NOT trade — starts debounce timer
        result = await monitor._handle_price_update("tok_yes")
        assert result is None
        deps["executor"].execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_debounce_clears_on_skip(self):
        monitor, deps = _build_monitor(debounce_seconds=5.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()

        # First: TRADE — starts debounce
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="TRADE",
            edge=0.10, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        await monitor._handle_price_update("tok_yes")
        assert "tok_yes" in monitor._pending_edges

        # Second: SKIP — clears debounce
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes", direction="BUY_YES", action="SKIP",
            edge=0.02, kelly_size=0.0, timestamp=datetime.now(timezone.utc),
        )
        await monitor._handle_price_update("tok_yes")
        assert "tok_yes" not in monitor._pending_edges


class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_second_trade(self):
        monitor, deps = _build_monitor(cooldown_seconds=900.0)
        deps["signal_cache"].update({"tok_yes": _make_cached_signal()})
        deps["ws_feed"].get_latest_price.return_value = _make_price()
        deps["edge_detector"].evaluate.return_value = TradingSignal(
            market_id="tok_yes",
            direction="BUY_YES",
            action="TRADE",
            edge=0.10,
            kelly_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        deps["position_sizer"].compute.return_value = 2.50
        deps["executor"].execute = AsyncMock(return_value=MagicMock())
        deps["paper_trader"].record_trade.return_value = "trade_id_1"

        # First trade succeeds
        result1 = await monitor._handle_price_update("tok_yes")
        assert result1 is not None

        # Second trade blocked by cooldown
        result2 = await monitor._handle_price_update("tok_yes")
        assert result2 is None
        assert deps["executor"].execute.call_count == 1  # only called once
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_price_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.orchestrator.price_monitor'`

- [ ] **Step 3: Write the implementation**

```python
# src/orchestrator/price_monitor.py
"""Continuous price monitor — evaluates edges against cached model probs.

Subscribes to the WebSocket feed and checks for tradeable edges whenever
market prices update, using model probabilities cached by the forecast
pipeline. Includes debounce (edge must persist) and cooldown (per-contract
post-trade lockout) to prevent overtrading.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
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
            # Also subscribe to NO tokens for BUY_NO routing
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

        # Look up cached model probability
        cached = self._signal_cache.get(token_id)
        if cached is None:
            return None

        # Staleness check
        if self._signal_cache.forecast_age_seconds > self._max_forecast_age_s:
            logger.warning(
                "Forecast too old (%.0fs) — skipping price monitor trade",
                self._signal_cache.forecast_age_seconds,
            )
            return None

        # Get live price from WS
        live_price = self._ws_feed.get_latest_price(token_id)
        if live_price is None:
            return None

        # Compute hours to resolution (live, not cached)
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

        # Run edge detection — use contract's volume_24h, NOT WS price's
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

        # CUSUM check (alarm only reset by scheduled pipeline — intentional)
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

        # Execute
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_price_monitor.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/price_monitor.py tests/unit/test_price_monitor.py
git commit -m "feat: add PriceMonitor for continuous edge detection"
```

---

### Task 7: Wire everything together in main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add imports**

Add at the top of `main.py`, with the other imports:

```python
from src.data.ws_feed import WebSocketFeed
from src.orchestrator.signal_cache import SignalCache
from src.orchestrator.price_monitor import PriceMonitor
from src.trading.exposure_tracker import ExposureTracker
```

- [ ] **Step 2: Create shared components and update pipeline construction**

In `main.py`, after the `cusum = CUSUMMonitor(threshold=2.0)` line (line 67), add:

```python
    signal_cache = SignalCache()
    exposure_tracker = ExposureTracker()
```

Update the `TradingPipeline(...)` constructor (around line 70) to add the two new arguments:

```python
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
    )
```

- [ ] **Step 3: Create and start PriceMonitor**

After the initial cycle (after `logger.info("Initial cycle: %s", result)`, around line 106), add:

```python
    # Start continuous price monitor
    ws_feed = None
    price_monitor = None
    if settings.PRICE_MONITOR_ENABLED:
        ws_feed = WebSocketFeed()
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
            debounce_seconds=settings.PRICE_MONITOR_DEBOUNCE_S,
            cooldown_seconds=settings.PRICE_MONITOR_COOLDOWN_S,
            max_forecast_age_s=settings.PRICE_MONITOR_MAX_FORECAST_AGE_S,
            bankroll=settings.BANKROLL,
        )
        await price_monitor.start()
        logger.info("Price monitor started (debounce=%.0fs, cooldown=%.0fs)",
                     settings.PRICE_MONITOR_DEBOUNCE_S,
                     settings.PRICE_MONITOR_COOLDOWN_S)
```

- [ ] **Step 4: Clean shutdown**

Update the `finally` block (around line 114) to also stop the price monitor:

```python
    try:
        await server.serve()
    finally:
        if price_monitor is not None:
            await price_monitor.stop()
        if ws_feed is not None:
            await ws_feed.close()
        await scheduler.stop()
```

- [ ] **Step 5: Verify it loads**

Run: `python -c "from main import main; print('imports OK')"`
Expected: `imports OK` (verifies no import errors)

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: wire PriceMonitor into main.py startup"
```

---

### Task 8: Integration test — full flow

**Files:**
- Create: `tests/integration/test_price_monitor_flow.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_price_monitor_flow.py
"""Integration test: forecast cycle → signal cache → price monitor → trade."""
import asyncio
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.models import (
    EnsembleForecast,
    MarketContract,
    MarketPrice,
    RegimeClassification,
    TradingSignal,
)
from src.data.ws_feed import WebSocketFeed
from src.orchestrator.price_monitor import PriceMonitor
from src.orchestrator.signal_cache import SignalCache
from src.prediction.calibration import CUSUMMonitor
from src.trading.edge_detector import EdgeDetector
from src.trading.exposure_tracker import ExposureTracker
from src.trading.position_sizer import PositionSizer
from src.trading.executor import OrderExecutor
from src.verification.prediction_log import PredictionLog
from src.verification.paper_trader import PaperTrader
from src.orchestrator.signal_cache import CachedSignal


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
    await asyncio.sleep(0.05)  # let tasks start

    # Simulate a forecast cycle updating the cache
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

    await asyncio.sleep(0.1)  # let resubscription task react

    ws_feed.resubscribe.assert_called_once()
    call_args = ws_feed.resubscribe.call_args[0][0]
    assert "tok_yes" in call_args
    assert "tok_no" in call_args

    await monitor.stop()


@pytest.mark.asyncio
async def test_exposure_shared_between_paths():
    """Both pipeline and price monitor should see the same exposure tracker."""
    tracker = ExposureTracker()

    # Simulate pipeline adding exposure
    tracker.add(10.0)

    # Price monitor should see 10.0
    assert tracker.current == 10.0

    # Simulate price monitor adding more
    tracker.add(5.0)

    # Pipeline should see 15.0
    assert tracker.current == 15.0


async def _empty_async_iter():
    """An async iterator that never yields."""
    while True:
        await asyncio.sleep(999)
        yield {}  # never reached
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/integration/test_price_monitor_flow.py -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_price_monitor_flow.py
git commit -m "test: integration tests for price monitor flow"
```

---

### Task 9: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS with no regressions

- [ ] **Step 2: Fix any failures**

If any existing tests fail due to the new `signal_cache`/`exposure_tracker` params on `TradingPipeline.__init__`, they should still work because both params are optional (default `None`). If not, add `signal_cache=None, exposure_tracker=None` to any test constructors that need it.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: test compatibility with new pipeline params"
```
