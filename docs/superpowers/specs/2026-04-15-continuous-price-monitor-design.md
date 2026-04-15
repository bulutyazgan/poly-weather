# Continuous Price Monitor — Design Spec

**Date:** 2026-04-15
**Status:** Draft

## Problem

The bot currently runs 6-7 times/day aligned with NWP model updates. Between cycles, market prices on Polymarket fluctuate constantly. If a market dislocation occurs (e.g., someone dumps shares), the bot misses the wider edge until the next scheduled wake-up — potentially hours later.

Edge = `model_prob - market_price`. The forecast side changes slowly (6-7x/day), but the market side changes constantly. We should monitor both.

## Solution

Decouple forecast updates from price monitoring. The forecast scheduler continues running on the NWP schedule. A new `PriceMonitor` subscribes to the existing `WebSocketFeed` and continuously evaluates edges against cached model probabilities.

## Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| React to WS price | Debounce (10s) | Weather markets are slow; filters WS glitches without HTTP round-trip |
| Forecast cache granularity | Per-contract model probs | Simpler than caching distributions; new mid-cycle contracts are rare |
| Position limits | Shared limits + per-contract cooldown | Prevents overtrading the same dislocation; shared ExposureTracker handles sizing |
| Price source | WebSocket feed (Approach A) | Already built with reconnection/backoff; real-time vs 30s HTTP polling |
| CUSUM reset | Only by scheduled pipeline | Conservative: price monitor respects alarm but only `run_cycle()` resets it |

## Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                       main.py                             │
│                                                           │
│  ┌──────────────┐       ┌────────────────────┐           │
│  │  Scheduler    │       │  PriceMonitor      │           │
│  │  (NWP times)  │       │  (continuous)      │           │
│  │               │       │                    │           │
│  │  run_cycle()──┼──────▶│  reads from cache  │           │
│  │  writes cache │       │  compares vs WS    │           │
│  │  updates exp. │       │  debounce → trade  │           │
│  └──────┬───────┘       │  updates exp.      │           │
│         │                └────────┬───────────┘           │
│         │                         │                        │
│         ▼                         ▼                        │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ SignalCache   │  │ ExposureTracker│  │ WebSocketFeed│  │
│  │ (in-memory)   │  │ (shared state) │  │ (existing)   │  │
│  └──────────────┘  └────────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Component 1: SignalCache

A simple in-memory store that bridges the forecast layer and price monitor.

```python
@dataclass
class CachedSignal:
    model_prob: float               # final calibrated probability
    regime: RegimeClassification    # includes ensemble_spread_percentile
    contract: MarketContract        # volume, resolution time, bucket info, no_token_id
    station_id: str
    forecast_time: datetime         # when this was computed

class SignalCache:
    _updated: asyncio.Event         # set on every update(), watched by PriceMonitor

    def update(self, signals: dict[str, CachedSignal]) -> None:
        """Full replacement of cache (not merge). Called after each forecast cycle.
        Sets _updated event to notify watchers (e.g., PriceMonitor resubscription)."""

    def get_all(self) -> dict[str, CachedSignal]:
        """Return all cached signals. Key is token_id."""

    def get(self, token_id: str) -> CachedSignal | None:
        """Return cached signal for a specific token."""

    @property
    def forecast_age_seconds(self) -> float:
        """Seconds since last forecast update. Used for staleness checks."""
```

**Cache lifecycle:**
- Fully replaced (not merged) on each forecast cycle — stale contracts don't linger
- `_updated` event notifies the PriceMonitor to resubscribe to new token IDs
- No locking needed — asyncio single-thread means dict writes are atomic
- Lives in `src/orchestrator/signal_cache.py`

**Note:** `CachedSignal` does NOT store `direction` — direction depends on live price and must be recomputed each tick (market_prob changes → direction may flip).

### Component 2: ExposureTracker

Solves the critical concurrency issue: `PositionSizer` is stateless — it takes `current_exposure` as a parameter. Without a shared tracker, both the pipeline and price monitor could size trades against the same stale exposure value.

```python
class ExposureTracker:
    """Thread-safe (asyncio-safe) running tally of portfolio exposure."""

    def __init__(self, initial: float = 0.0) -> None:
        self._exposure = initial

    @property
    def current(self) -> float:
        return self._exposure

    def add(self, amount: float) -> None:
        """Increment exposure after a trade is executed."""
        self._exposure += amount

    def reset(self, value: float) -> None:
        """Reset to known value (e.g., after reconciliation)."""
        self._exposure = value
```

Both `TradingPipeline.run_cycle()` and `PriceMonitor` read `tracker.current` just before calling `position_sizer.compute()`, then call `tracker.add(size_usd)` immediately after successful execution. Since both run in the same asyncio event loop, the read-execute-add sequence is atomic within each `await` boundary.

Lives in `src/trading/exposure_tracker.py`.

### Component 3: PriceMonitor

The core new component. Subscribes to WS updates, evaluates edges against cached probs, debounces, and executes.

```python
class PriceMonitor:
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
        cooldown_seconds: float = 900.0,  # 15 minutes
        max_forecast_age_s: float = 28800.0,  # 8 hours
    ):
        self._pending_edges: dict[str, datetime] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._trade_lock: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
```

**Main loop (`run`):**

```
start two concurrent tasks:
  1. _watch_prices()        — the main edge-detection loop
  2. _watch_resubscription() — resubscribes WS when cache updates

_watch_resubscription():
    forever:
        await signal_cache._updated.wait()
        signal_cache._updated.clear()
        token_ids = list(signal_cache.get_all().keys())
        # Also subscribe to NO tokens for BUY_NO routing
        for cached in signal_cache.get_all().values():
            if cached.contract.no_token_id:
                token_ids.append(cached.contract.no_token_id)
        await ws_feed.resubscribe(token_ids)

_watch_prices():
    async for update in ws_feed.listen():
        token_id = update.get("asset_id", "")
        cached = signal_cache.get(token_id)
        if cached is None:
            continue

        # Staleness check
        if signal_cache.forecast_age_seconds > max_forecast_age_s:
            continue

        live_price = ws_feed.get_latest_price(token_id)
        if live_price is None:
            continue

        # Recompute hours_to_resolution live (not cached!)
        resolution_dt = contract.end_date_utc or end-of-day fallback
        hours_to_resolution = (resolution_dt - now).total_seconds() / 3600.0

        # Run edge detection (same EdgeDetector, reused)
        signal = edge_detector.evaluate(
            model_prob=cached.model_prob,
            market_prob=live_price.mid,
            regime=cached.regime,
            volume_24h=cached.contract.volume_24h,  # from forecast snapshot, NOT WS
            hours_to_resolution=hours_to_resolution,
            market_id=token_id,
            market_bid=live_price.bid,
            market_ask=live_price.ask,
        )

        if signal.action != "TRADE":
            self._pending_edges.pop(token_id, None)  # clear debounce
            continue

        # Debounce: edge must persist for debounce_seconds
        if token_id not in self._pending_edges:
            self._pending_edges[token_id] = now
            continue
        if (now - self._pending_edges[token_id]).total_seconds() < debounce_seconds:
            continue

        # Cooldown: skip if recently traded
        if token_id in self._cooldowns and now < self._cooldowns[token_id]:
            continue

        # CUSUM check (alarm is only reset by scheduled pipeline — intentional)
        if cusum is not None and cusum.alarm:
            continue

        # Per-token lock prevents overlapping execution with scheduled pipeline
        async with self._trade_lock[token_id]:
            # Size position
            size_usd = position_sizer.compute(
                edge=signal.edge,
                market_prob=live_price.mid,
                bankroll=...,  # from config
                current_exposure=exposure_tracker.current,
                ensemble_spread_pctile=cached.regime.ensemble_spread_percentile,
                direction=signal.direction,
                active_station_count=1,  # conservative: no multi-station correlation info
            )

            # Build sized signal (matching pipeline pattern at pipeline.py:251-258)
            sized_signal = TradingSignal(
                market_id=signal.market_id,
                direction=signal.direction,
                action=signal.action,
                edge=signal.edge,
                kelly_size=size_usd,
                timestamp=signal.timestamp,
            )

            # BUY_NO token routing (matching pipeline pattern at pipeline.py:262-268)
            if signal.direction == "BUY_NO" and cached.contract.no_token_id:
                no_price = ws_feed.get_latest_price(cached.contract.no_token_id)
                if no_price is None:
                    continue
                exec_token = cached.contract.no_token_id
                exec_price = no_price
            else:
                exec_token = token_id
                exec_price = live_price

            trade_record = await executor.execute(sized_signal, exec_token, exec_price)

            if trade_record is not None:
                exposure_tracker.add(size_usd)

                # Paper trader recording (matching pipeline.py:276-282)
                paper_trader.record_trade(
                    signal=sized_signal,
                    contract=cached.contract,
                    entry_price=live_price.mid,
                    amount_usd=size_usd,
                    model_probability=cached.model_prob,
                )

                # Prediction log
                log_entry = SignalLogEntry(
                    signal=sized_signal,
                    station_id=cached.station_id,
                    regime=cached.regime,
                    model_probability=cached.model_prob,
                    market_probability=live_price.mid,
                    contract=cached.contract,
                )
                prediction_log.log(log_entry)

                # Set cooldown and clear debounce
                self._cooldowns[token_id] = now + timedelta(seconds=cooldown_seconds)
                self._pending_edges.pop(token_id, None)
```

**Key behaviors:**
- Debounce: edge must persist for 10s. If price bounces back, pending timer is cleared.
- Cooldown: after trading a contract, ignore it for 15 minutes.
- CUSUM: respects the alarm but does NOT reset it — only `run_cycle()` resets. This is intentional: if the model is drifting, we want the scheduled pipeline (with fresh forecast data) to decide when to resume trading.
- BUY_NO routing: follows the same pattern as pipeline.py:262-268, using `contract.no_token_id` and fetching the NO token's live price from the WS feed.
- `volume_24h`: always uses `cached.contract.volume_24h` from the forecast snapshot, NOT the WS `MarketPrice.volume_24h` (which is always 0.0 for price_change events).
- `hours_to_resolution`: recomputed live each tick from `contract.end_date_utc`, not cached from forecast time.
- Per-token `asyncio.Lock`: prevents the pipeline and price monitor from executing on the same token simultaneously.

**Lives in:** `src/orchestrator/price_monitor.py`

### Component 4: Pipeline Changes

Minimal changes to the existing pipeline:

1. **`TradingPipeline.__init__`** accepts optional `SignalCache` and `ExposureTracker`.
2. **`TradingPipeline.run_cycle`** Pass 1 — at the end, after computing all model probs, writes `CachedSignal` entries to the `SignalCache`.
3. **`TradingPipeline.run_cycle`** Pass 2 — reads `exposure_tracker.current` for `current_exposure` instead of the caller-supplied parameter. After each successful trade, calls `exposure_tracker.add(size_usd)`.
4. Pass 2 logic is otherwise unchanged — the scheduled cycle still trades on its own.

### Component 5: WebSocketFeed Enhancement

One addition to the existing `WebSocketFeed`:

```python
async def resubscribe(self, token_ids: list[str]) -> None:
    """Update subscriptions by restarting the connection.

    Cancels the current connection task and starts a new one with the
    updated token list. This is the simplest correct approach given
    that _send_subscription only runs at connection time.
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

This forces a reconnect with the new token list. Simple and correct — the reconnect logic with backoff already exists.

### Component 6: main.py Orchestration

```python
# After building pipeline (existing code)...

signal_cache = SignalCache()
exposure_tracker = ExposureTracker()

pipeline = TradingPipeline(
    ...,
    signal_cache=signal_cache,
    exposure_tracker=exposure_tracker,
)

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
)

# After initial cycle (which populates the signal cache)...
await price_monitor.start()  # background task
await scheduler.start()       # existing
```

## Configuration

New settings in `Settings`:

```python
PRICE_MONITOR_ENABLED: bool = True
PRICE_MONITOR_DEBOUNCE_S: float = 10.0
PRICE_MONITOR_COOLDOWN_S: float = 900.0       # 15 min per contract
PRICE_MONITOR_MAX_FORECAST_AGE_S: float = 28800.0  # 8 hours
```

## Staleness Protection

The price monitor does NOT trade on ancient forecasts. If `signal_cache.forecast_age_seconds > max_forecast_age_s` (default 8 hours), it logs a warning and skips all trades until the next successful forecast cycle.

## What Doesn't Change

- **Scheduler** — still runs on NWP schedule, still calls `run_cycle()`
- **EdgeDetector** — reused as-is by both paths
- **PositionSizer** — reused as-is (stateless pure function)
- **OrderExecutor** — reused as-is
- **CUSUM** — shared instance; price monitor respects alarm, only pipeline resets it
- **PredictionLog** — shared instance, both paths log through it
- **Two-pass pipeline** — Pass 2 still runs during scheduled cycles

## New Files

| File | Purpose |
|------|---------|
| `src/orchestrator/signal_cache.py` | `SignalCache` + `CachedSignal` |
| `src/orchestrator/price_monitor.py` | `PriceMonitor` |
| `src/trading/exposure_tracker.py` | `ExposureTracker` |
| `tests/unit/test_signal_cache.py` | Unit tests |
| `tests/unit/test_price_monitor.py` | Unit tests |
| `tests/unit/test_exposure_tracker.py` | Unit tests |

## Modified Files

| File | Change |
|------|--------|
| `src/data/ws_feed.py` | Add `resubscribe()` method |
| `src/orchestrator/pipeline.py` | Accept + write to `SignalCache`; use `ExposureTracker` |
| `src/config/settings.py` | Add 3 new settings |
| `main.py` | Wire up new components |

## Testing Strategy

1. **Unit: SignalCache** — update/get/replace/staleness/event notification
2. **Unit: ExposureTracker** — add/reset/current
3. **Unit: PriceMonitor debounce** — edge appears, persists, clears, re-appears
4. **Unit: PriceMonitor cooldown** — trade, then verify skip during cooldown
5. **Unit: PriceMonitor CUSUM** — alarm blocks trades, alarm not reset by monitor
6. **Unit: PriceMonitor BUY_NO routing** — correct token and price used for NO trades
7. **Unit: PriceMonitor staleness** — trades blocked when forecast too old
8. **Integration: Pipeline → SignalCache → PriceMonitor** — forecast cycle populates cache, simulated WS update triggers trade
9. **Integration: Resubscription** — new forecast cycle changes markets, WS reconnects with new tokens
10. **Integration: Concurrent execution** — pipeline and monitor don't double-trade same token (per-token lock)
