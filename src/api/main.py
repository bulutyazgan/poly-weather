"""FastAPI application for the Polymarket Weather Prediction Trading System."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, time as dt_time, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from src.api.event_bus import EventBus
from src.config.stations import get_station, get_stations
from src.orchestrator.signal_cache import SignalCache
from src.prediction.calibration import BrierScore, CUSUMMonitor, ReliabilityDiagram
from src.trading.executor import OrderExecutor
from src.trading.exposure_tracker import ExposureTracker
from src.data.ws_feed import WebSocketFeed
from src.verification.paper_trader import PaperTrader
from src.verification.prediction_log import PredictionLog

# Conditional import — scheduler needs pipeline which has heavy deps,
# but we only need the type for annotation purposes.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestrator.scheduler import PipelineScheduler

app = FastAPI(
    title="TradeBot",
    version="0.1.0",
    description="Polymarket Weather Prediction Trading System",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# In-memory state shared across requests — injected via set_state()
_prediction_log: PredictionLog | None = None
_paper_trader: PaperTrader | None = None
_scheduler: PipelineScheduler | None = None
_executor: OrderExecutor | None = None
_cusum: CUSUMMonitor | None = None
_signal_cache: SignalCache | None = None
_exposure_tracker: ExposureTracker | None = None
_ws_feed: WebSocketFeed | None = None
_price_monitor = None  # avoid circular import, use Any
_event_bus: EventBus | None = None


def set_state(
    log: PredictionLog | None,
    trader: PaperTrader | None,
    scheduler: PipelineScheduler | None,
    executor: OrderExecutor | None = None,
    cusum: CUSUMMonitor | None = None,
    signal_cache: SignalCache | None = None,
    exposure_tracker: ExposureTracker | None = None,
    ws_feed: WebSocketFeed | None = None,
    price_monitor=None,
    event_bus: EventBus | None = None,
) -> None:
    """Inject shared state (called from main.py startup or tests)."""
    global _prediction_log, _paper_trader, _scheduler, _executor, _cusum, _signal_cache
    global _exposure_tracker, _ws_feed, _price_monitor, _event_bus
    _prediction_log = log
    _paper_trader = trader
    _scheduler = scheduler
    _executor = executor
    _cusum = cusum
    _signal_cache = signal_cache
    _exposure_tracker = exposure_tracker
    _ws_feed = ws_feed
    _price_monitor = price_monitor
    _event_bus = event_bus


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def get_status():
    """Return bot liveness: scheduler health, last cycle, next event, cache age."""
    scheduler_status = _scheduler.status if _scheduler else {"running": False}
    cache_age = _signal_cache.forecast_age_seconds if _signal_cache else None
    return {
        "scheduler": scheduler_status,
        "signal_cache_age_seconds": round(cache_age, 0) if cache_age is not None else None,
        "signal_count": _prediction_log.count() if _prediction_log else 0,
    }


@app.get("/api/stations")
async def list_stations():
    return list(get_stations().values())


@app.get("/api/stations/{city}")
async def get_station_detail(city: str):
    try:
        return get_station(city)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Station '{city}' not found")


@app.get("/api/performance")
async def get_performance():
    """Return trading performance metrics."""
    return {
        "total_pnl": _paper_trader.total_pnl() if _paper_trader else 0.0,
        "win_rate": _paper_trader.win_rate() if _paper_trader else 0.0,
        "trade_count": len(_paper_trader.get_resolved_trades()) if _paper_trader else 0,
        "signal_count": _prediction_log.count() if _prediction_log else 0,
        "fill_rate": _executor.get_fill_rate() if _executor else None,
        "adverse_selection_ratio": _executor.get_adverse_selection_ratio() if _executor else None,
        "being_picked_off": _executor.is_being_picked_off() if _executor else False,
    }


@app.get("/api/signals")
async def list_signals(
    station: str | None = None,
    confidence: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Return signal log entries, optionally filtered and paginated."""
    if not _prediction_log:
        return []
    if station:
        entries = _prediction_log.get_by_station(station)
    elif confidence:
        entries = _prediction_log.get_by_regime(confidence)
    else:
        entries = _prediction_log.get_all()
    return [_entry_to_dict(e) for e in entries[offset : offset + limit]]


@app.get("/api/schedule")
async def get_schedule():
    """Return list of scheduled pipeline events."""
    if not _scheduler:
        return []
    events = _scheduler.get_scheduled_events()
    # Serialize datetime.time objects to strings
    return [
        {
            "time": e["time"].isoformat() if isinstance(e["time"], dt_time) else str(e["time"]),
            "event_type": e["event_type"],
            "description": e["description"],
        }
        for e in events
    ]


@app.get("/api/trades")
async def list_trades(status: str | None = None, limit: int = 100):
    """Return paper trade records, optionally filtered by status.

    Query params:
        status: "resolved", "pending", or None (all)
        limit: max records returned (default 100)
    """
    if not _paper_trader:
        return []

    trades = list(_paper_trader._trades.values())

    if status == "resolved":
        trades = [t for t in trades if t["resolved"]]
    elif status == "pending":
        trades = [t for t in trades if not t["resolved"]]

    # Most recent first (by trade_id creation order, reversed)
    trades = trades[-limit:][::-1]

    return [_trade_to_dict(t) for t in trades]


@app.get("/api/cusum")
async def get_cusum():
    """Return CUSUM monitor state for model degradation tracking."""
    if not _cusum:
        return {"alarm": False, "cusum_pos": 0.0, "cusum_neg": 0.0, "threshold": 2.0, "pct_of_threshold": 0.0}

    peak = max(_cusum.cusum_pos, _cusum.cusum_neg)
    pct = (peak / _cusum.threshold * 100) if _cusum.threshold > 0 else 0.0

    return {
        "alarm": _cusum.alarm,
        "cusum_pos": round(_cusum.cusum_pos, 4),
        "cusum_neg": round(_cusum.cusum_neg, 4),
        "threshold": _cusum.threshold,
        "pct_of_threshold": round(pct, 1),
    }


@app.get("/api/calibration")
async def get_calibration():
    """Return calibration metrics from resolved paper trades.

    Requires actual resolved outcomes — returns null fields until enough
    resolved trades exist. Never uses proxy outcomes.
    """
    no_data = {
        "brier_score": None,
        "brier_skill_score": None,
        "reliability_diagram": None,
        "resolved_count": 0,
    }

    if not _paper_trader:
        return no_data

    resolved = _paper_trader.get_resolved_trades()
    if len(resolved) < 10:
        return {**no_data, "resolved_count": len(resolved)}

    forecasts = []
    outcomes = []
    for t in resolved:
        signal = t["signal"]
        model_prob = t.get("model_probability")
        if model_prob is not None:
            # Use the model's actual probability forecast for calibration
            forecasts.append(model_prob)
        else:
            # Fallback for trades recorded before model_probability was stored
            price = t["entry_price"]
            if signal.direction == "BUY_YES":
                forecasts.append(price)
            else:
                forecasts.append(1.0 - price)
        outcomes.append(bool(t["outcome"]))

    try:
        bs = BrierScore.compute(forecasts, outcomes)
        bss = BrierScore.skill_score(forecasts, outcomes)
        rd = ReliabilityDiagram.compute(forecasts, outcomes)
    except (ValueError, ZeroDivisionError):
        return {**no_data, "resolved_count": len(resolved)}

    return {
        "brier_score": bs,
        "brier_skill_score": bss,
        "reliability_diagram": rd,
        "resolved_count": len(resolved),
    }


@app.get("/api/events")
async def sse_events():
    """Server-Sent Events stream for real-time dashboard updates."""
    if _event_bus is None:
        return StreamingResponse(
            iter([]),
            media_type="text/event-stream",
        )

    async def event_stream():
        queue = _event_bus.subscribe()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: {msg['event']}\ndata: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    hb = json.dumps({"event": "heartbeat", "data": {"ts": datetime.now(tz=timezone.utc).isoformat()}, "timestamp": datetime.now(tz=timezone.utc).isoformat()})
                    yield f"event: heartbeat\ndata: {hb}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _event_bus.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/exposure")
async def get_exposure():
    """Return current portfolio exposure and risk metrics."""
    if not _exposure_tracker:
        return {
            "current_exposure_usd": 0.0,
            "realized_pnl": 0.0,
            "is_halted": False,
            "bankroll": 300.0,
            "max_drawdown_pct": 0.15,
            "exposure_pct": 0.0,
            "drawdown_pct": 0.0,
        }
    bankroll = _exposure_tracker._bankroll
    current = _exposure_tracker.current
    realized = _exposure_tracker.realized_pnl
    return {
        "current_exposure_usd": round(current, 2),
        "realized_pnl": round(realized, 2),
        "is_halted": _exposure_tracker.is_halted,
        "bankroll": bankroll,
        "max_drawdown_pct": _exposure_tracker._max_drawdown_pct,
        "exposure_pct": round(current / bankroll * 100, 1) if bankroll > 0 else 0.0,
        "drawdown_pct": round(abs(min(realized, 0.0)) / bankroll * 100, 1) if bankroll > 0 else 0.0,
    }


@app.get("/api/cached-signals")
async def get_cached_signals():
    """Return all cached model signals with live prices."""
    if not _signal_cache:
        return []
    now = datetime.now(tz=timezone.utc)
    result = []
    for token_id, cached in _signal_cache.get_all().items():
        live_bid = None
        live_ask = None
        live_mid = None
        price_source = "none"
        if _ws_feed:
            lp = _ws_feed.get_latest_price(token_id)
            if lp is not None:
                live_bid = lp.bid
                live_ask = lp.ask
                live_mid = lp.mid
                price_source = "ws"

        # Fall back to Gamma API prices from the contract snapshot
        if live_bid is None and cached.contract.gamma_best_bid is not None:
            live_bid = cached.contract.gamma_best_bid
            price_source = "gamma"
        if live_ask is None and cached.contract.gamma_best_ask is not None:
            live_ask = cached.contract.gamma_best_ask
            if price_source == "none":
                price_source = "gamma"
        if live_mid is None and live_bid is not None and live_ask is not None:
            live_mid = (live_bid + live_ask) / 2.0
        elif live_mid is None and cached.contract.gamma_outcome_price is not None:
            live_mid = cached.contract.gamma_outcome_price

        # Compute current edge
        current_edge = None
        if live_ask is not None:
            current_edge = round(cached.model_prob - live_ask, 4)

        resolution_dt = cached.contract.end_date_utc
        if resolution_dt is None:
            from datetime import time as dt_time_
            resolution_dt = datetime.combine(
                cached.contract.resolution_date,
                dt_time_(23, 59, 59),
                tzinfo=timezone.utc,
            )
        hours_to_resolution = max(0.0, (resolution_dt - now).total_seconds() / 3600.0)

        bucket = f"{cached.contract.temp_bucket_low}-{cached.contract.temp_bucket_high}"
        if cached.contract.temp_bucket_low == float('-inf'):
            bucket = f"<{cached.contract.temp_bucket_high}"
        elif cached.contract.temp_bucket_high == float('inf'):
            bucket = f">{cached.contract.temp_bucket_low}"

        result.append({
            "token_id": token_id,
            "station_id": cached.station_id,
            "city": cached.contract.city,
            "question": cached.contract.question,
            "temp_bucket": bucket,
            "model_prob": round(cached.model_prob, 4),
            "regime": cached.regime.regime,
            "regime_confidence": cached.regime.confidence,
            "active_flags": cached.regime.active_flags,
            "ensemble_spread_pctile": round(cached.regime.ensemble_spread_percentile, 1),
            "forecast_time": cached.forecast_time.isoformat(),
            "forecast_age_s": round((now - cached.forecast_time).total_seconds(), 0),
            "live_bid": live_bid,
            "live_ask": live_ask,
            "live_mid": live_mid,
            "current_edge": current_edge,
            "resolution_date": cached.contract.resolution_date.isoformat(),
            "hours_to_resolution": round(hours_to_resolution, 1),
        })
    return result


@app.get("/api/price-monitor")
async def get_price_monitor():
    """Return price monitor operational status."""
    if _price_monitor is None:
        return {
            "running": False,
            "pending_edges": {},
            "cooldowns": {},
            "ws_connected": False,
            "subscribed_tokens": 0,
        }
    now = datetime.now(tz=timezone.utc)
    pending = {}
    for token_id, first_seen in _price_monitor._pending_edges.items():
        pending[token_id] = {
            "first_seen": first_seen.isoformat(),
            "elapsed_s": round((now - first_seen).total_seconds(), 1),
        }
    cooldowns = {}
    for token_id, expires in _price_monitor._cooldowns.items():
        remaining = (expires - now).total_seconds()
        if remaining > 0:
            cooldowns[token_id] = {
                "expires": expires.isoformat(),
                "remaining_s": round(remaining, 1),
            }
    ws_connected = False
    subscribed = 0
    if _ws_feed:
        ws_connected = _ws_feed._task is not None and not _ws_feed._task.done()
        subscribed = len(_ws_feed._token_ids)
    return {
        "running": _price_monitor._running,
        "pending_edges": pending,
        "cooldowns": cooldowns,
        "ws_connected": ws_connected,
        "subscribed_tokens": subscribed,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: float) -> float | None:
    """Convert inf/-inf to None for JSON serialization."""
    if v == float("inf") or v == float("-inf"):
        return None
    return v


def _trade_to_dict(trade: dict) -> dict:
    """Convert a paper trade record to a JSON-serializable dict."""
    signal = trade["signal"]
    contract = trade["contract"]
    return {
        "trade_id": trade["trade_id"],
        "direction": signal.direction,
        "question": contract.question,
        "city": contract.city,
        "resolution_date": contract.resolution_date.isoformat(),
        "temp_bucket_low": _safe_float(contract.temp_bucket_low),
        "temp_bucket_high": _safe_float(contract.temp_bucket_high),
        "entry_price": trade["entry_price"],
        "amount_usd": trade["amount_usd"],
        "model_probability": trade.get("model_probability"),
        "resolved": trade["resolved"],
        "outcome": trade["outcome"],
        "pnl": trade["pnl"],
    }


def _entry_to_dict(entry) -> dict:
    """Convert a SignalLogEntry to a JSON-serializable dict."""
    return {
        "market_id": entry.signal.market_id,
        "direction": entry.signal.direction,
        "action": entry.signal.action,
        "edge": entry.signal.edge,
        "kelly_size": entry.signal.kelly_size,
        "signal_timestamp": entry.signal.timestamp.isoformat(),
        "station_id": entry.station_id,
        "regime": entry.regime.regime,
        "regime_confidence": entry.regime.confidence,
        "model_probability": entry.model_probability,
        "market_probability": entry.market_probability,
        "skip_reason": entry.signal.skip_reason,
        "logged_at": entry.timestamp.isoformat(),
    }


# ---------------------------------------------------------------------------
# Static files — serve built React frontend (if available)
# ---------------------------------------------------------------------------
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
