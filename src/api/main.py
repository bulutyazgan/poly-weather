"""FastAPI application for the Polymarket Weather Prediction Trading System."""
from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config.stations import get_station, get_stations
from src.prediction.calibration import BrierScore, CUSUMMonitor, ReliabilityDiagram
from src.trading.executor import OrderExecutor
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


def set_state(
    log: PredictionLog | None,
    trader: PaperTrader | None,
    scheduler: PipelineScheduler | None,
    executor: OrderExecutor | None = None,
    cusum: CUSUMMonitor | None = None,
) -> None:
    """Inject shared state (called from main.py startup or tests)."""
    global _prediction_log, _paper_trader, _scheduler, _executor, _cusum
    _prediction_log = log
    _paper_trader = trader
    _scheduler = scheduler
    _executor = executor
    _cusum = cusum


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        "temp_bucket_low": contract.temp_bucket_low,
        "temp_bucket_high": contract.temp_bucket_high,
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
        "logged_at": entry.timestamp.isoformat(),
    }


# ---------------------------------------------------------------------------
# Static files — serve built React frontend (if available)
# ---------------------------------------------------------------------------
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
