"""FastAPI application for the Polymarket Weather Prediction Trading System."""
from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config.stations import get_station, get_stations
from src.prediction.calibration import BrierScore, ReliabilityDiagram
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


def set_state(
    log: PredictionLog | None,
    trader: PaperTrader | None,
    scheduler: PipelineScheduler | None,
) -> None:
    """Inject shared state (called from main.py startup or tests)."""
    global _prediction_log, _paper_trader, _scheduler
    _prediction_log = log
    _paper_trader = trader
    _scheduler = scheduler


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
        price = t["entry_price"]
        # Use entry price as the model's implied probability
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
