"""Schedule pipeline runs around NWP model update times.

Also ensures the bot re-evaluates at least every MARKET_REFRESH_MINUTES
even when no NWP update is due, so market price changes and new contracts
are picked up continuously.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time as dt_time, timedelta, timezone

from src.orchestrator.pipeline import TradingPipeline
from src.verification.resolution_checker import ResolutionChecker

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Schedule pipeline runs around NWP model update times.

    GFS model runs: 00, 06, 12, 18 UTC. Data available ~4-5h after init.
    ECMWF model runs: 00, 12 UTC. Data available ~6-8h after init.
    """

    GFS_RUN_TIMES = [
        dt_time(4, 30),   # ~4.5h after 00Z init
        dt_time(10, 30),  # ~4.5h after 06Z init
        dt_time(16, 30),  # ~4.5h after 12Z init
        dt_time(22, 30),  # ~4.5h after 18Z init
    ]
    ECMWF_RUN_TIMES = [
        dt_time(6, 0),    # ~6h after 00Z init
        dt_time(18, 0),   # ~6h after 12Z init
    ]
    MORNING_REFINEMENT = dt_time(14, 30)  # After 12Z HRRR available
    RESOLUTION_CHECK = dt_time(2, 0)  # 02:00 UTC — after overnight settlement
    MARKET_REFRESH_MINUTES: int = 90  # max gap between pipeline runs

    def __init__(
        self,
        pipeline: TradingPipeline,
        resolution_checker: ResolutionChecker | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._resolution_checker = resolution_checker
        self._events: list[dict] = self._build_events()
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_run_time: datetime | None = None
        self._last_run_result: dict | None = None
        self._last_error: str | None = None
        self._next_event_type: str = ""
        self._next_event_time: datetime | None = None

    def _build_events(self) -> list[dict]:
        """Build the full list of scheduled events."""
        events: list[dict] = []

        for t in self.GFS_RUN_TIMES:
            events.append({
                "time": t,
                "event_type": "gfs_update",
                "description": f"GFS model update available at {t.strftime('%H:%M')} UTC",
            })

        for t in self.ECMWF_RUN_TIMES:
            events.append({
                "time": t,
                "event_type": "ecmwf_update",
                "description": f"ECMWF model update available at {t.strftime('%H:%M')} UTC",
            })

        events.append({
            "time": self.MORNING_REFINEMENT,
            "event_type": "morning_refinement",
            "description": "Same-day refinement after 12Z HRRR available",
        })

        if self._resolution_checker is not None:
            events.append({
                "time": self.RESOLUTION_CHECK,
                "event_type": "resolution_check",
                "description": "Check resolved markets and compute PnL",
            })

        return events

    def get_scheduled_events(self) -> list[dict]:
        """Return list of {time: dt_time, event_type: str, description: str}."""
        return list(self._events)

    @property
    def status(self) -> dict:
        """Return scheduler health for the /api/status endpoint."""
        now = datetime.now(tz=timezone.utc)
        next_min = None
        if self._next_event_time is not None:
            next_min = max(0.0, (self._next_event_time - now).total_seconds() / 60.0)
        return {
            "running": self._running,
            "last_run_time": self._last_run_time.isoformat() if self._last_run_time else None,
            "last_run_result": self._last_run_result,
            "last_error": self._last_error,
            "next_event_type": self._next_event_type,
            "next_event_time": self._next_event_time.isoformat() if self._next_event_time else None,
            "next_event_minutes": round(next_min, 1) if next_min is not None else None,
        }

    def _get_sorted_times(self) -> list[tuple[dt_time, str]]:
        """Return all event times sorted, with event type."""
        times = []
        for e in self._events:
            times.append((e["time"], e["event_type"]))
        times.sort(key=lambda x: x[0])
        return times

    def _seconds_until(self, target: dt_time) -> float:
        """Seconds from now until next occurrence of target time (UTC)."""
        now = datetime.now(tz=timezone.utc)
        today_target = datetime.combine(now.date(), target, tzinfo=timezone.utc)
        if today_target <= now:
            today_target += timedelta(days=1)
        return (today_target - now).total_seconds()

    async def run_event(self, event_type: str, **kwargs) -> dict:
        """Execute a scheduled event.

        Resolution checks run before every pipeline cycle (not just at
        02:00 UTC) so that resolved trades free exposure immediately.
        Without this, the $60 exposure cap fills after 2 pipeline cycles
        and blocks all trading for the remaining 13+ cycles of the day.
        """
        logger.info("Running scheduled event: %s", event_type)

        if event_type == "resolution_check" and self._resolution_checker is not None:
            result = await self._resolution_checker.check_resolutions()
        else:
            # Always check resolutions before the pipeline cycle so
            # exposure freed by settled markets is available for new trades.
            if self._resolution_checker is not None:
                try:
                    res_result = await self._resolution_checker.check_resolutions()
                    if res_result.get("resolved", 0) > 0:
                        logger.info(
                            "Pre-cycle resolution check freed %d trades",
                            res_result["resolved"],
                        )
                except Exception:
                    logger.exception("Pre-cycle resolution check failed")
            result = await self.pipeline.run_cycle(**kwargs)

        logger.info("Event %s complete: %s", event_type, result)
        return result

    async def start(self, **pipeline_kwargs) -> None:
        """Start the scheduling loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._loop(**pipeline_kwargs))
        logger.info("Scheduler started with %d events", len(self._events))

    async def stop(self) -> None:
        """Stop the scheduling loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduler stopped")

    async def _loop(self, **pipeline_kwargs) -> None:
        """Main scheduling loop — sleep until next event, run pipeline, repeat.

        If the next NWP event is more than MARKET_REFRESH_MINUTES away,
        the loop wakes early for a ``market_refresh`` cycle so the bot
        re-evaluates with current market prices and discovers new contracts.
        """
        sorted_times = self._get_sorted_times()
        while self._running:
            # Find next NWP event
            next_wait = float("inf")
            next_event = "unknown"
            for t, event_type in sorted_times:
                wait = self._seconds_until(t)
                if wait < next_wait:
                    next_wait = wait
                    next_event = event_type

            # Cap at MARKET_REFRESH_MINUTES so the bot never goes silent
            max_wait = self.MARKET_REFRESH_MINUTES * 60
            if next_wait > max_wait:
                next_wait = max_wait
                next_event = "market_refresh"

            now = datetime.now(tz=timezone.utc)
            self._next_event_type = next_event
            self._next_event_time = now + timedelta(seconds=next_wait)

            logger.info(
                "Next event: %s in %.0f minutes",
                next_event,
                next_wait / 60,
            )
            await asyncio.sleep(next_wait)

            if not self._running:
                break

            try:
                result = await self.run_event(next_event, **pipeline_kwargs)
                self._last_run_time = datetime.now(tz=timezone.utc)
                self._last_run_result = result
                self._last_error = None
            except Exception as exc:
                logger.exception("Scheduled event %s failed", next_event)
                self._last_run_time = datetime.now(tz=timezone.utc)
                self._last_error = f"{next_event}: {exc}"
