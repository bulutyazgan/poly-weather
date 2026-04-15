"""Schedule pipeline runs around NWP model update times."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time as dt_time, timedelta, timezone

from src.orchestrator.pipeline import TradingPipeline

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

    def __init__(self, pipeline: TradingPipeline) -> None:
        self.pipeline = pipeline
        self._events: list[dict] = self._build_events()
        self._running = False
        self._task: asyncio.Task | None = None

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

        return events

    def get_scheduled_events(self) -> list[dict]:
        """Return list of {time: dt_time, event_type: str, description: str}."""
        return list(self._events)

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
        """Execute a scheduled event by running a pipeline cycle."""
        logger.info("Running scheduled event: %s", event_type)
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
        """Main scheduling loop — sleep until next event, run pipeline, repeat."""
        sorted_times = self._get_sorted_times()
        while self._running:
            # Find next event
            next_wait = float("inf")
            next_event = "unknown"
            for t, event_type in sorted_times:
                wait = self._seconds_until(t)
                if wait < next_wait:
                    next_wait = wait
                    next_event = event_type

            logger.info(
                "Next event: %s in %.0f minutes",
                next_event,
                next_wait / 60,
            )
            await asyncio.sleep(next_wait)

            if not self._running:
                break

            try:
                await self.run_event(next_event, **pipeline_kwargs)
            except Exception:
                logger.exception("Scheduled event %s failed", next_event)
