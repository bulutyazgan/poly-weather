"""Resolve paper trades by checking Polymarket outcomes.

Polls the Gamma API for resolved markets and calls paper_trader.resolve()
for each trade whose market has settled.
"""
from __future__ import annotations

import logging
from src.data.polymarket_client import GammaClient
from src.trading.exposure_tracker import ExposureTracker
from src.verification.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class ResolutionChecker:
    """Check pending paper trades for resolution and compute PnL."""

    def __init__(
        self,
        gamma: GammaClient,
        paper_trader: PaperTrader,
        exposure_tracker: ExposureTracker | None = None,
        event_bus=None,
    ) -> None:
        self._gamma = gamma
        self._paper_trader = paper_trader
        self._exposure_tracker = exposure_tracker
        self._event_bus = event_bus

    async def check_resolutions(self) -> dict:
        """Check all unresolved trades for market resolution.

        Returns summary: {checked, resolved, errors}.
        """
        checked = 0
        resolved = 0
        errors = 0

        unresolved = self._get_unresolved_trades()
        if not unresolved:
            return {"checked": 0, "resolved": 0, "errors": 0}

        logger.info("Checking %d unresolved trades for resolution", len(unresolved))

        for trade_id, trade in unresolved:
            checked += 1
            contract = trade["contract"]

            try:
                outcome = await self._gamma.fetch_market_resolution(
                    contract.condition_id
                )
            except Exception:
                logger.exception(
                    "Failed to check resolution for trade %s (condition %s)",
                    trade_id, contract.condition_id,
                )
                errors += 1
                continue

            if outcome is None:
                # Not yet resolved
                continue

            pnl = self._paper_trader.resolve(trade_id, outcome)
            if self._exposure_tracker is not None:
                self._exposure_tracker.record_pnl(pnl, amount_usd=trade["amount_usd"])
            resolved += 1
            if self._event_bus:
                self._event_bus.publish("trade_resolved", {
                    "trade_id": trade_id,
                    "outcome": outcome,
                    "pnl": round(pnl, 2),
                    "direction": trade["signal"].direction,
                    "city": contract.city,
                })
            logger.info(
                "Resolved trade %s: outcome=%s, pnl=$%.2f, direction=%s, market=%s",
                trade_id,
                outcome,
                pnl,
                trade["signal"].direction,
                contract.question,
            )

        summary = {"checked": checked, "resolved": resolved, "errors": errors}
        logger.info("Resolution check complete: %s", summary)
        return summary

    def _get_unresolved_trades(self) -> list[tuple[str, dict]]:
        """Return list of (trade_id, trade_dict) for unresolved trades."""
        return [
            (tid, t)
            for tid, t in self._paper_trader._trades.items()
            if not t["resolved"]
        ]
