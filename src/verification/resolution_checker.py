"""Resolve paper trades by checking Polymarket outcomes.

Polls the Gamma API for resolved markets and calls paper_trader.resolve()
for each trade whose market has settled.
"""
from __future__ import annotations

import logging
from src.data.polymarket_client import GammaClient
from src.verification.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class ResolutionChecker:
    """Check pending paper trades for resolution and compute PnL."""

    def __init__(self, gamma: GammaClient, paper_trader: PaperTrader) -> None:
        self._gamma = gamma
        self._paper_trader = paper_trader

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
            resolved += 1
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
