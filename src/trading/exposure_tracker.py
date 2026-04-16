"""Shared portfolio exposure tracking for concurrent trading paths."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ExposureTracker:
    """Running tally of portfolio exposure with drawdown circuit breaker.

    Safe for asyncio (single-threaded event loop). Both trading paths read
    ``current`` before sizing and call ``add()`` after successful execution.

    Drawdown guard: tracks cumulative realized P&L.  When losses exceed
    ``max_drawdown_pct`` of the bankroll, ``is_halted`` becomes True and
    no new trades should be placed until manually reset.
    """

    def __init__(
        self,
        bankroll: float = 300.0,
        initial: float = 0.0,
        max_drawdown_pct: float = 0.15,
        event_bus=None,
    ) -> None:
        self._bankroll = bankroll
        self._exposure = initial
        self._max_drawdown_pct = max_drawdown_pct
        self._realized_pnl = 0.0
        self._event_bus = event_bus

    @property
    def current(self) -> float:
        """Current total exposure in USD."""
        return self._exposure

    @property
    def realized_pnl(self) -> float:
        """Cumulative realized P&L since tracker creation."""
        return self._realized_pnl

    @property
    def is_halted(self) -> bool:
        """True if cumulative losses exceed the drawdown limit."""
        return self._realized_pnl < -(self._max_drawdown_pct * self._bankroll)

    def add(self, amount: float) -> None:
        """Increment exposure after a trade is executed."""
        self._exposure += amount
        if self._event_bus:
            self._event_bus.publish("exposure_change", {
                "current_exposure": round(self._exposure, 2),
                "realized_pnl": round(self._realized_pnl, 2),
                "is_halted": self.is_halted,
                "bankroll": self._bankroll,
            })

    def record_pnl(self, pnl: float, amount_usd: float = 0.0) -> None:
        """Record realized P&L from a resolved trade and release exposure.

        ``amount_usd`` is the original bet size that was added via ``add()``
        when the trade was placed.  Passing it here frees that capital so the
        portfolio exposure cap reflects only *open* positions.
        """
        self._realized_pnl += pnl
        self._exposure = max(0.0, self._exposure - amount_usd)
        if self._event_bus:
            self._event_bus.publish("exposure_change", {
                "current_exposure": round(self._exposure, 2),
                "realized_pnl": round(self._realized_pnl, 2),
                "is_halted": self.is_halted,
                "bankroll": self._bankroll,
            })
        if self.is_halted:
            logger.warning(
                "Drawdown circuit breaker tripped: realized P&L $%.2f "
                "exceeds %.0f%% of $%.0f bankroll — halting all trades",
                self._realized_pnl,
                self._max_drawdown_pct * 100,
                self._bankroll,
            )

    def reset(self, value: float) -> None:
        """Reset exposure to a known value (e.g., after reconciliation)."""
        self._exposure = value

    def reset_halt(self) -> None:
        """Manually clear the drawdown halt (e.g., after adding capital)."""
        self._realized_pnl = 0.0
        logger.info("Drawdown halt cleared — trading resumed")
