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
