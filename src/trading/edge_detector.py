"""Edge detection for weather prediction markets."""
from __future__ import annotations

from datetime import datetime, timezone

from src.data.models import RegimeClassification, TradingSignal


class EdgeDetector:
    """Evaluates whether a model-vs-market discrepancy is tradeable."""

    def __init__(
        self,
        high_threshold: float = 0.08,
        medium_threshold: float = 0.12,
        min_volume: float = 2000.0,
        min_hours: float = 2.0,
        max_market_certainty: float = 0.92,
        max_edge: float = 0.25,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.min_volume = min_volume
        self.min_hours = min_hours
        self.max_market_certainty = max_market_certainty
        self.max_edge = max_edge

    def evaluate(
        self,
        model_prob: float,
        market_prob: float,
        regime: RegimeClassification,
        volume_24h: float,
        hours_to_resolution: float,
        market_id: str = "",
        market_bid: float | None = None,
        market_ask: float | None = None,
    ) -> TradingSignal:
        """Check all entry conditions and return a TradingSignal.

        Edge is computed against the *execution price*, not the mid:
          BUY_YES edge = model_prob - ask   (we pay the ask to buy YES)
          BUY_NO  edge = bid - model_prob   (we sell YES at bid ≈ buy NO)

        When bid/ask are not available, falls back to ``|model_prob - mid|``.
        Direction is BUY_YES when model_prob > market_prob, else BUY_NO.
        """
        direction = "BUY_YES" if model_prob > market_prob else "BUY_NO"

        # Compute edge against execution price (spread-aware)
        if direction == "BUY_YES" and market_ask is not None:
            edge = model_prob - market_ask
        elif direction == "BUY_NO" and market_bid is not None:
            edge = market_bid - model_prob
        else:
            edge = abs(model_prob - market_prob)

        # Negative edge means the spread eats the entire advantage
        if edge < 0:
            edge = 0.0

        # Implausible edge guard: edges above max_edge almost certainly
        # indicate a model failure (e.g. 0% model prob against 50% market).
        #
        # Foregone conclusion guard: skip near-certain markets where the
        # price reflects information our forecast model doesn't have.
        if edge > self.max_edge:
            action = "SKIP"
        elif market_prob > self.max_market_certainty or market_prob < (1.0 - self.max_market_certainty):
            action = "SKIP"
        elif regime.confidence == "LOW":
            action = "SKIP"
        elif volume_24h < self.min_volume:
            action = "SKIP"
        elif hours_to_resolution < self.min_hours:
            action = "SKIP"
        else:
            threshold = (
                self.high_threshold
                if regime.confidence == "HIGH"
                else self.medium_threshold
            )
            action = "TRADE" if edge >= threshold else "SKIP"

        return TradingSignal(
            market_id=market_id,
            direction=direction,
            action=action,
            edge=edge,
            kelly_size=0.0,  # Sized separately by PositionSizer
            timestamp=datetime.now(timezone.utc),
        )
