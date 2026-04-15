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
    ) -> TradingSignal:
        """Check all entry conditions and return a TradingSignal.

        Edge is always ``|model_prob - market_prob|``.
        Direction is BUY_YES when model_prob > market_prob, else BUY_NO.
        """
        edge = abs(model_prob - market_prob)
        direction = "BUY_YES" if model_prob > market_prob else "BUY_NO"

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
