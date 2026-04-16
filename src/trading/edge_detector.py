"""Edge detection for weather prediction markets."""
from __future__ import annotations

from datetime import datetime, timezone

from src.data.models import RegimeClassification, TradingSignal


class EdgeDetector:
    """Evaluates whether a model-vs-market discrepancy is tradeable."""

    def __init__(
        self,
        high_threshold: float = 0.05,
        medium_threshold: float = 0.08,
        min_volume: float = 200.0,
        min_hours: float = 2.0,
        max_market_certainty: float = 0.92,
        max_edge: float = 0.25,
        taker_fee_rate: float = 0.02,
        max_spread: float = 0.10,
        min_execution_price: float = 0.10,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.min_volume = min_volume
        self.min_hours = min_hours
        self.max_market_certainty = max_market_certainty
        self.max_edge = max_edge
        self.taker_fee_rate = taker_fee_rate
        self.max_spread = max_spread
        self.min_execution_price = min_execution_price

    def _is_foregone_conclusion(
        self,
        model_prob: float,
        market_prob: float,
    ) -> bool:
        """Check if a market is a foregone conclusion we should avoid.

        Directional logic: only skip when model AGREES with the extreme
        market price.  When model disagrees (e.g. market=4%, model=28%),
        that's a tail value bet — exactly where weather models find edge.

        Skip when both model and market agree the outcome is near-certain
        or near-impossible.  Allow when model disagrees with an extreme
        market price — that disagreement IS the edge.
        """
        low_bound = 1.0 - self.max_market_certainty  # 0.08
        high_bound = self.max_market_certainty  # 0.92

        if market_prob > high_bound:
            # Market very confident YES.  Only skip if model also agrees.
            return model_prob > high_bound
        if market_prob < low_bound:
            # Market says very unlikely.  Only skip if model also agrees.
            return model_prob < low_bound
        return False

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
        **kwargs,
    ) -> TradingSignal:
        """Check all entry conditions and return a TradingSignal.

        Edge is computed against the *execution price*, not the mid:
          BUY_YES edge = model_prob - ask - fee   (we pay the ask to buy YES)
          BUY_NO  edge = bid - model_prob - fee   (we sell YES at bid ≈ buy NO)

        Fee per share follows Polymarket's formula:
          fee = taker_fee_rate × min(price, 1 − price)

        When bid/ask are not available, falls back to ``|model_prob - mid| - fee``.
        Direction is BUY_YES when model_prob > market_prob, else BUY_NO.
        """
        direction = "BUY_YES" if model_prob > market_prob else "BUY_NO"

        # Compute edge against execution price (spread-aware) minus taker fee
        if direction == "BUY_YES" and market_ask is not None:
            raw_edge = model_prob - market_ask
            fee = self.taker_fee_rate * min(market_ask, 1.0 - market_ask)
        elif direction == "BUY_NO" and market_bid is not None:
            raw_edge = market_bid - model_prob
            fee = self.taker_fee_rate * min(market_bid, 1.0 - market_bid)
        else:
            raw_edge = abs(model_prob - market_prob)
            fee = self.taker_fee_rate * min(market_prob, 1.0 - market_prob)

        edge = raw_edge - fee

        # Negative edge means spread + fees eat the entire advantage.
        # Track whether we had positive raw_edge but fees killed it —
        # this distinguishes "no model edge" from "edge exists but
        # uneconomical to capture" in the prediction log.
        spread_killed = edge < 0 and raw_edge > 0
        if edge < 0:
            edge = 0.0

        # Implausible edge guard: check raw_edge (before fees) because this
        # detects model failures — huge model-vs-market disagreement that fees
        # could otherwise mask.  A raw_edge of 0.26 with a 0.01 fee would slip
        # through a net-edge check at 0.25.
        #
        # Foregone conclusion guard: skip near-certain markets where the
        # price reflects information our forecast model doesn't have.
        # Spread guard: wide spreads signal illiquidity — real execution will
        # suffer slippage beyond what our edge calculation assumes.
        spread = (market_ask - market_bid) if (market_bid is not None and market_ask is not None) else 0.0

        # Execution price: what we actually pay per share
        if direction == "BUY_YES" and market_ask is not None:
            exec_price = market_ask
        elif direction == "BUY_NO" and market_bid is not None:
            exec_price = 1.0 - market_bid
        else:
            exec_price = min(market_prob, 1.0 - market_prob)

        skip_reason = ""
        if spread > self.max_spread:
            action = "SKIP"
            skip_reason = "wide_spread"
        elif exec_price < self.min_execution_price:
            action = "SKIP"
            skip_reason = "penny_option"
        elif raw_edge > self.max_edge:
            action = "SKIP"
            skip_reason = "implausible_edge"
        elif self._is_foregone_conclusion(model_prob, market_prob):
            action = "SKIP"
            skip_reason = "foregone_conclusion"
        elif regime.confidence == "LOW":
            action = "SKIP"
            skip_reason = "low_confidence_regime"
        elif volume_24h < self.min_volume:
            action = "SKIP"
            skip_reason = "low_volume"
        elif hours_to_resolution < self.min_hours:
            action = "SKIP"
            skip_reason = "too_close_to_resolution"
        else:
            threshold = (
                self.high_threshold
                if regime.confidence == "HIGH"
                else self.medium_threshold
            )
            if edge >= threshold:
                action = "TRADE"
            elif spread_killed:
                action = "SKIP"
                skip_reason = "spread_kills_edge"
            else:
                action = "SKIP"
                skip_reason = "below_threshold"

        return TradingSignal(
            market_id=market_id,
            direction=direction,
            action=action,
            edge=edge,
            kelly_size=0.0,  # Sized separately by PositionSizer
            timestamp=datetime.now(timezone.utc),
            skip_reason=skip_reason,
        )
