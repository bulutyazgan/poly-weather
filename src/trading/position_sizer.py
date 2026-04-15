"""Position sizing using fractional Kelly criterion."""
from __future__ import annotations


class PositionSizer:
    """Compute position size in USD using fractional Kelly with caps."""

    def __init__(
        self,
        kelly_fraction: float = 0.08,
        max_trade_usd: float = 3.0,
        max_bankroll_pct: float = 0.03,
        max_portfolio_exposure: float = 0.20,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_trade_usd = max_trade_usd
        self.max_bankroll_pct = max_bankroll_pct
        self.max_portfolio_exposure = max_portfolio_exposure

    def compute(
        self,
        edge: float,
        market_prob: float,
        bankroll: float,
        current_exposure: float = 0.0,
        ensemble_spread_pctile: float = 50.0,
        direction: str = "BUY_YES",
    ) -> float:
        """Compute position size in USD.

        Kelly formula:
            BUY_YES: kelly = edge / (1 - market_prob)
            BUY_NO:  kelly = edge / market_prob

        Raw size = kelly_fraction * kelly * bankroll

        Spread scaling:
            - spread_pctile < 20  -> multiply by 1.2 (high confidence)
            - spread_pctile > 50  -> multiply by 0.8 (low confidence)

        Caps applied (minimum of):
            - max_trade_usd
            - max_bankroll_pct * bankroll
            - max_portfolio_exposure * bankroll - current_exposure
        """
        if edge <= 0:
            return 0.0

        if direction == "BUY_YES":
            denom = 1.0 - market_prob
        else:  # BUY_NO
            denom = market_prob

        if denom <= 0:
            return 0.0

        kelly = edge / denom
        raw_size = self.kelly_fraction * kelly * bankroll

        # Spread scaling
        if ensemble_spread_pctile < 20:
            raw_size *= 1.2
        elif ensemble_spread_pctile > 50:
            raw_size *= 0.8

        # Apply caps
        remaining_exposure = self.max_portfolio_exposure * bankroll - current_exposure
        if remaining_exposure <= 0:
            return 0.0

        size = min(
            raw_size,
            self.max_trade_usd,
            self.max_bankroll_pct * bankroll,
            remaining_exposure,
        )

        return max(size, 0.0)
