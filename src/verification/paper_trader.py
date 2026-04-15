"""Paper trading tracker with PnL computation and counterfactual analysis.

Polymarket convention: buy YES or NO tokens at their price; they pay $1 if correct.
"""

from __future__ import annotations

import uuid

from src.data.models import MarketContract, TradingSignal


class PaperTrader:
    """Track paper trades and compute performance metrics."""

    def __init__(self) -> None:
        self._trades: dict[str, dict] = {}  # trade_id → trade record
        self._counterfactuals: list[dict] = []

    def record_trade(
        self,
        signal: TradingSignal,
        contract: MarketContract,
        entry_price: float,
        amount_usd: float,
        model_probability: float | None = None,
    ) -> str:
        """Record a paper trade. Returns trade_id."""
        trade_id = str(uuid.uuid4())
        self._trades[trade_id] = {
            "trade_id": trade_id,
            "signal": signal,
            "contract": contract,
            "entry_price": entry_price,
            "amount_usd": amount_usd,
            "model_probability": model_probability,
            "resolved": False,
            "outcome": None,
            "pnl": None,
        }
        return trade_id

    def resolve(self, trade_id: str, outcome: bool) -> float:
        """Resolve a trade with the actual outcome. Returns PnL.

        BUY_YES:
          win  (outcome=True):  pnl = amount * (1/price - 1)
          loss (outcome=False): pnl = -amount
        BUY_NO (entry_price is the YES price; NO token costs 1 - entry_price):
          win  (outcome=False): pnl = amount * (1/(1 - entry_price) - 1)
          loss (outcome=True):  pnl = -amount
        """
        trade = self._trades[trade_id]
        price = trade["entry_price"]
        amount = trade["amount_usd"]
        direction = trade["signal"].direction

        if direction == "BUY_YES":
            if price <= 0.0 or price >= 1.0:
                pnl = -amount  # Treat extreme prices as loss to avoid div-by-zero
            else:
                pnl = amount * (1.0 / price - 1.0) if outcome else -amount
        else:  # BUY_NO
            if price <= 0.0 or price >= 1.0:
                pnl = -amount
            else:
                pnl = amount * (1.0 / (1.0 - price) - 1.0) if not outcome else -amount

        trade["resolved"] = True
        trade["outcome"] = outcome
        trade["pnl"] = pnl
        return pnl

    def record_counterfactual(
        self,
        signal: TradingSignal,
        contract: MarketContract,
        hypothetical_price: float,
        hypothetical_size: float,
    ) -> None:
        """Record what would have happened for a SKIP signal."""
        self._counterfactuals.append({
            "signal": signal,
            "contract": contract,
            "hypothetical_price": hypothetical_price,
            "hypothetical_size": hypothetical_size,
            "resolved": False,
            "outcome": None,
            "pnl": None,
        })

    def resolve_counterfactual(self, index: int, outcome: bool) -> float:
        """Resolve a counterfactual with the actual outcome. Returns hypothetical PnL."""
        cf = self._counterfactuals[index]
        price = cf["hypothetical_price"]
        amount = cf["hypothetical_size"]
        direction = cf["signal"].direction

        if direction == "BUY_YES":
            if price <= 0.0 or price >= 1.0:
                pnl = -amount
            else:
                pnl = amount * (1.0 / price - 1.0) if outcome else -amount
        else:  # BUY_NO
            if price <= 0.0 or price >= 1.0:
                pnl = -amount
            else:
                pnl = amount * (1.0 / (1.0 - price) - 1.0) if not outcome else -amount

        cf["resolved"] = True
        cf["outcome"] = outcome
        cf["pnl"] = pnl
        return pnl

    def get_counterfactuals(self) -> list[dict]:
        """Return all counterfactual records."""
        return list(self._counterfactuals)

    def total_pnl(self) -> float:
        """Sum PnL across all resolved trades."""
        return sum(t["pnl"] for t in self._trades.values() if t["resolved"])

    def win_rate(self) -> float:
        """Fraction of resolved trades with positive PnL."""
        resolved = [t for t in self._trades.values() if t["resolved"]]
        if not resolved:
            return 0.0
        wins = sum(1 for t in resolved if t["pnl"] > 0)
        return wins / len(resolved)

    def get_resolved_trades(self) -> list[dict]:
        """Return all resolved trade records."""
        return [t for t in self._trades.values() if t["resolved"]]
