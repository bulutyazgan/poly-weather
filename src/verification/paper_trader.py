"""Paper trading tracker with PnL computation and counterfactual analysis.

Polymarket convention: buy YES or NO tokens at their price; they pay $1 if correct.
"""

from __future__ import annotations

import uuid

from src.data.models import MarketContract, TradingSignal


class PaperTrader:
    """Track paper trades and compute performance metrics."""

    def __init__(self, taker_fee_rate: float = 0.02) -> None:
        self.taker_fee_rate = taker_fee_rate
        self._trades: dict[str, dict] = {}  # trade_id → trade record
        self._counterfactuals: list[dict] = []

    def _compute_fee(self, entry_price: float, amount_usd: float, direction: str) -> float:
        """Polymarket taker fee: rate × min(price, 1−price) × shares.

        The fee is charged on entry regardless of outcome.
        """
        if direction == "BUY_YES":
            exec_price = entry_price
        else:  # BUY_NO — NO token costs (1 - YES_price)
            exec_price = 1.0 - entry_price

        if exec_price <= 0.0 or exec_price >= 1.0:
            return 0.0

        shares = amount_usd / exec_price
        return self.taker_fee_rate * min(exec_price, 1.0 - exec_price) * shares

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
        """Resolve a trade with the actual outcome. Returns net PnL after fees.

        BUY_YES:
          win  (outcome=True):  pnl = amount * (1/price - 1) - fee
          loss (outcome=False): pnl = -amount - fee
        BUY_NO (entry_price is the YES price; NO token costs 1 - entry_price):
          win  (outcome=False): pnl = amount * (1/(1 - entry_price) - 1) - fee
          loss (outcome=True):  pnl = -amount - fee

        Fee follows Polymarket's formula: taker_rate × min(price, 1−price) × shares.
        """
        trade = self._trades[trade_id]
        price = trade["entry_price"]
        amount = trade["amount_usd"]
        direction = trade["signal"].direction
        fee = self._compute_fee(price, amount, direction)

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

        pnl -= fee

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
        """Resolve a counterfactual with the actual outcome. Returns hypothetical net PnL."""
        cf = self._counterfactuals[index]
        price = cf["hypothetical_price"]
        amount = cf["hypothetical_size"]
        direction = cf["signal"].direction
        fee = self._compute_fee(price, amount, direction)

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

        pnl -= fee

        cf["resolved"] = True
        cf["outcome"] = outcome
        cf["pnl"] = pnl
        return pnl

    def get_counterfactuals(self) -> list[dict]:
        """Return all counterfactual records."""
        return list(self._counterfactuals)

    def total_pnl(self) -> float:
        """Sum PnL across all resolved trades."""
        return float(sum(t["pnl"] for t in self._trades.values() if t["resolved"]))

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
