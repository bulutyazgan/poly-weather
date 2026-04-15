"""Order execution for weather prediction markets."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

from src.data.models import MarketPrice, TradeRecord, TradingSignal
from src.data.polymarket_client import CLOBClient

logger = logging.getLogger(__name__)

# Staleness thresholds
_STALE_MODEL_HOURS = 3.0
_STALE_MARKET_MOVE_PCT = 0.03  # 3%
_RESOLUTION_PROXIMITY_HOURS = 4.0


class OrderExecutor:
    """Execute trading signals via the CLOB client."""

    def __init__(self, clob_client: CLOBClient, paper_trading: bool = True) -> None:
        if paper_trading and not clob_client.paper_trading:
            raise RuntimeError(
                "OrderExecutor is in paper mode but CLOBClient is live — "
                "this misconfiguration could result in real trades"
            )
        self.clob_client = clob_client
        self.paper_trading = paper_trading

    async def execute(
        self,
        signal: TradingSignal,
        token_id: str,
        market_price: MarketPrice,
    ) -> TradeRecord | None:
        """Execute a trade signal. Returns TradeRecord or None if SKIP."""
        if signal.action == "SKIP":
            return None

        # Determine order side and price
        if signal.direction == "BUY_YES":
            side = "BUY"
            price = market_price.ask  # lift the ask
        else:
            # BUY_NO means SELL the YES token
            side = "SELL"
            price = market_price.bid

        # Size is in USD; convert to shares: size_shares = amount_usd / price
        if price <= 0:
            logger.warning("Invalid price %s for token %s", price, token_id)
            return None

        size_shares = signal.kelly_size / price

        order_id = await self.clob_client.place_limit_order(
            token_id=token_id,
            side=side,
            price=price,
            size=size_shares,
        )

        return TradeRecord(
            trade_id=order_id,
            market_id=signal.market_id,
            direction=signal.direction,
            amount_usd=signal.kelly_size,
            price=price,
            timestamp=datetime.now(timezone.utc),
        )

    async def check_stale_quotes(
        self,
        last_model_update: datetime,
        current_prices: dict[str, float],
        previous_prices: dict[str, float],
    ) -> bool:
        """If model update >3h old and any market moved >3%, cancel all.

        Returns True if orders were cancelled.
        """
        age = datetime.now(timezone.utc) - last_model_update
        if age < timedelta(hours=_STALE_MODEL_HOURS):
            return False

        # Check for significant market moves
        for token_id, current in current_prices.items():
            previous = previous_prices.get(token_id)
            if previous is None or previous < 0.01:
                continue
            move = abs(current - previous) / previous
            if move > _STALE_MARKET_MOVE_PCT:
                logger.warning(
                    "Stale model (%.1fh) and market moved %.1f%% for %s — cancelling all orders",
                    age.total_seconds() / 3600,
                    move * 100,
                    token_id,
                )
                await self.clob_client.cancel_all_orders()
                return True

        return False

    async def check_resolution_proximity(self, hours_to_resolution: float) -> bool:
        """If <4h to resolution, cancel all orders. Returns True if cancelled."""
        if hours_to_resolution < _RESOLUTION_PROXIMITY_HOURS:
            logger.warning(
                "%.1fh to resolution — cancelling all orders", hours_to_resolution
            )
            await self.clob_client.cancel_all_orders()
            return True
        return False
