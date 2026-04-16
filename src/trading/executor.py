"""Order execution for weather prediction markets."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.data.models import MarketPrice, TradeRecord, TradingSignal
from src.data.polymarket_client import CLOBClient

logger = logging.getLogger(__name__)

# Staleness thresholds
_STALE_MODEL_HOURS = 3.0
_STALE_MARKET_MOVE_PCT = 0.03  # 3%
_RESOLUTION_PROXIMITY_HOURS = 4.0
_ADVERSE_SELECTION_THRESHOLD = 0.10


class OrderExecutor:
    """Execute trading signals via the CLOB client."""

    def __init__(
        self,
        clob_client: CLOBClient,
        paper_trading: bool = True,
        order_ttl_seconds: float = 1800.0,
    ) -> None:
        if paper_trading and not clob_client.paper_trading:
            raise RuntimeError(
                "OrderExecutor is in paper mode but CLOBClient is live — "
                "this misconfiguration could result in real trades"
            )
        self.clob_client = clob_client
        self.paper_trading = paper_trading
        self.order_ttl_seconds = order_ttl_seconds
        self._fill_log: list[dict] = []
        self._open_orders: dict[str, datetime] = {}  # order_id → placed_at
        self._order_tokens: dict[str, str] = {}  # order_id → token_id

    async def execute(
        self,
        signal: TradingSignal,
        token_id: str,
        market_price: MarketPrice,
    ) -> TradeRecord | None:
        """Execute a trade signal. Returns TradeRecord or None if SKIP."""
        if signal.action == "SKIP":
            return None

        # Reject duplicate: don't stack orders on the same token
        if self.has_open_order(token_id):
            logger.info("Duplicate order rejected — already have open order on %s", token_id)
            return None

        # Always BUY — the caller passes the correct token_id
        # (YES token for BUY_YES, NO token for BUY_NO)
        side = "BUY"
        price = round(market_price.ask, 2)  # CLOB requires 0.01 tick size

        # Size is in USD; convert to shares: size_shares = amount_usd / price
        if price <= 0:
            logger.warning("Invalid price %s for token %s", price, token_id)
            return None

        size_shares = round(signal.kelly_size / price, 2)

        order_id = await self.clob_client.place_limit_order(
            token_id=token_id,
            side=side,
            price=price,
            size=size_shares,
        )

        now = datetime.now(timezone.utc)
        self._open_orders[order_id] = now
        self._order_tokens[order_id] = token_id

        return TradeRecord(
            trade_id=order_id,
            market_id=signal.market_id,
            direction=signal.direction,
            amount_usd=signal.kelly_size,
            price=price,
            timestamp=datetime.now(timezone.utc),
        )

    def has_open_order(self, token_id: str) -> bool:
        """True if there's already an open order on this token."""
        return token_id in self._order_tokens.values()

    def record_fill(
        self,
        token_id: str,
        filled: bool,
        pnl: float | None = None,
    ) -> None:
        """Record whether an order was filled and its P&L outcome."""
        self._fill_log.append({
            "token_id": token_id,
            "filled": filled,
            "pnl": pnl,
        })

    def get_fill_rate(self) -> float:
        """Fraction of orders that were filled. Returns 0.0 if no records."""
        if not self._fill_log:
            return 0.0
        filled_count = sum(1 for r in self._fill_log if r["filled"])
        return filled_count / len(self._fill_log)

    def get_adverse_selection_ratio(self) -> float | None:
        """Difference between filled and unfilled win rates.

        Returns filled_win_rate - unfilled_win_rate.
        Negative means filled orders lose more often → being picked off.
        Returns None if insufficient data.
        """
        filled_pnl = [r["pnl"] for r in self._fill_log if r["filled"] and r["pnl"] is not None]
        unfilled_pnl = [r["pnl"] for r in self._fill_log if not r["filled"] and r["pnl"] is not None]

        if not filled_pnl or not unfilled_pnl:
            return None

        filled_win_rate = sum(1 for p in filled_pnl if p > 0) / len(filled_pnl)
        unfilled_win_rate = sum(1 for p in unfilled_pnl if p > 0) / len(unfilled_pnl)
        return filled_win_rate - unfilled_win_rate

    def is_being_picked_off(self) -> bool:
        """True if adverse selection ratio is worse than threshold."""
        ratio = self.get_adverse_selection_ratio()
        if ratio is None:
            return False
        return ratio < -_ADVERSE_SELECTION_THRESHOLD

    async def cancel_expired_orders(self) -> int:
        """Cancel orders older than order_ttl_seconds.

        GTC orders in thin weather markets can sit for hours after the
        edge has evaporated.  This sweep cancels only the specific
        expired orders — fresh orders (e.g. from PriceMonitor) are
        preserved.

        Returns the number of orders cancelled.
        """
        if not self._open_orders:
            return 0

        now = datetime.now(timezone.utc)
        expired = [
            oid for oid, placed_at in self._open_orders.items()
            if (now - placed_at).total_seconds() > self.order_ttl_seconds
        ]

        if not expired:
            return 0

        logger.warning(
            "Cancelling %d expired orders (TTL=%.0fs), keeping %d active",
            len(expired), self.order_ttl_seconds,
            len(self._open_orders) - len(expired),
        )

        for oid in expired:
            await self.clob_client.cancel_order(oid)
            del self._open_orders[oid]
            self._order_tokens.pop(oid, None)

        return len(expired)

    async def check_stale_quotes(
        self,
        last_model_update: datetime,
        current_prices: dict[str, float],
        previous_prices: dict[str, float],
    ) -> bool:
        """If model update >3h old and market moved >3%, cancel affected orders.

        Only cancels orders on tokens whose prices moved significantly —
        orders on unaffected tokens (including fresh PriceMonitor orders)
        survive.

        Returns True if any orders were cancelled.
        """
        age = datetime.now(timezone.utc) - last_model_update
        if age < timedelta(hours=_STALE_MODEL_HOURS):
            return False

        # Find tokens with significant market moves
        stale_tokens: set[str] = set()
        for token_id, current in current_prices.items():
            previous = previous_prices.get(token_id)
            if previous is None or previous < 0.01:
                continue
            move = abs(current - previous) / previous
            if move > _STALE_MARKET_MOVE_PCT:
                stale_tokens.add(token_id)

        if not stale_tokens:
            return False

        # Cancel only orders on the moved tokens
        to_cancel = [
            oid for oid, tid in self._order_tokens.items()
            if tid in stale_tokens
        ]
        if not to_cancel:
            return False

        logger.warning(
            "Stale model (%.1fh) and %d tokens moved >%.0f%% — "
            "cancelling %d orders, keeping %d",
            age.total_seconds() / 3600,
            len(stale_tokens),
            _STALE_MARKET_MOVE_PCT * 100,
            len(to_cancel),
            len(self._open_orders) - len(to_cancel),
        )
        for oid in to_cancel:
            await self.clob_client.cancel_order(oid)
            del self._open_orders[oid]
            self._order_tokens.pop(oid, None)

        return True

    async def check_resolution_proximity(
        self,
        hours_to_resolution: float,
        token_ids: set[str] | None = None,
    ) -> bool:
        """Cancel orders on tokens approaching resolution.

        When ``token_ids`` is provided, only cancel orders matching those
        tokens — orders on other tokens (e.g. tomorrow's contracts) survive.
        Falls back to cancelling all orders if no token set is given.

        Returns True if any orders were cancelled.
        """
        if hours_to_resolution >= _RESOLUTION_PROXIMITY_HOURS:
            return False

        if token_ids is not None:
            # Targeted cancel — only kill orders on near-resolution tokens
            to_cancel = [
                oid for oid, tid in self._order_tokens.items()
                if tid in token_ids
            ]
            if not to_cancel:
                return False
            logger.warning(
                "%.1fh to resolution — cancelling %d orders on %d tokens",
                hours_to_resolution, len(to_cancel), len(token_ids),
            )
            for oid in to_cancel:
                await self.clob_client.cancel_order(oid)
                self._open_orders.pop(oid, None)
                self._order_tokens.pop(oid, None)
            return True

        # Fallback: cancel everything (legacy behavior)
        logger.warning(
            "%.1fh to resolution — cancelling all orders", hours_to_resolution
        )
        await self.clob_client.cancel_all_orders()
        self._open_orders.clear()
        self._order_tokens.clear()
        return True
