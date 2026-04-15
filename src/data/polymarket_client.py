"""Polymarket API clients for weather prediction markets."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime, timezone

import httpx

from src.data.models import MarketContract, MarketPrice

logger = logging.getLogger(__name__)

# City name normalization mapping
CITY_MAP: dict[str, str] = {
    "New York City": "NYC",
    "New York": "NYC",
    "Chicago": "Chicago",
    "Chicago, IL": "Chicago",
    "Los Angeles": "LA",
    "Denver": "Denver",
    "Miami": "Miami",
}

# Month name → number
MONTH_MAP: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# Regex for temperature question parsing
_CITY_PATTERN = "|".join(re.escape(c) for c in sorted(CITY_MAP.keys(), key=len, reverse=True))
_DATE_PATTERN = r"(?P<month>\w+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
_BETWEEN_PATTERN = re.compile(
    rf"Will the high temperature in (?P<city>{_CITY_PATTERN})\s+on\s+{_DATE_PATTERN}\s+"
    rf"be between (?P<low>\d+)°F and (?P<high>\d+)°F\?",
    re.IGNORECASE,
)
_AT_OR_ABOVE_PATTERN = re.compile(
    rf"Will the high temperature in (?P<city>{_CITY_PATTERN})\s+on\s+{_DATE_PATTERN}\s+"
    rf"be at or above (?P<low>\d+)°F\?",
    re.IGNORECASE,
)
_BELOW_PATTERN = re.compile(
    rf"Will the high temperature in (?P<city>{_CITY_PATTERN})\s+on\s+{_DATE_PATTERN}\s+"
    rf"be below (?P<high>\d+)°F\?",
    re.IGNORECASE,
)


def _lazy_import_clob_client():
    """Lazy import to avoid importing py_clob_client unless needed."""
    from py_clob_client.client import ClobClient as _ClobClient
    return _ClobClient


# Alias for mocking in tests
PyClobClient = None  # Will be set on first live-mode init


class GammaClient:
    """Fetches market metadata from Polymarket Gamma API."""

    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch_weather_markets(self, active: bool = True) -> list[MarketContract]:
        """Fetch all active weather/temperature markets.

        Queries GET /markets with temperature tag filters.
        Parses each market question to extract weather data.
        Returns only "Yes" outcome tokens.
        """
        params: dict[str, str] = {}
        if active:
            params["active"] = "true"
            params["closed"] = "false"
        params["tag"] = "temperature"

        resp = await self.client.get("/markets", params=params)
        resp.raise_for_status()
        markets_data = resp.json()

        contracts: list[MarketContract] = []
        for market in markets_data:
            question = market.get("question", "")
            parsed = self.parse_temperature_question(question)
            if parsed is None:
                continue

            tokens = market.get("tokens", [])
            condition_id = market.get("condition_id", "")

            for token in tokens:
                if token.get("outcome") == "Yes":
                    contracts.append(
                        MarketContract(
                            token_id=token["token_id"],
                            condition_id=condition_id,
                            question=question,
                            city=parsed["city"],
                            resolution_date=parsed["resolution_date"],
                            temp_bucket_low=parsed["temp_bucket_low"],
                            temp_bucket_high=parsed["temp_bucket_high"],
                            outcome="Yes",
                        )
                    )
        return contracts

    @staticmethod
    def parse_temperature_question(question: str) -> dict | None:
        """Parse a Polymarket temperature question string.

        Returns dict with city, temp_bucket_low, temp_bucket_high, resolution_date
        or None if the question can't be parsed as a temperature market.
        """
        for pattern, handler in [
            (_BETWEEN_PATTERN, _handle_between),
            (_AT_OR_ABOVE_PATTERN, _handle_at_or_above),
            (_BELOW_PATTERN, _handle_below),
        ]:
            m = pattern.match(question)
            if m:
                return handler(m)
        return None


def _parse_date(m: re.Match) -> date | None:
    month_name = m.group("month")
    month = MONTH_MAP.get(month_name)
    if month is None:
        return None
    return date(int(m.group("year")), month, int(m.group("day")))


def _normalize_city(raw: str) -> str:
    return CITY_MAP.get(raw, raw)


def _handle_between(m: re.Match) -> dict | None:
    d = _parse_date(m)
    if d is None:
        return None
    return {
        "city": _normalize_city(m.group("city")),
        "temp_bucket_low": float(m.group("low")),
        "temp_bucket_high": float(m.group("high")),
        "resolution_date": d,
    }


def _handle_at_or_above(m: re.Match) -> dict | None:
    d = _parse_date(m)
    if d is None:
        return None
    return {
        "city": _normalize_city(m.group("city")),
        "temp_bucket_low": float(m.group("low")),
        "temp_bucket_high": float("inf"),
        "resolution_date": d,
    }


def _handle_below(m: re.Match) -> dict | None:
    d = _parse_date(m)
    if d is None:
        return None
    return {
        "city": _normalize_city(m.group("city")),
        "temp_bucket_low": float("-inf"),
        "temp_bucket_high": float(m.group("high")),
        "resolution_date": d,
    }


class CLOBClient:
    """Wrapper around py-clob-client for order management."""

    def __init__(self, api_url: str, private_key: str, paper_trading: bool = True):
        self.api_url = api_url
        self.paper_trading = paper_trading
        self.paper_orders: list[dict] = []
        self._http = httpx.AsyncClient(base_url=api_url, timeout=30.0)

        if not paper_trading:
            global PyClobClient
            if PyClobClient is None:
                PyClobClient = _lazy_import_clob_client()
            self.client = PyClobClient(api_url, key=private_key, chain_id=137)
        else:
            self.client = None

    async def close(self) -> None:
        await self._http.aclose()

    async def get_market_price(self, token_id: str) -> MarketPrice | None:
        """Get current best bid/ask and compute mid price."""
        book = await self._fetch_order_book(token_id)
        if book is None:
            return None

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            logger.warning("Empty order book for token %s", token_id)
            return None

        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        mid = (best_bid + best_ask) / 2.0

        # Sum volume for 24h approximation (using book depth as proxy)
        total_volume = sum(float(b["size"]) for b in bids) + sum(float(a["size"]) for a in asks)

        return MarketPrice(
            token_id=token_id,
            timestamp=datetime.now(timezone.utc),
            bid=best_bid,
            ask=best_ask,
            mid=mid,
            volume_24h=total_volume,
        )

    async def get_order_book_depth(self, token_id: str) -> dict:
        """Return total volume on each side of the book."""
        book = await self._fetch_order_book(token_id)
        if book is None:
            return {"bid_volume": 0.0, "ask_volume": 0.0}

        bid_volume = sum(float(b["size"]) for b in book.get("bids", []))
        ask_volume = sum(float(a["size"]) for a in book.get("asks", []))
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}

    async def place_limit_order(
        self, token_id: str, side: str, price: float, size: float
    ) -> str:
        """Place a limit order. Returns order_id.

        In paper mode: generate UUID, log the order, return UUID.
        In live mode: use py-clob-client to create and post a GTC limit order.
        """
        if self.paper_trading:
            order_id = str(uuid.uuid4())
            order = {
                "order_id": order_id,
                "token_id": token_id,
                "side": side,
                "price": price,
                "size": size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.paper_orders.append(order)
            logger.info("Paper order placed: %s", order)
            return order_id
        else:
            from py_clob_client.order_builder.constants import BUY, SELL

            order_side = BUY if side.upper() == "BUY" else SELL
            result = self.client.create_and_post_order(
                {
                    "tokenID": token_id,
                    "price": price,
                    "size": size,
                    "side": order_side,
                }
            )
            order_id = result.get("orderID", "")
            logger.info("Live order placed: %s", order_id)
            return order_id

    async def cancel_all_orders(self) -> int:
        """Cancel all outstanding orders. Returns count cancelled."""
        if self.paper_trading:
            count = len(self.paper_orders)
            self.paper_orders.clear()
            logger.info("Cancelled %d paper orders", count)
            return count
        else:
            result = self.client.cancel_all()
            cancelled = result.get("canceled_orders", [])
            count = len(cancelled)
            logger.info("Cancelled %d live orders", count)
            return count

    async def get_positions(self) -> list[dict]:
        """Get current open positions."""
        if self.paper_trading:
            # In paper mode, derive positions from paper orders
            return list(self.paper_orders)
        else:
            # In live mode, query the API
            resp = await self._http.get("/positions")
            resp.raise_for_status()
            return resp.json()

    async def _fetch_order_book(self, token_id: str) -> dict | None:
        """Fetch order book from CLOB API."""
        try:
            resp = await self._http.get("/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error("Failed to fetch order book for %s: %s", token_id, e)
            return None
