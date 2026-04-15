"""Polymarket API clients for weather prediction markets."""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx

from src.data.models import MarketContract, MarketPrice

logger = logging.getLogger(__name__)

# City name normalization mapping (Polymarket question text → our city key)
CITY_MAP: dict[str, str] = {
    "New York City": "NYC",
    "New York": "NYC",
    "Chicago": "Chicago",
    "Chicago, IL": "Chicago",
    "Los Angeles": "LA",
    "Denver": "Denver",
    "Miami": "Miami",
}

# Our city key → Polymarket event slug fragment (primary + alternates)
CITY_SLUG_MAP: dict[str, str] = {
    "NYC": "nyc",
    "Chicago": "chicago",
    "LA": "los-angeles",
    "Denver": "denver",
    "Miami": "miami",
}

# Alternate slug fragments to try when the primary slug returns no results.
# Polymarket sometimes uses different naming conventions for the same city.
_CITY_SLUG_ALTERNATES: dict[str, list[str]] = {
    "NYC": ["new-york-city", "new-york"],
    "LA": ["la", "los-angeles-ca"],
    "Chicago": ["chicago-il"],
    "Denver": ["denver-co"],
    "Miami": ["miami-fl"],
}

# Month number → lowercase name for slug construction
_MONTH_NAMES = [
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

# Regex for parsing groupItemTitle temperature buckets
_BUCKET_RANGE = re.compile(r"(\d+)-(\d+)°F")
_BUCKET_OR_BELOW = re.compile(r"(\d+)°F or below")
_BUCKET_OR_HIGHER = re.compile(r"(\d+)°F or higher")


def _safe_float(val: object) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _lazy_import_clob_client():
    """Lazy import to avoid importing py_clob_client unless needed."""
    from py_clob_client.client import ClobClient as _ClobClient
    return _ClobClient


# Alias for mocking in tests
PyClobClient = None  # Will be set on first live-mode init


def _build_event_slug(city: str, target_date: date, city_slug: str | None = None) -> str:
    """Build the Polymarket event slug for a city/date temperature market."""
    if city_slug is None:
        city_slug = CITY_SLUG_MAP.get(city)
    if city_slug is None:
        raise ValueError(f"No slug mapping for city: {city}")
    month_name = _MONTH_NAMES[target_date.month]
    return f"highest-temperature-in-{city_slug}-on-{month_name}-{target_date.day}-{target_date.year}"


def _parse_bucket(label: str) -> tuple[float, float] | None:
    """Parse a groupItemTitle like '80-81°F', '79°F or below', '98°F or higher'."""
    m = _BUCKET_RANGE.search(label)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _BUCKET_OR_BELOW.search(label)
    if m:
        return float("-inf"), float(m.group(1))
    m = _BUCKET_OR_HIGHER.search(label)
    if m:
        return float(m.group(1)), float("inf")
    return None


class GammaClient:
    """Fetches market metadata from Polymarket Gamma API."""

    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch_weather_markets(
        self,
        cities: list[str] | None = None,
        lookahead_days: int = 3,
    ) -> list[MarketContract]:
        """Fetch active temperature markets by constructing event slugs.

        Queries the Gamma API /events endpoint with exact slug lookups
        for each city/date combination. Returns Yes-outcome contracts
        for all temperature buckets found.

        Args:
            cities: City keys to query (defaults to all in CITY_SLUG_MAP).
            lookahead_days: Number of days from today to query (default 3).
        """
        if cities is None:
            cities = list(CITY_SLUG_MAP.keys())

        today = datetime.now(tz=timezone.utc).date()
        target_dates = [today + timedelta(days=d) for d in range(lookahead_days)]

        contracts: list[MarketContract] = []
        for city in cities:
            for target_date in target_dates:
                slug = _build_event_slug(city, target_date)
                event_contracts = await self._fetch_event_contracts(
                    slug, city, target_date
                )

                # If primary slug found nothing, try alternate slugs
                if not event_contracts:
                    for alt_slug in _CITY_SLUG_ALTERNATES.get(city, []):
                        alt_full = _build_event_slug(city, target_date, city_slug=alt_slug)
                        event_contracts = await self._fetch_event_contracts(
                            alt_full, city, target_date
                        )
                        if event_contracts:
                            logger.info(
                                "Found %s contracts via alternate slug %s",
                                city, alt_slug,
                            )
                            break

                if not event_contracts:
                    logger.warning(
                        "No contracts found for %s on %s (tried primary + alternates)",
                        city, target_date,
                    )

                contracts.extend(event_contracts)

        logger.info(
            "Fetched %d temperature contracts across %d cities, %d dates",
            len(contracts), len(cities), len(target_dates),
        )
        return contracts

    async def _fetch_event_contracts(
        self, slug: str, city: str, target_date: date
    ) -> list[MarketContract]:
        """Fetch contracts for a single event slug."""
        try:
            resp = await self.client.get("/events", params={"slug": slug})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Gamma API request failed for %s: %s", slug, exc)
            return []

        events = resp.json()
        if not events:
            logger.debug("No event found for slug: %s", slug)
            return []

        event = events[0]
        markets = event.get("markets", [])
        contracts: list[MarketContract] = []

        for market in markets:
            if market.get("closed"):
                continue

            label = market.get("groupItemTitle", "")
            bucket = _parse_bucket(label)
            if bucket is None:
                continue

            token_ids_raw = market.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(token_ids_raw)
            except (json.JSONDecodeError, TypeError):
                continue

            if not token_ids:
                continue

            # First token ID is the Yes outcome, second is No
            yes_token_id = token_ids[0]
            no_token_id = token_ids[1] if len(token_ids) > 1 else ""
            condition_id = market.get("conditionId", "")
            question = market.get("question", label)
            # Prefer 24h volume; fall back to total volume
            volume_str = market.get("volume24hr", market.get("volume", "0"))
            try:
                volume_24h = float(volume_str)
            except (ValueError, TypeError):
                volume_24h = 0.0

            # Extract Gamma-aggregated prices (far more reliable than raw CLOB book)
            gamma_best_bid = _safe_float(market.get("bestBid"))
            gamma_best_ask = _safe_float(market.get("bestAsk"))
            gamma_last_trade = _safe_float(market.get("lastTradePrice"))
            gamma_outcome_price: float | None = None
            outcome_prices_raw = market.get("outcomePrices")
            if outcome_prices_raw:
                try:
                    prices_list = json.loads(outcome_prices_raw) if isinstance(
                        outcome_prices_raw, str
                    ) else outcome_prices_raw
                    if prices_list:
                        gamma_outcome_price = float(prices_list[0])
                except (json.JSONDecodeError, TypeError, ValueError, IndexError):
                    pass

            # Parse exact resolution time from Gamma endDate
            end_date_utc = None
            end_date_str = market.get("endDate")
            if end_date_str:
                try:
                    end_date_utc = datetime.fromisoformat(
                        end_date_str.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            contracts.append(
                MarketContract(
                    token_id=yes_token_id,
                    no_token_id=no_token_id,
                    condition_id=condition_id,
                    question=question,
                    city=city,
                    resolution_date=target_date,
                    end_date_utc=end_date_utc,
                    temp_bucket_low=bucket[0],
                    temp_bucket_high=bucket[1],
                    outcome="Yes",
                    volume_24h=volume_24h,
                    gamma_best_bid=gamma_best_bid,
                    gamma_best_ask=gamma_best_ask,
                    gamma_outcome_price=gamma_outcome_price,
                    gamma_last_trade=gamma_last_trade,
                )
            )

        return contracts


    async def fetch_market_resolution(self, condition_id: str) -> bool | None:
        """Check if a market has resolved and return the YES outcome.

        Queries the Gamma API /markets endpoint by condition_id.
        Returns True if YES won, False if NO won, None if not yet resolved.
        """
        try:
            resp = await self.client.get(
                "/markets", params={"condition_id": condition_id}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Gamma resolution check failed for %s: %s", condition_id, exc)
            return None

        markets = resp.json()
        if not markets:
            return None

        market = markets[0] if isinstance(markets, list) else markets
        if not market.get("closed"):
            return None

        # Polymarket resolves to a winning token.  The 'winner' field
        # contains the winning token_id, or 'resolved_to' / 'resolution'
        # contains "Yes"/"No".
        winner = market.get("winner")
        if winner is not None:
            # winner is the token_id that won — compare to clobTokenIds
            token_ids_raw = market.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(token_ids_raw)
            except (json.JSONDecodeError, TypeError):
                token_ids = []
            if token_ids:
                # First token is YES
                return winner == token_ids[0]

        # Fallback: check resolution field
        resolution = market.get("resolution", market.get("resolved_to", ""))
        if resolution:
            return str(resolution).lower() in ("yes", "true", "1")

        return None


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

        # Book depth (total resting size) — NOT real 24h traded volume.
        # Real volume comes from Gamma API via MarketContract.volume_24h.
        book_depth = sum(float(b["size"]) for b in bids) + sum(float(a["size"]) for a in asks)

        return MarketPrice(
            token_id=token_id,
            timestamp=datetime.now(timezone.utc),
            bid=best_bid,
            ask=best_ask,
            mid=mid,
            volume_24h=book_depth,
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
