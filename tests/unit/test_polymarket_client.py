"""Unit tests for Polymarket client (Gamma + CLOB)."""
from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from src.data.models import MarketContract, MarketPrice
from src.data.polymarket_client import (
    CLOBClient,
    GammaClient,
    _build_event_slug,
    _parse_bucket,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _gamma_event(city_slug: str, target_date: date, markets: list[dict]) -> list[dict]:
    """Build a Gamma /events response (list of events) with sub-markets."""
    month_name = [
        "", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ][target_date.month]
    return [{
        "title": f"Highest temperature in {city_slug} on {month_name.capitalize()} {target_date.day}?",
        "slug": f"highest-temperature-in-{city_slug}-on-{month_name}-{target_date.day}-{target_date.year}",
        "markets": markets,
    }]


def _temp_market(
    label: str,
    question: str = "",
    condition_id: str = "0xcond1",
    yes_token: str = "tok_yes_1",
    no_token: str = "tok_no_1",
    closed: bool = False,
) -> dict:
    """Build a sub-market dict matching Gamma API structure."""
    return {
        "groupItemTitle": label,
        "question": question or f"Will the highest temperature be {label}?",
        "conditionId": condition_id,
        "clobTokenIds": json.dumps([yes_token, no_token]),
        "outcomePrices": json.dumps(["0.25", "0.75"]),
        "outcomes": json.dumps(["Yes", "No"]),
        "active": True,
        "closed": closed,
        "endDate": "2026-04-16T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# _parse_bucket
# ---------------------------------------------------------------------------


class TestParseBucket:
    def test_range(self):
        assert _parse_bucket("80-81°F") == (80.0, 81.0)

    def test_or_below(self):
        low, high = _parse_bucket("79°F or below")
        assert low == float("-inf")
        assert high == 79.0

    def test_or_higher(self):
        low, high = _parse_bucket("98°F or higher")
        assert low == 98.0
        assert high == float("inf")

    def test_unparseable(self):
        assert _parse_bucket("something else") is None


# ---------------------------------------------------------------------------
# _build_event_slug
# ---------------------------------------------------------------------------


class TestBuildEventSlug:
    def test_nyc(self):
        assert _build_event_slug("NYC", date(2026, 4, 15)) == \
            "highest-temperature-in-nyc-on-april-15-2026"

    def test_la(self):
        assert _build_event_slug("LA", date(2026, 7, 4)) == \
            "highest-temperature-in-los-angeles-on-july-4-2026"

    def test_chicago(self):
        assert _build_event_slug("Chicago", date(2026, 12, 25)) == \
            "highest-temperature-in-chicago-on-december-25-2026"

    def test_unknown_city_raises(self):
        with pytest.raises(ValueError, match="No slug mapping"):
            _build_event_slug("UnknownCity", date(2026, 1, 1))


# ---------------------------------------------------------------------------
# GammaClient — fetch_weather_markets
# ---------------------------------------------------------------------------


class TestFetchWeatherMarkets:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_contracts(self):
        """Fetching a single city/date returns parsed MarketContracts."""
        target = date(2026, 4, 16)
        markets = [
            _temp_market("79°F or below", yes_token="tok_below", condition_id="0xc1"),
            _temp_market("80-81°F", yes_token="tok_range", condition_id="0xc2"),
            _temp_market("98°F or higher", yes_token="tok_above", condition_id="0xc3"),
        ]
        event_data = _gamma_event("nyc", target, markets)

        # Mock the slug request for NYC on April 16
        respx.get(f"{GAMMA_BASE}/events").mock(
            return_value=httpx.Response(200, json=event_data)
        )

        client = GammaClient(base_url=GAMMA_BASE)
        with patch("src.data.polymarket_client.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contracts = await client.fetch_weather_markets(
                cities=["NYC"], lookahead_days=1
            )
        await client.close()

        assert len(contracts) == 3
        # Check the "or below" bucket
        below = next(c for c in contracts if c.token_id == "tok_below")
        assert below.city == "NYC"
        assert below.temp_bucket_low == float("-inf")
        assert below.temp_bucket_high == 79.0
        assert below.resolution_date == target
        assert below.outcome == "Yes"

        # Check the range bucket
        rng = next(c for c in contracts if c.token_id == "tok_range")
        assert rng.temp_bucket_low == 80.0
        assert rng.temp_bucket_high == 81.0

        # Check NO token IDs are populated
        assert below.no_token_id == "tok_no_1"
        assert rng.no_token_id == "tok_no_1"

        # Check the "or higher" bucket
        above = next(c for c in contracts if c.token_id == "tok_above")
        assert above.temp_bucket_low == 98.0
        assert above.temp_bucket_high == float("inf")

    @respx.mock
    @pytest.mark.asyncio
    async def test_skips_closed_markets(self):
        target = date(2026, 4, 16)
        markets = [
            _temp_market("80-81°F", yes_token="tok_open", closed=False),
            _temp_market("82-83°F", yes_token="tok_closed", closed=True),
        ]
        event_data = _gamma_event("nyc", target, markets)

        respx.get(f"{GAMMA_BASE}/events").mock(
            return_value=httpx.Response(200, json=event_data)
        )

        client = GammaClient(base_url=GAMMA_BASE)
        with patch("src.data.polymarket_client.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contracts = await client.fetch_weather_markets(
                cities=["NYC"], lookahead_days=1
            )
        await client.close()

        assert len(contracts) == 1
        assert contracts[0].token_id == "tok_open"

    @respx.mock
    @pytest.mark.asyncio
    async def test_handles_no_event_found(self):
        """When no event exists for a slug, returns empty list."""
        respx.get(f"{GAMMA_BASE}/events").mock(
            return_value=httpx.Response(200, json=[])
        )

        client = GammaClient(base_url=GAMMA_BASE)
        with patch("src.data.polymarket_client.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contracts = await client.fetch_weather_markets(
                cities=["NYC"], lookahead_days=1
            )
        await client.close()

        assert contracts == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """HTTP errors are caught and logged, return empty."""
        respx.get(f"{GAMMA_BASE}/events").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        client = GammaClient(base_url=GAMMA_BASE)
        with patch("src.data.polymarket_client.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 16, 12, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            contracts = await client.fetch_weather_markets(
                cities=["NYC"], lookahead_days=1
            )
        await client.close()

        assert contracts == []


# ---------------------------------------------------------------------------
# CLOBClient — market data
# ---------------------------------------------------------------------------

ORDER_BOOK_RESPONSE = {
    "bids": [
        {"price": "0.55", "size": "100"},
        {"price": "0.54", "size": "200"},
    ],
    "asks": [
        {"price": "0.57", "size": "150"},
        {"price": "0.58", "size": "250"},
    ],
}


class TestCLOBClientMarketData:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_market_price(self):
        respx.get(f"{CLOB_BASE}/book").mock(
            return_value=httpx.Response(200, json=ORDER_BOOK_RESPONSE)
        )

        client = CLOBClient(api_url=CLOB_BASE, private_key="0xfake", paper_trading=True)
        price = await client.get_market_price("tok_123")

        assert price is not None
        assert isinstance(price, MarketPrice)
        assert price.token_id == "tok_123"
        assert price.bid == 0.55
        assert price.ask == 0.57
        assert price.mid == pytest.approx(0.56, abs=0.001)

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_order_book_depth(self):
        respx.get(f"{CLOB_BASE}/book").mock(
            return_value=httpx.Response(200, json=ORDER_BOOK_RESPONSE)
        )

        client = CLOBClient(api_url=CLOB_BASE, private_key="0xfake", paper_trading=True)
        depth = await client.get_order_book_depth("tok_123")

        assert depth["bid_volume"] == pytest.approx(300.0)
        assert depth["ask_volume"] == pytest.approx(400.0)

        await client.close()


# ---------------------------------------------------------------------------
# CLOBClient — order management
# ---------------------------------------------------------------------------


class TestCLOBClientOrders:
    @pytest.mark.asyncio
    async def test_place_limit_order_paper_mode(self):
        client = CLOBClient(api_url=CLOB_BASE, private_key="0xfake", paper_trading=True)
        order_id = await client.place_limit_order(
            token_id="tok_123", side="BUY", price=0.55, size=10.0
        )

        # Should return a valid UUID string
        uuid.UUID(order_id)  # raises if invalid
        assert len(client.paper_orders) == 1
        assert client.paper_orders[0]["token_id"] == "tok_123"
        assert client.paper_orders[0]["side"] == "BUY"
        assert client.paper_orders[0]["price"] == 0.55
        assert client.paper_orders[0]["size"] == 10.0

        await client.close()

    @pytest.mark.asyncio
    async def test_place_limit_order_live_mode(self):
        with patch("src.data.polymarket_client.PyClobClient") as MockClobClient:
            mock_instance = MagicMock()
            mock_instance.create_and_post_order.return_value = {"orderID": "live_order_123"}
            MockClobClient.return_value = mock_instance

            client = CLOBClient(api_url=CLOB_BASE, private_key="0xfake", paper_trading=False)
            order_id = await client.place_limit_order(
                token_id="tok_123", side="BUY", price=0.55, size=10.0
            )

            assert order_id == "live_order_123"
            mock_instance.create_and_post_order.assert_called_once()

            await client.close()

    @pytest.mark.asyncio
    async def test_cancel_all_orders_paper(self):
        client = CLOBClient(api_url=CLOB_BASE, private_key="0xfake", paper_trading=True)
        # Add some paper orders
        await client.place_limit_order("tok_1", "BUY", 0.5, 10)
        await client.place_limit_order("tok_2", "SELL", 0.6, 5)
        assert len(client.paper_orders) == 2

        cancelled = await client.cancel_all_orders()
        assert cancelled == 2
        assert len(client.paper_orders) == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_all_orders_live(self):
        with patch("src.data.polymarket_client.PyClobClient") as MockClobClient:
            mock_instance = MagicMock()
            mock_instance.cancel_all.return_value = {"canceled_orders": ["o1", "o2", "o3"]}
            MockClobClient.return_value = mock_instance

            client = CLOBClient(api_url=CLOB_BASE, private_key="0xfake", paper_trading=False)
            cancelled = await client.cancel_all_orders()

            assert cancelled == 3
            mock_instance.cancel_all.assert_called_once()

            await client.close()
