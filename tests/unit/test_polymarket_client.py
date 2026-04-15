"""Unit tests for Polymarket client (Gamma + CLOB)."""
from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from src.data.models import MarketContract, MarketPrice
from src.data.polymarket_client import CLOBClient, GammaClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _gamma_event_payload(markets: list[dict]) -> dict:
    """Build a minimal Gamma /events response."""
    return markets


def _weather_market(
    question: str,
    condition_id: str = "cond_1",
    token_id_yes: str = "tok_yes_1",
    token_id_no: str = "tok_no_1",
    active: bool = True,
    closed: bool = False,
) -> dict:
    return {
        "condition_id": condition_id,
        "question": question,
        "active": active,
        "closed": closed,
        "tokens": [
            {"token_id": token_id_yes, "outcome": "Yes"},
            {"token_id": token_id_no, "outcome": "No"},
        ],
    }


# ---------------------------------------------------------------------------
# GammaClient — parse_temperature_question
# ---------------------------------------------------------------------------


class TestParseTemperatureQuestion:
    def test_between_range(self):
        q = "Will the high temperature in New York City on April 16, 2026 be between 72°F and 73°F?"
        result = GammaClient.parse_temperature_question(q)
        assert result is not None
        assert result["city"] == "NYC"
        assert result["temp_bucket_low"] == 72.0
        assert result["temp_bucket_high"] == 73.0
        assert result["resolution_date"] == date(2026, 4, 16)

    def test_at_or_above(self):
        q = "Will the high temperature in Chicago on April 16, 2026 be at or above 80°F?"
        result = GammaClient.parse_temperature_question(q)
        assert result is not None
        assert result["city"] == "Chicago"
        assert result["temp_bucket_low"] == 80.0
        assert result["temp_bucket_high"] == float("inf")
        assert result["resolution_date"] == date(2026, 4, 16)

    def test_below(self):
        q = "Will the high temperature in Denver on April 16, 2026 be below 60°F?"
        result = GammaClient.parse_temperature_question(q)
        assert result is not None
        assert result["city"] == "Denver"
        assert result["temp_bucket_low"] == float("-inf")
        assert result["temp_bucket_high"] == 60.0
        assert result["resolution_date"] == date(2026, 4, 16)

    def test_city_mapping_los_angeles(self):
        q = "Will the high temperature in Los Angeles on April 20, 2026 be between 65°F and 70°F?"
        result = GammaClient.parse_temperature_question(q)
        assert result is not None
        assert result["city"] == "LA"

    def test_city_mapping_miami(self):
        q = "Will the high temperature in Miami on April 20, 2026 be between 85°F and 90°F?"
        result = GammaClient.parse_temperature_question(q)
        assert result is not None
        assert result["city"] == "Miami"

    def test_non_weather_question_returns_none(self):
        q = "Will Bitcoin hit $100,000 by end of 2026?"
        result = GammaClient.parse_temperature_question(q)
        assert result is None

    def test_chicago_il_variant(self):
        q = "Will the high temperature in Chicago, IL on April 16, 2026 be at or above 80°F?"
        result = GammaClient.parse_temperature_question(q)
        assert result is not None
        assert result["city"] == "Chicago"


# ---------------------------------------------------------------------------
# GammaClient — fetch_weather_markets
# ---------------------------------------------------------------------------


class TestFetchWeatherMarkets:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_weather_markets(self):
        weather_q = "Will the high temperature in New York City on April 16, 2026 be between 72°F and 73°F?"
        market = _weather_market(weather_q, condition_id="cond_abc", token_id_yes="tok_y1")

        respx.get(f"{GAMMA_BASE}/markets").mock(
            return_value=httpx.Response(200, json=[market])
        )

        client = GammaClient(base_url=GAMMA_BASE)
        contracts = await client.fetch_weather_markets()

        assert len(contracts) == 1
        c = contracts[0]
        assert isinstance(c, MarketContract)
        assert c.token_id == "tok_y1"
        assert c.condition_id == "cond_abc"
        assert c.city == "NYC"
        assert c.temp_bucket_low == 72.0
        assert c.temp_bucket_high == 73.0
        assert c.resolution_date == date(2026, 4, 16)
        assert c.outcome == "Yes"

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_weather_markets_filters_non_weather(self):
        weather_q = "Will the high temperature in Miami on April 18, 2026 be between 80°F and 85°F?"
        non_weather_q = "Will the S&P 500 close above 5000?"

        respx.get(f"{GAMMA_BASE}/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    _weather_market(weather_q, token_id_yes="tok_w"),
                    _weather_market(non_weather_q, token_id_yes="tok_nw"),
                ],
            )
        )

        client = GammaClient(base_url=GAMMA_BASE)
        contracts = await client.fetch_weather_markets()

        assert len(contracts) == 1
        assert contracts[0].token_id == "tok_w"

        await client.close()


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
