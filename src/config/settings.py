"""Application settings using pydantic-settings."""
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Trading system configuration loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Required secrets
    POLYGON_WALLET_PRIVATE_KEY: SecretStr
    DATABASE_URL: str

    # Trading parameters
    KELLY_FRACTION: float = 0.08
    MAX_TRADE_USD: float = 3.0
    MAX_BANKROLL_PCT: float = 0.03
    MAX_PORTFOLIO_EXPOSURE: float = 0.20
    HIGH_REGIME_EDGE_THRESHOLD: float = 0.08
    MEDIUM_REGIME_EDGE_THRESHOLD: float = 0.12
    MIN_HOURS_TO_RESOLUTION: int = 2
    MIN_MARKET_VOLUME: int = 2000
    PAPER_TRADING: bool = True

    # API URLs
    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    NWS_API_URL: str = "https://api.weather.gov"
    OPENWEATHER_API_URL: str = "https://api.openweathermap.org"
