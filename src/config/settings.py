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
    KELLY_FRACTION: float = 0.04
    MIN_TRADE_USD: float = 0.25
    MAX_TRADE_USD: float = 1.50
    MAX_BANKROLL_PCT: float = 0.01
    MAX_PORTFOLIO_EXPOSURE: float = 0.35
    MAX_DRAWDOWN_PCT: float = 0.15
    HIGH_REGIME_EDGE_THRESHOLD: float = 0.05
    MEDIUM_REGIME_EDGE_THRESHOLD: float = 0.08
    MIN_HOURS_TO_RESOLUTION: int = 2
    MIN_MARKET_VOLUME: int = 200
    MAX_MARKET_CERTAINTY: float = 0.92
    MAX_EDGE: float = 0.25
    MAX_SPREAD: float = 0.10
    TAKER_FEE_RATE: float = 0.02
    ORDER_TTL_SECONDS: float = 1800.0
    PAPER_TRADING: bool = True

    # Price monitor
    PRICE_MONITOR_ENABLED: bool = True
    PRICE_MONITOR_DEBOUNCE_S: float = 10.0
    PRICE_MONITOR_COOLDOWN_S: float = 900.0  # 15 min per contract
    PRICE_MONITOR_MAX_FORECAST_AGE_S: float = 28800.0  # 8 hours
    PRICE_MONITOR_MAX_PRICE_AGE_S: float = 30.0  # reject cached prices older than 30s
    BANKROLL: float = 300.0  # total bankroll in USD

    # API URLs
    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    POLYMARKET_GAMMA_URL: str = "https://gamma-api.polymarket.com"
    NWS_API_URL: str = "https://api.weather.gov"
    OPENWEATHER_API_URL: str = "https://api.openweathermap.org"
