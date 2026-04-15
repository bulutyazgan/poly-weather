# TradeBot

A scientifically rigorous weather prediction trading system for Polymarket temperature markets. Uses causal atmospheric physics understanding -- not correlations -- to produce calibrated probability estimates and exploit market inefficiencies.

## Philosophy

This is a research-first trading system. The core principle: **prove the edge exists before risking capital**.

The system implements a formal null hypothesis hierarchy that must be rejected in sequence:

| Level | Hypothesis | What It Tests |
|-------|-----------|---------------|
| H0-1 | Model does not beat climatological base rates | Basic forecast skill |
| H0-2 | Model does not beat raw NWP ensemble output | Value of bias correction |
| H0-3 | Model does not beat NWP without regime filtering | Value of regime classifier |
| H0-4 | Model's edge over market prices is not significant | Tradeable edge exists |

Every signal is logged (including SKIPs) to prevent survivorship bias. The system starts in paper trading mode and stays there until statistical significance is reached.

## Architecture

```
Weather Data                      Market Data
(GFS, ECMWF, HRRR, METAR)       (Polymarket CLOB)
         |                              |
         v                              v
   Probability Engine             Market Prices
   (MOS-anchored ensemble)              |
         |                              |
         v                              v
   Regime Classifier  ------>  Edge Detector
   (7 physical flags)          (pre-registered thresholds)
                                        |
                                        v
                                Position Sizer
                               (fractional Kelly 8%)
                                        |
                                        v
                                Order Executor
                               (limit orders, stale quote protection)
                                        |
                                        v
                               Verification System
                              (Brier scores, CUSUM, paper trading)
```

### Weather Prediction Engine

**MOS-Anchored Ensemble Probability Estimation**:
1. GFS-MOS deterministic forecast as the central anchor (40+ years of bias correction)
2. ECMWF ensemble spread (60% weight) + GFS ensemble spread (40% weight) for uncertainty
3. Lapse-rate correction for station elevation vs. model grid elevation
4. Station-specific calibration via isotonic regression

GFS and ECMWF ensembles are kept **separate** throughout the pipeline -- they have different systematic biases, perturbation methods, and calibration properties.

### Regime Classifier

A continuous confidence score built from:

- **Primary signal**: Ensemble spread percentile (the models' own uncertainty estimate)
- **Physical override flags**:
  - `frontal_passage` -- pressure tendency >3mb/3h + wind shift >90deg (LOW)
  - `convective` -- CAPE >1000 + precip forecast (LOW)
  - `bimodal_ensemble` -- cluster detection in ensemble members (LOW)
  - `santa_ana` -- 700mb NE wind >25kt + RH <20% (HIGH override, LA only)
  - `post_frontal_clear` -- rising pressure + clearing skies (HIGH override)
  - `lake_breeze_risk` -- onshore wind + spring season (LOW, Chicago only)
  - `chinook` -- 700mb W/WNW wind >30kt (HIGH override, Denver only)

### Trading Engine

- **Edge thresholds**: HIGH regime >= 8%, MEDIUM regime >= 12%
- **Position sizing**: Fractional Kelly at 8% with direction-aware formula
  - BUY_YES: `kelly = edge / (1 - market_prob)`
  - BUY_NO: `kelly = edge / market_prob`
- **Hard caps**: $3/trade, 3% bankroll/trade, 20% total portfolio exposure
- **Stale quote protection**: Cancel all orders if model >3h old + market moved >3%
- **Resolution proximity**: Cancel all orders within 4h of resolution

## Monitored Stations

| City | Station | Elevation | Key Challenges |
|------|---------|-----------|----------------|
| NYC | KNYC | 154 ft | UHI effects, sea breeze penetration |
| Chicago | KORD | 672 ft | Lake Michigan thermal influence |
| LA | KLAX | 126 ft | Marine layer, Santa Ana events |
| Denver | KDEN | 5,431 ft | Chinook winds, rapid transitions |
| Miami | KMIA | 9 ft | Small diurnal range, easiest station |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the dashboard)

### Setup

```bash
# Clone and install
git clone <repo-url> && cd tradebot
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your keys (see Configuration section)

# Run tests
pytest tests/

# Build frontend
cd frontend && npm install && npm run build && cd ..

# Start the bot (paper trading mode by default)
python main.py
```

The API server starts on `http://127.0.0.1:8000` with the dashboard served at the root.

For frontend development with hot reload:

```bash
cd frontend
npm run dev    # Starts on http://localhost:3000, proxies API to :8000
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
POLYGON_WALLET_PRIVATE_KEY=your_wallet_private_key
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/tradebot

# Trading (defaults shown -- do not change until paper trading validates)
PAPER_TRADING=true
KELLY_FRACTION=0.08
MAX_TRADE_USD=3.0
MAX_BANKROLL_PCT=0.03
MAX_PORTFOLIO_EXPOSURE=0.20
HIGH_REGIME_EDGE_THRESHOLD=0.08
MEDIUM_REGIME_EDGE_THRESHOLD=0.12
MIN_HOURS_TO_RESOLUTION=2
MIN_MARKET_VOLUME=2000
```

All trading parameters are pre-registered. Do not tune them on historical market data.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stations` | GET | List all monitored stations |
| `/api/stations/{city}` | GET | Station details (NYC, Chicago, LA, Denver, Miami) |
| `/api/performance` | GET | P&L, win rate, trade/signal counts |
| `/api/signals` | GET | Signal log (params: `station`, `confidence`, `limit`, `offset`) |
| `/api/schedule` | GET | Scheduled pipeline events (GFS/ECMWF/HRRR timing) |
| `/api/calibration` | GET | Brier score, skill score, reliability diagram |

## Pipeline Schedule

The bot runs pipeline cycles at times aligned with NWP model data availability:

| Time (UTC) | Event | Source |
|------------|-------|--------|
| 04:30 | GFS 00Z update | ~4.5h after 00Z init |
| 06:00 | ECMWF 00Z update | ~6h after 00Z init |
| 10:30 | GFS 06Z update | ~4.5h after 06Z init |
| 14:30 | Morning refinement | After 12Z HRRR available |
| 16:30 | GFS 12Z update | ~4.5h after 12Z init |
| 18:00 | ECMWF 12Z update | ~6h after 12Z init |
| 22:30 | GFS 18Z update | ~4.5h after 18Z init |

## Project Structure

```
src/
  config/
    settings.py           # Environment config (pydantic-settings)
    stations.py           # Station definitions with microclimate metadata
  data/
    models.py             # Pydantic data models (10 models)
    weather_client.py     # Open-Meteo (GFS/ECMWF/HRRR) + Mesonet (METAR)
    polymarket_client.py  # Gamma API (markets) + CLOB (orders)
  prediction/
    probability_engine.py # MOS-anchored ensemble probability distribution
    regime_classifier.py  # Continuous confidence + 7 physical flags
    calibration.py        # Brier score, isotonic regression, CUSUM
  trading/
    edge_detector.py      # Pre-registered entry rules
    position_sizer.py     # Fractional Kelly with caps
    executor.py           # Limit orders, stale quote protection
  orchestrator/
    data_collector.py     # Parallel async data collection
    pipeline.py           # Full cycle: collect -> predict -> trade -> log
    scheduler.py          # NWP model-aligned scheduling loop
  verification/
    paper_trader.py       # PnL tracking + counterfactual analysis
    prediction_log.py     # Full audit trail (all signals including SKIPs)
    hypothesis_tester.py  # H0-1 through H0-4 sequential testing
  api/
    main.py               # FastAPI with 8 endpoints + static frontend
frontend/
  src/
    components/           # React dashboard (6 components)
    hooks/useApi.ts       # Auto-refreshing API hook
tests/
  unit/                   # 155 tests across 10 test files
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific module
pytest tests/unit/test_trading.py -v
```


## License

Private -- not for redistribution.
