# Pre-Registration Specification

**System**: Polymarket Weather Prediction Trading Bot
**Date**: 2026-04-15
**Status**: Pre-registered (do not modify after paper trading begins)

---

## 1. Null Hypothesis Hierarchy

Tests are evaluated in order. Each level must pass before proceeding.

| Level | Null Hypothesis | Test Statistic | Rejection Criterion |
|-------|----------------|----------------|---------------------|
| H0-1 | Model does not beat climatological base rates | Brier Skill Score (BSS) vs. climatology | BSS > 0, p < 0.05 (one-sided permutation test, N=100 resolved forecasts) |
| H0-2 | Model does not beat raw NWP ensemble output | BSS(model) - BSS(raw_ensemble) | Difference > 0, p < 0.05 (paired bootstrap, N=200) |
| H0-3 | Model does not beat NWP without regime filtering | BSS(regime_filtered) - BSS(unfiltered) | Difference > 0, p < 0.05 (paired bootstrap, N=300) |
| H0-4 | Model's edge over market prices is not significant | Mean signed edge on resolved trades | Edge > 0, p < 0.05 with Benjamini-Hochberg correction across city-regime combinations |

## 2. Pre-Registered Parameters

These parameters are fixed before any live/paper data is observed. They must not be tuned on historical market data.

### 2.1 Weather Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ECMWF ensemble weight | 0.60 | Better calibrated for temperature |
| GFS ensemble weight | 0.40 | More frequent updates (4x/day) |
| MOS anchor | GFS-MOS Tmax/Tmin | 40+ years of station-specific bias correction |
| Climatological std fallback | 4.0 °F | Conservative when no ensemble data |
| Lapse rate correction | Per-station (see stations.py) | Elevation difference between model grid and station |

### 2.2 Morning Refinement

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Morning observation correlation | 0.60 | Empirical 12Z error → Tmax propagation |
| Cloud cover adjustment max | ±3.0 °F | Physical bound on cloud radiative forcing |
| HRRR blend weight | 0.40 | Captures actual boundary layer state |
| Minimum std floor | 1.5 °F | Irreducible forecast uncertainty |
| Maximum confidence boost | 0.30 | Cap on regime confidence improvement |

### 2.3 Regime Classifier

| Parameter | Value |
|-----------|-------|
| HIGH confidence threshold | Spread < 30th percentile, no LOW flags |
| MEDIUM confidence threshold | Spread 30th–60th percentile, no LOW flags |
| LOW confidence threshold | Spread > 60th percentile OR any LOW flag |
| Frontal passage: pressure tendency | > 3 mb/3h + wind shift > 90° |
| Convective: CAPE threshold | > 1000 J/kg + precip forecast |
| Bimodal ensemble: dip test | p < 0.05 |
| Santa Ana (LA): 700mb wind | > 25 kt from N/NE, RH < 20% |
| Chinook (Denver): 700mb wind | > 30 kt from W/WNW |
| Lake breeze (Chicago): wind dir | 45°–135° (NE through SE), Apr–Jun |
| Post-frontal clear: pressure | Rising > 2 mb/3h + clearing skies |

### 2.4 Trading Engine

| Parameter | Value | Env Variable |
|-----------|-------|--------------|
| HIGH regime edge threshold | 0.08 (8%) | HIGH_REGIME_EDGE_THRESHOLD |
| MEDIUM regime edge threshold | 0.12 (12%) | MEDIUM_REGIME_EDGE_THRESHOLD |
| Kelly fraction | 0.08 (8%) | KELLY_FRACTION |
| Maximum trade size | $3.00 | MAX_TRADE_USD |
| Maximum bankroll per trade | 3% | MAX_BANKROLL_PCT |
| Maximum portfolio exposure | 20% | MAX_PORTFOLIO_EXPOSURE |
| Minimum hours to resolution | 2h | MIN_HOURS_TO_RESOLUTION |
| Minimum market volume | 2,000 shares | MIN_MARKET_VOLUME |
| Stale model threshold | 3h | (hardcoded) |
| Stale market move threshold | 3% | (hardcoded) |
| Resolution proximity cancel | 4h | (hardcoded) |
| Adverse selection threshold | 10% gap | (hardcoded) |

### 2.5 Correlation Discounts

| Active Stations | Discount Factor |
|----------------|-----------------|
| 1 | 1.00 |
| 2 | 0.70 |
| 3 | 0.50 |
| 4+ | 0.40 |

### 2.6 Calibration

| Parameter | Value |
|-----------|-------|
| Calibration method | Isotonic regression |
| Training window | Out-of-sample only (strict temporal separation) |
| Reliability diagram bins | 10 |
| CUSUM threshold | 2.0 |
| CUSUM drift | 0.0 |

## 3. Sample Size Requirements

Based on power analysis for detecting a 2% mean edge:

| Metric | Requirement |
|--------|-------------|
| Minimum resolved forecasts for H0-1 | 100 |
| Minimum resolved forecasts for H0-2 | 200 |
| Minimum resolved forecasts for H0-3 | 300 |
| Minimum independent bets for H0-4 | 500–1,000 |
| Expected fill rate | ~2 trades/day |
| Estimated time to H0-4 evaluation | 8–16 months |
| Independence correction | ~40–60% of raw count (weather autocorrelation) |

## 4. Falsification Criteria

The system is abandoned or returned to theory development if:

1. After 100 resolved forecasts, BSS vs. climatology is not significant at p < 0.05 (H0-1 fails)
2. After 200 resolved forecasts, bias-corrected model does not improve over raw NWP (H0-2 fails) — simplify to raw MOS
3. After 300 resolved forecasts, regime filtering does not improve BSS (H0-3 fails) — remove regime classifier
4. After the pre-specified evaluation period (500–1,000 independent bets), market edge is not significant at p < 0.05 with BH correction across all city-regime combinations (H0-4 fails) — no tradeable edge exists

**No "tweaking thresholds and trying again."** If H0-4 fails, the system is either abandoned or returned to Phase 0 with a new pre-registration.

## 5. Stopping Rules

### 5.1 Paper Trading Phase
- Duration: determined by power analysis (see Section 3)
- NO parameter changes during evaluation period
- ALL signals logged (including SKIPs and LOW-regime counterfactuals)
- CUSUM sequential monitoring with threshold 2.0 and false alarm rate controlled by drift parameter

### 5.2 Go-Live Criteria
All of the following must be satisfied:
- H0-1 through H0-3 rejected at p < 0.05
- H0-4 rejected at p < 0.05 with BH correction
- CUSUM alarm not active
- Fill rate > 30%
- Adverse selection ratio > -0.10 (not being systematically picked off)
- Positive paper P&L over the evaluation period

### 5.3 Live Trading Kill Switches
- CUSUM alarm triggers → block all new trades until manual review and reset
- Adverse selection ratio < -0.10 → halt trading, investigate fill patterns
- 5 consecutive losing days → reduce all position sizes by 50% for 1 week
- Portfolio drawdown > 30% → halt all trading

## 6. Monitored Stations

| City | ICAO | Elevation | Key Regime Challenges |
|------|------|-----------|----------------------|
| NYC | KNYC | 154 ft | UHI, sea breeze penetration |
| Chicago | KORD | 672 ft | Lake Michigan thermal influence |
| LA | KLAX | 126 ft | Marine layer, Santa Ana events |
| Denver | KDEN | 5,431 ft | Chinook winds, rapid transitions |
| Miami | KMIA | 9 ft | Small diurnal range, easiest station |

## 7. Validation Strategy

- **H0-1 through H0-3**: Option C (weather-only backtest) using historical NWP archives (NOAA NOMADS for GFS, AWS Open Data for ECMWF) matched to historical METAR observations
- **H0-4**: Option A (forward-only paper trading) collecting (forecast, market price, outcome) triples from deployment day one

This separation is necessary because Polymarket's historical price data degrades to 12h granularity for resolved markets.

## 8. Data Logging Requirements

Every signal is logged with:
- Regime classification + all input parameters
- Ensemble output (GFS and ECMWF separately)
- MOS forecast
- Calibrated probability
- Market price at signal time
- Edge calculation
- Position size calculation
- Order placed (or SKIP reason)
- Fill status
- Resolution outcome

**Critical**: ALL signals including SKIPs and paper trades in SKIP regimes are logged to prevent survivorship bias.
