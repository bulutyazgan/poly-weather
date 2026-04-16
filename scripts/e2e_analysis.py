"""End-to-end analysis: run one pipeline cycle and diagnose predictions."""
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import Settings
from src.config.stations import get_stations
from src.data.weather_client import OpenMeteoClient, MesonetClient
from src.data.polymarket_client import GammaClient, CLOBClient
from src.data.models import EnsembleForecast, MarketContract, MOSForecast
from src.orchestrator.data_collector import DataCollector
from src.prediction.probability_engine import ProbabilityEngine
from src.prediction.regime_classifier import RegimeClassifier
from src.prediction.calibration import CUSUMMonitor
from src.trading.edge_detector import EdgeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e_analysis")

# Reuse pipeline helpers
from src.orchestrator.pipeline import _pick_ensemble_for_date, _synthesize_mos, _resolution_utc


async def run_analysis():
    settings = Settings()
    now = datetime.now(tz=timezone.utc)
    stations = get_stations()

    # Init clients
    weather = OpenMeteoClient()
    mesonet = MesonetClient()
    gamma = GammaClient()
    clob = CLOBClient(
        api_url=settings.POLYMARKET_API_URL,
        private_key=settings.POLYGON_WALLET_PRIVATE_KEY.get_secret_value(),
        paper_trading=True,
    )

    collector = DataCollector(weather=weather, mesonet=mesonet, gamma=gamma, clob=clob)
    prob_engine = ProbabilityEngine()
    regime_classifier = RegimeClassifier()
    edge_detector = EdgeDetector(
        high_threshold=settings.HIGH_REGIME_EDGE_THRESHOLD,
        medium_threshold=settings.MEDIUM_REGIME_EDGE_THRESHOLD,
        min_volume=settings.MIN_MARKET_VOLUME,
        min_hours=settings.MIN_HOURS_TO_RESOLUTION,
        max_market_certainty=settings.MAX_MARKET_CERTAINTY,
        max_edge=settings.MAX_EDGE,
        taker_fee_rate=settings.TAKER_FEE_RATE,
        max_spread=settings.MAX_SPREAD,
    )

    # ── 1. Collect data ──────────────────────────────────────────────
    logger.info("Collecting snapshots for %d stations...", len(stations))
    try:
        snapshots = await collector.collect_snapshot()
    except Exception:
        logger.exception("Failed to collect snapshots")
        return

    snap_by_station = {s.station_id: s for s in snapshots}

    # ── 2. Analyze each station ──────────────────────────────────────
    results = []

    for city, station in stations.items():
        snap = snap_by_station.get(station.station_id)
        if snap is None:
            logger.warning("No snapshot for %s (%s)", city, station.station_id)
            continue

        # Log raw data availability
        data_report = {
            "city": city,
            "station_id": station.station_id,
            "has_gfs_ensemble": snap.gfs_ensemble is not None,
            "has_ecmwf_ensemble": snap.ecmwf_ensemble is not None,
            "gfs_all_count": len(snap.gfs_ensemble_all),
            "ecmwf_all_count": len(snap.ecmwf_ensemble_all),
            "hrrr_count": len(snap.hrrr) if snap.hrrr else 0,
            "contract_count": len(snap.market_contracts),
            "price_count": len(snap.market_prices),
        }
        logger.info("Data report for %s: %s", city, json.dumps(data_report, indent=2))

        # Classify regime — match pipeline logic for between-model spread
        ensemble_spread = snap.gfs_ensemble.std if snap.gfs_ensemble else 4.0
        ensemble_members = snap.gfs_ensemble.members if snap.gfs_ensemble else None
        gfs_peak_today = _pick_ensemble_for_date(snap.gfs_ensemble_all, now.date())
        ecmwf_peak_today = _pick_ensemble_for_date(snap.ecmwf_ensemble_all, now.date())
        between_spread = 0.0
        if gfs_peak_today and ecmwf_peak_today:
            between_spread = abs(gfs_peak_today.mean - ecmwf_peak_today.mean)
        regime = regime_classifier.classify(
            station_id=station.station_id,
            valid_date=now.date(),
            ensemble_spread=ensemble_spread,
            ensemble_members=ensemble_members,
            station_flags=station.flags,
            between_model_spread=between_spread,
        )
        logger.info(
            "%s regime: confidence=%s, spread_pctile=%.1f, flags=%s",
            city, regime.confidence, regime.ensemble_spread_percentile,
            regime.active_flags,
        )

        # Log ensemble details
        if snap.gfs_ensemble:
            logger.info(
                "%s GFS ensemble: mean=%.1f°F, std=%.1f°F, members=%d",
                city, snap.gfs_ensemble.mean, snap.gfs_ensemble.std,
                len(snap.gfs_ensemble.members) if snap.gfs_ensemble.members else 0,
            )
        if snap.ecmwf_ensemble:
            logger.info(
                "%s ECMWF ensemble: mean=%.1f°F, std=%.1f°F, members=%d",
                city, snap.ecmwf_ensemble.mean, snap.ecmwf_ensemble.std,
                len(snap.ecmwf_ensemble.members) if snap.ecmwf_ensemble.members else 0,
            )

        # Process each contract
        for contract in snap.market_contracts:
            resolution_dt = _resolution_utc(contract)
            if resolution_dt <= now:
                continue

            hours_to_resolution = (resolution_dt - now).total_seconds() / 3600.0

            # Build distribution for this contract's date
            mos = _synthesize_mos(snap, station, now, contract.resolution_date)
            gfs_for_date = _pick_ensemble_for_date(snap.gfs_ensemble_all, contract.resolution_date)
            ecmwf_for_date = _pick_ensemble_for_date(snap.ecmwf_ensemble_all, contract.resolution_date)

            distribution = prob_engine.compute_distribution(
                mos=mos,
                gfs_ensemble=gfs_for_date,
                ecmwf_ensemble=ecmwf_for_date,
                station=station,
                valid_date=contract.resolution_date,
            )

            model_prob = prob_engine.compute_bucket_probability(
                distribution,
                contract.temp_bucket_low,
                contract.temp_bucket_high,
            )

            price = snap.market_prices.get(contract.token_id)
            if price is None:
                continue

            market_prob = price.mid

            # Evaluate signal
            signal = edge_detector.evaluate(
                model_prob=model_prob,
                market_prob=market_prob,
                regime=regime,
                volume_24h=contract.volume_24h,
                hours_to_resolution=hours_to_resolution,
                market_id=contract.token_id,
                market_bid=price.bid,
                market_ask=price.ask,
                dist_std=distribution.std(),
            )

            entry = {
                "city": city,
                "bucket": f"{contract.temp_bucket_low}-{contract.temp_bucket_high}°F",
                "resolution_date": str(contract.resolution_date),
                "hours_to_resolution": round(hours_to_resolution, 1),
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_prob, 4),
                "market_bid": round(price.bid, 4) if price.bid else None,
                "market_ask": round(price.ask, 4) if price.ask else None,
                "spread": round(price.ask - price.bid, 4) if price.bid and price.ask else None,
                "edge": round(signal.edge, 4),
                "action": signal.action,
                "direction": signal.direction,
                "skip_reason": signal.skip_reason,
                "volume_24h": contract.volume_24h,
                "dist_mean": round(distribution.mean(), 2),
                "dist_std": round(distribution.std(), 2),
                "mos_high": round(mos.high_f, 2),
                "gfs_peak_mean": round(gfs_for_date.mean, 2) if gfs_for_date else None,
                "ecmwf_peak_mean": round(ecmwf_for_date.mean, 2) if ecmwf_for_date else None,
                "regime_confidence": regime.confidence,
            }
            results.append(entry)

    # ── 3. Print analysis ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"E2E ANALYSIS — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Stations: {len(stations)}, Snapshots: {len(snapshots)}")
    print("=" * 100)

    if not results:
        print("\nNO CONTRACTS FOUND — market may be closed or no active weather markets.")
        return

    # Group by city
    by_city = {}
    for r in results:
        by_city.setdefault(r["city"], []).append(r)

    for city, entries in by_city.items():
        print(f"\n{'─' * 80}")
        print(f"  {city} — {len(entries)} contracts")
        print(f"  Distribution: mean={entries[0]['dist_mean']}°F, std={entries[0]['dist_std']}°F")
        print(f"  MOS high: {entries[0]['mos_high']}°F")
        if entries[0]['gfs_peak_mean']:
            print(f"  GFS peak: {entries[0]['gfs_peak_mean']}°F")
        if entries[0]['ecmwf_peak_mean']:
            print(f"  ECMWF peak: {entries[0]['ecmwf_peak_mean']}°F")
        print(f"  Regime: {entries[0]['regime_confidence']}")
        print(f"{'─' * 80}")

        # Sort by bucket
        entries.sort(key=lambda e: e["bucket"])

        trades = [e for e in entries if e["action"] == "TRADE"]
        skips = [e for e in entries if e["action"] == "SKIP"]

        print(f"\n  {'Bucket':<20} {'Model':>8} {'Market':>8} {'Spread':>8} {'Edge':>8} {'Action':<10} {'Direction':<10} {'Skip Reason':<20}")
        print(f"  {'─' * 106}")
        for e in entries:
            spread_str = f"{e['spread']:.3f}" if e['spread'] else "N/A"
            skip_str = e['skip_reason'] or ""
            marker = ">>>" if e['action'] == "TRADE" else "   "
            print(
                f"{marker} {e['bucket']:<20} {e['model_prob']:>8.4f} {e['market_prob']:>8.4f} "
                f"{spread_str:>8} {e['edge']:>8.4f} {e['action']:<10} {e['direction']:<10} {skip_str:<20}"
            )

        if trades:
            print(f"\n  TRADES ({len(trades)}):")
            for t in trades:
                print(
                    f"    {t['direction']} {t['bucket']} — edge={t['edge']:.3f}, "
                    f"model={t['model_prob']:.3f} vs market={t['market_prob']:.3f}, "
                    f"vol={t['volume_24h']}, hrs_to_res={t['hours_to_resolution']:.1f}"
                )

    # ── 4. Aggregate diagnostics ─────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 100}")

    all_trades = [r for r in results if r["action"] == "TRADE"]
    all_skips = [r for r in results if r["action"] == "SKIP"]

    print(f"\nTotal signals: {len(results)}")
    print(f"  TRADE: {len(all_trades)}")
    print(f"  SKIP:  {len(all_skips)}")

    # Skip reason breakdown
    skip_reasons = {}
    for s in all_skips:
        reason = s["skip_reason"] or "below_threshold"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    print(f"\nSkip reasons:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Edge distribution
    edges = [r["edge"] for r in results]
    if edges:
        print(f"\nEdge distribution:")
        print(f"  min={min(edges):.4f}, max={max(edges):.4f}, mean={sum(edges)/len(edges):.4f}")
        positive_edges = [e for e in edges if e > 0]
        print(f"  Positive edges: {len(positive_edges)}/{len(edges)}")

    # Model vs market disagreement
    disagreements = [(r["model_prob"] - r["market_prob"]) for r in results]
    if disagreements:
        print(f"\nModel-Market disagreement (model - market):")
        print(f"  min={min(disagreements):.4f}, max={max(disagreements):.4f}")
        print(f"  mean={sum(disagreements)/len(disagreements):.4f}")
        print(f"  Model higher: {sum(1 for d in disagreements if d > 0)}")
        print(f"  Market higher: {sum(1 for d in disagreements if d < 0)}")

    # Per-date analysis (multi-day coverage?)
    dates = set(r["resolution_date"] for r in results)
    print(f"\nResolution dates covered: {sorted(dates)}")

    # Bid-ask spread analysis
    spreads = [r["spread"] for r in results if r["spread"] is not None]
    if spreads:
        print(f"\nBid-Ask spreads:")
        print(f"  min={min(spreads):.4f}, max={max(spreads):.4f}, mean={sum(spreads)/len(spreads):.4f}")
        wide = sum(1 for s in spreads if s > 0.10)
        print(f"  Wide (>10%): {wide}/{len(spreads)}")

    # Write raw results to JSON for further analysis
    out_path = Path(__file__).parent.parent / "e2e_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_analysis())
