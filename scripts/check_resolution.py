"""Check resolution outcomes for e2e trades against Polymarket."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


async def main():
    results_path = Path("results/e2e_run_20260416_112406_v3_diversified.json")
    with open(results_path) as f:
        data = json.load(f)

    trades = data["trades"]
    print(f"Checking {len(trades)} trades for resolution...\n")

    # Build event slugs for each unique city/date
    city_slug = {"NYC": "nyc", "Chicago": "chicago", "Miami": "miami", "LA": "los-angeles", "Denver": "denver"}
    month_names = ["", "january", "february", "march", "april", "may", "june",
                   "july", "august", "september", "october", "november", "december"]

    seen_slugs = {}
    for t in trades:
        city = t["city"]
        res_date = t["resolution_date"]
        key = f"{city}_{res_date}"
        if key not in seen_slugs:
            # Parse date
            from datetime import date as d
            parts = res_date.split("-")
            dt = d(int(parts[0]), int(parts[1]), int(parts[2]))
            slug = f"highest-temperature-in-{city_slug[city]}-on-{month_names[dt.month]}-{dt.day}-{dt.year}"
            seen_slugs[key] = slug

    async with httpx.AsyncClient(base_url="https://gamma-api.polymarket.com", timeout=30.0) as client:
        # Fetch each event and check resolution status
        for key, slug in seen_slugs.items():
            print(f"=== {key} (slug: {slug}) ===")
            try:
                resp = await client.get("/events", params={"slug": slug})
                resp.raise_for_status()
                events = resp.json()
                if not events:
                    print("  No event found\n")
                    continue

                event = events[0]
                markets = event.get("markets", [])
                print(f"  Found {len(markets)} markets")

                for market in markets:
                    label = market.get("groupItemTitle", "")
                    closed = market.get("closed", False)
                    outcome_prices = market.get("outcomePrices", "")
                    winner = market.get("winner", "")
                    resolved_to = market.get("resolved_to", market.get("resolution", ""))
                    best_bid = market.get("bestBid", "")
                    best_ask = market.get("bestAsk", "")

                    # Parse outcome prices
                    try:
                        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    except:
                        prices = []

                    yes_price = float(prices[0]) if prices else None

                    # Determine resolution
                    if closed and yes_price is not None:
                        if yes_price > 0.95:
                            resolution = "YES WON"
                        elif yes_price < 0.05:
                            resolution = "NO WON"
                        else:
                            resolution = f"UNCLEAR (price={yes_price})"
                    elif closed:
                        resolution = f"CLOSED (winner={winner}, resolved_to={resolved_to})"
                    else:
                        resolution = "NOT RESOLVED"

                    print(f"  {label:25s} closed={closed} yes_price={yes_price} → {resolution}")

            except Exception as e:
                print(f"  Error: {e}")
            print()

        # Now evaluate each trade
        print("\n" + "="*80)
        print("TRADE-BY-TRADE P&L EVALUATION")
        print("="*80 + "\n")

        total_pnl = 0.0
        wins = 0
        losses = 0

        for t in trades:
            city = t["city"]
            res_date = t["resolution_date"]
            key = f"{city}_{res_date}"
            slug = seen_slugs[key]
            bucket = t["bucket"]
            direction = t["direction"]
            edge = t["edge"]
            model_prob = t["model_prob"]
            market_bid = t["market_bid"]
            market_ask = t["market_ask"]

            # Fetch the specific market
            try:
                resp = await client.get("/events", params={"slug": slug})
                events = resp.json()
                if not events:
                    print(f"  {city} {bucket} {direction}: EVENT NOT FOUND")
                    continue

                event = events[0]
                markets = event.get("markets", [])

                # Find matching market
                matched = None
                for market in markets:
                    label = market.get("groupItemTitle", "")
                    # Match bucket to label
                    if bucket.startswith("-inf"):
                        # e.g. "-inf-77.5°F" matches "77°F or below"
                        bucket_high = float(bucket.split("-")[-1].replace("°F", ""))
                        target_val = int(bucket_high - 0.5)  # reverse continuity correction
                        if f"{target_val}°F or below" in label:
                            matched = market
                            break
                    elif bucket.endswith("inf"):
                        bucket_low = float(bucket.split("-")[0])
                        target_val = int(bucket_low + 0.5)
                        if f"{target_val}°F or higher" in label:
                            matched = market
                            break
                    else:
                        # Range bucket like "77.5-79.5°F"
                        parts = bucket.replace("°F", "").split("-")
                        low_val = int(float(parts[0]) + 0.5)
                        high_val = int(float(parts[1]) - 0.5)
                        if f"{low_val}-{high_val}°F" in label or f"{low_val}–{high_val}°F" in label:
                            matched = market
                            break

                if not matched:
                    # Try fuzzy match
                    print(f"  {city} {bucket} {direction}: NO MATCHING MARKET FOUND")
                    print(f"    Available: {[m.get('groupItemTitle','') for m in markets]}")
                    continue

                closed = matched.get("closed", False)
                outcome_prices = matched.get("outcomePrices", "")
                try:
                    prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                except:
                    prices = []
                yes_price = float(prices[0]) if prices else None

                if not closed:
                    print(f"  {city} {bucket} {direction}: NOT YET RESOLVED")
                    continue

                # Determine if YES won
                yes_won = yes_price is not None and yes_price > 0.95

                # Calculate P&L
                if direction == "BUY_YES":
                    entry_price = market_ask  # paid the ask
                    if yes_won:
                        pnl_per_share = 1.0 - entry_price  # won: collect $1
                        outcome = "WIN"
                        wins += 1
                    else:
                        pnl_per_share = -entry_price  # lost: lose entry
                        outcome = "LOSS"
                        losses += 1
                else:  # BUY_NO
                    entry_price = 1.0 - market_bid  # cost to buy NO = 1 - bid
                    if yes_won:
                        pnl_per_share = -entry_price  # NO lost
                        outcome = "LOSS"
                        losses += 1
                    else:
                        pnl_per_share = 1.0 - entry_price  # NO won: collect $1
                        outcome = "WIN"
                        wins += 1

                total_pnl += pnl_per_share

                print(f"  {city:8s} {bucket:20s} {direction:8s} edge={edge:.4f} "
                      f"model={model_prob:.3f} mkt={t['market_prob']:.3f} "
                      f"→ YES {'WON' if yes_won else 'LOST'} → {outcome} "
                      f"P&L={pnl_per_share:+.3f}")

            except Exception as e:
                print(f"  {city} {bucket}: Error: {e}")

        print(f"\n{'='*80}")
        print(f"SUMMARY: {wins}W / {losses}L  |  Total P&L per share: {total_pnl:+.3f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
