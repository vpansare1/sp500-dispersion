"""
Trace why a dispersion metric spikes on one specific day.

For each stock of interest, this prints its computed horizon return on every
trading day in a small window around the target date, showing BOTH ends of the
lookback each day: the current price, the exact lookback date the horizon
pointer lands on, and the price there. A one-day spike in a stock's computed
return -- while the surrounding days are calm -- reveals a mechanical artifact
(a bad tick, or a lookback date that lands on a corrupted/adjustment-boundary
price) rather than a real move.

Also recomputes the cross-sectional decile spread and xs-std for each day in
the window so you can see the metric spike line up with a specific stock.

Usage:
    python trace_stock_dates.py --tickers SNDK LITE WDC MU --around 2026-07-14
    python trace_stock_dates.py --tickers SNDK --around 2026-07-14 --pad 5 --horizon 12M
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import dispersion_lib as dl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--around", required=True, help="center date YYYY-MM-DD")
    ap.add_argument("--pad", type=int, default=4, help="trading days each side")
    ap.add_argument("--horizon", default="12M", choices=list(dl.HORIZONS))
    args = ap.parse_args()

    window = dl.HORIZONS[args.horizon]
    center = pd.Timestamp(args.around)

    print("Fetching full constituent set + prices...")
    members = dl.get_sp500_constituents()
    tickers = members["ticker"].tolist()
    # need >2y so the 252-day lookback exists for dates around `center`
    start = (center - pd.Timedelta("500d")).strftime("%Y-%m-%d")
    end = (center + pd.Timedelta("15d")).strftime("%Y-%m-%d")
    prices = dl.download_prices(tickers, start=start, end=end).dropna(axis=1, how="all")

    idx = prices.index
    if center not in idx:
        # snap to nearest available trading day
        center = idx[idx.get_indexer([center], method="nearest")[0]]
    ci = idx.get_loc(center)
    lo, hi = max(window, ci - args.pad), min(len(idx) - 1, ci + args.pad)

    # ---- per-stock trace across the date window --------------------------
    for tk in args.tickers:
        if tk not in prices.columns:
            print(f"\n{tk}: not in price data\n")
            continue
        print(f"\n===== {tk} ({args.horizon} return by date) =====")
        rows = []
        for i in range(lo, hi + 1):
            d = idx[i]
            back = idx[i - window]
            pn, pb = prices[tk].iloc[i], prices[tk].iloc[i - window]
            ret = (pn / pb - 1.0) if (pb and not np.isnan(pb)) else np.nan
            rows.append({
                "date": d.date(),
                "price": round(pn, 2) if not np.isnan(pn) else None,
                "lookback_date": back.date(),
                "lookback_price": round(pb, 2) if not np.isnan(pb) else None,
                "return": f"{ret:+.1%}" if not np.isnan(ret) else "n/a",
                "<<": "  <-- target" if d == center else "",
            })
        print(pd.DataFrame(rows).to_string(index=False))

    # ---- cross-sectional metric across the same window -------------------
    print(f"\n===== cross-sectional dispersion by date ({args.horizon}) =====")
    mrows = []
    for i in range(lo, hi + 1):
        d = idx[i]
        rets = prices.iloc[i] / prices.iloc[i - window] - 1.0
        rets_g = dl.clean_extreme_returns(rets)  # current guard applied
        mrows.append({
            "date": d.date(),
            "decile_spread_raw": f"{dl.decile_spread(rets):+.1%}",
            "decile_spread_guard": f"{dl.decile_spread(rets_g):+.1%}",
            "xs_std_raw": f"{dl.cross_sectional_std(rets):.1%}",
            "mad": f"{dl.cross_sectional_mad(rets):.1%}",
            "<<": "  <-- target" if d == center else "",
        })
    print(pd.DataFrame(mrows).to_string(index=False))
    print("\nRead: if a stock's return jumps for the target day only while its "
          "price is normal, look at that day's lookback_price/lookback_date -- "
          "a bad reference there is the artifact.")


if __name__ == "__main__":
    main()
