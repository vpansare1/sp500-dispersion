"""
Diagnose a dispersion spike: for a given horizon, show which constituents land
in the top and bottom deciles of realized return, so you can tell a genuine
extreme move from a bad price/split adjustment.

A single stock showing a return like +900% over 12 months, especially one that
doesn't line up with a real news event, is almost always a Yahoo adjusted-close
glitch (one end of the window adjusted for a split, the other not). This script
flags names whose return exceeds a sanity threshold and shows the raw prices at
both ends of the window so you can eyeball it.

Usage:
    python diagnose_spike.py                 # 12M horizon, latest date
    python diagnose_spike.py --horizon 12M --top 15
    python diagnose_spike.py --date 2026-07-14
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import dispersion_lib as dl

SANITY = 4.0  # flag any single-stock return above +400% as suspect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="12M", choices=list(dl.HORIZONS))
    ap.add_argument("--date", default=None, help="as-of date YYYY-MM-DD (default: latest)")
    ap.add_argument("--top", type=int, default=12, help="names to show per tail")
    args = ap.parse_args()

    window = dl.HORIZONS[args.horizon]

    print("Fetching constituents + prices (trailing ~13 months)...")
    members = dl.get_sp500_constituents()
    tickers = members["ticker"].tolist()
    start = (pd.Timestamp.now("UTC") - pd.Timedelta("400d")).strftime("%Y-%m-%d")
    prices = dl.download_prices(tickers, start=start).dropna(axis=1, how="all")

    if args.date:
        asof = pd.Timestamp(args.date)
        prices = prices.loc[:asof]
    asof = prices.index[-1]
    if len(prices) <= window:
        raise SystemExit(f"only {len(prices)} rows; need > {window} for {args.horizon}")

    p_now = prices.iloc[-1]
    p_then = prices.iloc[-1 - window]
    then_date = prices.index[-1 - window]
    rets = (p_now / p_then - 1.0).replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\nAs of {asof.date()} vs {then_date.date()} ({args.horizon}), "
          f"{len(rets)} stocks\n")

    def show(label, s):
        print(f"--- {label} ({len(s)}) ---")
        tbl = pd.DataFrame({
            "return": s.map("{:+.1%}".format),
            f"price {then_date.date()}": p_then.reindex(s.index).round(2),
            f"price {asof.date()}": p_now.reindex(s.index).round(2),
            "sector": members.set_index("ticker")["sector"].reindex(s.index),
        })
        print(tbl.to_string())
        print()

    top = rets.sort_values(ascending=False).head(args.top)
    bottom = rets.sort_values().head(args.top)
    show("TOP returns", top)
    show("BOTTOM returns", bottom)

    suspect = rets[rets.abs() > SANITY].sort_values(ascending=False)
    if len(suspect):
        print(f"*** {len(suspect)} name(s) exceed +/-{SANITY:.0%} -- likely "
              f"split/adjustment glitches, inspect prices above: ***")
        print(", ".join(f"{t} ({r:+.0%})" for t, r in suspect.items()))
    else:
        print(f"No single-stock returns exceed +/-{SANITY:.0%}; "
              "spike is broad-based, not one bad print.")

    # show the metric values with and without the suspect names
    spread_all = dl.decile_spread(rets)
    xs_all = dl.cross_sectional_std(rets)
    if len(suspect):
        clean = rets.drop(suspect.index)
        print(f"\ndecile spread: {spread_all:+.1%} all  ->  "
              f"{dl.decile_spread(clean):+.1%} excluding suspects")
        print(f"xs std       : {xs_all:.1%} all  ->  "
              f"{dl.cross_sectional_std(clean):.1%} excluding suspects")


if __name__ == "__main__":
    main()
