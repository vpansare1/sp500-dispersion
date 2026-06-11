"""
Daily update -- run by GitHub Actions after the close. Updates BOTH datasets
and rebuilds every chart:

Cap-weighted (built forward, since free historical market caps don't exist):
    1. Fetch current constituents + market caps.
    2. Compute today's cap-weighted dispersion row, append idempotently to
       data/capweighted_dispersion.csv.
    3. Archive today's market caps to data/market_caps/YYYY-MM-DD.csv.gz.

Equal-weighted (extends the local backfill incrementally):
    4. Using the same trailing ~13 months of prices, compute equal-weighted
       metrics for every date after the last row of
       data/equal_weighted_dispersion.csv and append them. Because prices are
       recoverable, this self-heals: if the action skips a few days, the next
       run fills the gap (unlike cap weights, which can't be backfilled).

Charts:
    5. Rebuild all output/*.html from the accumulated CSVs.

Run:  python daily_update.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import dispersion_lib as dl

DATA_DIR = Path("data")
CAPS_DIR = DATA_DIR / "market_caps"
OUT_DIR = Path("output")
CW_CSV = DATA_DIR / "capweighted_dispersion.csv"
EW_CSV = DATA_DIR / "equal_weighted_dispersion.csv"
SPX_CSV = DATA_DIR / "spx.csv"
LOOKBACK = "400d"  # ~13 months of trading data covers the 12M horizon
SPX_START = "1994-01-01"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=True)


def update_cap_weighted(prices: pd.DataFrame, caps: pd.Series,
                        members: pd.DataFrame) -> pd.DataFrame:
    """Append today's cap-weighted row (idempotent) and archive market caps."""
    asof = prices.index[-1]
    row: dict[str, float] = {}
    for label, window in dl.HORIZONS.items():
        if len(prices) <= window:
            print(f"not enough history for {label}", file=sys.stderr)
            continue
        rets = prices.iloc[-1] / prices.iloc[-1 - window] - 1.0
        row[f"cw_spread_{label}"] = dl.decile_spread(rets, weights=caps)
        row[f"cw_xs_std_{label}"] = dl.cross_sectional_std(rets, weights=caps)
        row[f"ew_spread_{label}"] = dl.decile_spread(rets)
        row[f"ew_xs_std_{label}"] = dl.cross_sectional_std(rets)
        row[f"mad_{label}"] = dl.cross_sectional_mad(rets)
        row[f"idr_{label}"] = dl.interdecile_range(rets)
    row["n_stocks"] = int(prices.iloc[-1].notna().sum())
    row["total_mcap"] = float(caps.sum())

    new = pd.DataFrame(row, index=pd.Index([asof.normalize()], name="date"))
    hist = _read_csv(CW_CSV)
    if hist is not None:
        hist = pd.concat([hist[~hist.index.isin(new.index)], new]).sort_index()
    else:
        hist = new
    hist.to_csv(CW_CSV, float_format="%.6f")
    print(f"cap-weighted dataset: {len(hist)} rows")

    snap = members.set_index("ticker").join(caps).dropna(subset=["market_cap"])
    snap.to_csv(CAPS_DIR / f"{asof.date()}.csv.gz", compression="gzip")
    return hist


def update_equal_weighted(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-weighted metrics over the trailing window and append all
    rows newer than the existing CSV (self-healing across missed runs)."""
    combined = dl.compute_equal_weighted_metrics(prices, corr_step=1)
    # keep rows where at least the 1M cross-section exists
    combined = combined.dropna(subset=["spread_1M"])

    hist = _read_csv(EW_CSV)
    if hist is None:
        print("WARNING: no equal-weighted backfill found; starting a short "
              "series from the trailing window. Run equal_weighted_history.py "
              "locally for the full ~30y history.", file=sys.stderr)
        merged = combined
    else:
        new = combined.loc[combined.index > hist.index.max()]
        merged = pd.concat([hist, new[hist.columns.intersection(new.columns)]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        print(f"equal-weighted dataset: +{len(new)} rows -> {len(merged)} total")
    merged.to_csv(EW_CSV, index_label="date", float_format="%.6f")
    return merged


def refresh_spx() -> pd.Series:
    """Full S&P 500 index history for chart overlays (single ticker, cheap)."""
    spx = dl.download_prices(["^GSPC"], start=SPX_START).iloc[:, 0]
    spx = spx.rename("S&P 500").dropna()
    spx.to_csv(SPX_CSV, index_label="date")
    return spx


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    CAPS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    members = dl.get_sp500_constituents()
    tickers = members["ticker"].tolist()
    print(f"{len(tickers)} constituents")

    print("Fetching market caps...")
    caps = dl.get_market_caps(tickers)
    if len(caps) < 450:
        print(f"only {len(caps)} market caps retrieved -- aborting to avoid "
              "writing a bad row", file=sys.stderr)
        return 1

    print("Downloading trailing prices...")
    start = (pd.Timestamp.now("UTC") - pd.Timedelta(LOOKBACK)).strftime("%Y-%m-%d")
    prices = dl.download_prices(tickers, start=start)
    prices = prices.dropna(axis=1, how="all")
    print(f"as of {prices.index[-1].date()}: {prices.shape[1]} stocks")

    cw_hist = update_cap_weighted(prices, caps, members)
    ew_hist = update_equal_weighted(prices)

    print("Rebuilding charts...")
    try:
        spx = refresh_spx()
    except Exception as exc:  # noqa: BLE001 - fall back to cached levels
        print(f"SPX refresh failed ({exc}); using cached data/spx.csv",
              file=sys.stderr)
        cached = _read_csv(SPX_CSV)
        if cached is None:
            print("no cached SPX either -- aborting", file=sys.stderr)
            return 1
        spx = cached.iloc[:, 0].rename("S&P 500")
    dl.build_cap_weighted_charts(cw_hist, OUT_DIR)
    dl.build_equal_weighted_charts(ew_hist, spx, OUT_DIR)

    print("Daily update complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
