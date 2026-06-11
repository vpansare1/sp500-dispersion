"""
Equal-weighted historical dispersion backfill (~30 years). Run ONCE locally;
after that, daily_update.py (via GitHub Actions) extends the series forward
one day at a time and keeps the charts current.

Uses *current* S&P 500 constituents over the full history. This introduces
survivorship bias -- see README -- but is the pragmatic choice without a paid
point-in-time constituents dataset. Equal weighting needs no cap history, so
the whole series can be reconstructed in one run.

Outputs:
    data/equal_weighted_dispersion.csv     all metrics, daily
    data/spx.csv                           S&P 500 index levels (chart overlay)
    output/*.html                          all equal-weighted plotly charts

Run:  python equal_weighted_history.py
"""

from __future__ import annotations

from pathlib import Path

import dispersion_lib as dl

START = "1994-01-01"          # 12M horizon means usable data starts ~1995
DATA_DIR = Path("data")
OUT_DIR = Path("output")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("Fetching constituents...")
    members = dl.get_sp500_constituents()
    tickers = members["ticker"].tolist()
    print(f"{len(tickers)} tickers")

    print("Downloading ~30y of prices (this takes a few minutes)...")
    prices = dl.download_prices(tickers, start=START)
    prices = prices.dropna(axis=1, how="all")
    print(f"price matrix: {prices.shape[0]} days x {prices.shape[1]} stocks")

    spx = dl.download_prices(["^GSPC"], start=START).iloc[:, 0].rename("S&P 500")
    spx.to_csv(DATA_DIR / "spx.csv", index_label="date")

    print("Computing all metrics (a few minutes)...")
    combined = dl.compute_equal_weighted_metrics(prices, corr_step=5)
    combined.to_csv(DATA_DIR / "equal_weighted_dispersion.csv",
                    index_label="date", float_format="%.6f")

    print("Building charts...")
    dl.build_equal_weighted_charts(combined, spx, OUT_DIR)
    print("Done. Open output/regime_dashboard.html to start.")


if __name__ == "__main__":
    main()
