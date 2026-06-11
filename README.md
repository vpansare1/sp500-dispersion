# S&P 500 Dispersion Monitor

Tracks cross-sectional return dispersion across S&P 500 constituents to help
identify regime changes (bubbles, rotations, macro-driven vs stock-specific
markets).

## Layout

```
dispersion_lib.py             shared metric + chart functions
equal_weighted_history.py     one-shot ~30y equal-weighted backfill (run locally once)
daily_update.py               daily GH Action: appends the cap-weighted row,
                              incrementally extends the equal-weighted series,
                              and rebuilds ALL charts
.github/workflows/            daily cron (07:00 UTC = 2am EST / 3am EDT)
data/                         CSV datasets (committed by the action)
data/market_caps/             daily market-cap snapshots (gzip CSV)
output/                       standalone plotly HTML charts
```

## Quick start

```bash
pip install -r requirements.txt
python equal_weighted_history.py   # ~5-10 min: builds the 30y history locally
git add data output && git commit && git push
```

From then on the daily action does everything: it appends one cap-weighted
row, extends the equal-weighted history with any missing recent dates (the
trailing 13 months of prices it downloads is enough to compute every metric,
so this is incremental -- no 30y re-download), refreshes the S&P 500 overlay
series, and rebuilds every chart in `output/`. The equal-weighted update is
self-healing: if the cron skips a day, the next run fills the gap from
prices. Cap-weighted gaps stay gaps (same-day market caps aren't
recoverable). The raw market-cap snapshot is archived daily so you can
recompute anything later under different definitions.

If you skip the local backfill, the action still works -- the equal-weighted
series just starts from ~today instead of 1995.

## Dispersion definitions

| Metric | Function | Notes |
|---|---|---|
| Decile spread | `decile_spread` | mean(top 10%) − mean(bottom 10%) by horizon return. Intuitive; tail-driven. |
| Cross-sectional std | `cross_sectional_std` | Std of all constituent returns. The standard academic/industry definition; the cap-weighted version is the realized analogue of CBOE's DSPX. **Recommended primary series.** |
| MAD | `cross_sectional_mad` | Median absolute deviation — robust to a few extreme movers. |
| Interdecile range | `interdecile_range` | P90 − P10 — robust cousin of the decile spread. |

Horizons: 1, 3, 6, 12 months (21/63/126/252 trading days).

## Companion regime metrics (also computed)

- **Average pairwise correlation** (126d rolling, exact average via the
  standardized-row-sum identity, evaluated every 5 days). Correlation and
  dispersion are near-mirror images; divergences are informative. Low
  correlation + high dispersion = stock-picker's / late-bubble market; high
  correlation + high dispersion = macro panic.
- **Vol dispersion**: cross-sectional std (and P90−P10) of 21d realized vols.
  Spikes when a subset of names decouples (dot-com 1999, meme stocks 2021,
  AI names 2024-25) even while index vol stays calm.
- **Average single-stock vol** vs index vol (implicit in the dashboard).
- **Regime flags**: rolling 3-year percentile / z-score of any series;
  the dashboard highlights top-decile dispersion regimes in red.

## Caveats

- **Survivorship bias** (equal-weighted backfill): using today's members over
  30 years overstates past returns of the cross-section and slightly
  understates dispersion in crises (the worst tail — delisted names like
  Enron, Lehman, SVB — is missing). Levels are still useful for *relative*
  regime comparison; just don't treat 1990s readings as precise.
- **Yahoo Finance** is unofficial: throttling and schema changes happen.
  The snapshot aborts (exit 1) rather than write a row if <450 market caps
  come back.
- **GitHub cron is best-effort**: runs can be delayed or skipped on busy
  days. Missed days are simply gaps in the cap-weighted set (prices can be
  backfilled, same-day market caps cannot). `workflow_dispatch` lets you run
  manually.
- The 1M decile spread is noisy; the 6M/12M series are usually the cleaner
  regime indicators.

## Ideas for extension

- **Sector vs idiosyncratic decomposition**: dispersion of sector-average
  returns vs within-sector dispersion (the Wikipedia table already includes
  GICS sector). Tells you whether a regime is a sector rotation or
  single-name driven.
- **Implied dispersion**: CBOE DSPX and the implied-correlation indices
  (COR1M/COR3M) as forward-looking overlays vs these realized series.
- **Forward-return conditioning**: bucket dates by dispersion percentile and
  tabulate subsequent 6-12M index returns / drawdowns — the actual test of
  whether high dispersion flags bubbles.
- **Breadth**: % of members above their 200dma; divergence between breadth
  and index level is a classic late-cycle signal that pairs naturally with
  rising dispersion.
- **Winner concentration**: share of index return explained by the top 10
  contributors (needs the cap snapshots this repo is already archiving).
