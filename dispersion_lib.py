"""
Core library for S&P 500 cross-sectional dispersion metrics.

Dispersion definitions implemented
----------------------------------
1. Decile spread        : mean return of top 10% of stocks minus mean return of
                          bottom 10% (by realized return over the horizon).
                          Intuitive, but driven entirely by the tails.
2. Cross-sectional std  : std-dev of all constituent returns on each date.
                          This is the "classic" definition (the CBOE DSPX
                          dispersion index is the implied-vol analogue).
                          Recommended as the primary series.
3. MAD                  : cross-sectional median absolute deviation. Robust to
                          outliers (a single +400% biotech won't dominate).
4. Interdecile range    : P90 - P10 of the return distribution. Robust version
                          of the decile spread (quantiles instead of tail means).

Companion regime metrics
------------------------
- Rolling average pairwise correlation (the mirror image of dispersion:
  high dispersion usually coincides with low correlation; divergences are
  interesting).
- Cross-sectional dispersion of realized volatility ("vol of vols" across
  names -- spikes when a subset of the market goes crazy, e.g. dot-com,
  meme stocks, AI names).
- Z-score / rolling percentile of any series for regime flagging.

All cross-sectional functions accept an optional weight vector so the same
code serves the equal-weighted backfill and the cap-weighted daily snapshot.
"""

from __future__ import annotations

import io
import time
import warnings

import numpy as np
import pandas as pd

# Trading-day approximations for the return horizons.
HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

TRADING_DAYS = 252
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (sp500-dispersion research script)"


# ---------------------------------------------------------------------------
# Universe & data download
# ---------------------------------------------------------------------------

def get_sp500_constituents() -> pd.DataFrame:
    """Scrape current S&P 500 constituents from Wikipedia.

    Returns a DataFrame with columns: ticker (Yahoo-style, e.g. BRK-B),
    name, sector.
    """
    import requests

    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].str.replace(".", "-", regex=False).str.strip(),
            "name": table["Security"],
            "sector": table["GICS Sector"],
        }
    )
    return out.drop_duplicates("ticker").reset_index(drop=True)


def download_prices(tickers: list[str], start: str = "1994-01-01",
                    end: str | None = None, batch_size: int = 100) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance in batches.

    Returns a DataFrame: index = dates, columns = tickers.
    """
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        data = yf.download(batch, start=start, end=end, auto_adjust=True,
                           progress=False, group_by="column", threads=True)
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(batch[0])
        frames.append(closes)
        time.sleep(1)  # be polite to Yahoo
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index()


def get_market_caps(tickers: list[str]) -> pd.Series:
    """Fetch current market caps (USD) via yfinance fast_info."""
    import yfinance as yf

    caps = {}
    for t in tickers:
        try:
            mc = yf.Ticker(t).fast_info.get("marketCap")
            if mc and mc > 0:
                caps[t] = float(mc)
        except Exception as exc:  # noqa: BLE001 - best-effort scrape
            warnings.warn(f"market cap failed for {t}: {exc}")
        time.sleep(0.05)
    return pd.Series(caps, name="market_cap")


# ---------------------------------------------------------------------------
# Cross-sectional dispersion metrics (single date)
# ---------------------------------------------------------------------------

def _clean(returns_cs: pd.Series, min_names: int) -> pd.Series | None:
    r = returns_cs.replace([np.inf, -np.inf], np.nan).dropna()
    return r if len(r) >= min_names else None


def decile_spread(returns_cs: pd.Series, weights: pd.Series | None = None,
                  pct: float = 0.10, min_names: int = 50) -> float:
    """Top-decile mean return minus bottom-decile mean return.

    If `weights` is given, the means within each decile are weight-averaged
    (decile membership is still defined by return rank, by count of names).
    """
    r = _clean(returns_cs, min_names)
    if r is None:
        return np.nan
    n = max(int(round(len(r) * pct)), 1)
    r_sorted = r.sort_values()
    bottom, top = r_sorted.iloc[:n], r_sorted.iloc[-n:]
    if weights is None:
        return float(top.mean() - bottom.mean())
    wt = weights.reindex(top.index).fillna(0.0)
    wb = weights.reindex(bottom.index).fillna(0.0)
    if wt.sum() <= 0 or wb.sum() <= 0:
        return np.nan
    return float(np.average(top, weights=wt) - np.average(bottom, weights=wb))


def cross_sectional_std(returns_cs: pd.Series, weights: pd.Series | None = None,
                        min_names: int = 50) -> float:
    """(Weighted) cross-sectional standard deviation of returns.

    Cap-weighted version: sqrt( sum_i w_i * (r_i - r_index)^2 ), the realized
    analogue of the CBOE DSPX dispersion index.
    """
    r = _clean(returns_cs, min_names)
    if r is None:
        return np.nan
    if weights is None:
        return float(r.std(ddof=1))
    w = weights.reindex(r.index).fillna(0.0)
    if w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    mu = float((w * r).sum())
    return float(np.sqrt((w * (r - mu) ** 2).sum()))


def cross_sectional_mad(returns_cs: pd.Series, min_names: int = 50) -> float:
    """Median absolute deviation around the cross-sectional median."""
    r = _clean(returns_cs, min_names)
    if r is None:
        return np.nan
    return float((r - r.median()).abs().median())


def interdecile_range(returns_cs: pd.Series, min_names: int = 50) -> float:
    """90th minus 10th percentile of the cross-sectional return distribution."""
    r = _clean(returns_cs, min_names)
    if r is None:
        return np.nan
    return float(r.quantile(0.90) - r.quantile(0.10))


# ---------------------------------------------------------------------------
# Rolling (time-series) computations
# ---------------------------------------------------------------------------

def horizon_returns(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Total return over `window` trading days for every stock, every day."""
    return prices / prices.shift(window) - 1.0


def rolling_dispersion(prices: pd.DataFrame,
                       horizons: dict[str, int] = HORIZONS,
                       metric: str = "decile_spread",
                       weights: pd.Series | None = None,
                       min_names: int = 50) -> pd.DataFrame:
    """Apply a cross-sectional dispersion metric every day for each horizon.

    Returns DataFrame indexed by date with one column per horizon label.
    """
    funcs = {
        "decile_spread": lambda r: decile_spread(r, weights, min_names=min_names),
        "xs_std": lambda r: cross_sectional_std(r, weights, min_names=min_names),
        "mad": lambda r: cross_sectional_mad(r, min_names=min_names),
        "idr": lambda r: interdecile_range(r, min_names=min_names),
    }
    fn = funcs[metric]
    out = {}
    for label, window in horizons.items():
        rets = horizon_returns(prices, window)
        out[label] = rets.apply(fn, axis=1)
    return pd.DataFrame(out)


def rolling_avg_pairwise_corr(daily_returns: pd.DataFrame, window: int = 126,
                              step: int = 5, min_names: int = 100) -> pd.Series:
    """Rolling average pairwise correlation across all constituents.

    Uses the identity: for standardized returns Z (T x N),
        sum_{i,j} corr_ij = || Z @ 1 ||^2 / (T - 1)
    so the average off-diagonal correlation is
        (sum(C) - N) / (N^2 - N)
    -- O(N*T) per window instead of O(N^2 * T).

    Evaluated every `step` days to keep a 30-year backfill fast.
    """
    vals = {}
    idx = daily_returns.index
    for end in range(window, len(idx) + 1, step):
        sub = daily_returns.iloc[end - window:end].dropna(axis=1)
        n = sub.shape[1]
        if n < min_names:
            continue
        z = (sub - sub.mean()) / sub.std(ddof=1)
        z = z.values
        s = z.sum(axis=1)                     # row sums across stocks
        total = float(s @ s) / (window - 1)   # sum of full correlation matrix
        vals[idx[end - 1]] = (total - n) / (n * n - n)
    return pd.Series(vals, name=f"avg_pairwise_corr_{window}d")


def rolling_vol_dispersion(daily_returns: pd.DataFrame, vol_window: int = 21,
                           min_names: int = 50) -> pd.DataFrame:
    """Cross-sectional dispersion of (annualized) realized volatility.

    Returns columns:
        avg_vol      - cross-sectional mean of single-stock realized vol
        vol_xs_std   - cross-sectional std of those vols (vol dispersion)
        vol_idr      - P90 - P10 of vols
    """
    vols = daily_returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    enough = vols.notna().sum(axis=1) >= min_names
    out = pd.DataFrame({
        "avg_vol": vols.mean(axis=1),
        "vol_xs_std": vols.std(axis=1),
        "vol_idr": vols.quantile(0.90, axis=1) - vols.quantile(0.10, axis=1),
    })
    return out.where(enough)


def regime_percentile(series: pd.Series, lookback: int = 756) -> pd.Series:
    """Rolling percentile rank (0-1) of the latest value vs trailing window.

    > 0.9 = unusually high dispersion regime, < 0.1 = unusually low/compressed.
    """
    return series.rolling(lookback, min_periods=lookback // 2).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True
    )


def zscore(series: pd.Series, lookback: int = 756) -> pd.Series:
    mu = series.rolling(lookback, min_periods=lookback // 2).mean()
    sd = series.rolling(lookback, min_periods=lookback // 2).std()
    return (series - mu) / sd


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Combined equal-weighted metric block (shared by backfill and daily update)
# ---------------------------------------------------------------------------

def compute_equal_weighted_metrics(prices: pd.DataFrame,
                                   corr_step: int = 5,
                                   min_names: int = 50) -> pd.DataFrame:
    """All equal-weighted dispersion + companion metrics for a price matrix.

    Returns a DataFrame with the canonical column schema used in
    data/equal_weighted_dispersion.csv:
        spread_{1M,3M,6M,12M}, xs_std_{...}, mad_12M, idr_12M,
        avg_pairwise_corr, avg_vol, vol_xs_std, vol_idr, n_stocks

    `corr_step` controls how often the (exact) average pairwise correlation
    is evaluated; use 5 for a 30y backfill, 1 for short incremental windows.
    """
    daily_rets = prices.pct_change(fill_method=None)
    spread = rolling_dispersion(prices, metric="decile_spread", min_names=min_names)
    xs_std = rolling_dispersion(prices, metric="xs_std", min_names=min_names)
    mad12 = rolling_dispersion(prices, horizons={"12M": 252}, metric="mad",
                               min_names=min_names)["12M"]
    idr12 = rolling_dispersion(prices, horizons={"12M": 252}, metric="idr",
                               min_names=min_names)["12M"]
    corr = rolling_avg_pairwise_corr(daily_rets, window=126, step=corr_step,
                                     min_names=min(min_names * 2, 100))
    vd = rolling_vol_dispersion(daily_rets, vol_window=21, min_names=min_names)

    return pd.concat(
        {
            **{f"spread_{c}": spread[c] for c in spread},
            **{f"xs_std_{c}": xs_std[c] for c in xs_std},
            "mad_12M": mad12,
            "idr_12M": idr12,
            "avg_pairwise_corr": corr.reindex(spread.index).ffill(limit=corr_step),
            "avg_vol": vd["avg_vol"],
            "vol_xs_std": vd["vol_xs_std"],
            "vol_idr": vd["vol_idr"],
            "n_stocks": prices.notna().sum(axis=1).astype(float),
        },
        axis=1,
    )


# ---------------------------------------------------------------------------
# Chart builders (write standalone HTML files + combined index dashboard)
# ---------------------------------------------------------------------------

def _equal_weighted_figures(df: pd.DataFrame, spx: pd.Series) -> list[tuple[str, "go.Figure"]]:
    """(filename-stem, figure) pairs for every equal-weighted chart."""
    horizons = list(HORIZONS)
    figs = []

    figs.append(("regime_dashboard", plot_regime_dashboard(
        dispersion=df["spread_12M"], corr=df["avg_pairwise_corr"].dropna(),
        vol_disp=df["vol_xs_std"], benchmark=spx)))

    spread = df[[f"spread_{h}" for h in horizons]]
    spread = spread.rename(columns=lambda c: c.replace("spread_", ""))
    figs.append(("eq_decile_spread", plot_dispersion(
        spread, "Equal-weighted return dispersion - top 10% minus bottom 10%",
        "Decile spread", benchmark=spx)))

    xs = df[[f"xs_std_{h}" for h in horizons]]
    xs = xs.rename(columns=lambda c: c.replace("xs_std_", ""))
    figs.append(("eq_xs_std", plot_dispersion(
        xs, "Equal-weighted cross-sectional std of returns",
        "Cross-sectional std", benchmark=spx)))

    figs.append(("eq_robust_dispersion", plot_dispersion(
        df[["mad_12M", "idr_12M"]].rename(
            columns={"mad_12M": "MAD (12M)", "idr_12M": "P90-P10 (12M)"}),
        "Robust dispersion measures, 12M horizon", "Dispersion", benchmark=spx,
        subtitle="'Robust' = insensitive to outliers: a few extreme movers barely shift these,<br>"
                 "unlike the decile spread/std. MAD = median absolute deviation of constituent "
                 "returns from the median; P90-P10 = 90th minus 10th percentile return.")))

    figs.append(("pairwise_correlation", plot_dispersion(
        df[["avg_pairwise_corr"]].rename(
            columns={"avg_pairwise_corr": "Avg pairwise corr (126d)"}),
        "Average pairwise correlation across S&P 500 constituents",
        "Correlation", benchmark=spx)))

    figs.append(("vol_dispersion", plot_dispersion(
        df[["avg_vol", "vol_xs_std", "vol_idr"]],
        "Cross-sectional volatility metrics (21d realized vol)",
        "Annualized vol", benchmark=spx)))
    return figs


def _cap_weighted_figures(hist: pd.DataFrame) -> list[tuple[str, "go.Figure"]]:
    """(filename-stem, figure) pairs for the cap-weighted charts."""
    spread_cols = [c for c in hist.columns if c.startswith("cw_spread_")]
    std_cols = [c for c in hist.columns if c.startswith("cw_xs_std_")]
    return [
        ("cap_weighted_dispersion", plot_dispersion(
            hist[spread_cols].rename(columns=lambda c: c.replace("cw_spread_", "")),
            "Cap-weighted return dispersion - top 10% minus bottom 10% "
            "(dataset accumulates daily)", "Decile spread")),
        ("cap_weighted_xs_std", plot_dispersion(
            hist[std_cols].rename(columns=lambda c: c.replace("cw_xs_std_", "")),
            "Cap-weighted cross-sectional std of returns",
            "Cross-sectional std")),
    ]


def build_equal_weighted_charts(df: pd.DataFrame, spx: pd.Series,
                                out_dir) -> None:
    """Rebuild every standalone equal-weighted chart file."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    for name, fig in _equal_weighted_figures(df, spx):
        fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")


def build_cap_weighted_charts(hist: pd.DataFrame, out_dir) -> None:
    """Rebuild standalone cap-weighted chart files from the daily CSV."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    for name, fig in _cap_weighted_figures(hist):
        fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Combined single-page dashboard (output/index.html)
# ---------------------------------------------------------------------------

def _stat_row(s: pd.Series) -> dict | None:
    """Latest value plus regime context for one metric series."""
    s = s.dropna()
    if s.empty:
        return None
    latest = float(s.iloc[-1])
    p3 = regime_percentile(s)
    z3 = zscore(s)
    return {
        "latest": latest,
        "pct_3y": float(p3.iloc[-1]) if pd.notna(p3.iloc[-1]) else np.nan,
        "z_3y": float(z3.iloc[-1]) if pd.notna(z3.iloc[-1]) else np.nan,
        "pct_full": float((s.iloc[:-1] < latest).mean()) if len(s) > 1 else np.nan,
    }


def _fmt(x, kind: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "&ndash;"
    if kind == "pct":
        return f"{x:.1%}"
    if kind == "rank":
        return f"{x:.0%}"
    return f"{x:+.2f}"


def _summary_table_html(ew: pd.DataFrame, cw: pd.DataFrame | None) -> str:
    rows = []

    def add(label, series, group):
        st = _stat_row(series)
        if st is None:
            return
        hot = ' class="hot"' if (pd.notna(st["pct_3y"]) and st["pct_3y"] >= 0.9) else ""
        cold = ' class="cold"' if (pd.notna(st["pct_3y"]) and st["pct_3y"] <= 0.1) else ""
        rows.append(
            f"<tr{hot or cold}><td>{group}</td><td>{label}</td>"
            f"<td>{_fmt(st['latest'], 'pct')}</td>"
            f"<td>{_fmt(st['pct_3y'], 'rank')}</td>"
            f"<td>{_fmt(st['z_3y'], 'z')}</td>"
            f"<td>{_fmt(st['pct_full'], 'rank')}</td></tr>"
        )

    for h in HORIZONS:
        add(f"Decile spread {h}", ew[f"spread_{h}"], "Equal-weighted")
    for h in HORIZONS:
        add(f"Cross-sectional std {h}", ew[f"xs_std_{h}"], "Equal-weighted")
    add("MAD 12M", ew["mad_12M"], "Equal-weighted")
    add("P90&ndash;P10 12M", ew["idr_12M"], "Equal-weighted")
    add("Avg pairwise corr (126d)", ew["avg_pairwise_corr"], "Correlation")
    add("Avg stock vol (21d)", ew["avg_vol"], "Volatility")
    add("Vol dispersion (xs std)", ew["vol_xs_std"], "Volatility")

    if cw is not None and len(cw):
        for h in HORIZONS:
            col = f"cw_spread_{h}"
            if col in cw:
                add(f"Decile spread {h}", cw[col], "Cap-weighted")
        for h in HORIZONS:
            col = f"cw_xs_std_{h}"
            if col in cw:
                add(f"Cross-sectional std {h}", cw[col], "Cap-weighted")

    return (
        '<table><thead><tr><th>Group</th><th>Metric</th><th>Latest</th>'
        '<th>3y %ile</th><th>3y z-score</th><th>Full-history %ile</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


_INDEX_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;
     background:#fafafa;color:#222}
header{background:#fff;border-bottom:1px solid #e3e3e3;padding:18px 28px;
       position:sticky;top:0;z-index:10}
header h1{margin:0;font-size:20px}
header .asof{color:#777;font-size:13px;margin-top:4px}
nav{margin-top:8px;font-size:13px}
nav a{margin-right:14px;color:#1f77b4;text-decoration:none}
nav a:hover{text-decoration:underline}
main{max-width:1200px;margin:0 auto;padding:20px 28px 60px}
section{background:#fff;border:1px solid #e6e6e6;border-radius:8px;
        margin:22px 0;padding:10px 14px}
h2{font-size:16px;margin:6px 4px 10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #eee}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{border-bottom:2px solid #ccc}
tr.hot td{background:#fdecea}
tr.cold td{background:#e8f1fb}
.note{color:#777;font-size:12px;margin:8px 4px}
.desc{color:#555;font-size:13px;line-height:1.5;margin:6px 6px 2px;
      max-width:980px}
.dl a{margin-right:16px;font-size:13px;color:#1f77b4}
"""


_CHART_DESCRIPTIONS = {
    "regime_dashboard":
        "The overview: index level, 12M return dispersion (red segments mark "
        "top-decile dispersion regimes vs the trailing 3 years), average "
        "pairwise correlation, and vol dispersion on one shared time axis. "
        "Reading the combination: low correlation + high dispersion is a "
        "stock-picker's / late-bubble market; high correlation + high "
        "dispersion is a macro panic.",
    "eq_decile_spread":
        "Mean return of the top 10% of constituents minus the bottom 10%, "
        "ranked by realized return over each horizon, equal-weighted. "
        "Intuitive but tail-driven: a small cohort of extreme movers can "
        "dominate it.",
    "eq_xs_std":
        "Standard deviation of all ~500 constituent returns on each date. "
        "Uses the whole return distribution rather than just the tails; the "
        "standard industry definition of dispersion and the recommended "
        "primary series.",
    "eq_robust_dispersion":
        "\"Robust\" means insensitive to outliers: these measures barely move "
        "when a handful of stocks make extreme moves, unlike the decile "
        "spread or std. MAD is the median absolute deviation of constituent "
        "returns from the cross-sectional median; P90&ndash;P10 is the 90th minus "
        "10th percentile return. When the decile spread rises but these "
        "don't, the action is concentrated in a few names; when all rise "
        "together, the whole market is genuinely differentiating.",
    "pairwise_correlation":
        "Average correlation between every pair of constituents, computed "
        "exactly over rolling 126-day windows of daily returns. The mirror "
        "image of dispersion \u2014 divergences between the two are themselves "
        "a signal.",
    "vol_dispersion":
        "Cross-sectional spread of 21-day realized volatilities: average "
        "single-stock vol, the std of vols across names, and the P90&ndash;P10 "
        "vol range. Spikes when a subset of the market decouples (dot-com, "
        "meme stocks, AI names) even while index vol stays calm \u2014 often "
        "leads return dispersion.",
    "cap_weighted_dispersion":
        "Same decile-spread construction, but returns within each decile are "
        "market-cap weighted. Built forward one row per day by the GitHub "
        "Action (historical cap weights aren't freely available), so this "
        "series starts at the first action run and grows daily.",
    "cap_weighted_xs_std":
        "Cap-weighted cross-sectional std: \u221a\u03a3w\u1d62(r\u1d62\u2212r_index)\u00b2 \u2014 the "
        "realized analogue of the CBOE DSPX dispersion index. Accumulates "
        "daily alongside the chart above.",
}


def build_index_dashboard(ew: pd.DataFrame, cw: pd.DataFrame | None,
                          spx: pd.Series, out_dir) -> None:
    """One self-contained page: summary stats table + every chart.

    Written to <out_dir>/index.html (so enabling GitHub Pages on the repo
    serves it directly). plotly.js is loaded once from the CDN and shared by
    all figures on the page.
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    sections = _equal_weighted_figures(ew, spx)
    if cw is not None and len(cw):
        sections += _cap_weighted_figures(cw)

    nav, body, first = [], [], True
    for name, fig in sections:
        title = fig.layout.title.text or name
        nav.append(f'<a href="#{name}">{title.split(" - ")[0]}</a>')
        inner = fig.to_html(full_html=False, div_id=name + "_plot",
                            include_plotlyjs="cdn" if first else False,
                            default_height="560px"
                            if name != "regime_dashboard" else "1100px")
        first = False
        desc = _CHART_DESCRIPTIONS.get(name, "")
        desc_html = f'<p class="desc">{desc}</p>' if desc else ""
        body.append(f'<section id="{name}">{desc_html}{inner}</section>')

    asof = ew.index.max()
    cw_note = ""
    if cw is not None and len(cw):
        cw_note = (f'<p class="note">Cap-weighted history accumulating since '
                   f'{cw.index.min().date()} ({len(cw)} trading days); regime '
                   f'percentiles appear once enough history builds up.</p>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S&amp;P 500 Dispersion Monitor</title>
<style>{_INDEX_CSS}</style></head>
<body>
<header>
  <h1>S&amp;P 500 Dispersion Monitor</h1>
  <div class="asof">Data through {asof.date()} &middot; {int(ew["n_stocks"].iloc[-1])} constituents</div>
  <nav><a href="#summary">Summary</a>{''.join(nav)}</nav>
</header>
<main>
<section id="summary">
  <h2>Latest readings &amp; regime context</h2>
  {_summary_table_html(ew, cw)}
  <p class="note">3y %ile / z-score: latest value vs trailing 756 trading days.
  Rows shaded red are in the top decile of their 3-year range (high-dispersion
  regime); blue rows are in the bottom decile (compressed regime).</p>
  {cw_note}
  <p class="dl">Datasets:
    <a href="../data/equal_weighted_dispersion.csv">equal_weighted_dispersion.csv</a>
    <a href="../data/capweighted_dispersion.csv">capweighted_dispersion.csv</a>
  </p>
</section>
{''.join(body)}
</main></body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_dispersion(df: pd.DataFrame, title: str, yaxis_title: str = "Dispersion",
                    benchmark: pd.Series | None = None,
                    benchmark_name: str = "S&P 500",
                    subtitle: str | None = None) -> "go.Figure":
    """Multi-horizon dispersion lines, optional index level on a log right axis.

    Header layout (top to bottom, non-overlapping): title [+ subtitle],
    legend row, range-selector buttons, then the plot.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=str(col), mode="lines",
            line=dict(width=1.4, color=_COLORS[i % len(_COLORS)]),
        ))
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values, name=benchmark_name,
            mode="lines", yaxis="y2",
            line=dict(width=1, color="rgba(120,120,120,0.55)"),
        ))
        fig.update_layout(yaxis2=dict(title=benchmark_name, overlaying="y",
                                      side="right", type="log", showgrid=False))
    title_block = dict(text=title, yref="container", yanchor="top", y=0.985)
    if subtitle:
        title_block["subtitle"] = dict(text=subtitle,
                                       font=dict(size=12, color="#666"))
    fig.update_layout(
        title=title_block, template="plotly_white", hovermode="x unified",
        yaxis=dict(title=yaxis_title, tickformat=".1%"),
        # one header row above the plot: range-selector buttons on the left,
        # legend right-aligned -- vertically separated from the title at any
        # window height (a stacked layout drifts because paper coordinates
        # scale with plot height)
        legend=dict(orientation="h", yanchor="bottom", y=1.03,
                    xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(
            x=0, xanchor="left", y=1.03, yanchor="bottom",
            buttons=[
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all"),
            ])),
        margin=dict(t=140 if subtitle else 115),
    )
    return fig


def plot_regime_dashboard(dispersion: pd.Series, corr: pd.Series,
                          vol_disp: pd.Series, benchmark: pd.Series,
                          dispersion_name: str = "12M decile spread") -> "go.Figure":
    """4-panel dashboard: index, dispersion (+ percentile shading), correlation,
    vol dispersion. Shared x-axis for visual regime comparison."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pct = regime_percentile(dispersion)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.28, 0.28, 0.22, 0.22],
        subplot_titles=(
            "S&P 500 (log scale)", f"Return dispersion: {dispersion_name}",
            "Average pairwise correlation (126d)",
            "Cross-sectional std of 21d realized vol",
        ),
    )
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="S&P 500",
                             line=dict(color="#444", width=1.2)), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    fig.add_trace(go.Scatter(x=dispersion.index, y=dispersion.values,
                             name=dispersion_name,
                             line=dict(color="#1f77b4", width=1.4)), row=2, col=1)
    # shade extreme-dispersion regimes (rolling 3y percentile > 90%)
    hi = pct > 0.90
    fig.add_trace(go.Scatter(
        x=dispersion.index, y=dispersion.where(hi).values, name="Top-decile regime",
        mode="lines", line=dict(color="#d62728", width=2.2)), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)

    fig.add_trace(go.Scatter(x=corr.index, y=corr.values, name="Avg pairwise corr",
                             line=dict(color="#2ca02c", width=1.4)), row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)

    fig.add_trace(go.Scatter(x=vol_disp.index, y=vol_disp.values,
                             name="Vol dispersion",
                             line=dict(color="#9467bd", width=1.4)), row=4, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)

    fig.update_layout(template="plotly_white", height=1100, hovermode="x unified",
                      title=dict(text="S&P 500 dispersion regime dashboard",
                                 yref="container", yanchor="top", y=0.992),
                      legend=dict(orientation="h", yanchor="bottom", y=1.022,
                                  xanchor="right", x=1),
                      margin=dict(t=110), showlegend=True)
    return fig
