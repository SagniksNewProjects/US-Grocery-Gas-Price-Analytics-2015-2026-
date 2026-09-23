"""
main.py
=======
US Grocery & Gas Price Analytics — single-file Streamlit application.

Contains both the analytics backend (data pipeline, rolling averages,
inflation adjustment, lag correlation engine, forecasting, insight engine)
and the Streamlit frontend (KPI cards, interactive Plotly charts, tabs).

Run:
    streamlit run main.py

Data required (place in the same folder as this file):
    us_average_prices_wide.csv
    us_average_prices_monthly.csv
    us_average_prices_item_summary.csv

Source: US Bureau of Labor Statistics (BLS) Average Retail Prices, 2015–2026.
GitHub: https://github.com/SagniksNewProjects/US-Grocery-Gas-Price-Analytics-2015-2026-
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ANALYTICS ENGINE
# ════════════════════════════════════════════════════════════════════════════════
# All functions are pure: they accept DataFrames / scalars and return DataFrames
# or plain Python objects.  No Streamlit calls appear in this section.

# ── Constants ────────────────────────────────────────────────────────────────

_INFLATION_ANNUAL_RATE: float = 0.035   # ~3.5 % /yr  →  ~35 % cumulative decade


# ── 1a. Data pipeline ────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every column name."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect the date column, coerce it to datetime, and set it as the index.

    Priority order:
      1. Exact name match: 'date', 'period', 'date_period'
      2. Name contains 'date' or 'period' (but NOT bare 'year' / 'month')

    Skips integer-only columns (year=2015, month=1) that look like dates
    in name but contain integer data.
    """
    priority = [c for c in df.columns if c.lower() in ("date", "period", "date_period")]
    fallback = [
        c for c in df.columns
        if c not in priority
        and ("date" in c.lower() or "period" in c.lower())
        and c.lower() not in ("year", "month")
    ]
    for col in priority + fallback:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df = df.copy()
                df[col] = parsed
                df = df.dropna(subset=[col]).set_index(col).sort_index()
                return df
        except Exception:
            continue
    return df


def load_wide(path: str | Path) -> pd.DataFrame:
    """
    Load the wide-format CSV (rows = months, columns = price items).
    Returns a datetime-indexed DataFrame of floats, forward-filled.
    """
    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df)
    df = _parse_date_column(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.ffill().bfill()


def load_monthly(path: str | Path) -> pd.DataFrame:
    """
    Load the long-format monthly CSV (date, year, month, item, unit,
    category, price, series_id) and pivot it to wide format.
    Returns a datetime-indexed DataFrame (rows=months, columns=items).
    """
    df = pd.read_csv(path, low_memory=False)
    df = _normalise_columns(df)

    item_col = next(
        (c for c in df.columns if c.lower() == "item"),
        next((c for c in df.columns if "item" in c.lower() or "product" in c.lower()), None),
    )
    val_col = next(
        (c for c in df.columns if c.lower() == "price"),
        next((c for c in df.columns if "price" in c.lower() or "value" in c.lower()), None),
    )

    df = _parse_date_column(df)

    if item_col and val_col and item_col in df.columns and val_col in df.columns:
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        wide = df.pivot_table(index=df.index, columns=item_col, values=val_col, aggfunc="mean")
        wide.index.name = "date"
        return wide.ffill().bfill()

    return df.select_dtypes(include="number").ffill().bfill()


def load_item_summary(path: str | Path) -> pd.DataFrame:
    """Load the pre-aggregated item summary CSV."""
    df = pd.read_csv(path, low_memory=False)
    return _normalise_columns(df)


def merge_datasets(wide_path: str | Path, monthly_path: str | Path) -> pd.DataFrame:
    """
    Merge the wide and monthly DataFrames into a single unified time-series.
    Prefers wide values; supplements with monthly where coverage is missing.
    Raises ValueError (with per-source error details) if both loaders fail.
    """
    wide_err = monthly_err = None

    try:
        wide = load_wide(wide_path)
    except Exception as e:
        wide, wide_err = pd.DataFrame(), e

    try:
        monthly = load_monthly(monthly_path)
    except Exception as e:
        monthly, monthly_err = pd.DataFrame(), e

    if wide.empty and monthly.empty:
        raise ValueError(
            f"Both datasets failed to load.\n  wide: {wide_err}\n  monthly: {monthly_err}"
        )
    if wide.empty:
        return monthly
    if monthly.empty:
        return wide

    combined = wide.combine_first(monthly)
    return combined.sort_index().ffill().bfill()


# ── 1b. Macro stability filter ───────────────────────────────────────────────

def compute_rolling(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """12-month rolling mean across all columns (min_periods=3)."""
    return df.rolling(window=window, min_periods=3).mean()


# ── 1c. Inflation adjustment ─────────────────────────────────────────────────

def _inflation_index(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Compound monthly inflation multiplier anchored to the earliest date.
    Based on _INFLATION_ANNUAL_RATE (~3.5 %/yr ≈ 35 % over a decade).
    """
    r_m = (1 + _INFLATION_ANNUAL_RATE) ** (1 / 12) - 1
    origin = dates.min()
    t = ((dates.year - origin.year) * 12 + (dates.month - origin.month)).astype(float)
    return pd.Series((1 + r_m) ** t, index=dates, name="inflation_factor")


def adjust_for_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Deflate nominal prices to real prices (base = first date in index)."""
    return df.div(_inflation_index(df.index), axis=0)


# ── 1d. Lagged correlation engine ────────────────────────────────────────────

def compute_lagged_correlation(
    df: pd.DataFrame,
    fuel_col: str,
    grocery_col: str,
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Pearson correlation between fuel_col and grocery_col at lags 0…max_lag.
    Uses np.corrcoef on raw ndarrays to avoid Pandas 3.x Series.corr(ndarray)
    TypeError.

    Returns a DataFrame with columns: lag, correlation, interpretation.
    """
    if fuel_col not in df.columns or grocery_col not in df.columns:
        return pd.DataFrame(columns=["lag", "correlation", "interpretation"])

    fuel   = df[fuel_col].dropna()
    grocer = df[grocery_col].dropna()
    common = fuel.index.intersection(grocer.index)
    fuel   = fuel.loc[common]
    grocer = grocer.loc[common]

    rows = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = fuel.corr(grocer)
        else:
            f_arr = fuel.values[lag:].astype(float)
            g_arr = grocer.values[:-lag].astype(float)
            n     = min(len(f_arr), len(g_arr))
            f_arr, g_arr = f_arr[:n], g_arr[:n]
            mask  = ~(np.isnan(f_arr) | np.isnan(g_arr))
            corr  = float(np.corrcoef(f_arr[mask], g_arr[mask])[0, 1]) if mask.sum() >= 3 else float("nan")

        interp = "Strong" if abs(corr) >= 0.7 else ("Moderate" if abs(corr) >= 0.4 else "Weak")
        rows.append({"lag": lag, "correlation": round(corr, 4), "interpretation": interp})

    return pd.DataFrame(rows)


def full_correlation_matrix(
    df: pd.DataFrame, fuel_col: str, max_lag: int = 6
) -> pd.DataFrame:
    """
    Build a lag × item correlation matrix (rows=items, columns=lag_0…lag_N).
    Used to power the heatmap in Tab 2.
    """
    if fuel_col not in df.columns:
        return pd.DataFrame()

    fuel_arr = df[fuel_col].values.astype(float)
    result   = {}

    for col in [c for c in df.columns if c != fuel_col]:
        groc_arr = df[col].values.astype(float)
        row = {}
        for lag in range(max_lag + 1):
            f_a = fuel_arr[lag:] if lag else fuel_arr
            g_a = groc_arr[:-lag] if lag else groc_arr
            n   = min(len(f_a), len(g_a))
            f_a, g_a = f_a[:n], g_a[:n]
            mask = ~(np.isnan(f_a) | np.isnan(g_a))
            row[f"lag_{lag}"] = (
                float(np.corrcoef(f_a[mask], g_a[mask])[0, 1]) if mask.sum() >= 3 else float("nan")
            )
        result[col] = row

    return pd.DataFrame(result).T.round(3)


# ── 1e. KPI calculations ─────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, col: str) -> Dict:
    """
    Return a dict of KPI values for a single price column.

    Keys: peak_price, peak_date, min_price, min_date, start_price,
          end_price, decade_change_pct, latest_price, latest_date, volatility.
    """
    if col not in df.columns:
        return {}
    series = df[col].dropna()
    if series.empty:
        return {}

    start  = series.iloc[0]
    end    = series.iloc[-1]
    change = ((end - start) / start * 100) if start != 0 else 0

    return {
        "peak_price":        round(series.max(), 3),
        "peak_date":         series.idxmax().strftime("%b %Y"),
        "min_price":         round(series.min(), 3),
        "min_date":          series.idxmin().strftime("%b %Y"),
        "start_price":       round(start, 3),
        "end_price":         round(end, 3),
        "decade_change_pct": round(change, 1),
        "latest_price":      round(end, 3),
        "latest_date":       series.index[-1].strftime("%b %Y"),
        "volatility":        round(series.std(), 4),
    }


# ── 1f. Forecasting engine ───────────────────────────────────────────────────

def _linear_forecast(
    series: pd.Series, months_ahead: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Linear-regression fallback forecast when Prophet is unavailable.
    Returns (yhat, lower_95, upper_95) as date-indexed Series.
    """
    ts = series.dropna().reset_index()
    ts.columns = ["ds", "y"]
    ts["t"] = np.arange(len(ts))

    model = LinearRegression().fit(ts[["t"]], ts["y"])
    last  = ts["ds"].iloc[-1]
    ft    = np.arange(len(ts), len(ts) + months_ahead).reshape(-1, 1)
    fds   = pd.date_range(start=last, periods=months_ahead + 1, freq="MS")[1:]
    yhat  = model.predict(ft)
    sigma = (ts["y"].values - model.predict(ts[["t"]])).std()

    return (
        pd.Series(yhat,                  index=fds, name="forecast"),
        pd.Series(yhat - 1.96 * sigma,   index=fds, name="lower"),
        pd.Series(yhat + 1.96 * sigma,   index=fds, name="upper"),
    )


def forecast_item(
    df: pd.DataFrame, col: str, months_ahead: int = 36
) -> pd.DataFrame:
    """
    Forecast prices for `col` up to `months_ahead` months into the future.

    Attempts Facebook Prophet (multiplicative seasonality, 95 % CI).
    Falls back to linear regression + residual-based CI if Prophet is absent.

    Returns a DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, source
    where source ∈ {'historical', 'forecast'}.
    """
    if col not in df.columns:
        return pd.DataFrame()
    series = df[col].dropna()

    # ── Try Prophet ──────────────────────────────────────────────────────────
    try:
        from prophet import Prophet  # type: ignore
        ts = series.reset_index()
        ts.columns = ["ds", "y"]
        ts["ds"] = pd.to_datetime(ts["ds"])
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.95,
            changepoint_prior_scale=0.15,
        )
        m.fit(ts)
        future = m.make_future_dataframe(periods=months_ahead, freq="MS")
        pred   = m.predict(future)
        pred["source"] = np.where(pred["ds"].isin(ts["ds"]), "historical", "forecast")
        return pred[["ds", "yhat", "yhat_lower", "yhat_upper", "source"]]
    except Exception:
        pass

    # ── Fallback: linear regression ──────────────────────────────────────────
    fc, lo, hi = _linear_forecast(series, months_ahead)
    hist = series.reset_index()
    hist.columns = ["ds", "yhat"]
    hist["yhat_lower"] = hist["yhat"]
    hist["yhat_upper"] = hist["yhat"]
    hist["source"]     = "historical"
    future = pd.DataFrame({
        "ds": fc.index, "yhat": fc.values,
        "yhat_lower": lo.values, "yhat_upper": hi.values, "source": "forecast",
    })
    return pd.concat([hist, future], ignore_index=True)


# ── 1g. Insight engine ───────────────────────────────────────────────────────

def generate_insights(
    kpis: Dict,
    selected_item: str,
    lag_df: pd.DataFrame | None,
    mode: str,
    inflation_on: bool,
) -> List[str]:
    """
    Generate a list of dynamic business & policy insight strings based on
    the current KPIs, lag correlation result, and active user filters.
    """
    if not kpis:
        return ["No data available for selected item."]

    insights: List[str] = []
    change = kpis.get("decade_change_pct", 0)
    vol    = kpis.get("volatility", 0)
    item   = selected_item.split(",")[0] if selected_item else "This item"

    if change > 50:
        insights.append(
            f"🚨 **Severe Price Escalation:** {item} has surged {change:.1f}% over the decade — "
            "nearly double typical inflation. Procurement teams should explore long-term fixed-price contracts."
        )
    elif change > 25:
        insights.append(
            f"⚠️ **Above-Inflation Growth:** {item} rose {change:.1f}% — outpacing the 35% baseline CPI. "
            "Retailers should hedge inventory 3–6 months ahead during downward-trend windows."
        )
    elif change > 0:
        insights.append(
            f"✅ **Stable Pricing:** {item} increased {change:.1f}% — broadly in line with consumer inflation. "
            "Standard reorder cycles remain appropriate."
        )
    else:
        insights.append(
            f"📉 **Deflationary Pressure:** {item} declined {abs(change):.1f}% in real terms — "
            "a strong buyer's market. Opportunistic bulk-purchasing could yield margin gains."
        )

    if vol > 0.5:
        insights.append(
            f"📊 **High Volatility (σ = {vol:.2f}):** Price swings are large. "
            "Policy-makers should monitor supply-chain disruptions and consider strategic reserves."
        )
    elif vol > 0.2:
        insights.append(
            f"📈 **Moderate Volatility (σ = {vol:.2f}):** Seasonal hedging (futures or forward purchases "
            "in Q3) is advisable to smooth cost structures."
        )

    if inflation_on:
        insights.append(
            "💡 **Inflation-Adjusted View Active:** You are viewing real prices (2015 baseline). "
            "Nominal sticker-price increases overstate true cost growth; "
            "real changes reveal genuine affordability trends."
        )

    if mode == "Trend (12-mo Rolling Avg)":
        insights.append(
            "📉 **Trend Mode:** Short-term noise is removed. Use this view to identify structural "
            "multi-year shifts suitable for annual budget forecasting."
        )

    if lag_df is not None and not lag_df.empty:
        best     = lag_df.loc[lag_df["correlation"].abs().idxmax()]
        lag_val  = int(best["lag"])
        corr_val = float(best["correlation"])
        if lag_val == 0:
            insights.append(
                f"⛽ **Simultaneous Fuel–Grocery Link (r = {corr_val:.2f}):** "
                "Gasoline price changes transmit to grocery costs immediately — "
                "logistics cost pass-through is near-instantaneous."
            )
        elif lag_val <= 2:
            insights.append(
                f"⛽ **Short Fuel→Grocery Lag ({lag_val} months, r = {corr_val:.2f}):** "
                "Grocery prices respond to fuel spikes within 1–2 months. "
                "Retailers should pre-stock before fuel price upswings."
            )
        else:
            insights.append(
                f"⛽ **Extended Fuel→Grocery Lag ({lag_val} months, r = {corr_val:.2f}):** "
                f"Supply-chain buffering delays pass-through. "
                f"A {lag_val}-month forward-buying window exists when fuel prices trend up."
            )

    insights.append(
        "🏛️ **Policy Note:** The USDA recommends that food-insecure households allocate ≤ 30% of income "
        "to food expenditure. Rising real prices disproportionately affect lower-income quintiles; "
        "targeted SNAP benefit adjustments may be warranted when real grocery prices exceed the 5-year moving average."
    )
    return insights


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STREAMLIT FRONTEND
# ════════════════════════════════════════════════════════════════════════════════

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="US Grocery & Gas Price Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — only <style> blocks are passed through st.markdown ───────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
    #MainMenu, footer, header  { visibility: hidden; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
    .main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1600px; }
    .js-plotly-plot .plotly .modebar       { opacity: 0.4; }
    .js-plotly-plot .plotly .modebar:hover { opacity: 1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── UI helpers ────────────────────────────────────────────────────────────────

_SH = (
    "font-size:0.70rem;font-weight:600;letter-spacing:0.12em;"
    "text-transform:uppercase;color:#8b949e;"
    "margin:1.8rem 0 0.4rem 0;padding-bottom:0.3rem;"
    "border-bottom:1px solid #21262d;"
)


def section_header(text: str) -> None:
    """Render a small all-caps section label via st.html() (bypasses sanitizer)."""
    st.html(f'<p style="{_SH}">{text}</p>')


def short_name(col: str) -> str:
    """Return a compact display name from a BLS column string."""
    return col.split(",")[0].strip() if "," in col else col[:45]


# ── Plotly dark theme ─────────────────────────────────────────────────────────

CHART_PALETTE = [
    "#58a6ff", "#3fb950", "#e3b341", "#f85149", "#a371f7",
    "#79c0ff", "#56d364", "#ffa657", "#ff7b72", "#d2a8ff",
]

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="Inter, system-ui, sans-serif", color="#c9d1d9", size=12),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zeroline=False),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zeroline=False),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font_size=12),
        colorway=CHART_PALETTE,
        margin=dict(l=48, r=20, t=48, b=40),
    )
)

ACCENT = {
    "fuel": "#e3b341", "grocery": "#58a6ff", "trend": "#a371f7",
    "real": "#3fb950",  "forecast": "#ffa657", "ci": "rgba(255,166,87,0.15)",
}

# ── Data paths ────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent


def _get_path(filename: str, override: str | None) -> Path:
    if override and Path(override).exists():
        return Path(override)
    candidate = _DATA_DIR / filename
    return candidate if candidate.exists() else Path(filename)


# ── Cached data loading ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading datasets…", ttl=3600)
def _load_all(wide: str, monthly: str, summary: str):
    return merge_datasets(wide, monthly), load_item_summary(summary)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.html(
        "<div style='padding:1.2rem 0 1.5rem 0;'>"
        "<div style='font-size:1.35rem;font-weight:800;color:#e6edf3;"
        "letter-spacing:-0.02em;'>🛒 GroceryMetrics</div>"
        "<div style='font-size:0.75rem;color:#8b949e;margin-top:0.2rem;'>"
        "US Prices · 2015–2026</div></div>"
    )

    st.markdown("#### 📂 Data Sources")
    _w_ov = st.text_input("Wide CSV path",    placeholder="auto-detect")
    _m_ov = st.text_input("Monthly CSV path", placeholder="auto-detect")
    _s_ov = st.text_input("Summary CSV path", placeholder="auto-detect")

    wide_path    = _get_path("us_average_prices_wide.csv",         _w_ov or None)
    monthly_path = _get_path("us_average_prices_monthly.csv",      _m_ov or None)
    summary_path = _get_path("us_average_prices_item_summary.csv", _s_ov or None)

    st.divider()
    st.markdown("#### 🎛️ Analytics Controls")

    price_mode = st.radio(
        "Price Display",
        options=["Raw Prices", "Trend (12-mo Rolling Avg)"],
        index=0,
        help="Toggle between raw monthly prices and a 12-month rolling average.",
    )
    inflation_adj = st.toggle(
        "Inflation-Adjust (Real Prices)", value=False,
        help="Deflate nominal prices to real 2015-baseline values (35% decade CPI).",
    )

    st.divider()
    st.markdown("#### 📅 Date Range")
    date_start = st.slider("Start Year", min_value=2015, max_value=2025, value=2015)
    date_end   = st.slider("End Year",   min_value=2015, max_value=2026, value=2026)

    st.divider()
    st.markdown("#### ⛽ Fuel vs Grocery Lag")
    max_lag = st.slider("Max Lag (months)", min_value=1, max_value=12, value=6)

    st.divider()
    st.markdown("#### 🔮 Forecast")
    forecast_months = st.slider("Months to Forecast", min_value=6, max_value=60, value=36, step=6)

    st.divider()
    st.html(
        "<div style='font-size:0.68rem;color:#484f58;padding-top:0.5rem;'>"
        "Data: US BLS Average Retail Prices · Built with Streamlit &amp; Plotly</div>"
    )


# ════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════

try:
    df_raw, df_summary = _load_all(str(wide_path), str(monthly_path), str(summary_path))
except Exception as exc:
    st.error(
        f"**Could not load datasets.** Place the three CSV files in the same folder "
        f"as `main.py`, or enter their full paths in the sidebar.\n\n`{exc}`"
    )
    st.stop()

mask = (df_raw.index.year >= date_start) & (df_raw.index.year <= date_end)
df   = df_raw.loc[mask].copy()

if df.empty:
    st.warning("No data in the selected date range. Adjust the year sliders.")
    st.stop()

# ── Apply price-mode transformations ─────────────────────────────────────────

df_display = compute_rolling(df) if price_mode == "Trend (12-mo Rolling Avg)" else df.copy()
if inflation_adj:
    df_display = adjust_for_inflation(df_display)

# ── Detect fuel column ────────────────────────────────────────────────────────

all_cols = df_display.columns.tolist()
_fuel_kw = ["gasoline", "unleaded", "fuel oil", "automotive diesel"]
fuel_col = next((c for c in all_cols if any(k in c.lower() for k in _fuel_kw)), all_cols[0] if all_cols else None)

col_display_map = {c: short_name(c) for c in all_cols}
display_col_map = {v: k for k, v in col_display_map.items()}

# ── Sidebar: item selector ────────────────────────────────────────────────────

with st.sidebar:
    st.divider()
    st.markdown("#### 🛍️ Item Selection")

    selected_display = st.selectbox(
        "Grocery / Gas Item",
        options=[col_display_map[c] for c in all_cols],
        index=0,
    )
    selected_col = display_col_map.get(selected_display, all_cols[0])

    compare_display = st.multiselect(
        "Compare with (up to 5)",
        options=[col_display_map[c] for c in all_cols if c != selected_col],
        default=[col_display_map[c] for c in all_cols[1:4] if c != selected_col][:3],
        max_selections=5,
    )
    compare_cols = [display_col_map[d] for d in compare_display]


# ════════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════════════════════

kpis = compute_kpis(df_display, selected_col)

_KPI_ACCENTS = {
    "accent-blue": "#1f6feb", "accent-red":    "#b62324",
    "accent-green": "#196c2e", "accent-purple": "#6e40c9", "accent-gold": "#9e6a03",
}
_BADGE_COL = {"up": "#f85149", "down": "#3fb950", "flat": "#e3b341"}


def render_kpi(label: str, value: str, sub: str, accent: str,
               badge: str = "", badge_class: str = "") -> None:
    """Render a styled KPI card using st.html() to avoid Streamlit sanitization."""
    bar  = _KPI_ACCENTS.get(accent, "#1f6feb")
    bcol = _BADGE_COL.get(badge_class, "#8b949e")
    bdg  = (
        f'<div style="color:{bcol};font-size:0.80rem;font-weight:600;'
        f'margin:0.15rem 0 0.1rem 0;">{badge}</div>'
    ) if badge else ""
    st.html(
        f'<div style="background:linear-gradient(145deg,#161b22,#0d1117);'
        f'border:1px solid #30363d;border-radius:12px;padding:1.2rem 1.4rem;'
        f'overflow:hidden;border-top:3px solid {bar};margin-bottom:0.5rem;">'
        f'<div style="font-size:0.70rem;font-weight:600;letter-spacing:0.10em;'
        f'text-transform:uppercase;color:#8b949e;margin-bottom:0.35rem;">{label}</div>'
        f'<div style="font-size:1.85rem;font-weight:700;color:#e6edf3;'
        f'line-height:1.1;letter-spacing:-0.02em;">{value}</div>'
        f'{bdg}'
        f'<div style="font-size:0.75rem;color:#8b949e;margin-top:0.35rem;">{sub}</div>'
        f'</div>'
    )


# Hero banner
_bs = (
    "display:inline-block;background:#1f6feb22;border:1px solid #1f6feb66;"
    "color:#58a6ff;border-radius:999px;padding:0.18rem 0.75rem;font-size:0.72rem;"
    "font-weight:600;letter-spacing:0.08em;text-transform:uppercase;"
    "margin-right:0.4rem;margin-bottom:0.6rem;"
)
_pl = "Real Prices" if inflation_adj else "Nominal Prices"
st.html(
    f'<div style="margin-bottom:0.3rem;">'
    f'<span style="{_bs}">US BLS Data</span><span style="{_bs}">2015–2026</span>'
    f'<span style="{_bs}">{_pl}</span><span style="{_bs}">{price_mode}</span></div>'
    f'<div style="font-size:2.0rem;font-weight:800;color:#e6edf3;'
    f'letter-spacing:-0.03em;line-height:1.15;margin-bottom:0.2rem;">'
    f'US Grocery &amp; Gas Price Analytics</div>'
    f'<div style="font-size:0.92rem;color:#8b949e;margin-top:0.3rem;margin-bottom:1rem;">'
    f'Macro price trends · Inflation adjustment · Fuel→Grocery lag · 3-year price forecasts</div>'
)

section_header(f"Key Performance Indicators — {selected_display}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_kpi("Latest Price",  f"${kpis.get('latest_price','—')}", kpis.get("latest_date",""), "accent-blue")
with c2:
    render_kpi("All-Time Peak", f"${kpis.get('peak_price','—')}", f"Reached {kpis.get('peak_date','')}", "accent-red")
with c3:
    render_kpi("All-Time Low",  f"${kpis.get('min_price','—')}", f"Reached {kpis.get('min_date','')}", "accent-green")
with c4:
    chg = kpis.get("decade_change_pct", 0)
    bc  = "up" if chg > 0 else ("down" if chg < 0 else "flat")
    bt  = (f"▲ +{chg:.1f}%" if chg > 0 else f"▼ {chg:.1f}%") if chg != 0 else "─ Flat"
    render_kpi("Period Change", f"{chg:+.1f}%",
               f"From ${kpis.get('start_price','')} → ${kpis.get('end_price','')}",
               "accent-purple", bt, bc)
with c5:
    v = float(kpis.get("volatility", 0.0) or 0.0)
    render_kpi("Price Volatility", f"{v:.4f}",
               f"{'High' if v>0.5 else 'Medium' if v>0.2 else 'Low'} σ (std dev of prices)",
               "accent-gold")

st.write("")

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Price Trends",
    "🔗  Fuel ↔ Grocery Correlation",
    "🔮  Price Forecast",
    "📋  Data Explorer",
])


# ── Tab 1: Price Trends ───────────────────────────────────────────────────────

with tab1:
    section_header("Selected Item Price History")

    series     = df_display[selected_col].dropna()
    raw_series = df[selected_col].dropna() if price_mode == "Trend (12-mo Rolling Avg)" else None

    fig_main = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
    if raw_series is not None:
        rp = adjust_for_inflation(raw_series.to_frame()).iloc[:, 0] if inflation_adj else raw_series
        fig_main.add_trace(go.Scatter(
            x=rp.index, y=rp.values, mode="lines", name="Raw Price",
            line=dict(color=ACCENT["grocery"], width=1, dash="dot"), opacity=0.35,
        ))
    fig_main.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", name=selected_display,
        line=dict(color=ACCENT["grocery"], width=2.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
    ))
    if fuel_col and fuel_col != selected_col and fuel_col in df_display.columns:
        fs = df_display[fuel_col].dropna()
        fig_main.add_trace(go.Scatter(
            x=fs.index, y=fs.values, mode="lines", name=short_name(fuel_col),
            line=dict(color=ACCENT["fuel"], width=2), yaxis="y2",
            hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
        ))
        fig_main.update_layout(yaxis2=dict(
            title="Gasoline ($/gal)", overlaying="y", side="right",
            showgrid=False, tickfont=dict(color="#e3b341"),
        ))
    fig_main.update_layout(
        title=dict(text=f"{selected_display} — {'Real' if inflation_adj else 'Nominal'} Price History", font_size=14),
        xaxis_title="",
        yaxis_title=f"Price (USD {'Real' if inflation_adj else 'Nominal'})",
        height=420, hovermode="x unified",
    )
    st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": True})

    if compare_cols:
        section_header("Multi-Item Price Comparison")
        fig_comp = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
        fig_comp.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=selected_display,
            line=dict(color=CHART_PALETTE[0], width=2.5),
            hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
        ))
        for i, col in enumerate(compare_cols[:5]):
            if col in df_display.columns:
                s = df_display[col].dropna()
                fig_comp.add_trace(go.Scatter(
                    x=s.index, y=s.values, mode="lines", name=short_name(col),
                    line=dict(color=CHART_PALETTE[(i+1) % len(CHART_PALETTE)], width=1.8),
                    hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
                ))
        fig_comp.update_layout(
            title=dict(text="Comparative Price Trends", font_size=14),
            xaxis_title="",
            yaxis_title=f"Price (USD {'Real' if inflation_adj else 'Nominal'})",
            height=400, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": True})

    section_header("Year-over-Year % Price Change")
    yoy = df_display[selected_col].pct_change(12).dropna() * 100
    fig_yoy = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
    fig_yoy.add_trace(go.Bar(
        x=yoy.index, y=yoy.values,
        marker_color=[ACCENT["grocery"] if v >= 0 else ACCENT["fuel"] for v in yoy.values],
        name="YoY %", hovertemplate="<b>%{x|%b %Y}</b><br>%{y:+.2f}%<extra></extra>",
    ))
    fig_yoy.add_hline(y=0,   line_color="#30363d", line_width=1)
    fig_yoy.add_hline(y=3.5, line_color="#e3b341", line_dash="dot", line_width=1,
                      annotation_text="Avg inflation (3.5%)", annotation_position="right")
    fig_yoy.update_layout(
        title=dict(text=f"Year-over-Year Change — {selected_display}", font_size=14),
        xaxis_title="", yaxis_title="% Change vs. 12 months prior",
        height=320, hovermode="x unified",
    )
    st.plotly_chart(fig_yoy, use_container_width=True, config={"displayModeBar": True})


# ── Tab 2: Fuel ↔ Grocery Correlation ────────────────────────────────────────

with tab2:
    section_header("Lagged Correlation: Gasoline → Grocery Prices")

    if not fuel_col:
        st.warning("No gasoline column detected in the dataset.")
    else:
        col_info, col_chart = st.columns([1, 2], gap="large")

        with col_info:
            st.html(
                f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                f'padding:1.2rem 1.4rem;font-size:0.85rem;color:#c9d1d9;line-height:1.7;">'
                f'<div style="font-weight:600;margin-bottom:0.5rem;color:#e6edf3;">What is a Lag Correlation?</div>'
                f'A <b>lag</b> measures how many months grocery prices take to react '
                f'to gasoline price changes.<br><br>'
                f'<b>Lag 0</b> = same-month response (instantaneous)<br>'
                f'<b>Lag 3</b> = groceries respond 3 months later<br><br>'
                f'A correlation of <b>≥ 0.7</b> is considered strong.<br><br>'
                f'<b>Fuel column:</b><br>'
                f'<span style="color:#e3b341;font-size:0.78rem;">{short_name(fuel_col)}</span>'
                f'</div>'
            )

        with col_chart:
            lag_df = compute_lagged_correlation(df_display, fuel_col, selected_col, max_lag)
            if not lag_df.empty:
                bar_colors = [
                    "#3fb950" if abs(v) >= 0.7 else "#e3b341" if abs(v) >= 0.4 else "#8b949e"
                    for v in lag_df["correlation"]
                ]
                fig_lag = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
                fig_lag.add_trace(go.Bar(
                    x=lag_df["lag"], y=lag_df["correlation"], marker_color=bar_colors,
                    text=lag_df["correlation"].apply(lambda v: f"{v:.3f}"),
                    textposition="outside", name="Correlation",
                    hovertemplate="<b>Lag %{x} months</b><br>r = %{y:.4f}<extra></extra>",
                ))
                for y_ref, col_ref, lbl in [
                    (0.7, "#3fb950", "Strong (0.7)"), (0.4, "#e3b341", "Moderate (0.4)"),
                    (0.0, "#30363d", None),
                    (-0.4, "#e3b341", None), (-0.7, "#3fb950", None),
                ]:
                    kw = dict(line_dash="dot") if y_ref != 0.0 else {}
                    if lbl:
                        kw["annotation_text"] = lbl
                        kw["annotation_position"] = "right"
                    fig_lag.add_hline(y=y_ref, line_color=col_ref, line_width=1, **kw)
                fig_lag.update_layout(
                    title=dict(text=f"Gasoline → {selected_display}: Lag Correlation (0–{max_lag} mo)", font_size=13),
                    xaxis=dict(title="Lag (months)", tickmode="linear"),
                    yaxis=dict(title="Pearson r", range=[-1.05, 1.05]),
                    height=360,
                )
                st.plotly_chart(fig_lag, use_container_width=True, config={"displayModeBar": False})
                st.dataframe(
                    lag_df.style
                        .background_gradient(subset=["correlation"], cmap="RdYlGn", vmin=-1, vmax=1)
                        .format({"correlation": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )

        section_header("Full Lag-Correlation Heatmap (All Items)")
        with st.spinner("Computing full correlation matrix…"):
            matrix_df = full_correlation_matrix(df_display, fuel_col, max_lag=min(max_lag, 6))

        if not matrix_df.empty:
            matrix_df["_max"] = matrix_df.abs().max(axis=1)
            top = matrix_df.nlargest(20, "_max").drop(columns="_max")
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=top.values.tolist(), x=top.columns.tolist(),
                    y=[short_name(c) for c in top.index],
                    colorscale=[
                        [0.0, "#b62324"], [0.35, "#21262d"], [0.5, "#21262d"],
                        [0.65, "#21262d"], [1.0, "#196c2e"],
                    ],
                    zmid=0, zmin=-1, zmax=1,
                    text=top.round(2).values.tolist(),
                    texttemplate="%{text}", textfont={"size": 9},
                    hovertemplate="<b>%{y}</b><br>%{x}: r = %{z:.3f}<extra></extra>",
                    colorbar=dict(title="r", thickness=12, len=0.8),
                ),
                layout=PLOTLY_TEMPLATE["layout"],
            )
            fig_heat.update_layout(
                title=dict(text="Gasoline Price Lag Correlations (Top 20 Items)", font_size=13),
                height=520, xaxis=dict(title="Lag"),
                yaxis=dict(title="", autorange="reversed"),
                margin=dict(l=200, r=60, t=60, b=40),
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})


# ── Tab 3: Price Forecast ─────────────────────────────────────────────────────

with tab3:
    section_header(f"Price Forecast — {selected_display}")

    fc_col1, fc_col2 = st.columns([3, 1])

    with fc_col2:
        st.html(
            '<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
            'padding:1.2rem 1.4rem;font-size:0.83rem;color:#c9d1d9;line-height:1.7;margin-top:0.5rem;">'
            '<div style="font-weight:600;color:#e6edf3;margin-bottom:0.5rem;">Forecast Method</div>'
            'Uses <b>Facebook Prophet</b> with multiplicative seasonality and a '
            '0.15 changepoint prior for adaptive trend detection.<br><br>'
            'Falls back to <b>linear regression</b> + ±1.96σ CI if Prophet is absent.<br><br>'
            'Shaded band = 95% prediction interval.'
            '</div>'
        )

    with fc_col1:
        with st.spinner(f"Forecasting {selected_display}…"):
            fc_df = forecast_item(df_display, selected_col, forecast_months)

        if fc_df.empty:
            st.warning("Forecast could not be generated for this item.")
        else:
            hist_df = fc_df[fc_df["source"] == "historical"]
            pred_df = fc_df[fc_df["source"] == "forecast"]

            fig_fc = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
            fig_fc.add_trace(go.Scatter(
                x=hist_df["ds"], y=hist_df["yhat"], mode="lines", name="Historical",
                line=dict(color=ACCENT["grocery"], width=2),
                hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
            ))
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([pred_df["ds"], pred_df["ds"].iloc[::-1]]),
                y=pd.concat([pred_df["yhat_upper"], pred_df["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor=ACCENT["ci"],
                line=dict(color="rgba(0,0,0,0)"), showlegend=True,
                name="95% CI", hoverinfo="skip",
            ))
            fig_fc.add_trace(go.Scatter(
                x=pred_df["ds"], y=pred_df["yhat"], mode="lines", name="Forecast",
                line=dict(color=ACCENT["forecast"], width=2.5, dash="dash"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:.3f}<extra></extra>",
            ))
            if not hist_df.empty and not pred_df.empty:
                fig_fc.add_vline(
                    x=hist_df["ds"].iloc[-1], line_color="#30363d", line_dash="dot",
                    annotation_text="Forecast Start", annotation_position="top right",
                    annotation_font_color="#8b949e", annotation_font_size=10,
                )
            fig_fc.update_layout(
                title=dict(text=f"{selected_display} — {forecast_months}-Month Forecast", font_size=14),
                xaxis_title="",
                yaxis_title=f"Price (USD {'Real' if inflation_adj else 'Nominal'})",
                height=450, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": True})

            section_header("Forecast Summary (Annual Snapshots)")
            if not pred_df.empty:
                pc = pred_df.copy()
                pc["Year"] = pc["ds"].dt.year
                snap = (
                    pc.groupby("Year")
                    .agg(Avg=("yhat","mean"), Lo=("yhat_lower","mean"), Hi=("yhat_upper","mean"))
                    .round(3).reset_index()
                )
                snap.columns = ["Year", "Avg Forecast ($)", "Lower 95% CI ($)", "Upper 95% CI ($)"]
                st.dataframe(
                    snap.style.background_gradient(subset=["Avg Forecast ($)"], cmap="Blues"),
                    use_container_width=True, hide_index=True,
                )

    if compare_cols:
        section_header("Forecast Comparison — Selected Items")
        fig_fc_multi = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
        for i, col in enumerate([selected_col] + compare_cols[:4]):
            if col not in df_display.columns:
                continue
            with st.spinner(f"Forecasting {short_name(col)}…"):
                fc = forecast_item(df_display, col, forecast_months)
            if fc.empty:
                continue
            color = CHART_PALETTE[i % len(CHART_PALETTE)]
            h = fc[fc["source"] == "historical"]
            p = fc[fc["source"] == "forecast"]
            fig_fc_multi.add_trace(go.Scatter(
                x=h["ds"], y=h["yhat"], mode="lines", name=f"{short_name(col)} (hist)",
                line=dict(color=color, width=1.5), opacity=0.6,
                hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
            ))
            fig_fc_multi.add_trace(go.Scatter(
                x=p["ds"], y=p["yhat"], mode="lines", name=f"{short_name(col)} (forecast)",
                line=dict(color=color, width=2.5, dash="dash"),
                hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>",
            ))
        fig_fc_multi.update_layout(
            title=dict(text="Multi-Item Price Forecast Comparison", font_size=13),
            xaxis_title="", yaxis_title="Price (USD)",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc_multi, use_container_width=True, config={"displayModeBar": True})


# ── Tab 4: Data Explorer ──────────────────────────────────────────────────────

with tab4:
    section_header("Raw Data Table")

    cols_show = [c for c in ([selected_col] + compare_cols) if c in df_display.columns]

    if df_summary is not None and not df_summary.empty:
        with st.expander("📋 Item Summary Statistics", expanded=False):
            st.dataframe(df_summary, use_container_width=True)

    view_df = df_display[cols_show].copy()
    view_df.index   = view_df.index.strftime("%Y-%m")
    view_df.columns = [short_name(c) for c in view_df.columns]
    st.dataframe(
        view_df.style.format("${:.3f}").background_gradient(cmap="Blues", axis=0),
        use_container_width=True, height=420,
    )

    st.download_button(
        label="⬇️  Download Filtered Dataset (CSV)",
        data=df_display.to_csv().encode("utf-8"),
        file_name="us_prices_filtered.csv",
        mime="text/csv",
    )

    section_header("Descriptive Statistics")
    desc = df_display[cols_show].describe().round(4)
    desc.columns = [short_name(c) for c in desc.columns]
    st.dataframe(desc.style.background_gradient(cmap="Blues"), use_container_width=True)

    section_header("Price Distribution")
    fig_box = go.Figure(layout=PLOTLY_TEMPLATE["layout"])
    for i, col in enumerate(cols_show[:8]):
        if col not in df_display.columns:
            continue
        fig_box.add_trace(go.Box(
            y=df_display[col].dropna().values, name=short_name(col),
            boxpoints="outliers",
            marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
            line_color=CHART_PALETTE[i % len(CHART_PALETTE)],
        ))
    fig_box.update_layout(
        title=dict(text="Price Distribution by Item", font_size=13),
        yaxis_title="Price (USD)", height=380, showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# AI BUSINESS & POLICY DECISIONS
# ════════════════════════════════════════════════════════════════════════════════

st.write("")

with st.expander("🤖  AI Business & Policy Decisions", expanded=False):

    lag_for_insights = (
        compute_lagged_correlation(df_display, fuel_col, selected_col, max_lag)
        if fuel_col else None
    )
    insights = generate_insights(
        kpis=kpis, selected_item=selected_col,
        lag_df=lag_for_insights, mode=price_mode, inflation_on=inflation_adj,
    )

    st.html(
        f'<div style="font-size:0.78rem;color:#8b949e;margin-bottom:1rem;">'
        f'Insights for <b style="color:#e6edf3;">{selected_display}</b> · '
        f'{"Real" if inflation_adj else "Nominal"} prices · {price_mode}</div>'
    )
    for insight in insights:
        st.html(
            f'<div style="background:#161b22;border-left:3px solid #58a6ff;'
            f'border-radius:0 8px 8px 0;padding:0.75rem 1rem;margin-bottom:0.65rem;'
            f'font-size:0.88rem;color:#c9d1d9;line-height:1.6;">{insight}</div>'
        )

    if fuel_col and fuel_col in df_display.columns and selected_col in df_display.columns:
        section_header("Fuel vs. Grocery Price Scatter")
        sdf = df_display[[fuel_col, selected_col]].dropna().copy()
        sdf.columns = ["fuel", "grocery"]
        sdf["year"] = sdf.index.year.astype(int)
        fig_sc = px.scatter(
            sdf, x="fuel", y="grocery", color="year",
            color_continuous_scale="Plasma",
            labels={"fuel": short_name(fuel_col), "grocery": selected_display, "color": "Year"},
            trendline="ols", template=None,
        )
        fig_sc.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title=dict(text=f"{short_name(fuel_col)} vs. {selected_display}", font_size=13),
            height=380,
            coloraxis_colorbar=dict(title="Year", thickness=10, len=0.7),
        )
        fig_sc.update_traces(marker=dict(size=5, opacity=0.75), selector=dict(mode="markers"))
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})


# ── Footer ────────────────────────────────────────────────────────────────────

st.html(
    '<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid #21262d;'
    'text-align:center;font-size:0.72rem;color:#484f58;">'
    'US Grocery &amp; Gas Price Analytics &nbsp;·&nbsp; '
    'Data: US Bureau of Labor Statistics &nbsp;·&nbsp; '
    'Built with Streamlit, Plotly &amp; Prophet'
    '</div>'
)
