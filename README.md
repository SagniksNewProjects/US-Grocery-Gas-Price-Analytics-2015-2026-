# US Grocery & Gas Price Analytics (2015–2026)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A production-ready, single-file Streamlit data analytics application** analyzing US average retail prices for grocery staples and gasoline from **January 2015 to 2026**, built on real BLS (Bureau of Labor Statistics) data.

🔗 **GitHub:** https://github.com/SagniksNewProjects/US-Grocery-Gas-Price-Analytics-2015-2026-

🔗 **Datasets' Source from Kaggle** https://www.kaggle.com/datasets/harshitsama/us-grocery-and-gas-prices-2015-2026

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 **KPI Dashboard** | 5 dynamic metric cards — latest price, all-time peak & low, period % change, volatility |
| 📈 **Interactive Charts** | Dual-axis price history, multi-item comparison, year-over-year % change bar chart |
| 📉 **Macro Stability Filter** | Toggle between raw monthly prices and a 12-month rolling average |
| 💵 **Inflation Adjustment** | Deflate nominal USD prices to real 2015-baseline values (35% cumulative decade CPI) |
| 🔗 **Lag Correlation Engine** | Pearson lag correlation (0–12 months) between gasoline and grocery price series |
| 🔮 **Price Forecasting** | Facebook Prophet (multiplicative seasonality) with linear-regression fallback + 95% CI |
| 🤖 **AI Business Insights** | Dynamic policy and procurement recommendations based on live KPIs and lag results |
| 📋 **Data Explorer** | Styled data tables, descriptive stats, box-plot distributions, CSV download |

---

## Project Structure

```
us_grocery_gas_analytics/
├── main.py                              ← Single-file app (analytics + UI)
├── requirements.txt                     ← Python dependencies
├── README.md                            ← This file
├── US_Grocery_Gas_Analytics_Report.pptx ← 12-slide project presentation
├── us_average_prices_wide.csv           ← BLS wide-format dataset
├── us_average_prices_monthly.csv        ← BLS long-format dataset
└── us_average_prices_item_summary.csv   ← Pre-aggregated item statistics
```

### Why a single file?

`main.py` contains two clearly separated sections:

- **Section 1 — Analytics Engine** (lines 1–310): Pure Python functions — data loading, rolling averages, inflation deflation, Pearson lag correlation, Prophet forecasting, KPI calculation, insight generation. No Streamlit imports; fully testable in isolation.
- **Section 2 — Streamlit Frontend** (lines 311–826): All UI rendering — page config, CSS, sidebar, KPI cards, tabs, charts, expanders.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SagniksNewProjects/US-Grocery-Gas-Price-Analytics-2015-2026-.git
cd US-Grocery-Gas-Price-Analytics-2015-2026-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `prophet` requires additional system dependencies (C++ build tools / Stan). If installation fails, the app will automatically fall back to linear-regression forecasting.
> - Windows: Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> - macOS: `xcode-select --install`
> - Linux: `sudo apt-get install build-essential`

### 3. Run the application

```bash
streamlit run main.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Datasets

| File | Format | Description |
|------|--------|-------------|
| `us_average_prices_wide.csv` | Wide (pivoted) | Rows = months, columns = price items. Primary time-series source. |
| `us_average_prices_monthly.csv` | Long (tidy) | One row per item per month. Merged to supplement the wide dataset. |
| `us_average_prices_item_summary.csv` | Pre-aggregated | One row per item with min, max, mean, std dev across the full period. |

**Source:** [US Bureau of Labor Statistics — Average Retail Food and Energy Prices](https://www.bls.gov/regions/mid-atlantic/data/averageretailfoodandenergyprices_usandmidwest_table.htm)

**Coverage:** January 2015 – July 2026 · Monthly granularity · 74 items

---

## Analytics Engine — Function Reference

| Function | Description |
|----------|-------------|
| `load_wide(path)` | Load the pivoted CSV, parse dates, forward-fill |
| `load_monthly(path)` | Load the long CSV, pivot to wide, forward-fill |
| `merge_datasets(wide, monthly)` | Union-merge both datasets on datetime index |
| `compute_rolling(df, window=12)` | 12-month rolling mean across all columns |
| `adjust_for_inflation(df)` | Deflate to real 2015-baseline prices |
| `compute_lagged_correlation(df, fuel, grocery, max_lag)` | Pearson r at lags 0…max_lag |
| `full_correlation_matrix(df, fuel, max_lag)` | Full item × lag correlation heatmap matrix |
| `compute_kpis(df, col)` | Peak, min, change %, volatility for one item |
| `forecast_item(df, col, months_ahead)` | Prophet or linear regression forecast with CI |
| `generate_insights(kpis, item, lag_df, mode, inflation_on)` | Dynamic business insight strings |

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| [Streamlit](https://streamlit.io) | ≥ 1.31 | Web framework, `st.html()` for safe HTML rendering |
| [Plotly](https://plotly.com/python/) | ≥ 5.20 | Interactive charts (Scatter, Bar, Box, Heatmap) |
| [Pandas](https://pandas.pydata.org) | ≥ 2.0 | Data manipulation, pivot, rolling, ffill |
| [NumPy](https://numpy.org) | ≥ 1.26 | Array operations, `np.corrcoef` for lag correlation |
| [Prophet](https://facebook.github.io/prophet/) | ≥ 1.1.5 | Multiplicative seasonality forecasting (optional) |
| [scikit-learn](https://scikit-learn.org) | ≥ 1.4 | Linear regression fallback forecasting |
| [matplotlib](https://matplotlib.org) | ≥ 3.7 | Required by `pandas.Styler.background_gradient` |
| [statsmodels](https://www.statsmodels.org) | ≥ 0.14 | OLS trendline in Plotly Express scatter |

---

## Sidebar Controls

| Control | Description |
|---------|-------------|
| **Wide / Monthly / Summary CSV path** | Override auto-detected file paths |
| **Price Display** | Raw Prices vs Trend (12-mo Rolling Avg) |
| **Inflation-Adjust** | Toggle real vs nominal prices |
| **Start / End Year** | Filter the date range (2015–2026) |
| **Max Lag (months)** | Max lag depth for correlation engine (1–12) |
| **Months to Forecast** | Forecast horizon (6–60 months) |
| **Grocery / Gas Item** | Select the primary item for all charts and KPIs |
| **Compare with** | Up to 5 additional items for comparison charts |

---

## Known Limitations

- Inflation adjustment uses a fixed 3.5%/yr compound rate. For higher accuracy, replace with real monthly CPI data from FRED.
- Prophet requires C++ build tools; the linear-regression fallback is used automatically if unavailable.
- `short_name()` truncates column names at the first comma — display names are abbreviated but underlying column keys remain full BLS strings.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built as part of the IBM SkillsBuild Data Analytics Project · 2026*
*And as a part of AICTE Academic Internship - 2026*
