import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

BASE_CURRENCY = "EUR"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
ECB_DATA_BASE = "https://data-api.ecb.europa.eu/service/data"

# -----------------------------
# Load portfolio
# -----------------------------
@st.cache_data
def load_portfolio(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    required = [
        "ticker", "name", "asset_class", "theme", "quantity", "native_currency",
        "cost_basis_native", "fee_native", "tax_bucket", "price_source", "api_id"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ontbrekende kolommen in CSV: {missing}")

    numeric_cols = ["quantity", "cost_basis_native", "fee_native"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# ECB FX
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_ecb_usd_eur() -> Optional[float]:
    """
    ECB series key:
    EXR / D.USD.EUR.SP00.A
    """
    url = f"{ECB_DATA_BASE}/EXR/D.USD.EUR.SP00.A"
    params = {"lastNObservations": 1}
    headers = {"Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    series_dict = data["dataSets"][0]["series"]
    if not series_dict:
        return None

    first_series = next(iter(series_dict.values()))
    obs = first_series.get("observations", {})
    if not obs:
        return None

    first_obs = next(iter(obs.values()))
    value = first_obs[0]
    return float(value)


# -----------------------------
# CoinGecko
# -----------------------------
@st.cache_data(ttl=300)
def fetch_coingecko_prices(ids: List[str], vs_currency: str = "eur") -> Dict[str, float]:
    if not ids:
        return {}

    url = f"{COINGECKO_BASE}/simple/price"
    params = {
        "ids": ",".join(ids),
        "vs_currencies": vs_currency.lower(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    out = {}
    for coin_id in ids:
        value = data.get(coin_id, {}).get(vs_currency.lower())
        if value is not None:
            out[coin_id] = float(value)
    return out


@st.cache_data(ttl=1800)
def fetch_coingecko_history(coin_id: str, days: int = 90, vs_currency: str = "eur") -> pd.Series:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency.lower(), "days": days}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    prices = data.get("prices", [])
    if not prices:
        return pd.Series(dtype=float)

    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.date
    series = df.groupby("date")["price"].last()
    series.index = pd.to_datetime(series.index)
    series.name = coin_id
    return series.sort_index()


# -----------------------------
# Alpha Vantage
# -----------------------------
@st.cache_data(ttl=300)
def fetch_alpha_vantage_quote(symbol: str, api_key: str) -> Optional[float]:
    if not api_key:
        return None

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }
    r = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    price = data.get("Global Quote", {}).get("05. price")
    if price in (None, ""):
        return None
    return float(price)


@st.cache_data(ttl=1800)
def fetch_alpha_vantage_history(symbol: str, api_key: str, days: int = 90) -> pd.Series:
    if not api_key:
        return pd.Series(dtype=float)

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }
    r = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    ts = data.get("Time Series (Daily)", {})
    if not ts:
        return pd.Series(dtype=float)

    records = []
    for d, row in ts.items():
        close = row.get("4. close")
        if close is not None:
            records.append((pd.to_datetime(d), float(close)))

    if not records:
        return pd.Series(dtype=float)

    s = pd.Series(dict(records)).sort_index()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    s = s[s.index >= cutoff]
    s.name = symbol
    return s


# -----------------------------
# FRED
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_fred_latest(series_id: str, api_key: str) -> Tuple[Optional[float], Optional[str]]:
    if not api_key:
        return None, None

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    r = requests.get(FRED_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    if not obs:
        return None, None

    value = obs[0].get("value")
    date = obs[0].get("date")
    if value in (None, ".", ""):
        return None, date
    return float(value), date


@st.cache_data(ttl=3600)
def fetch_fred_history(series_id: str, api_key: str, days: int = 365) -> pd.Series:
    if not api_key:
        return pd.Series(dtype=float)

    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for obs in data.get("observations", []):
        value = obs.get("value")
        if value not in (None, ".", ""):
            rows.append((pd.to_datetime(obs["date"]), float(value)))

    if not rows:
        return pd.Series(dtype=float)

    s = pd.Series(dict(rows)).sort_index()
    s.name = series_id
    return s


# -----------------------------
# Portfolio enrichment
# -----------------------------
def enrich_portfolio(df: pd.DataFrame, alpha_key: str, usd_to_eur: float) -> pd.DataFrame:
    out = df.copy()

    cg_ids = out.loc[out["price_source"] == "coingecko", "api_id"].dropna().tolist()
    av_symbols = out.loc[out["price_source"] == "alphavantage", "api_id"].dropna().tolist()

    cg_prices = fetch_coingecko_prices(cg_ids, vs_currency="eur") if cg_ids else {}
    av_prices = {}
    if av_symbols and alpha_key:
        for sym in av_symbols:
            try:
                av_prices[sym] = fetch_alpha_vantage_quote(sym, alpha_key)
            except Exception:
                av_prices[sym] = None

    live_prices = []
    native_ccy_effective = []
    fx_to_base = []

    for _, row in out.iterrows():
        source = row["price_source"]
        api_id = row["api_id"]
        native_ccy = str(row["native_currency"]).upper()

        if source == "coingecko":
            price_native = cg_prices.get(api_id)
            native_eff = "EUR"
        else:
            price_native = av_prices.get(api_id)
            native_eff = native_ccy

        if native_eff == BASE_CURRENCY:
            fx = 1.0
        elif native_eff == "USD":
            fx = usd_to_eur
        else:
            fx = 1.0

        live_prices.append(price_native)
        native_ccy_effective.append(native_eff)
        fx_to_base.append(fx)

    out["native_currency_effective"] = native_ccy_effective
    out["price_native"] = pd.to_numeric(live_prices, errors="coerce")
    out["fx_to_base"] = fx_to_base

    out["market_value_native"] = out["quantity"] * out["price_native"]
    out["market_value_base"] = out["market_value_native"] * out["fx_to_base"]

    out["cost_value_native"] = out["quantity"] * out["cost_basis_native"]
    out["cost_value_base"] = out["cost_value_native"] * out["fx_to_base"]

    out["fee_base"] = out["fee_native"] * out["fx_to_base"]

    out["pnl_base_gross"] = out["market_value_base"] - out["cost_value_base"]
    out["pnl_base_net"] = out["pnl_base_gross"] - out["fee_base"]
    out["return_pct_net"] = out["pnl_base_net"] / out["cost_value_base"]

    total = out["market_value_base"].sum(skipna=True)
    out["weight_pct"] = out["market_value_base"] / total if total else math.nan

    out["fx_exposure_base"] = out.apply(
        lambda row: row["market_value_base"]
        if str(row["native_currency_effective"]).upper() != BASE_CURRENCY else 0.0,
        axis=1,
    )
    return out


def portfolio_summary(df: pd.DataFrame) -> dict:
    total_value = df["market_value_base"].sum(skipna=True)
    total_cost = df["cost_value_base"].sum(skipna=True)
    total_fees = df["fee_base"].sum(skipna=True)
    net_pnl = df["pnl_base_net"].sum(skipna=True)

    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "net_return_pct": net_pnl / total_cost if total_cost else math.nan,
        "fx_exposure": df["fx_exposure_base"].sum(skipna=True),
    }


def allocation_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)["market_value_base"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total = grouped["market_value_base"].sum()
    grouped["weight_pct"] = grouped["market_value_base"] / total if total else math.nan
    return grouped


# -----------------------------
# Historical portfolio proxy
# -----------------------------
def get_asset_history(row: pd.Series, alpha_key: str, usd_to_eur: float, days: int) -> pd.Series:
    source = row["price_source"]
    api_id = row["api_id"]
    native_ccy = str(row["native_currency"]).upper()
    qty = float(row["quantity"])

    if source == "coingecko":
        s = fetch_coingecko_history(api_id, days=days, vs_currency="eur")
        return s * qty

    s = fetch_alpha_vantage_history(api_id, alpha_key, days=days)
    if s.empty:
        return s

    fx = 1.0 if native_ccy == BASE_CURRENCY else usd_to_eur if native_ccy == "USD" else 1.0
    return s * qty * fx


def build_portfolio_history(df: pd.DataFrame, alpha_key: str, usd_to_eur: float, days: int = 90) -> pd.Series:
    pieces = []
    for _, row in df.iterrows():
        try:
            s = get_asset_history(row, alpha_key, usd_to_eur, days)
            if not s.empty:
                pieces.append(s.rename(row["ticker"]))
        except Exception:
            pass

    if not pieces:
        return pd.Series(dtype=float)

    hist = pd.concat(pieces, axis=1).sort_index()
    portfolio = hist.sum(axis=1, skipna=True)
    portfolio.name = "Portfolio"
    return portfolio


# -----------------------------
# Plot helper
# -----------------------------
def plot_series(series: pd.Series, title: str, y_label: str = ""):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Multi-Asset Dashboard Step 3", layout="wide")
st.title("Multi-Asset Portfolio Dashboard — Step 3")
st.caption("Automatische ECB FX, macro-cards en historische grafieken")

st.sidebar.header("API keys")
alpha_key = st.sidebar.text_input("Alpha Vantage API key", type="password")
fred_key = st.sidebar.text_input("FRED API key", type="password")

st.sidebar.header("Periode")
days = st.sidebar.selectbox("Historische periode", [30, 90, 180, 365], index=1)

uploaded = st.file_uploader("Upload portfolio.csv", type=["csv"])
if uploaded is None:
    st.info("Upload eerst je portfolio.csv-bestand.")
    st.stop()

# Load
try:
    raw = load_portfolio(uploaded)
except Exception as e:
    st.error(f"CSV fout: {e}")
    st.stop()

# ECB FX
try:
    ecb_usd_eur = fetch_ecb_usd_eur()
except Exception:
    ecb_usd_eur = None

manual_fx = st.sidebar.number_input(
    "Fallback USD → EUR",
    min_value=0.50,
    max_value=1.50,
    value=0.92,
    step=0.01
)

usd_to_eur = ecb_usd_eur if ecb_usd_eur is not None else manual_fx

# Enrich
try:
    df = enrich_portfolio(raw, alpha_key=alpha_key, usd_to_eur=usd_to_eur)
    summary = portfolio_summary(df)
except Exception as e:
    st.error(f"Verwerkingsfout: {e}")
    st.stop()

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Totale waarde", f"EUR {summary['total_value']:,.2f}")
c2.metric("Netto P/L", f"EUR {summary['net_pnl']:,.2f}", f"{summary['net_return_pct']:.2%}")
c3.metric("Totale kosten", f"EUR {summary['total_fees']:,.2f}")
c4.metric("FX exposure", f"EUR {summary['fx_exposure']:,.2f}")
c5.metric("USD→EUR (ECB)", f"{usd_to_eur:.4f}")

# Macro cards
st.subheader("Macro")
m1, m2, m3, m4 = st.columns(4)

fed_funds, fed_date = fetch_fred_latest("FEDFUNDS", fred_key) if fred_key else (None, None)
cpi, cpi_date = fetch_fred_latest("CPIAUCSL", fred_key) if fred_key else (None, None)
unrate, unrate_date = fetch_fred_latest("UNRATE", fred_key) if fred_key else (None, None)

# Gold proxy from portfolio sources: GLD ETF via Alpha Vantage, as proxy
gold_proxy = None
if alpha_key:
    try:
        gold_proxy = fetch_alpha_vantage_quote("GLD", alpha_key)
    except Exception:
        gold_proxy = None

m1.metric("Fed Funds", "-" if fed_funds is None else f"{fed_funds:.2f}%")
m2.metric("CPIAUCSL", "-" if cpi is None else f"{cpi:.2f}")
m3.metric("UNRATE", "-" if unrate is None else f"{unrate:.2f}%")
m4.metric("Gold proxy (GLD)", "-" if gold_proxy is None else f"{gold_proxy:.2f}")

with st.expander("Macro details"):
    st.write(f"FEDFUNDS datum: {fed_date}")
    st.write(f"CPIAUCSL datum: {cpi_date}")
    st.write(f"UNRATE datum: {unrate_date}")
    st.write("Gold proxy gebruikt GLD ETF in plaats van spot goud.")

# Holdings
st.subheader("Holdings")
st.dataframe(
    df[
        [
            "ticker", "name", "asset_class", "theme",
            "native_currency_effective", "quantity", "price_native",
            "fx_to_base", "market_value_base", "cost_value_base",
            "fee_base", "pnl_base_net", "return_pct_net",
            "weight_pct", "tax_bucket", "price_source", "api_id"
        ]
    ],
    use_container_width=True
)

# Allocations
left, right = st.columns(2)

with left:
    st.subheader("Allocatie per asset class")
    alloc_asset = allocation_by(df, "asset_class")
    st.bar_chart(alloc_asset.set_index("asset_class")["market_value_base"])

    st.subheader("Allocatie per valuta")
    alloc_ccy = allocation_by(df, "native_currency_effective")
    st.bar_chart(alloc_ccy.set_index("native_currency_effective")["market_value_base"])

with right:
    st.subheader("Allocatie per thema")
    alloc_theme = allocation_by(df, "theme")
    st.bar_chart(alloc_theme.set_index("theme")["market_value_base"])

    st.subheader("Tax buckets")
    alloc_tax = allocation_by(df, "tax_bucket")
    st.bar_chart(alloc_tax.set_index("tax_bucket")["market_value_base"])

# Portfolio history
st.subheader("Historische grafieken")

portfolio_hist = build_portfolio_history(df, alpha_key=alpha_key, usd_to_eur=usd_to_eur, days=days)
if portfolio_hist.empty:
    st.warning("Geen portfoliohistorie beschikbaar. Controleer Alpha Vantage key en symbols.")
else:
    plot_series(portfolio_hist, f"Portfolio waarde ({days} dagen)", y_label="EUR")

# Macro history
if fred_key:
    macro_choice = st.selectbox(
        "Macroserie",
        [
            ("FEDFUNDS", "Fed Funds"),
            ("CPIAUCSL", "US CPI"),
            ("UNRATE", "US Unemployment"),
        ],
        format_func=lambda x: x[1]
    )
    macro_series = fetch_fred_history(macro_choice[0], fred_key, days=max(days, 365))
    if not macro_series.empty:
        plot_series(macro_series, f"{macro_choice[1]} historie", y_label=macro_choice[0])
    else:
        st.info("Geen macrohistorie geladen.")
else:
    st.info("Voeg een FRED API key toe voor macrohistorie.")

# Single asset history
asset_labels = [f"{r['ticker']} — {r['name']}" for _, r in df.iterrows()]
selected_label = st.selectbox("Asset grafiek", asset_labels)
selected_idx = asset_labels.index(selected_label)
selected_row = df.iloc[selected_idx]

try:
    if selected_row["price_source"] == "coingecko":
        s = fetch_coingecko_history(selected_row["api_id"], days=days, vs_currency="eur")
        if not s.empty:
            plot_series(s, f"{selected_row['ticker']} prijs ({days} dagen)", y_label="EUR")
    else:
        s = fetch_alpha_vantage_history(selected_row["api_id"], alpha_key, days=days)
        if not s.empty:
            plot_series(s, f"{selected_row['ticker']} prijs ({days} dagen)", y_label=selected_row["native_currency"])
except Exception as e:
    st.warning(f"Assetgrafiek kon niet geladen worden: {e}")

# Diagnostics
st.subheader("Datakwaliteit")
missing_prices = df[df["price_native"].isna()][["ticker", "name", "price_source", "api_id"]]
if missing_prices.empty:
    st.success("Voor alle assets is een prijs gevonden.")
else:
    st.warning("Voor sommige assets is geen live prijs gevonden.")
    st.dataframe(missing_prices, use_container_width=True)
