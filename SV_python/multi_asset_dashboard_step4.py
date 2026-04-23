import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

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
        "cost_basis_native", "fee_native", "tax_bucket", "price_source", "api_id",
        "factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ontbrekende kolommen in CSV: {missing}")

    numeric_cols = [
        "quantity", "cost_basis_native", "fee_native",
        "factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# ECB FX
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_ecb_usd_eur() -> Optional[float]:
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
    return float(first_obs[0])


# -----------------------------
# CoinGecko
# -----------------------------
@st.cache_data(ttl=300)
def fetch_coingecko_prices(ids: List[str], vs_currency: str = "eur") -> Dict[str, float]:
    if not ids:
        return {}

    url = f"{COINGECKO_BASE}/simple/price"
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency.lower()}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    out = {}
    for coin_id in ids:
        v = data.get(coin_id, {}).get(vs_currency.lower())
        if v is not None:
            out[coin_id] = float(v)
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
    s = df.groupby("date")["price"].last()
    s.index = pd.to_datetime(s.index)
    s.name = coin_id
    return s.sort_index()


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

    rows = []
    for d, row in ts.items():
        close = row.get("4. close")
        if close is not None:
            rows.append((pd.to_datetime(d), float(close)))

    if not rows:
        return pd.Series(dtype=float)

    s = pd.Series(dict(rows)).sort_index()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    s = s[s.index >= cutoff]
    s.name = symbol
    return s


# -----------------------------
# FRED
# -----------------------------
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
# Current portfolio enrichment
# -----------------------------
def enrich_portfolio(df: pd.DataFrame, alpha_key: str, usd_to_eur: float) -> pd.DataFrame:
    out = df.copy()

    cg_ids = out.loc[out["price_source"] == "coingecko", "api_id"].dropna().tolist()
    av_symbols = out.loc[out["price_source"] == "alphavantage", "api_id"].dropna().tolist()

    cg_prices = fetch_coingecko_prices(cg_ids, vs_currency="eur") if cg_ids else {}
    av_prices = {}
    if alpha_key:
        for sym in av_symbols:
            try:
                av_prices[sym] = fetch_alpha_vantage_quote(sym, alpha_key)
            except Exception:
                av_prices[sym] = None

    price_native = []
    native_currency_effective = []
    fx_to_base = []

    for _, row in out.iterrows():
        source = row["price_source"]
        api_id = row["api_id"]
        native_ccy = str(row["native_currency"]).upper()

        if source == "coingecko":
            p = cg_prices.get(api_id)
            native_eff = "EUR"
        else:
            p = av_prices.get(api_id)
            native_eff = native_ccy

        if native_eff == BASE_CURRENCY:
            fx = 1.0
        elif native_eff == "USD":
            fx = usd_to_eur
        else:
            fx = 1.0

        price_native.append(p)
        native_currency_effective.append(native_eff)
        fx_to_base.append(fx)

    out["price_native"] = pd.to_numeric(price_native, errors="coerce")
    out["native_currency_effective"] = native_currency_effective
    out["fx_to_base"] = fx_to_base

    out["market_value_native"] = out["quantity"] * out["price_native"]
    out["market_value_base"] = out["market_value_native"] * out["fx_to_base"]
    out["cost_value_native"] = out["quantity"] * out["cost_basis_native"]
    out["cost_value_base"] = out["cost_value_native"] * out["fx_to_base"]
    out["fee_base"] = out["fee_native"] * out["fx_to_base"]

    out["pnl_base_net"] = out["market_value_base"] - out["cost_value_base"] - out["fee_base"]
    out["return_pct_net"] = out["pnl_base_net"] / out["cost_value_base"]

    total = out["market_value_base"].sum(skipna=True)
    out["weight_pct"] = out["market_value_base"] / total if total else math.nan

    out["fx_exposure_base"] = out.apply(
        lambda row: row["market_value_base"]
        if str(row["native_currency_effective"]).upper() != BASE_CURRENCY else 0.0,
        axis=1,
    )
    return out


# -----------------------------
# Historical series
# -----------------------------
def get_asset_history(row: pd.Series, alpha_key: str, usd_to_eur: float, days: int) -> pd.Series:
    source = row["price_source"]
    api_id = row["api_id"]
    native_ccy = str(row["native_currency"]).upper()

    if source == "coingecko":
        s = fetch_coingecko_history(api_id, days=days, vs_currency="eur")
        return s

    s = fetch_alpha_vantage_history(api_id, alpha_key, days=days)
    if s.empty:
        return s

    if native_ccy == BASE_CURRENCY:
        fx = 1.0
    elif native_ccy == "USD":
        fx = usd_to_eur
    else:
        fx = 1.0

    return s * fx


def build_price_matrix(df: pd.DataFrame, alpha_key: str, usd_to_eur: float, days: int = 90) -> pd.DataFrame:
    series_list = []

    for _, row in df.iterrows():
        try:
            s = get_asset_history(row, alpha_key=alpha_key, usd_to_eur=usd_to_eur, days=days)
            if not s.empty:
                s = s.rename(row["ticker"])
                series_list.append(s)
        except Exception:
            pass

    if not series_list:
        return pd.DataFrame()

    prices = pd.concat(series_list, axis=1).sort_index()
    return prices


def build_position_value_matrix(prices: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    qty_map = df.set_index("ticker")["quantity"].to_dict()

    out = prices.copy()
    for col in out.columns:
        out[col] = out[col] * qty_map.get(col, 0)

    return out


# -----------------------------
# Quant helpers
# -----------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return prices.pct_change().dropna(how="all")


def compute_drawdown(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    running_max = series.cummax()
    return series / running_max - 1.0


def rolling_volatility(returns: pd.Series, window: int = 30, annualization: int = 252) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window).std() * (annualization ** 0.5)


def portfolio_factor_exposure(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["weight_pct"] = temp["weight_pct"].fillna(0)

    factors = ["factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"]
    rows = []
    for factor in factors:
        exposure = (temp["weight_pct"] * temp[factor]).sum()
        rows.append({"factor": factor.replace("factor_", ""), "exposure": exposure})
    return pd.DataFrame(rows)


def contribution_to_risk(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()

    aligned_cols = [c for c in returns.columns if c in weights.index]
    if not aligned_cols:
        return pd.DataFrame()

    r = returns[aligned_cols].dropna()
    if r.empty:
        return pd.DataFrame()

    w = weights.loc[aligned_cols].fillna(0).values.reshape(-1, 1)
    cov = r.cov().values
    port_var = float((w.T @ cov @ w)[0, 0])

    if port_var <= 0:
        return pd.DataFrame()

    marginal = cov @ w
    contrib = (w * marginal) / port_var
    out = pd.DataFrame({
        "ticker": aligned_cols,
        "risk_contribution": contrib.flatten()
    }).sort_values("risk_contribution", ascending=False)
    return out


def plot_line(series: pd.Series, title: str, y_label: str = ""):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Multi-Asset Dashboard Step 4", layout="wide")
st.title("Multi-Asset Portfolio Dashboard — Step 4")
st.caption("Drawdown, rolling volatility, correlaties en factorblootstelling")

st.sidebar.header("API keys")
alpha_key = st.sidebar.text_input("Alpha Vantage API key", type="password")
fred_key = st.sidebar.text_input("FRED API key", type="password")

st.sidebar.header("Parameters")
days = st.sidebar.selectbox("Historische periode", [30, 90, 180, 365], index=1)
vol_window = st.sidebar.selectbox("Volatility window", [20, 30, 60], index=1)

uploaded = st.file_uploader("Upload portfolio.csv", type=["csv"])
if uploaded is None:
    st.info("Upload eerst je portfolio.csv-bestand.")
    st.stop()

try:
    raw = load_portfolio(uploaded)
except Exception as e:
    st.error(f"CSV fout: {e}")
    st.stop()

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

try:
    df = enrich_portfolio(raw, alpha_key=alpha_key, usd_to_eur=usd_to_eur)
except Exception as e:
    st.error(f"Verwerkingsfout: {e}")
    st.stop()

# Top metrics
total_value = df["market_value_base"].sum(skipna=True)
total_cost = df["cost_value_base"].sum(skipna=True)
total_fees = df["fee_base"].sum(skipna=True)
net_pnl = df["pnl_base_net"].sum(skipna=True)
net_return = net_pnl / total_cost if total_cost else math.nan
fx_exposure = df["fx_exposure_base"].sum(skipna=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Totale waarde", f"EUR {total_value:,.2f}")
c2.metric("Netto P/L", f"EUR {net_pnl:,.2f}", f"{net_return:.2%}" if pd.notna(net_return) else "-")
c3.metric("Totale kosten", f"EUR {total_fees:,.2f}")
c4.metric("FX exposure", f"EUR {fx_exposure:,.2f}")
c5.metric("USD→EUR", f"{usd_to_eur:.4f}")

# Holdings table
st.subheader("Holdings")
st.dataframe(
    df[
        [
            "ticker", "name", "asset_class", "theme",
            "market_value_base", "weight_pct", "return_pct_net",
            "factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"
        ]
    ],
    use_container_width=True
)

# Price matrix and returns
prices = build_price_matrix(df, alpha_key=alpha_key, usd_to_eur=usd_to_eur, days=days)
position_values = build_position_value_matrix(prices, df)
portfolio_series = position_values.sum(axis=1, skipna=True) if not position_values.empty else pd.Series(dtype=float)
asset_returns = compute_returns(prices)
portfolio_returns = compute_returns(portfolio_series.to_frame("Portfolio"))["Portfolio"] if not portfolio_series.empty else pd.Series(dtype=float)

# First quant outputs
st.subheader("Portfolio quant")

q1, q2, q3 = st.columns(3)

if not portfolio_series.empty:
    dd = compute_drawdown(portfolio_series)
    max_dd = dd.min() if not dd.empty else math.nan
    vol = rolling_volatility(portfolio_returns, window=vol_window)
    latest_vol = vol.dropna().iloc[-1] if not vol.dropna().empty else math.nan
    total_ret = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1 if len(portfolio_series) > 1 else math.nan
else:
    dd = pd.Series(dtype=float)
    vol = pd.Series(dtype=float)
    max_dd = math.nan
    latest_vol = math.nan
    total_ret = math.nan

q1.metric("Portfolio return", "-" if pd.isna(total_ret) else f"{total_ret:.2%}")
q2.metric("Max drawdown", "-" if pd.isna(max_dd) else f"{max_dd:.2%}")
q3.metric("Rolling vol", "-" if pd.isna(latest_vol) else f"{latest_vol:.2%}")

if not portfolio_series.empty:
    plot_line(portfolio_series, f"Portfolio waarde ({days} dagen)", y_label="EUR")

if not dd.empty:
    plot_line(dd, "Portfolio drawdown", y_label="Drawdown")

if not vol.empty:
    plot_line(vol, f"Rolling volatility ({vol_window} dagen)", y_label="Volatility")

# Correlation matrix
st.subheader("Correlaties")
if asset_returns.empty or asset_returns.shape[1] < 2:
    st.info("Onvoldoende historische data voor correlatiematrix.")
else:
    corr = asset_returns.corr()
    st.dataframe(corr, use_container_width=True)

# Risk contribution
st.subheader("Risk contribution")
weights = df.set_index("ticker")["weight_pct"]
risk_contrib = contribution_to_risk(asset_returns, weights)
if risk_contrib.empty:
    st.info("Onvoldoende data voor risk contribution.")
else:
    st.dataframe(risk_contrib, use_container_width=True)
    plot_bar(risk_contrib, "ticker", "risk_contribution", "Bijdrage aan portefeuillerisico")

# Factor exposure
st.subheader("Factorblootstelling")
factor_df = portfolio_factor_exposure(df)
st.dataframe(factor_df, use_container_width=True)
plot_bar(factor_df, "factor", "exposure", "Portfolio factor exposure")

# Macro overlay
st.subheader("Macro context")
if fred_key:
    macro_options = {
        "Fed Funds": "FEDFUNDS",
        "US CPI": "CPIAUCSL",
        "Unemployment": "UNRATE",
    }
    macro_label = st.selectbox("Kies macroserie", list(macro_options.keys()))
    macro_series = fetch_fred_history(macro_options[macro_label], fred_key, days=max(days, 365))
    if not macro_series.empty:
        plot_line(macro_series, f"{macro_label} historie", y_label=macro_options[macro_label])

        if not portfolio_returns.empty:
            combined = pd.concat(
                [portfolio_returns.rename("portfolio"), macro_series.pct_change().rename("macro")],
                axis=1
            ).dropna()
            if not combined.empty:
                macro_corr = combined["portfolio"].corr(combined["macro"])
                st.metric("Portfolio vs macro return correlation", f"{macro_corr:.2f}")
    else:
        st.info("Geen macrodata beschikbaar.")
else:
    st.info("Voeg een FRED API key toe voor macrocontext.")

# Asset detail
st.subheader("Asset detail")
if not prices.empty:
    asset_choice = st.selectbox("Kies asset", list(prices.columns))
    asset_price = prices[asset_choice].dropna()
    asset_ret = asset_returns[asset_choice].dropna() if asset_choice in asset_returns.columns else pd.Series(dtype=float)

    if not asset_price.empty:
        plot_line(asset_price, f"{asset_choice} prijs ({days} dagen)", y_label="EUR")

    if not asset_ret.empty:
        asset_vol = rolling_volatility(asset_ret, window=vol_window)
        asset_dd = compute_drawdown(asset_price)

        if not asset_dd.empty:
            plot_line(asset_dd, f"{asset_choice} drawdown", y_label="Drawdown")
        if not asset_vol.empty:
            plot_line(asset_vol, f"{asset_choice} rolling volatility", y_label="Volatility")

# Diagnostics
st.subheader("Datakwaliteit")
missing_live = df[df["price_native"].isna()][["ticker", "name", "price_source", "api_id"]]
if missing_live.empty:
    st.success("Voor alle assets is een live prijs gevonden.")
else:
    st.warning("Voor sommige assets is geen live prijs gevonden.")
    st.dataframe(missing_live, use_container_width=True)

if prices.empty:
    st.warning("Geen historische prijsdata opgebouwd. Controleer symbols en API keys.")
