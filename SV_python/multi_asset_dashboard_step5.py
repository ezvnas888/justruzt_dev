import math
from typing import Dict, List, Optional

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
# Loading
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
        raise ValueError(f"Ontbrekende kolommen: {missing}")

    numeric_cols = [
        "quantity", "cost_basis_native", "fee_native",
        "factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# Current FX from ECB
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


@st.cache_data(ttl=1800)
def fetch_alpha_vantage_fx_daily(from_symbol: str, to_symbol: str, api_key: str, days: int = 90) -> pd.Series:
    if not api_key:
        return pd.Series(dtype=float)

    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }
    r = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    ts = data.get("Time Series FX (Daily)", {})
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
    s.name = f"{from_symbol}{to_symbol}"
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
# Current portfolio snapshot
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

    live_prices = []
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

        live_prices.append(p)
        native_currency_effective.append(native_eff)
        fx_to_base.append(fx)

    out["price_native"] = pd.to_numeric(live_prices, errors="coerce")
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
# Historical prices with dated FX
# -----------------------------
def get_asset_history_base(row: pd.Series, alpha_key: str, days: int) -> pd.Series:
    source = row["price_source"]
    api_id = row["api_id"]
    native_ccy = str(row["native_currency"]).upper()

    if source == "coingecko":
        # direct in EUR
        return fetch_coingecko_history(api_id, days=days, vs_currency="eur")

    price_native = fetch_alpha_vantage_history(api_id, alpha_key, days=days)
    if price_native.empty:
        return price_native

    if native_ccy == BASE_CURRENCY:
        return price_native

    if native_ccy == "USD":
        fx_hist = fetch_alpha_vantage_fx_daily("USD", "EUR", alpha_key, days=days)
        if fx_hist.empty:
            return price_native
        combined = pd.concat([price_native.rename("px"), fx_hist.rename("fx")], axis=1).sort_index().ffill().dropna()
        return combined["px"] * combined["fx"]

    return price_native


def build_price_matrix_base(df: pd.DataFrame, alpha_key: str, days: int = 90) -> pd.DataFrame:
    series_list = []

    for _, row in df.iterrows():
        try:
            s = get_asset_history_base(row, alpha_key=alpha_key, days=days)
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
# Quant
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


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 30) -> pd.Series:
    combined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    return combined["a"].rolling(window).corr(combined["b"])


def classify_regime(fred_key: str, days: int = 365) -> pd.DataFrame:
    if not fred_key:
        return pd.DataFrame()

    fed = fetch_fred_history("FEDFUNDS", fred_key, days=days)
    cpi = fetch_fred_history("CPIAUCSL", fred_key, days=days)
    unrate = fetch_fred_history("UNRATE", fred_key, days=days)

    if fed.empty or cpi.empty or unrate.empty:
        return pd.DataFrame()

    df = pd.concat(
        [fed.rename("fedfunds"), cpi.rename("cpi"), unrate.rename("unrate")],
        axis=1
    ).sort_index().ffill().dropna()

    df["inflation_regime"] = (df["cpi"].pct_change(12) > 0.03).map({True: "high", False: "low"})
    df["rates_regime"] = (df["fedfunds"].diff(3) > 0).map({True: "rising", False: "falling_or_flat"})
    df["labor_regime"] = (df["unrate"].diff(3) > 0).map({True: "weakening", False: "stable_or_improving"})
    return df


def scenario_shock_table(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    shocks = {
        "Rates +50bps": {
            "Crypto": -0.08,
            "Equity": -0.04,
            "ETF": -0.03,
            "Sovereign Bonds": -0.06,
        },
        "Gold +10%": {
            "Crypto": 0.00,
            "Equity": 0.00,
            "ETF": 0.05,
            "Sovereign Bonds": 0.00,
        },
        "BTC -20%": {
            "Crypto": -0.20,
            "Equity": 0.00,
            "ETF": 0.00,
            "Sovereign Bonds": 0.00,
        },
        "USD -5% vs EUR": {
            "Crypto": -0.05,
            "Equity": 0.00,
            "ETF": -0.05,
            "Sovereign Bonds": 0.00,
        },
        "Geopolitical shock": {
            "Crypto": -0.07,
            "Equity": -0.04,
            "ETF": 0.02,
            "Sovereign Bonds": 0.03,
        },
    }

    mapping = shocks.get(scenario_name, {})
    out = df.copy()
    out["scenario_return"] = out["asset_class"].map(mapping).fillna(0.0)
    out["scenario_pnl"] = out["market_value_base"] * out["scenario_return"]
    return out


# -----------------------------
# Plot helpers
# -----------------------------
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


def plot_heatmap(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Geen data voor heatmap.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cax = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    fig.colorbar(cax)
    st.pyplot(fig)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Multi-Asset Dashboard Step 5", layout="wide")
st.title("Multi-Asset Portfolio Dashboard — Step 5")
st.caption("Regimes, rolling correlations, factor heatmap en scenario-analyse")

st.sidebar.header("API keys")
alpha_key = st.sidebar.text_input("Alpha Vantage API key", type="password")
fred_key = st.sidebar.text_input("FRED API key", type="password")

st.sidebar.header("Parameters")
days = st.sidebar.selectbox("Historische periode", [90, 180, 365], index=1)
vol_window = st.sidebar.selectbox("Volatility window", [20, 30, 60], index=1)
corr_window = st.sidebar.selectbox("Rolling correlation window", [20, 30, 60], index=1)

uploaded = st.file_uploader("Upload portfolio.csv", type=["csv"])
if uploaded is None:
    st.info("Upload eerst je portfolio.csv-bestand.")
    st.stop()

raw = load_portfolio(uploaded)

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

df = enrich_portfolio(raw, alpha_key=alpha_key, usd_to_eur=usd_to_eur)

# top metrics
total_value = df["market_value_base"].sum(skipna=True)
total_cost = df["cost_value_base"].sum(skipna=True)
net_pnl = df["pnl_base_net"].sum(skipna=True)
net_return = net_pnl / total_cost if total_cost else math.nan
fx_exposure = df["fx_exposure_base"].sum(skipna=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Totale waarde", f"EUR {total_value:,.2f}")
m2.metric("Netto P/L", f"EUR {net_pnl:,.2f}", "-" if pd.isna(net_return) else f"{net_return:.2%}")
m3.metric("FX exposure", f"EUR {fx_exposure:,.2f}")
m4.metric("Actuele USD→EUR", "-" if usd_to_eur is None else f"{usd_to_eur:.4f}")

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

# historical data in EUR/base
prices_base = build_price_matrix_base(df, alpha_key=alpha_key, days=days)
position_values = build_position_value_matrix(prices_base, df)

if position_values.empty:
    st.warning("Geen historische prijsdata opgebouwd. Controleer je symbols en API keys.")
    st.stop()

portfolio_series = position_values.sum(axis=1, skipna=True)
asset_returns = compute_returns(prices_base)
portfolio_returns = compute_returns(portfolio_series.to_frame("Portfolio"))["Portfolio"]

# core quant
st.subheader("Portfolio quant")
dd = compute_drawdown(portfolio_series)
vol = rolling_volatility(portfolio_returns, window=vol_window)
max_dd = dd.min() if not dd.empty else math.nan
latest_vol = vol.dropna().iloc[-1] if not vol.dropna().empty else math.nan

q1, q2, q3 = st.columns(3)
q1.metric("Portfolio return", f"{(portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1):.2%}")
q2.metric("Max drawdown", "-" if pd.isna(max_dd) else f"{max_dd:.2%}")
q3.metric("Rolling vol", "-" if pd.isna(latest_vol) else f"{latest_vol:.2%}")

plot_line(portfolio_series, f"Portfolio waarde ({days} dagen)", y_label="EUR")
plot_line(dd, "Portfolio drawdown", y_label="Drawdown")
if not vol.empty:
    plot_line(vol, f"Rolling volatility ({vol_window}d)", y_label="Vol")

# rolling correlations
st.subheader("Rolling correlations")
if asset_returns.shape[1] >= 2:
    asset_list = list(asset_returns.columns)
    col1, col2 = st.columns(2)
    with col1:
        asset_a = st.selectbox("Asset A", asset_list, index=0)
    with col2:
        asset_b = st.selectbox("Asset B", asset_list, index=min(1, len(asset_list)-1))

    rc = rolling_correlation(asset_returns[asset_a], asset_returns[asset_b], window=corr_window)
    if not rc.empty:
        plot_line(rc, f"Rolling correlation: {asset_a} vs {asset_b}", y_label="Corr")
else:
    st.info("Onvoldoende historische series voor rolling correlations.")

# correlation heatmap
st.subheader("Correlation heatmap")
if asset_returns.shape[1] >= 2:
    corr = asset_returns.corr()
    st.dataframe(corr, use_container_width=True)
    plot_heatmap(corr, "Asset return correlations")
else:
    st.info("Onvoldoende data voor correlatiematrix.")

# factor exposure + heatmap
st.subheader("Factorblootstelling")
factor_df = portfolio_factor_exposure(df)
st.dataframe(factor_df, use_container_width=True)
plot_bar(factor_df, "factor", "exposure", "Portfolio factor exposure")

factor_matrix = df.set_index("ticker")[
    ["factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"]
]
plot_heatmap(factor_matrix, "Factor heatmap per asset")

# regime detection
st.subheader("Regimes")
regime_df = classify_regime(fred_key, days=max(days, 365)) if fred_key else pd.DataFrame()

if regime_df.empty:
    st.info("Voeg een FRED API key toe voor regimeclassificatie.")
else:
    latest = regime_df.iloc[-1]
    r1, r2, r3 = st.columns(3)
    r1.metric("Inflation regime", latest["inflation_regime"])
    r2.metric("Rates regime", latest["rates_regime"])
    r3.metric("Labor regime", latest["labor_regime"])

    # portfolio vs macro rolling corr
    macro_choice = st.selectbox("Macroserie", ["FEDFUNDS", "CPIAUCSL", "UNRATE"])
    macro_series = fetch_fred_history(macro_choice, fred_key, days=max(days, 365))
    if not macro_series.empty:
        macro_ret = macro_series.pct_change()
        port_macro = pd.concat(
            [portfolio_returns.rename("portfolio"), macro_ret.rename("macro")],
            axis=1
        ).dropna()
        if not port_macro.empty:
            rc_macro = rolling_correlation(port_macro["portfolio"], port_macro["macro"], window=corr_window)
            if not rc_macro.empty:
                plot_line(rc_macro, f"Rolling correlation: Portfolio vs {macro_choice}", y_label="Corr")

# scenario engine
st.subheader("Scenario-analyse")
scenario = st.selectbox(
    "Kies scenario",
    [
        "Rates +50bps",
        "Gold +10%",
        "BTC -20%",
        "USD -5% vs EUR",
        "Geopolitical shock",
    ]
)

scenario_df = scenario_shock_table(df, scenario)
scenario_total = scenario_df["scenario_pnl"].sum()
scenario_return = scenario_total / total_value if total_value else math.nan

s1, s2 = st.columns(2)
s1.metric("Scenario P/L", f"EUR {scenario_total:,.2f}")
s2.metric("Scenario return", "-" if pd.isna(scenario_return) else f"{scenario_return:.2%}")

st.dataframe(
    scenario_df[
        ["ticker", "name", "asset_class", "market_value_base", "scenario_return", "scenario_pnl"]
    ],
    use_container_width=True
)

plot_bar(
    scenario_df.sort_values("scenario_pnl", ascending=False),
    "ticker",
    "scenario_pnl",
    f"Scenario impact per asset — {scenario}"
)

# data quality
st.subheader("Datakwaliteit")
missing_live = df[df["price_native"].isna()][["ticker", "name", "price_source", "api_id"]]
if missing_live.empty:
    st.success("Voor alle assets is een live prijs gevonden.")
else:
    st.warning("Voor sommige assets is geen live prijs gevonden.")
    st.dataframe(missing_live, use_container_width=True)
