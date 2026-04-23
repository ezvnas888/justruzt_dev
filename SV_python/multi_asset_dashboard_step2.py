import math
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

BASE_CURRENCY = "EUR"


# -----------------------------
# Config
# -----------------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


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

    return df


# -----------------------------
# FX
# -----------------------------
@st.cache_data(ttl=3600)
def get_usd_to_eur_ecb_fallback() -> float:
    """
    Eenvoudige fallback:
    in productie zou je ECB SDMX of een eigen FX feed gebruiken.
    Voor nu gebruiken we een handmatige invoer in de sidebar als primaire route.
    """
    return 0.92


# -----------------------------
# Live data fetchers
# -----------------------------
@st.cache_data(ttl=300)
def fetch_coingecko_prices(ids: list[str], vs_currency: str = "eur") -> Dict[str, float]:
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
        price = data.get(coin_id, {}).get(vs_currency.lower())
        if price is not None:
            out[coin_id] = float(price)
    return out


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

    quote = data.get("Global Quote", {})
    price = quote.get("05. price")
    if price is None or price == "":
        return None
    return float(price)


def fetch_alpha_vantage_quotes(symbols: list[str], api_key: str) -> Dict[str, Optional[float]]:
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = fetch_alpha_vantage_quote(symbol, api_key)
        except Exception:
            out[symbol] = None
    return out


# -----------------------------
# Enrichment
# -----------------------------
def enrich_portfolio(
    df: pd.DataFrame,
    manual_usd_to_eur: float,
    alpha_vantage_key: str
) -> pd.DataFrame:
    out = df.copy()

    # 1) live prijzen ophalen
    cg_ids = out.loc[out["price_source"] == "coingecko", "api_id"].dropna().tolist()
    av_symbols = out.loc[out["price_source"] == "alphavantage", "api_id"].dropna().tolist()

    cg_prices_eur = fetch_coingecko_prices(cg_ids, vs_currency="eur") if cg_ids else {}
    av_prices_native = fetch_alpha_vantage_quotes(av_symbols, alpha_vantage_key) if av_symbols else {}

    live_prices = []
    fx_to_base = []

    for _, row in out.iterrows():
        native_ccy = str(row["native_currency"]).upper()
        source = row["price_source"]
        api_id = row["api_id"]

        # prijs
        price_native = None
        if source == "coingecko":
            # We halen hier direct EUR op, dus behandelen dat als native prijs in EUR
            price_native = cg_prices_eur.get(api_id)
            native_ccy = "EUR"
        elif source == "alphavantage":
            price_native = av_prices_native.get(api_id)

        live_prices.append(price_native)

        # FX
        if native_ccy == BASE_CURRENCY:
            fx = 1.0
        elif native_ccy == "USD":
            fx = manual_usd_to_eur
        else:
            fx = 1.0  # later uitbreiden naar GBP, CHF, etc.

        fx_to_base.append(fx)

    out["native_currency_effective"] = [
        "EUR" if s == "coingecko" else c
        for s, c in zip(out["price_source"], out["native_currency"])
    ]
    out["price_native"] = live_prices
    out["fx_to_base"] = fx_to_base

    # lege prijzen opvangen
    out["price_native"] = pd.to_numeric(out["price_native"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["cost_basis_native"] = pd.to_numeric(out["cost_basis_native"], errors="coerce")
    out["fee_native"] = pd.to_numeric(out["fee_native"], errors="coerce")

    out["market_value_native"] = out["quantity"] * out["price_native"]
    out["market_value_base"] = out["market_value_native"] * out["fx_to_base"]

    out["cost_value_native"] = out["quantity"] * out["cost_basis_native"]
    out["cost_value_base"] = out["cost_value_native"] * out["fx_to_base"]

    out["fee_base"] = out["fee_native"] * out["fx_to_base"]

    out["pnl_base_gross"] = out["market_value_base"] - out["cost_value_base"]
    out["pnl_base_net"] = out["pnl_base_gross"] - out["fee_base"]

    out["return_pct_gross"] = out["pnl_base_gross"] / out["cost_value_base"]
    out["return_pct_net"] = out["pnl_base_net"] / out["cost_value_base"]

    total = out["market_value_base"].sum(skipna=True)
    if total and not pd.isna(total):
        out["weight_pct"] = out["market_value_base"] / total
    else:
        out["weight_pct"] = math.nan

    out["fx_exposure_base"] = out.apply(
        lambda row: row["market_value_base"]
        if str(row["native_currency_effective"]).upper() != BASE_CURRENCY else 0.0,
        axis=1,
    )

    out["cost_drag_pct"] = out["fee_base"] / out["cost_value_base"]

    return out


def portfolio_summary(df: pd.DataFrame) -> dict:
    total_value = df["market_value_base"].sum(skipna=True)
    total_cost = df["cost_value_base"].sum(skipna=True)
    total_fees = df["fee_base"].sum(skipna=True)
    gross_pnl = df["pnl_base_gross"].sum(skipna=True)
    net_pnl = df["pnl_base_net"].sum(skipna=True)

    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_fees": total_fees,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "gross_return_pct": gross_pnl / total_cost if total_cost else math.nan,
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


def format_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


def format_eur(x: float) -> str:
    return "-" if pd.isna(x) else f"EUR {x:,.2f}"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Multi-Asset Dashboard Step 2", layout="wide")
st.title("Multi-Asset Portfolio Dashboard — Step 2")
st.caption("CSV import + live prijzen + FX naar EUR")

st.sidebar.header("Instellingen")
alpha_key = st.sidebar.text_input("Alpha Vantage API key", type="password")
manual_usd_to_eur = st.sidebar.number_input(
    "USD → EUR",
    min_value=0.50,
    max_value=1.50,
    value=0.92,
    step=0.01
)

uploaded = st.file_uploader("Upload portfolio.csv", type=["csv"])

if uploaded is None:
    st.info("Upload eerst je portfolio.csv-bestand.")
    st.stop()

try:
    raw = load_portfolio(uploaded)
    df = enrich_portfolio(raw, manual_usd_to_eur=manual_usd_to_eur, alpha_vantage_key=alpha_key)
    summary = portfolio_summary(df)
except Exception as e:
    st.error(f"Fout bij laden/verwerken: {e}")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Totale waarde", format_eur(summary["total_value"]))
c2.metric("Netto P/L", format_eur(summary["net_pnl"]), format_pct(summary["net_return_pct"]))
c3.metric("Bruto P/L", format_eur(summary["gross_pnl"]), format_pct(summary["gross_return_pct"]))
c4.metric("Totale kosten", format_eur(summary["total_fees"]))
c5.metric("FX exposure", format_eur(summary["fx_exposure"]))

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

st.subheader("Datakwaliteit")
missing_prices = df[df["price_native"].isna()][["ticker", "name", "price_source", "api_id"]]
if missing_prices.empty:
    st.success("Voor alle assets is een prijs gevonden.")
else:
    st.warning("Voor sommige assets is geen live prijs gevonden.")
    st.dataframe(missing_prices, use_container_width=True)
