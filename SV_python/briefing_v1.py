"""
Monday Morning Briefing — Prototype v1
Single data source: Twelve Data (stocks, ETFs, crypto pairs).
FX: ECB API. Macro: FRED. Synthesis: Claude Opus 4.7.

CSV format: price_source column is informational only.
             api_id must be a valid Twelve Data symbol:
               - Stocks/ETFs : ASML.AS, GDX, AAPL
               - Crypto      : BTC/EUR, ETH/EUR  (use /EUR to avoid FX conversion)
               - USD-priced  : GDX (native_currency=USD — ECB rate applied)
"""

import json
import math
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

BASE_CURRENCY = "EUR"
TWELVE_DATA_BASE = "https://api.twelvedata.com"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
ECB_DATA_BASE = "https://data-api.ecb.europa.eu/service/data"

BRIEFING_MODEL = "claude-opus-4-7"

BRIEFING_SYSTEM_PROMPT = """\
You are the editorial engine for Stellarvum, a private portfolio intelligence platform.

Your task: write the Monday Morning Briefing — a factual weekly portfolio summary — \
from the structured JSON data provided. You are writing a newspaper about the portfolio, not advice about it.

MANDATORY RULES (violations are errors, not style choices):
1. Past tense for all review sections (Week in Review, The Numbers).
2. Present tense only when stating a scheduled future fact (calendar sections).
3. FORBIDDEN words/phrases: buy, sell, reduce, add, consider, opportunity, risk-off, risk-on,
   you should, we recommend, could rise, might fall, expect, likely, suggests you,
   potentially, attractive, cheap, expensive.
4. Every number must appear verbatim in the payload — never re-round, never invent.
5. Plain language — write for an intelligent non-specialist investor, not a trader.
6. No investment advice. No predictions. No directional language of any kind.
7. If a data field is null or missing, omit the fact entirely — do not speculate or fill in.

OUTPUT FORMAT (Markdown, exact section order, no deviations):

## Your Week in Review
[2–3 sentences. What moved, what did not, in past tense factual tone. Name assets explicitly.]

## The Numbers
[Markdown table: Holding | WTD | MTD | YTD | Since Purchase | Value (EUR)
 Use — for any missing value. Percentages formatted as e.g. +2.3% or -1.1%.]

## Income Received
[List dividends/coupons received this week. If the payload income field says data is unavailable,
 write exactly: "Dividend and coupon data will appear here in a future release."]

## This Week's Calendar
[Bullet list from calendar_this_week. Format: - DATE — EVENT (source)
 If empty: "No scheduled macro releases in the calendar for this week."]

## On the Horizon
[Bullet list from calendar_horizon. Same format. Limit to 5 items.
 End with one sentence stating the current macro regime — factual, past tense.]
"""


# ---------------------------------------------------------------------------
# Twelve Data — single source for all live prices and history
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_td_quote(symbol: str, api_key: str) -> Optional[float]:
    """Current price for any Twelve Data symbol (stock, ETF, or crypto pair)."""
    if not api_key:
        return None
    try:
        r = requests.get(
            f"{TWELVE_DATA_BASE}/price",
            params={"symbol": symbol, "apikey": api_key},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        # Twelve Data returns {"code": 4xx, "message": "..."} on API-level errors
        if "code" in data and data["code"] != 200:
            return None
        price = data.get("price")
        return float(price) if price is not None else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def probe_td_api(api_key: str) -> str:
    """Quick single-symbol test to verify the API key and surface any error message."""
    if not api_key:
        return "no key"
    try:
        r = requests.get(
            f"{TWELVE_DATA_BASE}/price",
            params={"symbol": "AAPL", "apikey": api_key},
            timeout=10,
        )
        data = r.json()
        if "code" in data:
            return f"API error {data['code']}: {data.get('message', '')}"
        if "price" in data:
            return "ok"
        return f"unexpected response: {data}"
    except Exception as e:
        return f"network error: {e}"


@st.cache_data(ttl=1800)
def fetch_td_history(symbol: str, api_key: str, days: int = 90) -> pd.Series:
    """Daily close prices for any Twelve Data symbol."""
    if not api_key:
        return pd.Series(dtype=float)
    try:
        r = requests.get(
            f"{TWELVE_DATA_BASE}/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "outputsize": min(days + 10, 5000),
                "apikey": api_key,
            },
            timeout=30,
        )
        r.raise_for_status()
        values = r.json().get("values", [])
        if not values:
            return pd.Series(dtype=float)
        rows = [
            (pd.to_datetime(v["datetime"]), float(v["close"]))
            for v in values
            if v.get("close") is not None
        ]
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series(dict(rows)).sort_index()
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        s = s[s.index >= cutoff]
        s.name = symbol
        return s
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# ECB FX + FRED macro
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_ecb_usd_eur() -> Optional[float]:
    try:
        url = f"{ECB_DATA_BASE}/EXR/D.USD.EUR.SP00.A"
        r = requests.get(
            url,
            params={"lastNObservations": 1},
            headers={"Accept": "application/vnd.sdmx.data+json;version=1.0.0-wd"},
            timeout=20,
        )
        r.raise_for_status()
        series_dict = r.json()["dataSets"][0]["series"]
        if not series_dict:
            return None
        obs = next(iter(series_dict.values())).get("observations", {})
        return float(next(iter(obs.values()))[0]) if obs else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_fred_history(series_id: str, api_key: str, days: int = 365) -> pd.Series:
    if not api_key:
        return pd.Series(dtype=float)
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            FRED_BASE,
            params={"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start},
            timeout=30,
        )
        r.raise_for_status()
        rows = [
            (pd.to_datetime(obs["date"]), float(obs["value"]))
            for obs in r.json().get("observations", [])
            if obs.get("value") not in (None, ".", "")
        ]
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series(dict(rows)).sort_index()
        s.name = series_id
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=86400)
def fetch_fred_release_calendar(fred_key: str, days_ahead: int = 30) -> List[dict]:
    """Fetch upcoming release dates per tracked release ID (same API key as series data)."""
    if not fred_key:
        return []

    # FRED release_id → friendly label
    TRACKED_IDS = {
        10:  "US CPI (Inflation)",
        50:  "US Jobs Report",
        53:  "US GDP",
        54:  "US PCE / Personal Income",
        31:  "US PPI",
        56:  "US Retail Sales",
        18:  "Fed Funds Rate (H.15)",
    }

    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=days_ahead)
    today_str = today.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    events = []
    for release_id, label in TRACKED_IDS.items():
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/release/dates",
                params={
                    "release_id": release_id,
                    "api_key": fred_key,
                    "file_type": "json",
                    "realtime_start": today_str,
                    "realtime_end": end_str,
                    "sort_order": "asc",
                    "include_release_dates_with_no_data": "false",
                },
                timeout=15,
            )
            r.raise_for_status()
            for item in r.json().get("release_dates", []):
                date = item.get("date", "")
                if today_str <= date <= end_str:
                    events.append({"date": date, "event": label, "source": "FRED"})
        except Exception:
            continue

    return sorted(events, key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# Portfolio loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_portfolio(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    required = [
        "ticker", "name", "asset_class", "theme", "quantity", "native_currency",
        "cost_basis_native", "fee_native", "tax_bucket", "api_id",
        "factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in ["quantity", "cost_basis_native", "fee_native",
              "factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Portfolio computation
# ---------------------------------------------------------------------------

def enrich_portfolio(df: pd.DataFrame, td_key: str, usd_to_eur: float) -> pd.DataFrame:
    out = df.copy()
    live_prices, fx_list = [], []

    for _, row in out.iterrows():
        symbol = row["api_id"]
        native_ccy = str(row["native_currency"]).upper()
        price = fetch_td_quote(symbol, td_key)
        fx = usd_to_eur if native_ccy == "USD" else 1.0
        live_prices.append(price)
        fx_list.append(fx)

    out["price_native"] = pd.to_numeric(live_prices, errors="coerce")
    out["fx_to_base"] = fx_list
    out["market_value_native"] = out["quantity"] * out["price_native"]
    out["market_value_base"] = out["market_value_native"] * out["fx_to_base"]
    out["cost_value_native"] = out["quantity"] * out["cost_basis_native"]
    out["cost_value_base"] = out["cost_value_native"] * out["fx_to_base"]
    out["fee_base"] = out["fee_native"] * out["fx_to_base"]
    out["pnl_base_net"] = out["market_value_base"] - out["cost_value_base"] - out["fee_base"]
    out["return_pct_net"] = out["pnl_base_net"] / out["cost_value_base"]

    total = out["market_value_base"].sum(skipna=True)
    out["weight_pct"] = out["market_value_base"] / total if total else math.nan
    return out


def build_price_matrix_base(df: pd.DataFrame, td_key: str, usd_to_eur: float, days: int = 90) -> pd.DataFrame:
    series_list = []
    for _, row in df.iterrows():
        try:
            s = fetch_td_history(row["api_id"], td_key, days=days)
            if s.empty:
                continue
            native_ccy = str(row["native_currency"]).upper()
            if native_ccy == "USD":
                s = s * usd_to_eur
            series_list.append(s.rename(row["ticker"]))
        except Exception:
            pass
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1).sort_index()


def build_position_value_matrix(prices: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    qty_map = df.set_index("ticker")["quantity"].to_dict()
    out = prices.copy()
    for col in out.columns:
        out[col] = out[col] * qty_map.get(col, 0)
    return out


def classify_regime(fred_key: str, days: int = 365) -> pd.DataFrame:
    if not fred_key:
        return pd.DataFrame()
    fed = fetch_fred_history("FEDFUNDS", fred_key, days=days)
    cpi = fetch_fred_history("CPIAUCSL", fred_key, days=days)
    unrate = fetch_fred_history("UNRATE", fred_key, days=days)
    if fed.empty or cpi.empty or unrate.empty:
        return pd.DataFrame()
    df = pd.concat([fed.rename("fedfunds"), cpi.rename("cpi"), unrate.rename("unrate")], axis=1).sort_index().ffill().dropna()
    df["inflation_regime"] = (df["cpi"].pct_change(12) > 0.03).map({True: "high", False: "low"})
    df["rates_regime"] = (df["fedfunds"].diff(3) > 0).map({True: "rising", False: "falling or flat"})
    df["labor_regime"] = (df["unrate"].diff(3) > 0).map({True: "weakening", False: "stable or improving"})
    return df


def portfolio_factor_exposure(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["weight_pct"] = temp["weight_pct"].fillna(0)
    return pd.DataFrame([
        {"factor": f.replace("factor_", ""), "exposure": float((temp["weight_pct"] * temp[f]).sum())}
        for f in ["factor_inflation", "factor_rates", "factor_liquidity", "factor_geopolitics"]
    ])


# ---------------------------------------------------------------------------
# Period returns
# ---------------------------------------------------------------------------

def _period_start(today: pd.Timestamp, period: str) -> pd.Timestamp:
    if period == "WTD":
        return today - pd.Timedelta(days=today.weekday())
    if period == "MTD":
        return today.replace(day=1)
    if period == "YTD":
        return pd.Timestamp(today.year, 1, 1)
    return today


def compute_period_returns(prices_base: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if prices_base.empty:
        return pd.DataFrame(columns=["ticker", "name", "WTD", "MTD", "YTD"])
    today = prices_base.index[-1]
    results = []
    for _, row in df.iterrows():
        ticker, name = row["ticker"], row["name"]
        rec = {"ticker": ticker, "name": name}
        if ticker not in prices_base.columns:
            rec.update({"WTD": None, "MTD": None, "YTD": None})
        else:
            series = prices_base[ticker].dropna()
            current = float(series.iloc[-1]) if not series.empty else None
            for period in ("WTD", "MTD", "YTD"):
                window = series[series.index >= _period_start(today, period)]
                if current is None or len(window) < 1:
                    rec[period] = None
                else:
                    start_val = float(window.iloc[0])
                    rec[period] = round((current / start_val - 1) * 100, 2) if start_val != 0 else None
        results.append(rec)
    return pd.DataFrame(results)


def _portfolio_period_return(portfolio_series: pd.Series, period: str) -> Optional[float]:
    if portfolio_series.empty:
        return None
    today = portfolio_series.index[-1]
    window = portfolio_series[portfolio_series.index >= _period_start(today, period)]
    if len(window) < 1:
        return None
    start_val = float(window.iloc[0])
    end_val = float(portfolio_series.iloc[-1])
    return round((end_val / start_val - 1) * 100, 2) if start_val != 0 else None


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_briefing_payload(
    df: pd.DataFrame,
    period_returns: pd.DataFrame,
    portfolio_series: pd.Series,
    regime_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    calendar_events: List[dict],
    week_start: str,
) -> dict:
    today = pd.Timestamp.today().normalize()
    week_end_str = (today - pd.Timedelta(days=today.weekday()) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    pr_indexed = period_returns.set_index("ticker") if not period_returns.empty else pd.DataFrame()

    holdings = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        pr = pr_indexed.loc[ticker] if ticker in pr_indexed.index else None
        holdings.append({
            "ticker": ticker,
            "name": row["name"],
            "asset_class": row["asset_class"],
            "market_value_eur": round(float(row["market_value_base"]), 2) if pd.notna(row.get("market_value_base")) else None,
            "weight_pct": round(float(row["weight_pct"]) * 100, 1) if pd.notna(row.get("weight_pct")) else None,
            "since_purchase_pct": round(float(row["return_pct_net"]) * 100, 2) if pd.notna(row.get("return_pct_net")) else None,
            "WTD_pct": float(pr["WTD"]) if pr is not None and pd.notna(pr.get("WTD")) else None,
            "MTD_pct": float(pr["MTD"]) if pr is not None and pd.notna(pr.get("MTD")) else None,
            "YTD_pct": float(pr["YTD"]) if pr is not None and pd.notna(pr.get("YTD")) else None,
        })

    top_movers = sorted(
        [h for h in holdings if h["WTD_pct"] is not None],
        key=lambda x: abs(x["WTD_pct"]),
        reverse=True,
    )[:5]

    regime_summary = {}
    if not regime_df.empty:
        latest = regime_df.iloc[-1]
        regime_summary = {
            "inflation": str(latest.get("inflation_regime", "unknown")),
            "rates": str(latest.get("rates_regime", "unknown")),
            "labor": str(latest.get("labor_regime", "unknown")),
        }

    factors = {
        str(r["factor"]): round(float(r["exposure"]), 3)
        for _, r in factor_df.iterrows()
    } if not factor_df.empty else {}

    return {
        "meta": {
            "week_start": week_start,
            "generated_at": today.strftime("%Y-%m-%d"),
            "base_currency": "EUR",
        },
        "portfolio_summary": {
            "total_value_eur": round(float(df["market_value_base"].sum(skipna=True)), 2),
            "total_cost_eur": round(float(df["cost_value_base"].sum(skipna=True)), 2),
            "net_unrealised_pnl_eur": round(float(df["pnl_base_net"].sum(skipna=True)), 2),
            "WTD_pct": _portfolio_period_return(portfolio_series, "WTD"),
            "MTD_pct": _portfolio_period_return(portfolio_series, "MTD"),
            "YTD_pct": _portfolio_period_return(portfolio_series, "YTD"),
        },
        "holdings": holdings,
        "top_movers_wtd": top_movers,
        "income": {"note": "Dividend and coupon data not yet integrated. This section will be populated in v2."},
        "calendar_this_week": [e for e in calendar_events if e["date"] <= week_end_str],
        "calendar_horizon": [e for e in calendar_events if e["date"] > week_end_str],
        "macro_regime": regime_summary,
        "factor_exposure": factors,
    }


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

def synthesize_briefing(payload: dict, anthropic_key: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "**Error:** `anthropic` package not installed. Run `pip install anthropic`."

    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model=BRIEFING_MODEL,
        max_tokens=2000,
        system=BRIEFING_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Generate the Monday Morning Briefing for the week of {payload['meta']['week_start']}.\n\n"
                f"Payload:\n```json\n{json.dumps(payload, indent=2)}\n```"
            ),
        }],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Delivery helpers
# ---------------------------------------------------------------------------

def markdown_to_html(md: str, week_start: str) -> str:
    """Convert briefing markdown to a mobile-friendly HTML email / download."""
    try:
        import markdown as md_lib
        body = md_lib.markdown(md, extensions=["tables"])
    except ImportError:
        # Minimal fallback: wrap in <pre> so at least it's readable
        escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = f"<pre style='white-space:pre-wrap'>{escaped}</pre>"

    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Stellarvum Briefing — {week_start}</title>
          <style>
            body {{
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              font-size: 16px; line-height: 1.6; color: #1a1a1a;
              max-width: 680px; margin: 0 auto; padding: 20px 16px;
              background: #ffffff;
            }}
            h1 {{ font-size: 1.4em; color: #0a2540; border-bottom: 2px solid #0a2540; padding-bottom: 6px; }}
            h2 {{ font-size: 1.15em; color: #0a2540; margin-top: 1.6em; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; margin: 1em 0; }}
            th {{ background: #0a2540; color: #fff; padding: 8px 6px; text-align: left; }}
            td {{ padding: 7px 6px; border-bottom: 1px solid #e0e0e0; }}
            tr:nth-child(even) td {{ background: #f7f9fc; }}
            ul, ol {{ padding-left: 1.4em; }}
            li {{ margin: 4px 0; }}
            p {{ margin: 0.6em 0; }}
            .footer {{ font-size: 0.78em; color: #888; margin-top: 2em; border-top: 1px solid #e0e0e0; padding-top: 8px; }}
          </style>
        </head>
        <body>
          <h1>Stellarvum — Monday Morning Briefing</h1>
          <p class="footer">Week of {week_start} &nbsp;·&nbsp; Generated by Stellarvum v1</p>
          {body}
          <p class="footer">This document is for informational purposes only and does not constitute investment advice.</p>
        </body>
        </html>
    """)


def send_briefing_email(
    html: str,
    week_start: str,
    to_addr: str,
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
) -> str:
    """Send the HTML briefing by email. Returns '' on success or an error string."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Stellarvum Briefing — week of {week_start}"
        msg["From"] = smtp_user
        msg["To"] = to_addr
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=20) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        return ""
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stellarvum — Monday Morning Briefing", layout="wide")
st.title("Stellarvum — Monday Morning Briefing")
st.caption("Prototype v1 · Data: Twelve Data + ECB + FRED · Synthesis: Claude Opus 4.7")

st.sidebar.header("API Keys")
td_key = st.sidebar.text_input("Twelve Data key", type="password")
fred_key = st.sidebar.text_input("FRED key", type="password")
anthropic_key = st.sidebar.text_input("Anthropic (Claude) key", type="password")

st.sidebar.header("Parameters")
days = st.sidebar.selectbox("History window", [90, 180, 365], index=0)

today = pd.Timestamp.today().normalize()
default_week_start = (today - pd.Timedelta(days=today.weekday())).date()
week_start = st.sidebar.date_input("Week start (Monday)", value=default_week_start)

with st.sidebar.expander("Email Delivery (optional)"):
    email_enabled = st.checkbox("Send briefing by email", value=False)
    email_to = st.text_input("Recipient address", value="", placeholder="you@example.com")
    smtp_server = st.text_input("SMTP server", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP port (SSL)", value=465, step=1)
    smtp_user = st.text_input("SMTP username", placeholder="your.account@gmail.com")
    smtp_password = st.text_input("App password / SMTP password", type="password")
    st.caption("Gmail: use a 16-char App Password (not your login password).")

st.sidebar.markdown("""
---
**CSV api_id format**
- Stocks/ETFs: `ASML.AS`, `GDX`, `AAPL`
- Crypto (EUR): `BTC/EUR`, `ETH/EUR`
- Crypto (USD): `BTC/USD` + set native_currency=USD
""")

uploaded = st.file_uploader("Upload portfolio.csv", type=["csv"])
if uploaded is None:
    st.info("Upload your portfolio.csv to begin.")
    st.stop()

raw = load_portfolio(uploaded)

ecb_rate = fetch_ecb_usd_eur()
fallback_fx = st.sidebar.number_input("Fallback USD→EUR", min_value=0.50, max_value=1.50, value=0.92, step=0.01)
usd_to_eur = ecb_rate if ecb_rate is not None else fallback_fx

if not td_key:
    st.warning("Add your Twelve Data API key in the sidebar to load live prices.")
    st.stop()

td_status = probe_td_api(td_key)
if td_status != "ok":
    st.error(f"Twelve Data API: {td_status}")
    st.stop()

df = enrich_portfolio(raw, td_key=td_key, usd_to_eur=usd_to_eur)

total_value = df["market_value_base"].sum(skipna=True)
net_pnl = df["pnl_base_net"].sum(skipna=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total value", f"EUR {total_value:,.2f}")
c2.metric("Net P/L", f"EUR {net_pnl:,.2f}")
c3.metric("USD→EUR", f"{usd_to_eur:.4f}" if usd_to_eur else "—")
c4.metric("Holdings", str(len(df)))

st.subheader("Holdings snapshot")
st.dataframe(
    df[["ticker", "name", "asset_class", "market_value_base", "weight_pct", "return_pct_net"]].rename(
        columns={"market_value_base": "Value (EUR)", "weight_pct": "Weight", "return_pct_net": "Since Purchase"}
    ),
    use_container_width=True,
)

missing = df[df["price_native"].isna()][["ticker", "name", "api_id"]]
if not missing.empty:
    st.warning("No live price found for these assets — check api_id format:")
    st.dataframe(missing, use_container_width=True)

with st.spinner("Fetching price history…"):
    prices_base = build_price_matrix_base(df, td_key=td_key, usd_to_eur=usd_to_eur, days=days)

position_values = build_position_value_matrix(prices_base, df)
portfolio_series = position_values.sum(axis=1, skipna=True) if not position_values.empty else pd.Series(dtype=float)
period_returns = compute_period_returns(prices_base, df)

if not period_returns.empty:
    st.subheader("Period returns")
    st.dataframe(period_returns.set_index("ticker"), use_container_width=True)

regime_df = classify_regime(fred_key, days=max(days, 365)) if fred_key else pd.DataFrame()
factor_df = portfolio_factor_exposure(df)

if not regime_df.empty:
    latest = regime_df.iloc[-1]
    st.subheader("Macro regime")
    r1, r2, r3 = st.columns(3)
    r1.metric("Inflation", latest["inflation_regime"])
    r2.metric("Rates", latest["rates_regime"])
    r3.metric("Labor", latest["labor_regime"])

calendar_events = fetch_fred_release_calendar(fred_key, days_ahead=30) if fred_key else []
if calendar_events:
    st.subheader("Upcoming macro calendar")
    st.dataframe(pd.DataFrame(calendar_events), use_container_width=True)
elif fred_key:
    st.info("No tracked macro releases scheduled in the next 30 days.")
else:
    st.info("Add your FRED key (same key as above) to load the macro release calendar.")

st.divider()
st.subheader("Generate Briefing")

if not anthropic_key:
    st.warning("Add your Anthropic API key in the sidebar to generate the narrative.")
else:
    if st.button("Generate Monday Morning Briefing", type="primary"):
        with st.spinner("Building payload and calling Claude Opus 4.7…"):
            payload = build_briefing_payload(
                df=df,
                period_returns=period_returns,
                portfolio_series=portfolio_series,
                regime_df=regime_df,
                factor_df=factor_df,
                calendar_events=calendar_events,
                week_start=str(week_start),
            )
            narrative = synthesize_briefing(payload, anthropic_key)

        st.markdown("---")
        st.markdown(narrative)

        # --- Delivery ---
        html_doc = markdown_to_html(narrative, str(week_start))
        file_name = f"briefing_{week_start}.html"

        st.download_button(
            label="Download briefing (HTML — opens on mobile)",
            data=html_doc.encode("utf-8"),
            file_name=file_name,
            mime="text/html",
            help="Save to your device and open in any browser for a clean mobile view.",
        )

        if email_enabled:
            if not email_to or not smtp_user or not smtp_password:
                st.warning("Fill in recipient address, SMTP username, and password to send by email.")
            else:
                with st.spinner(f"Sending to {email_to}…"):
                    err = send_briefing_email(
                        html=html_doc,
                        week_start=str(week_start),
                        to_addr=email_to,
                        smtp_server=smtp_server,
                        smtp_port=int(smtp_port),
                        smtp_user=smtp_user,
                        smtp_password=smtp_password,
                    )
                if err:
                    st.error(f"Email failed: {err}")
                else:
                    st.success(f"Briefing sent to {email_to}")

        with st.expander("Raw payload (JSON)", expanded=False):
            st.json(payload)
