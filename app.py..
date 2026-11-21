import os
import time
import xml.etree.ElementTree as ET
import ccxt
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==========================================
# PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="SMC + FVG Institutional Crypto Terminal",
    layout="wide"
)

if "sent_signals" not in st.session_state:
    st.session_state.sent_signals = set()
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []


def get_secret(key_name: str) -> str:
    try:
        return st.secrets.get(key_name, os.environ.get(key_name, ""))
    except Exception:
        return os.environ.get(key_name, "")


# ==========================================
# SIDEBAR CONFIGURATION (EU MiCA USDC PAIRS)
# ==========================================
st.sidebar.title("⚙️ System Settings")
exchange_id = st.sidebar.selectbox(
    "Spot Execution Exchange",
    ["kraken", "coinbase", "kucoin", "gate"],
    index=0
)
selected_pair = st.sidebar.selectbox(
    "Active Chart Pair",
    ["BTC/USDC", "ETH/USDC"],
    index=0
)
auto_refresh = st.sidebar.checkbox("Auto-Scan Every 60s", value=True)
risk_reward = st.sidebar.slider("Min Risk-to-Reward (R:R)", 1.5, 4.0, 2.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Fundamental Filter Strictness")
require_volume_spike = st.sidebar.checkbox(
    "Require Above-Average Volume (>1.1x)",
    value=True
)
max_crowding_pct = st.sidebar.slider("Max Long/Short Crowding (%)", 60, 80, 68)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI News Sentiment Filter")
gemini_key = st.sidebar.text_input(
    "Gemini API Key",
    value=get_secret("GEMINI_KEY"),
    type="password"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📲 Telegram Alerts")
tg_bot_token = st.sidebar.text_input(
    "Telegram Bot Token",
    value=get_secret("TG_BOT_TOKEN"),
    type="password"
)
tg_chat_id = st.sidebar.text_input(
    "Telegram Chat ID",
    value=get_secret("TG_CHAT_ID")
)

if st.sidebar.button("Send Test Telegram Alert"):
    if tg_bot_token and tg_chat_id:
        tg_url = "https://api.telegram.org/bot" + tg_bot_token + "/sendMessage"
        res = requests.post(
            tg_url,
            json={
                "chat_id": tg_chat_id,
                "text": "Institutional SMC + Fundamental Scanner Connected!"
            },
            timeout=6
        )
        if res.status_code == 200:
            st.sidebar.success("Test message sent to Telegram!")
        else:
            st.sidebar.error("Telegram Error: " + res.text)
    else:
        st.sidebar.warning("Enter both Bot Token and Chat ID first.")


# ==========================================
# 1. MARKET DATA & MULTI-FACTOR FUNDAMENTALS
# ==========================================
@st.cache_resource
def get_exchanges(ex_id: str):
    spot = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 6000})
    perp = ccxt.krakenfutures({"enableRateLimit": True, "timeout": 6000})
    return spot, perp


spot_ex, perp_ex = get_exchanges(exchange_id)


def fetch_candles(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
    ohlcv = spot_ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    hl = df["high"] - df["low"]
    hc = np.abs(df["high"] - df["close"].shift())
    lc = np.abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    return df


@st.cache_data(ttl=60)
def get_deep_fundamentals(symbol: str) -> dict:
    base = symbol.split("/")[0]
    perp_symbol = base + "/USD:USD"

    funding_rate = 0.0
    oi_change_pct = 0.0
    long_pct = 50.0
    fng_value = 50
    fng_label = "Neutral"

    try:
        ticker = perp_ex.fetch_ticker(perp_symbol)
        info = ticker.get("info", {})
        raw_fr = float(info.get("fundingRate", 0.0))
        scaled_fr = raw_fr if abs(raw_fr) > 0.5 else raw_fr * 100
        funding_rate = round(scaled_fr, 4)
        raw_long = 50.0 + (funding_rate * 200)
        long_pct = round(min(max(raw_long, 25.0), 75.0), 1)
    except Exception:
        pass

    try:
        cg_id = "bitcoin" if base == "BTC" else "ethereum"
        cg_url = "https://api.coingecko.com/api/v3/simple/price"
        cg_params = {
            "ids": cg_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true"
        }
        cg_resp = requests.get(cg_url, params=cg_params, timeout=5).json()
        if cg_id in cg_resp:
            chg = cg_resp[cg_id].get("usd_24h_change", 0.0)
            oi_change_pct = round(float(chg), 2)
    except Exception:
        pass

    try:
        fng_url = "https://api.alternative.me/fng/?limit=1"
        fng_resp = requests.get(fng_url, timeout=5).json()
        if fng_resp.get("data"):
            fng_value = int(fng_resp["data"][0]["value"])
            fng_label = fng_resp["data"][0]["value_classification"]
    except Exception:
        pass

    score = 50
    if abs(oi_change_pct) > 0.5:
        score += 15
    if abs(funding_rate) < 0.05:
        score += 20
    if 38 <= long_pct <= 62:
        score += 15

    return {
        "funding_rate": funding_rate,
        "oi_change_pct": oi_change_pct,
        "long_pct": long_pct,
        "short_pct": round(100 - long_pct, 1),
        "fng_value": fng_value,
        "fng_label": fng_label,
        "fund_score": min(max(score, 0), 100)
    }


# ==========================================
# 2. FREE LIVE RSS NEWS & AI SENTIMENT FILTER
# ==========================================
@st.cache_data(ttl=180)
def fetch_news_and_sentiment(gem_key: str) -> dict:
    headlines = []
    status = "SAFE"
    summary = "Normal market conditions. No black-swan macro alerts."

    try:
        rss_url = "https://cointelegraph.com/rss"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(rss_url, headers=headers, timeout=5)
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item")[:6]:
            title = item.find("title")
            if title is not None and title.text:
                headlines.append(title.text.strip())
    except Exception:
        headlines = ["RSS feed temporarily unavailable"]

    if gem_key and GEMINI_AVAILABLE and headlines:
        try:
            client = genai.Client(
                api_key=gem_key,
                http_options={"timeout": 5000}
            )
            prompt = (
                "Analyze these live crypto headlines for a BTC/ETH trader. "
                "Reply with HIGH_IMPACT_PAUSE and 1 sentence ONLY if there "
                "is a major hack, regulatory ban, or crash. Otherwise reply "
                "SAFE and a 1-sentence market mood summary.\n\n"
                + "\n".join("- " + h for h in headlines)
            )
            models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
            for m_name in models:
                try:
                    ai_resp = client.models.generate_content(
                        model=m_name,
                        contents=prompt
                    )
                    text = ai_resp.text.strip()
                    if "HIGH_IMPACT_PAUSE" in text:
                        status = "HIGH_IMPACT_PAUSE"
                    clean = text.replace("HIGH_IMPACT_PAUSE", "")
                    summary = clean.replace("SAFE", "").strip(" :-")
                    break
                except Exception:
                    continue
        except Exception:
            summary = "AI check skipped (API timeout)"

    return {"status": status, "headlines": headlines, "summary": summary}
# ==========================================
# 3. COMBINED SMC + FVG + FUNDAMENTAL ENGINE
# ==========================================
def analyze_market(
    symbol: str,
    rr_ratio: float,
    news_status: str,
    req_vol: bool,
    max_crowd: int
) -> dict:
    df_htf = fetch_candles(symbol, "1h", limit=100
