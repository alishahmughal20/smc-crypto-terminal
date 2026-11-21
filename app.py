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
# PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="SMC + FVG Institutional Crypto Terminal", layout="wide")

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
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ System Settings")
exchange_id = st.sidebar.selectbox("Spot Execution Exchange", ["kraken", "okx", "bybit", "binance"], index=0)
selected_pair = st.sidebar.selectbox("Active Chart Pair", ["BTC/USDC", "ETH/USDC"], index=0)
auto_refresh = st.sidebar.checkbox("Auto-Scan Every 60s", value=True)
risk_reward = st.sidebar.slider("Min Risk-to-Reward (R:R)", 1.5, 4.0, 2.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Fundamental Filter Strictness")
require_volume_spike = st.sidebar.checkbox("Require Above-Average Volume (>1.1x)", value=True)
max_crowding_pct = st.sidebar.slider("Max Long/Short Crowding (%)", 60, 80, 68)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI News Sentiment Filter")
gemini_key = st.sidebar.text_input("Gemini API Key", value=get_secret("GEMINI_KEY"), type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("📲 Telegram Alerts")
tg_bot_token = st.sidebar.text_input("Telegram Bot Token", value=get_secret("TG_BOT_TOKEN"), type="password")
tg_chat_id = st.sidebar.text_input("Telegram Chat ID", value=get_secret("TG_CHAT_ID"))

if st.sidebar.button("Send Test Telegram Alert"):
    if tg_bot_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_bot_token}/sendMessage"
        res = requests.post(url, json={
            "chat_id": tg_chat_id,
            "text": "✅ *Institutional SMC + Fundamental Scanner Connected!*\nMonitoring BTC/USDC & ETH/USDC.",
            "parse_mode": "Markdown"
        })
        if res.status_code == 200:
            st.sidebar.success("Test message sent to Telegram!")
        else:
            st.sidebar.error(f"Telegram Error: {res.text}")
    else:
        st.sidebar.warning("Enter both Bot Token and Chat ID first.")

# ==========================================
# 1. MARKET DATA & MULTI-FACTOR FUNDAMENTALS
# ==========================================
@st.cache_resource
def get_exchanges(ex_id: str):
    spot = getattr(ccxt, ex_id)({"enableRateLimit": True})
    perp = ccxt.okx({"enableRateLimit": True})
    return spot, perp

spot_ex, perp_ex = get_exchanges(exchange_id)

def fetch_candles(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    ohlcv = spot_ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # 14-period ATR & 20-period Volume Moving Average
    hl = df["high"] - df["low"]
    hc = np.abs(df["high"] - df["close"].shift())
    lc = np.abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    return df

@st.cache_data(ttl=60)
def get_deep_fundamentals(symbol: str) -> dict:
    """
    Fetches 4 real-time fundamental & derivatives metrics:
    1. Perpetual Funding Rate
    2. Open Interest (OI) 15-minute % Change
    3. Long/Short Account Ratio
    4. Macro Crypto Fear & Greed Index
    """
    base = symbol.split("/")[0]  # 'BTC' or 'ETH'
    perp_symbol = f"{base}/USDT:USDT"
    
    funding_rate = 0.0
    oi_change_pct = 0.0
    long_pct = 50.0
    fng_value = 50
    fng_label = "Neutral"

    # 1. Perpetual Funding Rate
    try:
        funding = perp_ex.fetch_funding_rate(perp_symbol)
        funding_rate = round(funding.get("fundingRate", 0.0) * 100, 4)
    except Exception:
        pass

    # 2. Open Interest (OI) Momentum (Last 3 x 5m intervals = 15m flow)
    try:
        oi_hist = perp_ex.fetch_open_interest_history(perp_symbol, timeframe="5m", limit=4)
        if len(oi_hist) >= 2:
            oi_prev = oi_hist[0].get("openInterestAmount", 0)
            oi_now = oi_hist[-1].get("openInterestAmount", 0)
            if oi_prev > 0:
                oi_change_pct = round(((oi_now - oi_prev) / oi_prev) * 100, 2)
    except Exception:
        pass

    # 3. Long/Short Account Ratio (OKX Public Rubik Endpoint)
    try:
        ls_url = f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={base}&period=5m"
        ls_resp = requests.get(ls_url, timeout=6).json()
        if ls_resp.get("data"):
            ratio = float(ls_resp["data"][0][1])  # Long/Short ratio (e.g., 1.25)
            long_pct = round((ratio / (1 + ratio)) * 100, 1)
    except Exception:
        pass

    # 4. Macro Crypto Fear & Greed Index
    try:
        fng_resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6).json()
        if fng_resp.get("data"):
            fng_value = int(fng_resp["data"][0]["value"])
            fng_label = fng_resp["data"][0]["value_classification"]
    except Exception:
        pass

    # Compute Fundamental Health Score (0 to 100)
    score = 50
    if oi_change_pct > 0.15:
        score += 20  # Fresh capital entering the move
    elif oi_change_pct < -0.3:
        score -= 15
