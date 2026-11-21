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

    hl = df["high
