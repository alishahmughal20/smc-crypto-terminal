import os
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
# SIDEBAR CONFIGURATION (EU MiCA USDC PAIRS)
# ==========================================
st.sidebar.title("⚙️ System Settings")
exchange_id = st.sidebar.selectbox("Spot Execution Exchange", ["kraken", "coinbase", "kucoin", "gate"], index=0)
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
        }, timeout=6)
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
    spot = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 6000})
    perp = ccxt.krakenfutures({"enableRateLimit": True, "timeout": 6000})
    return spot, perp

spot_ex, perp_ex = get_exchanges(exchange_id)

def fetch_candles(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
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
    base = symbol.split("/")[0]  # 'BTC' or 'ETH'
    perp_symbol = f"{base}/USD:USD"
    
    funding_rate = 0.0
    oi_change_pct = 0.0
    long_pct = 50.0
    fng_value = 50
    fng_label = "Neutral"

    # 1. Kraken Futures Funding Rate & Crowding Estimation
    try:
        ticker = perp_ex.fetch_ticker(perp_symbol)
        info = ticker.get("info", {})
        raw_fr = float(info.get("fundingRate", 0.0))
        funding_rate = round(raw_fr if abs(raw_fr) > 0.5 else raw_fr * 100, 4)
        long_pct = round(min(max(50.0 + (funding_rate * 200), 25.0), 75.0), 1)
    except Exception:
        pass

    # 2. CoinGecko 24h Momentum Flow
    try:
        cg_id = "bitcoin" if base == "BTC" else "ethereum"
        cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd&include_24hr_change=true"
        cg_resp = requests.get(cg_url, timeout=5).json()
        if cg_id in cg_resp:
            oi_change_pct = round(float(cg_resp[cg_id].get("usd_24h_change", 0.0)), 2)
    except Exception:
        pass

    # 3. Macro Crypto Fear & Greed Index
    try:
        fng_resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
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
        resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        root = ET.fromstring(resp.content)
        for item in root.findall(".//item")[:6]:
            title = item.find("title")
            if title is not None and title.text:
                headlines.append(title.text.strip())
    except Exception as e:
        headlines = [f"RSS feed temporarily unavailable: {e}"]

    if gem_key and GEMINI_AVAILABLE and headlines:
        try:
            client = genai.Client(api_key=gem_key)
            prompt = (
                "Analyze these live crypto headlines for an intraday BTC/ETH trader. "
                "Reply with 'HIGH_IMPACT_PAUSE' followed by a 1-sentence reason ONLY if there is a major exchange hack, "
                "emergency regulatory ban, or sudden market crash. Otherwise reply 'SAFE' followed by a 1-sentence market mood summary.\n\n"
                "Headlines:\n" + "\n".join(f"- {h}" for h in headlines)
            )
            ai_resp = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
            text = ai_resp.text.strip()
            if "HIGH_IMPACT_PAUSE" in text:
                status = "HIGH_IMPACT_PAUSE"
            summary = text.replace("HIGH_IMPACT_PAUSE", "").replace("SAFE", "").strip(" :-")
        except Exception as e:
            summary = f"AI check skipped ({e})"

    return {"status": status, "headlines": headlines, "summary": summary}

# ==========================================
# 3. COMBINED SMC + FVG + FUNDAMENTAL ENGINE
# ==========================================
def analyze_market(symbol: str, rr_ratio: float, news_status: str, req_vol: bool, max_crowd: int) -> dict:
    df_htf = fetch_candles(symbol, "1h", limit=100)
    df_ltf = fetch_candles(symbol, "5m", limit=100)
    fund = get_deep_fundamentals(symbol)

    # 1H Dealing Range (Premium / Discount) & Trend
    recent_htf = df_htf.iloc[-24:]
    swing_high = recent_htf["high"].max()
    swing_low = recent_htf["low"].min()
    equilibrium = (swing_high + swing_low) / 2
    curr_price = df_ltf["close"].iloc[-1]
    range_pct = round(((curr_price - swing_low) / (swing_high - swing_low)) * 100, 1)
    ema_50 = df_htf["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    if curr_price > ema_50 and range_pct < 50:
        bias = "BULLISH (Discount Zone)"
    elif curr_price < ema_50 and range_pct > 50:
        bias = "BEARISH (Premium Zone)"
    else:
        bias = "NEUTRAL (Wait for Pullback)"

    # Detect 5m Fair Value Gaps (FVGs)
    fvgs = []
    for i in range(len(df_ltf) - 15, len(df_ltf) - 1):
        c_prev2 = df_ltf.iloc[i - 2]
        c_curr = df_ltf.iloc[i]
        
        if c_curr["low"] > c_prev2["high"]:
            fvgs.append({
                "type": "BULLISH",
                "start_time": c_prev2["timestamp"],
                "end_time": df_ltf["timestamp"].iloc[-1],
                "bottom": c_prev2["high"],
                "top": c_curr["low"],
                "mid": round((c_curr["low"] + c_prev2["high"]) / 2, 2),
                "index": i
            })
        elif c_curr["high"] < c_prev2["low"]:
            fvgs.append({
                "type": "BEARISH",
                "start_time": c_prev2["timestamp"],
                "end_time": df_ltf["timestamp"].iloc[-1],
                "bottom": c_curr["high"],
                "top": c_prev2["low"],
                "mid": round((c_prev2["low"] + c_curr["high"]) / 2, 2),
                "index": i
            })

    signal = None
    filter_status = "Waiting for 5m Sweep + FVG pattern"

    if news_status == "SAFE" and fvgs and fvgs[-1]["index"] == len(df_ltf) - 2:
        latest_fvg = fvgs[-1]
        c_prev2 = df_ltf.iloc[-4]
        c_prev1 = df_ltf.iloc[-3]
        atr = c_prev1["atr"]
        body = abs(c_prev1["close"] - c_prev1["open"])
        vol_ok = (not req_vol) or (c_prev1["volume"] >= 1.1 * c_prev1["vol_sma20"])

        if "BULLISH" in bias and latest_fvg["type"] == "BULLISH":
            swept_low = min(c_prev2["low"], c_prev1["low"]) <= df_ltf["low"].iloc[-15:-3].min()
            if not swept_low or body <= 1.1 * atr:
                filter_status = "FVG seen, but weak displacement or no liquidity sweep"
            elif not vol_ok:
                filter_status = "Blocked by Fundamental Filter: Low breakout volume"
            elif fund["funding_rate"] > 0.08:
                filter_status = f"Blocked by Fundamental Filter: Overheated funding ({fund['funding_rate']}%)"
            elif fund["long_pct"] > max_crowd:
                filter_status = f"Blocked by Fundamental Filter: Retail too long ({fund['long_pct']}%)"
            else:
                entry = latest_fvg["mid"]
                sl = round(min(c_prev2["low"], c_prev1["low"]) - 0.2 * atr, 2)
                risk = entry - sl
                if risk > 0:
                    signal = {
                        "id": f"{symbol}_BUY_{c_prev1['timestamp']}",
                        "symbol": symbol,
                        "action": "BUY (LONG)",
                        "entry": entry,
                        "sl": sl,
                        "tp1": round(entry + risk * 2.0, 2),
                        "tp2": round(max(entry + risk * rr_ratio, swing_high), 2),
                        "fvg": f"{latest_fvg['bottom']} - {latest_fvg['top']}",
                        "fund_score": f"{fund['fund_score']}/100 (24h {fund['oi_change_pct']:+}%)",
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                    filter_status = "ALL TECHNICAL + FUNDAMENTAL GATES PASSED!"

        elif "BEARISH" in bias and latest_fvg["type"] == "BEARISH":
            swept_high = max(c_prev2["high"], c_prev1["high"]) >= df_ltf["high"].iloc[-15:-3].max()
            if not swept_high or body <= 1.1 * atr:
                filter_status = "FVG seen, but weak displacement or no liquidity sweep"
            elif not vol_ok:
                filter_status = "Blocked by Fundamental Filter: Low breakdown volume"
            elif fund["funding_rate"] < -0.08:
                filter_status = f"Blocked by Fundamental Filter: Crowded shorts ({fund['funding_rate']}%)"
            elif fund["short_pct"] > max_crowd:
                filter_status = f"Blocked by Fundamental Filter: Retail too short ({fund['short_pct']}%)"
            else:
                entry = latest_fvg["mid"]
                sl = round(max(c_prev2["high"], c_prev1["high"]) + 0.2 * atr, 2)
                risk = sl - entry
                if risk > 0:
                    signal = {
                        "id": f"{symbol}_SELL_{c_prev1['timestamp']}",
                        "symbol": symbol,
                        "action": "SELL (SHORT)",
                        "entry": entry,
                        "sl": sl,
                        "tp1": round(entry - risk * 2.0, 2),
                        "tp2": round(min(entry - risk * rr_ratio, swing_low), 2),
                        "fvg": f"{latest_fvg['bottom']} - {latest_fvg['top']}",
                        "fund_score": f"{fund['fund_score']}/100 (24h {fund['oi_change_pct']:+}%)",
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                    filter_status = "ALL TECHNICAL + FUNDAMENTAL GATES PASSED!"

    return {
        "symbol": symbol,
        "df_ltf": df_ltf,
        "price": curr_price,
        "bias": bias,
        "range_pct": range_pct,
        "equilibrium": round(equilibrium, 2),
        "fund": fund,
        "fvgs": fvgs,
        "signal": signal,
        "filter_status": filter_status
    }

# ==========================================
# 4. TELEGRAM ALERT DISPATCHER
# ==========================================
def dispatch_telegram_signal(sig: dict, bot_token: str, chat_id: str):
    if not bot_token or not chat_id or sig["id"] in st.session_state.sent_signals:
        return

    emoji = "🟢" if "BUY" in sig["action"] else "🔴"
    msg = (
        f"{emoji} *SMC + FUNDAMENTAL SIGNAL: {sig['symbol']}*\n"
        f"*Direction:* {sig['action']}\n"
        f"*Limit Entry (50% FVG):* `${sig['entry']:,.2f}`\n"
        f"*Stop-Loss (SL):* `${sig['sl']:,.2f}`\n"
        f"*Take-Profit 1 (2R):* `${sig['tp1']:,.2f}`\n"
        f"*Take-Profit 2 (Liquidity):* `${sig['tp2']:,.2f}`\n"
        f"*FVG Zone:* `{sig['fvg']}`\n"
        f"*Fundamentals:* `{sig['fund_score']}`"
    )
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=6)
        if resp.status_code == 200:
            st.session_state.sent_signals.add(sig["id"])
    except Exception:
        pass

# ==========================================
# 5. NON-BLOCKING LIVE DASHBOARD RENDERER
# ==========================================
st.title("📊 Institutional SMC + FVG & Fundamental Crypto Terminal")

def render_live_dashboard():
    try:
        news_data = fetch_news_and_sentiment(gemini_key)
        btc_data = analyze_market("BTC/USDC", risk_reward, news_data["status"], require_volume_spike, max_crowding_pct)
        eth_data = analyze_market("ETH/USDC", risk_reward, news_data["status"], require_volume_spike, max_crowding_pct)

        for asset in [btc_data, eth_data]:
            if asset["signal"]:
                dispatch_telegram_signal(asset["signal"], tg_bot_token, tg_chat_id)
                if asset["signal"]["id"] not in [s["id"] for s in st.session_state.signal_history]:
                    st.session_state.signal_history.insert(0, asset["signal"])

        active_data = btc_data if selected_pair == "BTC/USDC" else eth_data
        f = active_data["fund"]

        # Row 1: Price & Macro Sentiment KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BTC/USDC Price", f"${btc_data['price']:,.2f}", btc_data["bias"])
        col2.metric("ETH/USDC Price", f"${eth_data['price']:,.2f}", eth_data["bias"])
        col3.metric("Fear & Greed Index", f"{f['fng_value']} / 100", f["fng_label"])
        col4.metric("News Gate Status", news_data["status"], f"Last scan: {datetime.now().strftime('%H:%M:%S')}")

        # Row 2: Live Intraday Fundamental Metrics for Selected Pair
        st.markdown(f"#### 🏛️ Live Fundamental & Derivatives Flow ({selected_pair})")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        fcol1.metric("Fundamental Score", f"{f['fund_score']} / 100", "Institutional Confluence")
        fcol2.metric("24h Market Momentum", f"{f['oi_change_pct']:+}%", "Capital Inflow / Outflow")
        fcol3.metric("Long / Short Bias", f"{f['long_pct']}% L / {f['short_pct']}% S", "Positioning Crowding")
        fcol4.metric("Perpetual Funding Rate", f"{f['funding_rate']}%", "Leverage Bias")

        # News & Scanner Gate Banner
        if news_data["status"] == "HIGH_IMPACT_PAUSE":
            st.error(f"🚨 **TRADING PAUSED — HIGH-IMPACT NEWS DETECTED:** {news_data['summary']}")
        else:
            st.info(f"📰 **AI News Filter:** {news_data['summary']}  |  **Engine Gate ({selected_pair}):** {active_data['filter_status']}")

        # Active Chart + FVG Visualizer
        df_plot = active_data["df_ltf"].iloc[-60:]

        fig = go.Figure(data=[go.Candlestick(
            x=df_plot["timestamp"],
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name=selected_pair
        )])

        for fvg in active_data["fvgs"]:
            if fvg["start_time"] >= df_plot["timestamp"].iloc[0]:
                fill_color = "rgba(0, 200, 83, 0.22)" if fvg["type"] == "BULLISH" else "rgba(255, 82, 82, 0.22)"
                line_color = "green" if fvg["type"] == "BULLISH" else "red"
                fig.add_shape(
                    type="rect",
                    x0=fvg["start_time"], x1=fvg["end_time"],
                    y0=fvg["bottom"], y1=fvg["top"],
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1, dash="dot"),
                )

        fig.add_hline(
            y=active_data["equilibrium"],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"1H Equilibrium (${active_data['equilibrium']:,.2f})"
        )

        fig.update_layout(
            title=f"{selected_pair} — 5m Execution Chart (Fair Value Gaps + 1H Equilibrium)",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        left_col, right_col = st.columns([1.3, 1])

        with left_col:
            st.subheader("🎯 Actionable Buy / Sell Signals")
            if st.session_state.signal_history:
                st.dataframe(pd.DataFrame(st.session_state.signal_history)[
                    ["time", "symbol", "action", "entry", "sl", "tp1", "tp2", "fund_score"]
                ], use_container_width=True)
            else:
                st.write("Scanning 5m liquidity sweeps, Fair Value Gaps, and market flow...")

        with right_col:
            st.subheader("🗞️ Live Crypto Headlines")
            for h in news_data["headlines"]:
                st.markdown(f"- {h}")

    except Exception as e:
        st.error(f"⚠️ Exchange Data Error ({exchange_id}): {e}")
        st.info("Tip: Select 'kraken' or 'coinbase' in the Spot Execution Exchange dropdown on the left.")

# Top-level fragment definition (never wrapped in an if-block)
@st.fragment(run_every="60s" if auto_refresh else None)
def auto_scanning_panel():
    render_live_dashboard()

auto_scanning_panel()
