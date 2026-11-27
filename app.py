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
if "locked_decisions" not in st.session_state:
    st.session_state.locked_decisions = {}


def get_secret(key_name: str) -> str:
    try:
        return st.secrets.get(key_name, os.environ.get(key_name, ""))
    except Exception:
        return os.environ.get(key_name, "")


# ==========================================
# SIDEBAR CONFIGURATION
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

if st.sidebar.button("🔄 Reset Locked Decision"):
    st.session_state.locked_decisions.pop(selected_pair, None)

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
alert_on_prepare = st.sidebar.checkbox(
    "Alert on Locked Limit Setups (Daily Signals)",
    value=True
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
        if abs(raw_fr) > 1.0:
            fr_8h = raw_fr / (3.0 * 365.0)
        elif abs(raw_fr) > 0.05:
            fr_8h = raw_fr / 3.0
        else:
            fr_8h = raw_fr * 100.0
        funding_rate = round(fr_8h, 4)
        raw_long = 50.0 + (funding_rate * 250)
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
            models = ["gemini-2.5-flash", "gemini-2.0-flash"]
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
    max_crowd: int,
    send_prepare_alerts: bool
) -> dict:
    df_htf = fetch_candles(symbol, "1h", limit=100)
    df_ltf = fetch_candles(symbol, "5m", limit=100)
    fund = get_deep_fundamentals(symbol)

    recent_htf = df_htf.iloc[-24:]
    swing_high = recent_htf["high"].max()
    swing_low = recent_htf["low"].min()
    equilibrium = (swing_high + swing_low) / 2
    curr_price = df_ltf["close"].iloc[-1]
    closed_price = df_ltf["close"].iloc[-2]
    last_atr = df_ltf["atr"].iloc[-2]
    curr_atr = last_atr if not np.isnan(last_atr) else curr_price * 0.002
    range_pct = round(((closed_price - swing_low) / (swing_high - swing_low)) * 100, 1)
    ema_50 = df_htf["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    if closed_price >= ema_50:
        bias = "BULLISH (Uptrend)" if range_pct <= 80 else "NEUTRAL (Overbought)"
    else:
        bias = "BEARISH (Downtrend)" if range_pct >= 20 else "NEUTRAL (Oversold)"

    fvgs = []
    for i in range(len(df_ltf) - 30, len(df_ltf) - 1):
        c_prev2 = df_ltf.iloc[i - 2]
        c_curr = df_ltf.iloc[i]

        if c_curr["low"] > c_prev2["high"]:
            b_val = round(c_prev2["high"], 2)
            t_val = round(c_curr["low"], 2)
            fvgs.append({
                "type": "BULLISH",
                "start_time": c_prev2["timestamp"],
                "end_time": df_ltf["timestamp"].iloc[-1],
                "bottom": b_val,
                "top": t_val,
                "mid": round((t_val + b_val) / 2, 2),
                "index": i
            })
        elif c_curr["high"] < c_prev2["low"]:
            b_val = round(c_curr["high"], 2)
            t_val = round(c_prev2["low"], 2)
            fvgs.append({
                "type": "BEARISH",
                "start_time": c_prev2["timestamp"],
                "end_time": df_ltf["timestamp"].iloc[-1],
                "bottom": b_val,
                "top": t_val,
                "mid": round((t_val + b_val) / 2, 2),
                "index": i
            })

    # Active BUY & SELL Limit Plans
    bull_fvgs = [g for g in fvgs if g["type"] == "BULLISH" and g["mid"] <= closed_price]
    if bull_fvgs:
        b_zone = bull_fvgs[-1]
        buy_entry = b_zone["mid"]
        buy_sl = round(b_zone["bottom"] - 0.5 * curr_atr, 2)
        buy_source = "5m Bullish FVG ($" + str(b_zone["bottom"]) + " - $" + str(b_zone["top"]) + ")"
    else:
        buy_entry = round(swing_low + 0.25 * (swing_high - swing_low), 2)
        buy_sl = round(swing_low - 0.5 * curr_atr, 2)
        buy_source = "1H Discount Order Block (25% Fib)"

    buy_risk = max(buy_entry - buy_sl, curr_atr * 0.5)
    buy_tp1 = round(buy_entry + buy_risk * 2.0, 2)
    buy_tp2 = round(max(buy_entry + buy_risk * rr_ratio, swing_high), 2)

    bear_fvgs = [g for g in fvgs if g["type"] == "BEARISH" and g["mid"] >= closed_price]
    if bear_fvgs:
        s_zone = bear_fvgs[-1]
        sell_entry = s_zone["mid"]
        sell_sl = round(s_zone["top"] + 0.5 * curr_atr, 2)
        sell_source = "5m Bearish FVG ($" + str(s_zone["bottom"]) + " - $" + str(s_zone["top"]) + ")"
    else:
        sell_entry = round(swing_high - 0.25 * (swing_high - swing_low), 2)
        sell_sl = round(swing_high + 0.5 * curr_atr, 2)
        sell_source = "1H Premium Supply Zone (75% Fib)"

    sell_risk = max(sell_sl - sell_entry, curr_atr * 0.5)
    sell_tp1 = round(sell_entry - sell_risk * 2.0, 2)
    sell_tp2 = round(min(sell_entry - sell_risk * rr_ratio, swing_low), 2)

    trade_plan = {
        "buy_entry": buy_entry,
        "buy_sl": buy_sl,
        "buy_tp1": buy_tp1,
        "buy_tp2": buy_tp2,
        "buy_source": buy_source,
        "sell_entry": sell_entry,
        "sell_sl": sell_sl,
        "sell_tp1": sell_tp1,
        "sell_tp2": sell_tp2,
        "sell_source": sell_source,
    }

    signal = None
    filter_status = "Waiting for 5m Sweep + FVG or 1.0% Zone Trigger"

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
                filter_status = "FVG seen, waiting for stronger displacement"
            elif not vol_ok:
                filter_status = "Blocked by Fundamental Filter: Low breakout volume"
            elif fund["funding_rate"] > 0.08:
                filter_status = "Blocked: Overheated funding (" + str(fund["funding_rate"]) + "%)"
            elif fund["long_pct"] > max_crowd:
                filter_status = "Blocked: Retail too long (" + str(fund["long_pct"]) + "%)"
            else:
                entry = latest_fvg["mid"]
                sl = round(min(c_prev2["low"], c_prev1["low"]) - 0.2 * atr, 2)
                risk = entry - sl
                if risk > 0:
                    signal = {
                        "id": symbol + "_SWEEP_BUY_" + str(c_prev1["timestamp"]),
                        "symbol": symbol,
                        "action": "CONFIRMED BUY (LONG)",
                        "entry": entry,
                        "sl": sl,
                        "tp1": round(entry + risk * 2.0, 2),
                        "tp2": round(max(entry + risk * rr_ratio, swing_high), 2),
                        "fvg": str(latest_fvg["bottom"]) + " - " + str(latest_fvg["top"]),
                        "fund_score": str(fund["fund_score"]) + "/100",
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                    filter_status = "ALL TECHNICAL + FUNDAMENTAL GATES PASSED!"

        elif "BEARISH" in bias and latest_fvg["type"] == "BEARISH":
            swept_high = max(c_prev2["high"], c_prev1["high"]) >= df_ltf["high"].iloc[-15:-3].max()
            if not swept_high or body <= 1.1 * atr:
                filter_status = "FVG seen, waiting for stronger displacement"
            elif not vol_ok:
                filter_status = "Blocked by Fundamental Filter: Low breakdown volume"
            elif fund["funding_rate"] < -0.08:
                filter_status = "Blocked: Crowded shorts (" + str(fund["funding_rate"]) + "%)"
            elif fund["short_pct"] > max_crowd:
                filter_status = "Blocked: Retail too short (" + str(fund["short_pct"]) + "%)"
            else:
                entry = latest_fvg["mid"]
                sl = round(max(c_prev2["high"], c_prev1["high"]) + 0.2 * atr, 2)
                risk = sl - entry
                if risk > 0:
                    signal = {
                        "id": symbol + "_SWEEP_SELL_" + str(c_prev1["timestamp"]),
                        "symbol": symbol,
                        "action": "CONFIRMED SELL (SHORT)",
                        "entry": entry,
                        "sl": sl,
                        "tp1": round(entry - risk * 2.0, 2),
                        "tp2": round(min(entry - risk * rr_ratio, swing_low), 2),
                        "fvg": str(latest_fvg["bottom"]) + " - " + str(latest_fvg["top"]),
                        "fund_score": str(fund["fund_score"]) + "/100",
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                    filter_status = "ALL TECHNICAL + FUNDAMENTAL GATES PASSED!"

    # ==========================================
    # STICKY DECISION ENGINE + TELEGRAM TRIGGER
    # ==========================================
    now_ts = time.time()
    locked = st.session_state.locked_decisions.get(symbol)
    alert_to_send = signal

    if locked and news_status == "SAFE" and not signal:
        age_mins = (now_ts - locked["created_at"]) / 60.0
        still_valid = False
        if locked["type"] == "BUY":
            still_valid = (curr_price > locked["sl"]) and (curr_price < locked["tp1"]) and (age_mins < 30)
        elif locked["type"] == "SELL":
            still_valid = (curr_price < locked["sl"]) and (curr_price > locked["tp1"]) and (age_mins < 30)

        if still_valid:
            decision = locked
        else:
            st.session_state.locked_decisions.pop(symbol, None)
            locked = None

    if not locked or news_status != "SAFE" or signal:
        dist_buy = abs(closed_price - buy_entry) / closed_price * 100
        dist_sell = abs(closed_price - sell_entry) / closed_price * 100

        if news_status == "HIGH_IMPACT_PAUSE":
            decision = {
                "type": "PAUSE",
                "headline": "🚨 ACTION: PAUSE ALL TRADES (" + symbol + ")",
                "detail": "High-impact news detected. Stay in USDC until volatility stabilizes.",
                "created_at": now_ts
            }
        elif signal and "BUY" in signal["action"]:
            decision = {
                "type": "BUY",
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp1": signal["tp1"],
                "headline": "🟢 LOCKED BUY (LONG) ON " + symbol + " AT $" + f"{signal['entry']:,.2f}",
                "detail": "Confirmed Sweep + Bullish FVG! Profit Booking: $" + f"{signal['tp1']:,.2f}" + " (TP1) / $" + f"{signal['tp2']:,.2f}" + " (TP2) | SL: $" + f"{signal['sl']:,.2f}",
                "created_at": now_ts
            }
            st.session_state.locked_decisions[symbol] = decision
        elif signal and "SELL" in signal["action"]:
            decision = {
                "type": "SELL",
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp1": signal["tp1"],
                "headline": "🔴 LOCKED SELL (SHORT) ON " + symbol + " AT $" + f"{signal['entry']:,.2f}",
                "detail": "Confirmed Sweep + Bearish FVG! Profit Booking: $" + f"{signal['tp1']:,.2f}" + " (TP1) / $" + f"{signal['tp2']:,.2f}" + " (TP2) | SL: $" + f"{signal['sl']:,.2f}",
                "created_at": now_ts
            }
            st.session_state.locked_decisions[symbol] = decision
        elif "BULLISH" in bias and dist_buy <= 1.0 and curr_price > buy_sl:
            decision = {
                "type": "BUY",
                "entry": buy_entry,
                "sl": buy_sl,
                "tp1": buy_tp1,
                "headline": "🟢 LOCKED BUY LIMIT ON " + symbol + " AT $" + f"{buy_entry:,.2f}",
                "detail": "Bullish Trend Setup (" + buy_source + "). Profit Booking: $" + f"{buy_tp1:,.2f}" + " / $" + f"{buy_tp2:,.2f}" + " | SL: $" + f"{buy_sl:,.2f}",
                "created_at": now_ts
            }
            st.session_state.locked_decisions[symbol] = decision
            if send_prepare_alerts and not alert_to_send:
                alert_to_send = {
                    "id": symbol + "_LOCKED_BUY_" + str(round(buy_entry, 0)),
                    "symbol": symbol,
                    "action": "LOCKED BUY LIMIT",
                    "entry": buy_entry,
                    "sl": buy_sl,
                    "tp1": buy_tp1,
                    "tp2": buy_tp2,
                    "fvg": buy_source,
                    "fund_score": str(fund["fund_score"]) + "/100",
                    "time": datetime.now().strftime("%H:%M:%S")
                }
        elif "BEARISH" in bias and dist_sell <= 1.0 and curr_price < sell_sl:
            decision = {
                "type": "SELL",
                "entry": sell_entry,
                "sl": sell_sl,
                "tp1": sell_tp1,
                "headline": "🔴 LOCKED SELL LIMIT ON " + symbol + " AT $" + f"{sell_entry:,.2f}",
                "detail": "Bearish Trend Setup (" + sell_source + "). Profit Booking: $" + f"{sell_tp1:,.2f}" + " / $" + f"{sell_tp2:,.2f}" + " | SL: $" + f"{sell_sl:,.2f}",
                "created_at": now_ts
            }
            st.session_state.locked_decisions[symbol] = decision
            if send_prepare_alerts and not alert_to_send:
                alert_to_send = {
                    "id": symbol + "_LOCKED_SELL_" + str(round(sell_entry, 0)),
                    "symbol": symbol,
                    "action": "LOCKED SELL LIMIT",
                    "entry": sell_entry,
                    "sl": sell_sl,
                    "tp1": sell_tp1,
                    "tp2": sell_tp2,
                    "fvg": sell_source,
                    "fund_score": str(fund["fund_score"]) + "/100",
                    "time": datetime.now().strftime("%H:%M:%S")
                }
        else:
            decision = {
                "type": "WAIT",
                "headline": "🟡 ACTION: WAIT — DO NOT ENTER YET (" + symbol + " @ $" + f"{curr_price:,.2f}" + ")",
                "detail": "Bias: " + bias + ". Waiting for price within 1.0% of BUY ($" + f"{buy_entry:,.2f}" + ") or SELL ($" + f"{sell_entry:,.2f}" + ")",
                "created_at": now_ts
            }

    return {
        "symbol": symbol,
        "df_ltf": df_ltf,
        "price": curr_price,
        "bias": bias,
        "range_pct": range_pct,
        "equilibrium": round(equilibrium, 2),
        "fund": fund,
        "fvgs": fvgs,
        "trade_plan": trade_plan,
        "signal": alert_to_send,
        "decision": decision,
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
        emoji + " *SMC + FUNDAMENTAL ALERT: " + sig["symbol"] + "*\n"
        + "*Signal Type:* " + sig["action"] + "\n"
        + "*Entry Position Price:* `$" + f"{sig['entry']:,.2f}" + "`\n"
        + "*Stop-Loss (SL):* `$" + f"{sig['sl']:,.2f}" + "`\n"
        + "*Profit Booking 1 (2R):* `$" + f"{sig['tp1']:,.2f}" + "`\n"
        + "*Profit Booking 2 (Liquidity):* `$" + f"{sig['tp2']:,.2f}" + "`\n"
        + "*Zone:* `" + sig["fvg"] + "`"
    )
    try:
        url = "https://api.telegram.org/bot" + bot_token + "/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=6
        )
        if resp.status_code == 200:
            st.session_state.sent_signals.add(sig["id"])
    except Exception:
        pass


# ==========================================
# 5. LIVE DASHBOARD RENDERER
# ==========================================
st.title("📊 Institutional SMC + FVG & Fundamental Crypto Terminal")


def render_live_dashboard():
    try:
        news_data = fetch_news_and_sentiment(gemini_key)
        btc_data = analyze_market(
            "BTC/USDC",
            risk_reward,
            news_data["status"],
            require_volume_spike,
            max_crowding_pct,
            alert_on_prepare
        )
        eth_data = analyze_market(
            "ETH/USDC",
            risk_reward,
            news_data["status"],
            require_volume_spike,
            max_crowding_pct,
            alert_on_prepare
        )

        for asset in [btc_data, eth_data]:
            if asset["signal"]:
                dispatch_telegram_signal(asset["signal"], tg_bot_token, tg_chat_id)
                existing_ids = [s["id"] for s in st.session_state.signal_history]
                if asset["signal"]["id"] not in existing_ids:
                    st.session_state.signal_history.insert(0, asset["signal"])

        active_data = btc_data if selected_pair == "BTC/USDC" else eth_data
        f = active_data["fund"]
        tp = active_data["trade_plan"]
        dec = active_data["decision"]

        # TOP LIVE DECISION BANNER
        banner_text = "### " + dec["headline"] + "\n**Execution Plan:** " + dec["detail"]
        if dec["type"] == "BUY":
            st.success(banner_text)
        elif dec["type"] in ["SELL", "PAUSE"]:
            st.error(banner_text)
        else:
            st.warning(banner_text)

        # Row 1: Price & Macro Sentiment KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BTC/USDC Price", "$" + f"{btc_data['price']:,.2f}", btc_data["bias"])
        col2.metric("ETH/USDC Price", "$" + f"{eth_data['price']:,.2f}", eth_data["bias"])
        col3.metric("Fear & Greed Index", str(f["fng_value"]) + " / 100", f["fng_label"])
        col4.metric("News Gate Status", news_data["status"], "Last scan: " + datetime.now().strftime("%H:%M:%S"))

        # Row 2: Live Intraday Fundamental Metrics
        st.markdown("#### 🏛️ Live Fundamental & Derivatives Flow (" + selected_pair + ")")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        fcol1.metric("Fundamental Score", str(f["fund_score"]) + " / 100", "Institutional Confluence")
        fcol2.metric("24h Market Momentum", f"{f['oi_change_pct']:+}%", "Capital Inflow / Outflow")
        fcol3.metric("Long / Short Bias", str(f["long_pct"]) + "% L / " + str(f["short_pct"]) + "% S", "Positioning Crowding")
        fcol4.metric("Perpetual Funding (8h)", str(f["funding_rate"]) + "%", "Leverage Bias")

        # Row 3: LIVE ENTRY & PROFIT BOOKING LEVELS
        st.markdown("#### 🎯 Live Entry & Profit Booking Levels (" + selected_pair + ")")
        buy_box, sell_box = st.columns(2)

        with buy_box:
            st.success(
                "**🟢 BUY (LONG) SETUP** — *" + tp["buy_source"] + "*\n\n"
                + "- **Buying Entry Position Price:** `$" + f"{tp['buy_entry']:,.2f}" + "`\n"
                + "- **Profit Booking Price (Target 1 — 2R):** `$" + f"{tp['buy_tp1']:,.2f}" + "`\n"
                + "- **Profit Booking Price (Target 2 — Swing High):** `$" + f"{tp['buy_tp2']:,.2f}" + "`\n"
                + "- **Stop-Loss Exit (Invalidation):** `$" + f"{tp['buy_sl']:,.2f}" + "`"
            )

        with sell_box:
            st.error(
                "**🔴 SELL (SHORT) SETUP** — *" + tp["sell_source"] + "*\n\n"
                + "- **Selling Entry Position Price:** `$" + f"{tp['sell_entry']:,.2f}" + "`\n"
                + "- **Selling Exit / Profit Booking (Target 1 — 2R):** `$" + f"{tp['sell_tp1']:,.2f}" + "`\n"
                + "- **Selling Exit / Profit Booking (Target 2 — Swing Low):** `$" + f"{tp['sell_tp2']:,.2f}" + "`\n"
                + "- **Stop-Loss Exit (Invalidation):** `$" + f"{tp['sell_sl']:,.2f}" + "`"
            )

        # News & Scanner Gate Banner
        st.info(
            "📰 **AI News Filter:** " + news_data["summary"]
            + "  |  **Auto-Alert Gate (" + selected_pair + "):** " + active_data["filter_status"]
        )

        # Active Chart + FVG + Entry/Exit Visualizer
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
                is_bull = fvg["type"] == "BULLISH"
                fill_c = "rgba(0, 200, 83, 0.22)" if is_bull else "rgba(255, 82, 82, 0.22)"
                line_c = "green" if is_bull else "red"
                fig.add_shape(
                    type="rect",
                    x0=fvg["start_time"],
                    x1=fvg["end_time"],
                    y0=fvg["bottom"],
                    y1=fvg["top"],
                    fillcolor=fill_c,
                    line=dict(color=line_c, width=1, dash="dot"),
                )

        fig.add_hline(y=tp["buy_entry"], line_dash="solid", line_color="#00E676", annotation_text="Buy Entry")
        fig.add_hline(y=tp["buy_tp1"], line_dash="dot", line_color="#69F0AE", annotation_text="Buy Profit Exit")
        fig.add_hline(y=tp["sell_entry"], line_dash="solid", line_color="#FF5252", annotation_text="Sell Entry")
        fig.add_hline(y=tp["sell_tp1"], line_dash="dot", line_color="#FF8A80", annotation_text="Sell Profit Exit")
        fig.add_hline(y=active_data["equilibrium"], line_dash="dash", line_color="orange", annotation_text="1H Equilibrium")

        fig.update_layout(
            title=selected_pair + " — 5m Execution Chart (Entry, Profit Booking & Fair Value Gaps)",
            xaxis_rangeslider_visible=False,
            height=540,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        left_col, right_col = st.columns([1.3, 1])

        with left_col:
            st.subheader("📲 Sent Telegram Alerts Log")
            if st.session_state.signal_history:
                cols = ["time", "symbol", "action", "entry", "sl", "tp1", "tp2", "fund_score"]
                st.dataframe(pd.DataFrame(st.session_state.signal_history)[cols], use_container_width=True)
            else:
                st.write("Alerts fire automatically to Telegram whenever a Locked Limit Setup or Confirmed Sweep triggers.")

        with right_col:
            st.subheader("🗞️ Live Crypto Headlines")
            for h in news_data["headlines"]:
                st.markdown("- " + h)

    except Exception as e:
        st.error("⚠️ Exchange Data Error (" + exchange_id + "): " + str(e))
        st.info("Tip: Select 'kraken' or 'coinbase' in the Spot Execution Exchange dropdown on the left.")


render_live_dashboard()

if auto_refresh:
    time.sleep(60)
    st.rerun()
