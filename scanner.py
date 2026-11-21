import os
import ccxt
import requests
import pandas as pd
import numpy as np
from datetime import datetime

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
PAIRS = ["BTC/USDC", "ETH/USDC"]
RR_RATIO = 2.5


def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Missing Telegram credentials.")
        return
    url = "https://api.telegram.org/bot" + TG_BOT_TOKEN + "/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
        timeout=10
    )
    print("Telegram status:", resp.status_code)


def fetch_df(ex, symbol: str, tf: str, limit: int = 100) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    hl = df["high"] - df["low"]
    hc = np.abs(df["high"] - df["close"].shift())
    lc = np.abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    return df


def scan_pair(ex, symbol: str):
    df_htf = fetch_df(ex, symbol, "1h", 100)
    df_ltf = fetch_df(ex, symbol, "5m", 100)

    recent_htf = df_htf.iloc[-24:]
    swing_high = recent_htf["high"].max()
    swing_low = recent_htf["low"].min()
    closed_price = df_ltf["close"].iloc[-2]
    prev_closed = df_ltf["close"].iloc[-3]
    atr = df_ltf["atr"].iloc[-2]
    if np.isnan(atr):
        atr = closed_price * 0.002

    range_pct = ((closed_price - swing_low) / (swing_high - swing_low)) * 100
    ema_50 = df_htf["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    if closed_price > ema_50 and range_pct < 50:
        bias = "BULLISH"
    elif closed_price < ema_50 and range_pct > 50:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    fvgs = []
    for i in range(len(df_ltf) - 30, len(df_ltf) - 1):
        c_prev2 = df_ltf.iloc[i - 2]
        c_curr = df_ltf.iloc[i]
        if c_curr["low"] > c_prev2["high"]:
            b_val = round(c_prev2["high"], 2)
            t_val = round(c_curr["low"], 2)
            fvgs.append({"type": "BULLISH", "bottom": b_val, "top": t_val, "mid": round((t_val + b_val) / 2, 2)})
        elif c_curr["high"] < c_prev2["low"]:
            b_val = round(c_curr["high"], 2)
            t_val = round(c_prev2["low"], 2)
            fvgs.append({"type": "BEARISH", "bottom": b_val, "top": t_val, "mid": round((t_val + b_val) / 2, 2)})

    # Buy setup check
    bull_fvgs = [g for g in fvgs if g["type"] == "BULLISH" and g["mid"] <= closed_price]
    if bull_fvgs:
        z = bull_fvgs[-1]
        buy_entry = z["mid"]
        buy_sl = round(z["bottom"] - 0.5 * atr, 2)
        buy_src = "5m Bullish FVG"
    else:
        buy_entry = round(swing_low + 0.25 * (swing_high - swing_low), 2)
        buy_sl = round(swing_low - 0.5 * atr, 2)
        buy_src = "1H Discount Zone"

    # Sell setup check
    bear_fvgs = [g for g in fvgs if g["type"] == "BEARISH" and g["mid"] >= closed_price]
    if bear_fvgs:
        z = bear_fvgs[-1]
        sell_entry = z["mid"]
        sell_sl = round(z["top"] + 0.5 * atr, 2)
        sell_src = "5m Bearish FVG"
    else:
        sell_entry = round(swing_high - 0.25 * (swing_high - swing_low), 2)
        sell_sl = round(swing_high + 0.5 * atr, 2)
        sell_src = "1H Premium Zone"

    # Stateless anti-spam: only alert when the latest 5m candle JUST entered the 0.65% zone
    dist_buy = abs(closed_price - buy_entry) / closed_price * 100
    prev_dist_buy = abs(prev_closed - buy_entry) / prev_closed * 100

    dist_sell = abs(closed_price - sell_entry) / closed_price * 100
    prev_dist_sell = abs(prev_closed - sell_entry) / prev_closed * 100

    if bias == "BULLISH" and dist_buy <= 0.65 and prev_dist_buy > 0.65 and closed_price > buy_sl:
        risk = max(buy_entry - buy_sl, atr * 0.5)
        tp1 = round(buy_entry + risk * 2.0, 2)
        tp2 = round(max(buy_entry + risk * RR_RATIO, swing_high), 2)
        msg = (
            "🟢 *24/7 BUY ALERT: " + symbol + "*\n"
            + "*Zone:* " + buy_src + "\n"
            + "*Entry:* `$" + f"{buy_entry:,.2f}" + "`\n"
            + "*Stop-Loss:* `$" + f"{buy_sl:,.2f}" + "`\n"
            + "*TP1 (2R):* `$" + f"{tp1:,.2f}" + "`\n"
            + "*TP2:* `$" + f"{tp2:,.2f}" + "`"
        )
        send_telegram(msg)

    elif bias == "BEARISH" and dist_sell <= 0.65 and prev_dist_sell > 0.65 and closed_price < sell_sl:
        risk = max(sell_sl - sell_entry, atr * 0.5)
        tp1 = round(sell_entry - risk * 2.0, 2)
        tp2 = round(min(sell_entry - risk * RR_RATIO, swing_low), 2)
        msg = (
            "🔴 *24/7 SELL ALERT: " + symbol + "*\n"
            + "*Zone:* " + sell_src + "\n"
            + "*Entry:* `$" + f"{sell_entry:,.2f}" + "`\n"
            + "*Stop-Loss:* `$" + f"{sell_sl:,.2f}" + "`\n"
            + "*TP1 (2R):* `$" + f"{tp1:,.2f}" + "`\n"
            + "*TP2:* `$" + f"{tp2:,.2f}" + "`"
        )
        send_telegram(msg)
    else:
        print(symbol, "| Bias:", bias, "| Price:", closed_price, "| No fresh trigger.")


if __name__ == "__main__":
    ex = ccxt.kraken({"enableRateLimit": True, "timeout": 10000})
    for p in PAIRS:
        try:
            scan_pair(ex, p)
        except Exception as e:
            print("Error scanning", p, ":", e)
