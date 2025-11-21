#!/usr/bin/env python3
"""
Delta Exchange (v2 India) momentum scanner + VWAP + MACD + Telegram + order placement
Railway-ready single-file bot.

Configure via environment variables (see .env.example).
Run continuously (Procfile sets worker). Use PAPER_MODE while testing.
"""

import os
import time
import hmac
import hashlib
import json
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# -----------------------------
# CONFIG FROM ENV
# -----------------------------
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = os.getenv("BASE_URL", "https://api.deltaex.org")  # India region
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() in ("1","true","yes")
CAPITAL = float(os.getenv("CAPITAL", "1000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MIN_AVG_VOLUME = float(os.getenv("MIN_AVG_VOLUME", "1000"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.5"))
CANDLE_RESOLUTION = int(os.getenv("CANDLE_RESOLUTION", "300"))
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "300"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
VWAP_WINDOW = int(os.getenv("VWAP_WINDOW", "20"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
USE_PREV_AS_CLOSED = os.getenv("USE_PREV_AS_CLOSED", "false").lower() in ("1","true","yes")
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))  # 0 = all
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("delta-bot")

# -----------------------------
# Helper: Delta v2 Auth
# Signature: METHOD + TIMESTAMP + PATH + QUERY_STRING + BODY  (HMAC-SHA256 hex)
# -----------------------------

def generate_signature(method, path, query_string, body_str):
    timestamp = str(int(time.time()))
    payload = method.upper() + timestamp + path + (query_string or "") + (body_str or "")
    sig = hmac.new(API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return sig, timestamp


def auth_headers(method, path, body_dict=None, query_string=""):
    body_str = ""
    if body_dict is not None:
        body_str = json.dumps(body_dict, separators=(',', ':'))
    sig, timestamp = generate_signature(method, path, query_string, body_str)
    headers = {
        "api-key": API_KEY,
        "signature": sig,
        "timestamp": timestamp,
        "Content-Type": "application/json",
        "User-Agent": "delta-railway-bot/1.0"
    }
    return headers, body_str

# -----------------------------
# Telegram helper
# -----------------------------

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.warning("Telegram send failed: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("Telegram error: %s", e)

# -----------------------------
# Market data
# -----------------------------

def get_futures():
    path = "/v2/products"
    url = BASE_URL + path
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        fut = [p for p in j.get("result", []) if p.get("contract_type") == "perpetual_futures"]
        return fut
    except Exception as e:
        logger.exception("get_futures error")
        return []


def get_candles(product_id, resolution=CANDLE_RESOLUTION, limit=CANDLE_LIMIT):
    path = "/v2/history/candles"
    qs = f"resolution={resolution}&limit={limit}&product_id={product_id}"
    url = BASE_URL + path + "?" + qs
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            logger.debug("Candle fetch status %s for %s", r.status_code, product_id)
            return None
        j = r.json()
        if "result" not in j or not j["result"]:
            return None
        df = pd.DataFrame(j["result"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception:
        logger.exception("get_candles error for %s", product_id)
        return None

# -----------------------------
# Indicators
# -----------------------------

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def macd(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def vwap(df, window=VWAP_WINDOW):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    rolling_pv = pv.rolling(window).sum()
    rolling_vol = df["volume"].rolling(window).sum()
    return rolling_pv / rolling_vol


def rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period).mean()
    avg_loss = loss.ewm(alpha=1/period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -----------------------------
# Orders
# -----------------------------

def place_market_order(product_id, size, side="buy"):
    path = "/v2/orders"
    url = BASE_URL + path
    body = {"product_id": int(product_id), "size": float(size), "side": side, "order_type": "market"}
    headers, body_str = auth_headers("POST", path, body)
    if PAPER_MODE:
        logger.info("[PAPER_MODE] Order not sent: %s", body)
        return {"paper_mode": True, "body": body}
    try:
        r = requests.post(url, headers=headers, data=body_str, timeout=20)
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}
    except Exception:
        logger.exception("place_market_order error")
        return {"error": "exception"}

# -----------------------------
# Evaluate symbol
# -----------------------------

def evaluate_symbol(prod):
    try:
        df = get_candles(prod["id"])
        if df is None or len(df) < max(VWAP_WINDOW + 3, MACD_SLOW + MACD_SIGNAL + 5):
            return None

        closed_i = -1
        prev_i = -2
        prev2_i = -3
        if USE_PREV_AS_CLOSED:
            closed_i = -2
            prev_i = -3
            prev2_i = -4

        df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
        df["vwap"] = vwap(df, VWAP_WINDOW)
        df["ema20"] = ema(df["close"], 20)
        df["rsi"] = rsi(df["close"], RSI_PERIOD)

        df = df.dropna().reset_index(drop=True)
        if len(df) < 4:
            return None

        last = df.iloc[closed_i]
        prev = df.iloc[prev_i]
        prev2 = df.iloc[prev2_i]

        avg_vol = df["volume"].rolling(VWAP_WINDOW).mean().iloc[-1]
        recent_vol = float(last["volume"])
        if np.isnan(avg_vol) or avg_vol < MIN_AVG_VOLUME:
            return None
        if recent_vol < VOLUME_MULTIPLIER * avg_vol:
            return None

        macd_cross_up = (prev["macd"] <= prev["macd_sig"]) and (last["macd"] > last["macd_sig"])
        macd_cross_down = (prev["macd"] >= prev["macd_sig"]) and (last["macd"] < last["macd_sig"])

        last_vwap = float(last["vwap"])
        last_price = float(last["close"])
        price_above_ema = last_price > float(last["ema20"])
        price_below_ema = last_price < float(last["ema20"])

        info = {
            "symbol": prod.get("symbol"),
            "product_id": prod.get("id"),
            "price": last_price,
            "avg_vol": float(avg_vol),
            "recent_vol": recent_vol,
            "time": str(last.get("time"))
        }

        if macd_cross_up and last_price > last_vwap and price_above_ema:
            info.update({"side": "buy", "reason": "MACD cross up + price>VWAP + price>EMA20"})
            return info
        if macd_cross_down and last_price < last_vwap and price_below_ema:
            info.update({"side": "sell", "reason": "MACD cross down + price<VWAP + price<EMA20"})
            return info

        return None

    except Exception:
        logger.exception("evaluate_symbol error for %s", prod.get("symbol"))
        return None

# -----------------------------
# Position sizing
# -----------------------------

def compute_size(price, capital=CAPITAL, risk_per_trade=RISK_PER_TRADE, min_size=0.0001):
    risk_amount = capital * risk_per_trade
    qty = risk_amount / price if price > 0 else min_size
    qty_rounded = round(qty, 6)
    if qty_rounded < min_size:
        qty_rounded = min_size
    return qty_rounded

# -----------------------------
# Main loop (single-run scanner) - can be invoked continuously
# -----------------------------

def scan_and_execute():
    now = datetime.utcnow().isoformat()
    text = f"🔎 Delta Momentum scan starting at {now} UTC"
    logger.info(text)
    send_telegram(text)

    futs = get_futures()
    if not futs:
        send_telegram("❌ Failed to fetch futures list or no futures found.")
        return

    # optionally limit universe to top liquidity to save rate-limit
    if MAX_SYMBOLS and MAX_SYMBOLS > 0:
        futs = sorted(futs, key=lambda x: x.get("turnover_usd_24h", 0), reverse=True)[:MAX_SYMBOLS]

    signals = []
    for p in futs:
        sig = evaluate_symbol(p)
        if sig:
            signals.append(sig)

    if not signals:
        send_telegram("⚠️ No momentum signals found in this run.")
        return

    signals = sorted(signals, key=lambda s: s["recent_vol"], reverse=True)

    for s in signals:
        msg = f"*Signal:* {s['symbol']}\n*Side:* {s['side']}\n*Price:* {s['price']}\n*Reason:* {s['reason']}\n*AvgVol:* {int(s['avg_vol'])}  *RecentVol:* {int(s['recent_vol'])}\n*Time:* {s['time']}"
        send_telegram(msg)

        size = compute_size(s['price'])
        send_telegram(f"Placing {'(PAPER) ' if PAPER_MODE else ''}{s['side']} market order for {s['symbol']} size={size}")
        order_resp = place_market_order(s['product_id'], size, side=s['side'])
        send_telegram(f"Order response for {s['symbol']}: `{order_resp}`")
        time.sleep(1.0)

# -----------------------------
# Entrypoint: run loop forever
# -----------------------------
if __name__ == "__main__":
    # sanity checks
    if not API_KEY or not API_SECRET:
        logger.error("API_KEY or API_SECRET not set. Exiting.")
        raise SystemExit("Missing API credentials")

    logger.info("Starting Delta Momentum Bot (Railway-ready). PAPER_MODE=%s", PAPER_MODE)
    # Run forever — Railway worker will keep the process alive
    while True:
        try:
            scan_and_execute()
        except Exception:
            logger.exception("Unexpected error in main loop")
            send_telegram("Fatal error in scanner — check logs")
        # sleep between runs (default 60s)
        time.sleep(SLEEP_SECONDS)
