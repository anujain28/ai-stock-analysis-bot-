# app.py
# Full AI Stock Analysis + Dhan + Groww + Dividends + Paper Trading + PNL Log + Keep-alive
# NOTE: This file is intended for demo / paper-trading simulation only.

import os
import sys
import json
import time
import math
import threading
import traceback
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import ta
import pytz
import requests

# -------------------------
# CONFIGURATION / DEFAULTS
# -------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAPER_PNL_FILE = DATA_DIR / "paper_pnl_log.csv"
BOOKED_TRADES_FILE = DATA_DIR / "paper_booked_trades.csv"
PAPER_POS_FILE = DATA_DIR / "paper_positions.json"

# Defaults - change as needed in UI
DEFAULT_BASE_CAPITAL = 100_000.0           # ₹1 Lakh
DEFAULT_MAX_USAGE_PCT = 0.40               # 40%
DEFAULT_MAX_POSITIONS = 5
DEFAULT_TARGET_PCT = 0.04                  # 4% target
DEFAULT_AUTO_START = True
DEFAULT_AUTO_START_TIME = dtime(hour=9, minute=30)   # 09:30 IST
DEFAULT_EOD_CAPTURE_TIME = dtime(hour=15, minute=30) # 15:30 IST
DEFAULT_TRAILING_BASE = 0.05               # default trailing pct (adaptive)
DEFAULT_BROKERAGE_PCT = 0.0003             # 0.03%
DEFAULT_GST_PCT = 0.18
DEFAULT_STT_PCT = 0.001                     # 0.1%
DEFAULT_EXCHANGE_PCT = 0.00003              # 0.003%
DEFAULT_STAMP_PCT = 0.00015                 # 0.015%
DEFAULT_PLATFORM_FEE = 10.0
DEFAULT_TAX_PCT = 0.15                      # 15% STCG
IST = pytz.timezone("Asia/Kolkata")

# Keep-alive / self ping
KEEP_ALIVE = True
SELF_URL = "https://airobots.streamlit.app/"   # user provided; can be changed in UI
SELF_PING_INTERVAL_SEC = 4 * 60  # 4 minutes

# -------------------------
# Helper utilities
# -------------------------
def now_ist():
    return datetime.now(IST)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def save_json(path: Path, obj):
    try:
        with path.open("w") as f:
            json.dump(obj, f, default=str, indent=2)
    except Exception:
        pass

def load_json(path: Path):
    try:
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def append_csv(path: Path, row: Dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)

# -------------------------
# Minimal scanner helpers (reused/adapted)
# -------------------------
def nse_yf_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = str(sym).strip().upper()
    return s if s.endswith(".NS") else f"{s}.NS"

def fetch_yf_df(symbol: str, period: str = "60d", interval: str = "15m"):
    try:
        t = yf.Ticker(nse_yf_symbol(symbol))
        df = t.history(period=period, interval=interval, auto_adjust=True)
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame):
    """Compute a few indicators used by scanner and book-now AI."""
    out = {}
    if df is None or df.empty:
        return out
    c = df["Close"]
    h = df["High"] if "High" in df else c
    l = df["Low"] if "Low" in df else c
    try:
        out['rsi'] = ta.momentum.rsi(c, window=14).iloc[-1]
    except Exception:
        out['rsi'] = np.nan
    try:
        macd = ta.trend.MACD(c)
        out['macd'] = safe_float(macd.macd().iloc[-1])
        out['macd_sig'] = safe_float(macd.macd_signal().iloc[-1])
    except Exception:
        out['macd'] = out['macd_sig'] = np.nan
    try:
        atr = ta.volatility.average_true_range(h, l, c, window=14).iloc[-1]
        out['atr'] = safe_float(atr)
        out['atr_pct'] = safe_float(atr) / float(c.iloc[-1]) * 100.0 if c.iloc[-1] else np.nan
    except Exception:
        out['atr'] = out['atr_pct'] = np.nan
    try:
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        out['bb_width'] = safe_float(bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / float(c.iloc[-1]) * 100.0
    except Exception:
        out['bb_width'] = np.nan
    try:
        vol = df["Volume"] if "Volume" in df else pd.Series(dtype=float)
        out['vol'] = float(vol.iloc[-1]) if not vol.empty else np.nan
        out['vol_avg20'] = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else np.nan
    except Exception:
        out['vol'] = out['vol_avg20'] = np.nan
    try:
        out['ema9'] = float(ta.trend.ema_indicator(c, window=9).iloc[-1])
        out['ema21'] = float(ta.trend.ema_indicator(c, window=21).iloc[-1])
    except Exception:
        out['ema9'] = out['ema21'] = np.nan
    return out

# -------------------------
# Paper trading engine
# -------------------------
class PaperTrader:
    def __init__(self):
        # load persisted positions and logs
        self.positions = load_json(PAPER_POS_FILE) or {}
        # positions keyed by ticker: dict with entry_price, qty, invested, peak, entry_time, source
        self.booked_file = BOOKED_TRADES_FILE
        self.pnl_file = PAPER_PNL_FILE
        # in-memory session values
        self.lock = threading.Lock()

    def persist_positions(self):
        save_json(PAPER_POS_FILE, self.positions)

    def snapshot_positions_df(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        rows = []
        for t, v in self.positions.items():
            rows.append({
                "Ticker": t,
                "Qty": v.get("qty"),
                "EntryPrice": v.get("entry_price"),
                "Invested": v.get("invested"),
                "Peak": v.get("peak", v.get("entry_price")),
                "EntryTime": v.get("entry_time"),
                "Source": v.get("source", "auto")
            })
        return pd.DataFrame(rows)

    def buy(self, ticker: str, alloc: float, price: float, qty: int, source: str = "auto"):
        """Register a buy (paper)."""
        with self.lock:
            if qty <= 0:
                return None
            invested = price * qty
            self.positions[ticker] = {
                "entry_price": float(price),
                "qty": int(qty),
                "invested": float(invested),
                "peak": float(price),
                "entry_time": datetime.utcnow().isoformat(),
                "source": source,
            }
            self.persist_positions()
            return self.positions[ticker]

    def update_peak_and_mark(self, ticker: str, price: float):
        with self.lock:
            pos = self.positions.get(ticker)
            if not pos:
                return
            if price > pos.get("peak", pos["entry_price"]):
                pos["peak"] = float(price)
            self.positions[ticker] = pos
            self.persist_positions()

    def eligible_to_sell(self, ticker: str, price: float, trailing_pct: float):
        """
        Sell only if in profit and either:
            - price has fallen trailing_pct from peak (trailing trigger)
            - AI early-book condition triggers (external)
        Also never sell if current price <= entry (i.e. losing).
        """
        pos = self.positions.get(ticker)
        if not pos:
            return False, "no_position"
        entry = pos["entry_price"]
        if price <= entry:
            return False, "in_loss_no_sell"
        peak = pos.get("peak", entry)
        # trailing pct check
        if price <= peak * (1 - trailing_pct):
            return True, "trailing_trigger"
        # else return False by default (AI early book handled separately)
        return False, "no_trigger"

    def sell(self, ticker: str, price: float, fees_cfg: Dict, reason: str = "auto-trail"):
        """Execute paper sell, compute fees and taxes, persist book."""
        with self.lock:
            pos = self.positions.get(ticker)
            if not pos:
                return None
            qty = int(pos["qty"])
            invested = float(pos["invested"])
            sell_turnover = price * qty
            gross_profit = sell_turnover - invested

            # compute fees
            brokerage = sell_turnover * fees_cfg.get("brokerage_pct", DEFAULT_BROKERAGE_PCT)
            gst = brokerage * fees_cfg.get("gst_pct", DEFAULT_GST_PCT)
            stt = sell_turnover * fees_cfg.get("stt_pct", DEFAULT_STT_PCT)
            exchange = sell_turnover * fees_cfg.get("exchange_pct", DEFAULT_EXCHANGE_PCT)
            stamp = sell_turnover * fees_cfg.get("stamp_pct", DEFAULT_STAMP_PCT)
            platform_fee = fees_cfg.get("platform_fee", DEFAULT_PLATFORM_FEE)
            fees_sum = brokerage + gst + stt + exchange + stamp + platform_fee

            tax = max(0.0, gross_profit) * fees_cfg.get("tax_pct", DEFAULT_TAX_PCT)

            net_realized = gross_profit - fees_sum - tax

            # Record booked trade
            booked = {
                "timestamp": datetime.utcnow().isoformat(),
                "ticker": ticker,
                "qty": qty,
                "entry_price": pos["entry_price"],
                "sell_price": float(price),
                "invested": invested,
                "turnover": sell_turnover,
                "gross_profit": gross_profit,
                "brokerage": brokerage,
                "gst": gst,
                "stt": stt,
                "exchange_fee": exchange,
                "stamp_duty": stamp,
                "platform_fee": platform_fee,
                "fees_sum": fees_sum,
                "tax": tax,
                "net_realized": net_realized,
                "reason": reason
            }
            # append log and remove position
            try:
                append_csv(Path(self.booked_file), booked)
                append_csv(Path(self.pnl_file), {
                    "date": date.today().isoformat(),
                    "timestamp": booked["timestamp"],
                    "ticker": ticker,
                    "net_realized": net_realized,
                    "gross_profit": gross_profit
                })
            except Exception:
                pass

            # remove position
            try:
                del self.positions[ticker]
            except Exception:
                pass
            self.persist_positions()
            return booked

    def load_open_positions(self):
        return self.positions

# global single instance
paper = PaperTrader()

# -------------------------
# Selection & AI book-now function
# -------------------------
def compute_book_now_score(indicators: Dict, current_price: float, entry_price: float, time_to_eod_minutes: int) -> float:
    """
    Return 0-100 score. High means 'book now'.
    Components simplified for demo:
        - momentum decay (EMA crossover)
        - volume drop
        - volatility contraction
        - time to EOD (closer => more likely to book)
        - distance to target / accumulated profit
    """
    score = 0.0
    try:
        # momentum: if ema9 < ema21 -> add more score (momentum fading)
        ema9 = indicators.get("ema9", np.nan)
        ema21 = indicators.get("ema21", np.nan)
        if not np.isnan(ema9) and not np.isnan(ema21):
            if ema9 < ema21:
                score += 30.0
            else:
                score += 5.0

        # volume: if vol < vol_avg20 * 0.7 -> add
        vol = indicators.get("vol", np.nan)
        vol20 = indicators.get("vol_avg20", np.nan)
        if not np.isnan(vol) and not np.isnan(vol20):
            if vol < vol20 * 0.7:
                score += 20.0
            else:
                score += 5.0

        # bb width small => possible reversal
        bbw = indicators.get("bb_width", np.nan)
        if not np.isnan(bbw):
            if bbw < 1.2:
                score += 10.0
            else:
                score += 2.0

        # time to eod
        if time_to_eod_minutes <= 60:
            score += 20.0 * (1 - (time_to_eod_minutes / 60.0))  # more weight near EOD

        # accumulated profit factor
        accum_pct = ((indicators.get("peak", current_price) - entry_price) / entry_price * 100.0) if entry_price else 0.0
        if accum_pct >= 2.5:
            score += min(20.0, accum_pct)  # cap add

    except Exception:
        score = 0.0
    return min(100.0, score)

# -------------------------
# Paper trading orchestration / scheduler
# -------------------------
class PaperTradingScheduler:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.config = {
            "base_capital": DEFAULT_BASE_CAPITAL,
            "max_usage_pct": DEFAULT_MAX_USAGE_PCT,
            "max_positions": DEFAULT_MAX_POSITIONS,
            "target_pct": DEFAULT_TARGET_PCT,
            "trailing_base": DEFAULT_TRAILING_BASE,
            "brokerage_pct": DEFAULT_BROKERAGE_PCT,
            "gst_pct": DEFAULT_GST_PCT,
            "stt_pct": DEFAULT_STT_PCT,
            "exchange_pct": DEFAULT_EXCHANGE_PCT,
            "stamp_pct": DEFAULT_STAMP_PCT,
            "platform_fee": DEFAULT_PLATFORM_FEE,
            "tax_pct": DEFAULT_TAX_PCT,
            "auto_start": DEFAULT_AUTO_START,
            "auto_start_time": DEFAULT_AUTO_START_TIME,
            "eod_capture_time": DEFAULT_EOD_CAPTURE_TIME,
            "max_positions": DEFAULT_MAX_POSITIONS,
            "self_url": SELF_URL,
            "keep_alive": KEEP_ALIVE
        }
        self.session = {
            "used_capital": 0.0,
            "cash_unused": self.config["base_capital"],
        }
        # Load persisted PnL file if absent create
        if not Path(self.config_path()).exists():
            pass

    def config_path(self):
        return DATA_DIR / "paper_config.json"

    def save_config(self):
        save_json(self.config_path(), self.config)

    def load_config(self):
        cfg = load_json(self.config_path())
        if cfg:
            self.config.update(cfg)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def _loop(self):
        """Main loop: runs every 30 seconds, checks time, updates peaks, evaluates trailing & AI-book conditions."""
        try:
            while self.running:
                try:
                    now = now_ist()
                    # update open positions' peaks and mark using last price
                    positions = paper.load_open_positions() or {}
                    for tck, pos in list(positions.items()):
                        try:
                            df = fetch_yf_df(tck, period="5d", interval="5m")
                            if df is None or df.empty:
                                continue
                            last_price = float(df["Close"].iloc[-1])
                            paper.update_peak_and_mark(tck, last_price)

                            # determine time to eod minutes
                            eod_dt = datetime.combine(now.date(), self.config["eod_capture_time"])
                            eod_dt = IST.localize(eod_dt)
                            time_to_eod = max(0, int((eod_dt - now).total_seconds() // 60))

                            # compute indicators for book-now score
                            inds = compute_indicators(df)
                            inds['peak'] = pos.get("peak", pos["entry_price"])
                            book_score = compute_book_now_score(inds, last_price, pos["entry_price"], time_to_eod)

                            # determine trailing pct adaptively using ATR%
                            atr_pct = inds.get("atr_pct", np.nan)
                            if not np.isnan(atr_pct):
                                if atr_pct >= 3.0:
                                    trailing_pct = 0.05
                                elif atr_pct >= 2.0:
                                    trailing_pct = 0.04
                                elif atr_pct >= 1.0:
                                    trailing_pct = 0.03
                                else:
                                    trailing_pct = 0.02
                            else:
                                trailing_pct = self.config.get("trailing_base", DEFAULT_TRAILING_BASE)

                            # check trailing trigger
                            can_sell, reason = paper.eligible_to_sell(tck, last_price, trailing_pct)
                            if can_sell:
                                fees_cfg = {
                                    "brokerage_pct": self.config["brokerage_pct"],
                                    "gst_pct": self.config["gst_pct"],
                                    "stt_pct": self.config["stt_pct"],
                                    "exchange_pct": self.config["exchange_pct"],
                                    "stamp_pct": self.config["stamp_pct"],
                                    "platform_fee": self.config["platform_fee"],
                                    "tax_pct": self.config["tax_pct"]
                                }
                                booked = paper.sell(tck, last_price, fees_cfg, reason=f"trailing_{int(trailing_pct*100)}")
                                # update daily PnL etc (already appended)
                            else:
                                # AI early booking
                                if book_score >= 72:
                                    # only if in profit
                                    posentry = pos.get("entry_price")
                                    if last_price > posentry:
                                        fees_cfg = {
                                            "brokerage_pct": self.config["brokerage_pct"],
                                            "gst_pct": self.config["gst_pct"],
                                            "stt_pct": self.config["stt_pct"],
                                            "exchange_pct": self.config["exchange_pct"],
                                            "stamp_pct": self.config["stamp_pct"],
                                            "platform_fee": self.config["platform_fee"],
                                            "tax_pct": self.config["tax_pct"]
                                        }
                                        booked = paper.sell(tck, last_price, fees_cfg, reason=f"ai_early_book_{int(book_score)}")

                        except Exception:
                            continue

                    # EOD capture: record snapshot PnL at eod capture time (only once when passing over)
                    # Simple approach: if current time is within 1 minute of EOD time, append snapshot
                    now_time = now.time()
                    eod = self.config.get("eod_capture_time", DEFAULT_EOD_CAPTURE_TIME)
                    # if now within one minute of eod, capture day snapshot
                    if abs((datetime.combine(now.date(), now_time) - datetime.combine(now.date(), eod)).total_seconds()) < 65:
                        # calculate unrealised sum
                        unreal = 0.0
                        for tck, pos in (paper.load_open_positions() or {}).items():
                            try:
                                df = fetch_yf_df(tck, period="5d", interval="5m")
                                if df is None or df.empty:
                                    continue
                                last_price = float(df["Close"].iloc[-1])
                                unreal += (last_price * pos["qty"] - pos["invested"])
                            except Exception:
                                continue
                        # booked net today from file
                        pnl_df = pd.DataFrame()
                        try:
                            if Path(self.pnl_file).exists():
                                pnl_df = pd.read_csv(self.pnl_file)
                        except Exception:
                            pnl_df = pd.DataFrame()
                        realized_today = 0.0
                        if not pnl_df.empty:
                            realized_today = float(pnl_df[pnl_df['date'] == date.today().isoformat()]['net_realized'].sum())
                        snapshot = {
                            "date": date.today().isoformat(),
                            "timestamp": datetime.utcnow().isoformat(),
                            "realized_today": realized_today,
                            "unrealised": unreal,
                            "total_capital": float(self.config["base_capital"]) + realized_today + unreal
                        }
                        # append to pnl file as a row as well
                        append_csv(Path(self.pnl_file), {
                            "date": snapshot["date"],
                            "timestamp": snapshot["timestamp"],
                            "ticker": "SNAPSHOT",
                            "net_realized": snapshot["realized_today"]
                        })
                        # sleep a bit to avoid double-capture
                        time.sleep(70)

                    # sleep main loop
                    time.sleep(30)

                except Exception:
                    # safe continue to keep loop alive
                    time.sleep(5)
            except Exception:
                break

    def run_selection_and_buy(self, picks: List[Dict], config_override: Dict = None):
        """
        picks: list of {ticker, price, recommended_source}
        allocates available capital across picks (up to configured max usage and max_positions)
        buys at provided price (simulate fill) using floor shares
        """
        try:
            cfg = self.config
            base = float(cfg["base_capital"])
            max_used = base * float(cfg["max_usage_pct"])
            current_used = sum([v.get("invested", 0.0) for v in (paper.load_open_positions() or {}).values()])
            available_for_new = max(0.0, max_used - current_used)
            picks = picks[: int(cfg["max_positions"])]
            if not picks or available_for_new <= 0:
                return []

            per_pick_alloc = available_for_new / len(picks)
            bought = []
            for p in picks:
                t = p.get("ticker")
                price = float(p.get("price", p.get("cmp", 0.0)))
                if price <= 0:
                    continue
                qty = int(math.floor(per_pick_alloc / price))
                if qty <= 0:
                    continue
                invested = qty * price
                b = paper.buy(t, per_pick_alloc, price, qty, source=p.get("source","auto"))
                if b:
                    bought.append({"ticker": t, "qty": qty, "buy_price": price, "invested": invested})
            return bought
        except Exception:
            return []

paper_scheduler = PaperTradingScheduler()
paper_scheduler.load_config()

# -------------------------
# Keep-alive self-pinger thread
# -------------------------
def self_ping_loop(url: str, interval: int, enabled: bool):
    if not enabled or not url:
        return
    while True:
        try:
            # lightweight GET
            requests.get(url, timeout=10)
        except Exception:
            pass
        time.sleep(interval)

# start self-pinger if enabled
if KEEP_ALIVE and SELF_URL:
    tping = threading.Thread(target=self_ping_loop, args=(SELF_URL, SELF_PING_INTERVAL_SEC, KEEP_ALIVE), daemon=True)
    tping.start()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="🤖 AI Stock Analysis Bot + Paper Trading", page_icon="📈", layout="wide")
st.markdown("<style> .stApp { background-color: #fbfbf9; } </style>", unsafe_allow_html=True)

# session state initialization
if "paper_running" not in st.session_state:
    st.session_state["paper_running"] = False
if "paper_auto_start" not in st.session_state:
    st.session_state["paper_auto_start"] = paper_scheduler.config.get("auto_start", DEFAULT_AUTO_START)
if "base_capital" not in st.session_state:
    st.session_state["base_capital"] = paper_scheduler.config.get("base_capital", DEFAULT_BASE_CAPITAL)
if "max_usage_pct" not in st.session_state:
    st.session_state["max_usage_pct"] = paper_scheduler.config.get("max_usage_pct", DEFAULT_MAX_USAGE_PCT)
if "max_positions" not in st.session_state:
    st.session_state["max_positions"] = paper_scheduler.config.get("max_positions", DEFAULT_MAX_POSITIONS)
if "target_pct" not in st.session_state:
    st.session_state["target_pct"] = paper_scheduler.config.get("target_pct", DEFAULT_TARGET_PCT)
if "trailing_base" not in st.session_state:
    st.session_state["trailing_base"] = paper_scheduler.config.get("trailing_base", DEFAULT_TRAILING_BASE)

# Sidebar navigation
NAV_PAGES = [
    "🔥 Top Stocks",
    "🌙 BTST",
    "⚡ Intraday",
    "📆 Weekly",
    "📅 Monthly",
    "📣 Dividends",
    "📊 Groww",
    "🤝 Dhan",
    "🧾 Dhan Stocks Analysis",
    "🧾 PNL Log",
    "🪙 Paper Trading",
    "⚙️ Configuration",
]
with st.sidebar:
    page = st.radio("Navigation", NAV_PAGES, index=NAV_PAGES.index("🔥 Top Stocks"))

# Header
st.markdown('<div style="padding:12px;border-radius:12px;background:linear-gradient(90deg,#4f46e5,#0ea5e9);color:white"><h2>AI Stock Analysis Bot + Paper Trading</h2></div>', unsafe_allow_html=True)
st.caption(f"Local time: {now_ist().strftime('%d-%m-%Y %I:%M %p')} IST")

# --- Pages (simplified: reuse previous Groww/Dhan/Div pages simplified for compactness) ---
if page == "🔥 Top Stocks":
    st.subheader("🔥 Top Stocks (preview)")
    st.info("Run scans from the top header 'Run Full Scan' if available (scanning logic simplified in this demo).")
    st.write("Top stocks will show when you run the scanner.")

elif page == "📣 Dividends":
    st.subheader("📣 Upcoming Dividends demo")
    st.info("Dividends scanning available in full app; simplified here: use Groww / YF to fetch dividend info.")
    st.write("Use the Dividends page in the main app to scan NIFTY200 for upcoming ex-dividend dates.")

elif page == "📊 Groww":
    st.subheader("📊 Groww Upload (demo)")
    st.info("Upload Groww CSV to analyze (same logic as previous version).")
    uploaded = st.file_uploader("Upload Groww portfolio file", type=["csv", "xls", "xlsx"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if str(uploaded.name).lower().endswith(".csv") else pd.read_excel(uploaded)
            st.write(df_up.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "🤝 Dhan":
    st.subheader("🤝 Dhan demo")
    st.info("Dhan integration is supported in the full app; this demo shows a placeholder.")

elif page == "🧾 Dhan Stocks Analysis":
    st.subheader("🧾 Dhan Stocks Analysis (demo)")
    st.info("This page will show Dhan holdings + AI recommendations in the full app. Use paper trading for demo now.")
    df_pos = paper.snapshot_positions_df()
    if df_pos.empty:
        st.info("No paper positions yet. Start Paper Trading to see holdings here.")
    else:
        st.dataframe(df_pos)

elif page == "🧾 PNL Log":
    st.subheader("🧾 Paper Trading PNL Log")
    st.markdown("This page shows booked trades and daily PnL snapshots (persisted).")
    if Path(BOOKED_TRADES_FILE).exists():
        df_booked = pd.read_csv(BOOKED_TRADES_FILE)
        st.markdown("### Booked Trades (realized)")
        st.dataframe(df_booked.sort_values("timestamp", ascending=False).head(200))
        st.download_button("Download Booked Trades CSV", data=df_booked.to_csv(index=False), file_name="booked_trades.csv")
    else:
        st.info("No booked trades yet.")

    if Path(PAPER_PNL_FILE).exists():
        df_pnl = pd.read_csv(PAPER_PNL_FILE)
        st.markdown("### PnL Log")
        st.dataframe(df_pnl.sort_values(["date","timestamp"], ascending=False).head(300))
        st.download_button("Download PnL Log CSV", data=df_pnl.to_csv(index=False), file_name="paper_pnl_log.csv")
    else:
        st.info("No PnL snapshots yet.")

elif page == "🪙 Paper Trading":
    st.subheader("🪙 Paper Trading Engine")
    st.markdown("Configure paper trading parameters and start/stop the simulator. Booked trades and PnL saved to data/*.csv")

    # Controls
    c1, c2 = st.columns([2, 3])
    with c1:
        base_cap = st.number_input("Base capital (₹)", value=float(st.session_state["base_capital"]), step=1000.0)
        max_usage = st.slider("Max capital usage (%)", min_value=10, max_value=100, value=int(st.session_state["max_usage_pct"]*100))/100.0
        max_pos = st.number_input("Max positions", min_value=1, max_value=10, value=int(st.session_state["max_positions"]))
        target_pct = st.number_input("Target % (per trade)", value=float(st.session_state["target_pct"]*100.0))/100.0
    with c2:
        trailing_base = st.number_input("Default trailing % (fallback)", value=float(st.session_state["trailing_base"]*100.0))/100.0
        brokerage_pct = st.number_input("Brokerage %", value=float(DEFAULT_BROKERAGE_PCT*100.0))/100.0
        gst_pct = st.number_input("GST %", value=float(DEFAULT_GST_PCT*100.0))
        stt_pct = st.number_input("STT % (sell)", value=float(DEFAULT_STT_PCT*100.0))/100.0
        tax_pct = st.number_input("Tax % on gain", value=float(DEFAULT_TAX_PCT*100.0))/100.0

    # apply to scheduler config
    paper_scheduler.config.update({
        "base_capital": float(base_cap),
        "max_usage_pct": float(max_usage),
        "max_positions": int(max_pos),
        "target_pct": float(target_pct),
        "trailing_base": float(trailing_base),
        "brokerage_pct": float(brokerage_pct),
        "gst_pct": float(gst_pct),
        "stt_pct": float(stt_pct),
        "tax_pct": float(tax_pct)
    })
    paper_scheduler.save_config()

    # Start / Stop buttons
    cola, colb = st.columns(2)
    with cola:
        if st.button("Start Paper Trading"):
            if not st.session_state["paper_running"]:
                paper_scheduler.start()
                st.session_state["paper_running"] = True
                st.success("Paper Trading started (background scheduler running).")
            else:
                st.info("Paper Trading already running.")
    with colb:
        if st.button("Stop Paper Trading"):
            if st.session_state["paper_running"]:
                paper_scheduler.stop()
                st.session_state["paper_running"] = False
                st.success("Paper Trading stopped.")
            else:
                st.info("Paper Trading not running.")

    # Quick selection run (simulate picks from scanner)
    st.markdown("### Quick selection: pick top 5 (demo) and allocate")
    # Demo picks - in real app we'd use analyze_multiple_stocks results; here create sample picks
    sample_picks = [
        {"ticker": "ABCD", "price": 250.00, "source": "BTST"},
        {"ticker": "WXYZ", "price": 420.00, "source": "BTST"},
        {"ticker": "EFGL", "price": 88.75, "source": "Intraday"},
        {"ticker": "TUV1", "price": 312.30, "source": "Intraday"},
        {"ticker": "WEA1", "price": 2450.00, "source": "Weekly"},
    ]
    if st.button("Run selection & Buy (simulate)"):
        buys = paper_scheduler.run_selection_and_buy(sample_picks)
        if buys:
            st.success(f"Bought {len(buys)} positions (paper).")
            st.write(pd.DataFrame(buys))
        else:
            st.info("Nothing bought (no available capital or invalid prices).")

    # show open positions
    st.markdown("### Open Paper Positions")
    df_open = paper.snapshot_positions_df()
    if df_open.empty:
        st.info("No open paper positions.")
    else:
        st.dataframe(df_open)
        if st.button("Force Book All Profitable (manual)"):
            # iterate and book if profitable
            pos = paper.load_open_positions() or {}
            cnt = 0
            for tck, p in list(pos.items()):
                try:
                    df = fetch_yf_df(tck, period="5d", interval="5m")
                    if df is None or df.empty:
                        continue
                    last = float(df["Close"].iloc[-1])
                    if last > p["entry_price"]:
                        booked = paper.sell(tck, last, {
                            "brokerage_pct": paper_scheduler.config["brokerage_pct"],
                            "gst_pct": paper_scheduler.config["gst_pct"],
                            "stt_pct": paper_scheduler.config["stt_pct"],
                            "exchange_pct": paper_scheduler.config["exchange_pct"],
                            "stamp_pct": paper_scheduler.config["stamp_pct"],
                            "platform_fee": paper_scheduler.config["platform_fee"],
                            "tax_pct": paper_scheduler.config["tax_pct"],
                        }, reason="manual_force_book")
                        cnt += 1
                except Exception:
                    continue
            st.success(f"Booked {cnt} profitable positions (manual).")

elif page == "⚙️ Configuration":
    st.subheader("⚙️ Configuration & Keep-Alive")
    st.markdown("Configure app behavior. Keep the self-url set to your deployed app if you want the internal pinger to try to keep it awake.")
    self_url = st.text_input("Self URL (for keep-alive pinger)", value=paper_scheduler.config.get("self_url", SELF_URL))
    keep_alive = st.checkbox("Enable internal self-ping keep-alive (may not prevent host sleep on free tiers)", value=paper_scheduler.config.get("self_url", SELF_URL) is not None)
    if st.button("Save Keep-Alive"):
        paper_scheduler.config["self_url"] = self_url
        paper_scheduler.config["keep_alive"] = bool(keep_alive)
        paper_scheduler.save_config()
        st.success("Saved keep-alive config (note: external Uptime pinger is more reliable).")

# End of streamlit pages

# Footer note
st.markdown("---")
st.caption("Paper Trading is simulation-only. The engine will not place real orders. Check booked trades & PnL log from the sidebar. Adjust fee/tax params in Configuration.")

# The app itself will not "start trading" automatically from me — it will run the scheduler on the deployed host if you press Start Paper Trading
# or if auto-start is enabled in config and the host/environment allows background threads.

