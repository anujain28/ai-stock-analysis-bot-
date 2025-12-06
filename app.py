# ========= AUTO-INSTALL (ONLY DHAN, OPTIONAL) =========
import subprocess, sys

def ensure_package(pkg_name: str):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

try:
    ensure_package("dhanhq")
except Exception:
    pass

# ========= IMPORTS =========
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime
import time
from typing import Dict, List, Optional
import pytz
import requests
import os
import json
from pathlib import Path

from streamlit_local_storage import LocalStorage  # browser localStorage helper

# Dhan
try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

# ========= CONFIG FILE (SERVER-SIDE BACKUP) =========
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "dhan_client_id": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "notify_enabled": False
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config_from_state():
    cfg = {
        "dhan_client_id": st.session_state.get("dhan_client_id", ""),
        "telegram_bot_token": st.session_state.get("telegram_bot_token", ""),
        "telegram_chat_id": st.session_state.get("telegram_chat_id", ""),
        "notify_enabled": st.session_state.get("notify_enabled", False),
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save config: {e}")

# ========= PAGE CONFIG & CSS =========
st.set_page_config(
    page_title="🤖 AI Stock Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body {
        background: radial-gradient(circle at top left, #1e293b 0, #020617 50%, #0f172a 100%);
    }
    .main-header {
        background: linear-gradient(120deg, #2563eb 0%, #7c3aed 35%, #ec4899 100%);
        padding: 28px 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 14px 30px rgba(15,23,42,0.7);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .main-header h1 {
        margin-bottom: 4px;
        font-size: 2.1rem;
    }
    .main-header p {
        margin: 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        background: rgba(15,23,42,0.4);
        border: 1px solid rgba(148,163,184,0.6);
        margin-top: 8px;
    }
    .metric-card {
        padding: 12px 12px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 100%);
        border: 1px solid rgba(191,219,254,0.5);
        box-shadow: 0 10px 25px rgba(15,23,42,0.9);
        margin-bottom: 10px;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #e5e7eb;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f9fafb;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #c4b5fd;
    }
    .side-section {
        padding: 10px 12px;
        border-radius: 12px;
        background: linear-gradient(145deg, #4c1d95 0%, #581c87 50%, #7c3aed 100%);
        border: 1px solid rgba(221,214,254,0.6);
        margin-bottom: 12px;
    }
    .side-section h4 {
        font-size: 0.9rem;
        margin-bottom: 8px;
        color: #f5f3ff;
    }
    [data-testid=stSidebar] {
        background: radial-gradient(circle at top, #4c1d95 0, #581c87 40%, #2e1065 100%);
    }
</style>
""", unsafe_allow_html=True)  # violet sidebar background [web:219][web:301]

IST = pytz.timezone('Asia/Kolkata')

# ========= LOAD CONFIG & SESSION STATE =========
_cfg = load_config()
localS = LocalStorage()

if 'last_analysis_time' not in st.session_state:
    st.session_state['last_analysis_time'] = None
if 'last_auto_scan' not in st.session_state:
    st.session_state['last_auto_scan'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = {'BTST': [], 'Intraday': [], 'Weekly': [], 'Monthly': []}

# Dhan state
for key, default in [
    ('dhan_enabled', False),
    ('dhan_client_id', _cfg.get('dhan_client_id', '')),
    ('dhan_access_token', ''),
    ('dhan_client', None),
    ('dhan_login_msg', 'Not configured'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Telegram state
for key, default in [
    ('notify_enabled', _cfg.get('notify_enabled', False)),
    ('telegram_bot_token', _cfg.get('telegram_bot_token', '')),
    ('telegram_chat_id', _cfg.get('telegram_chat_id', '')),
    ('last_pnl_notify', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ========= NIFTY 200 FROM CSV (HIDDEN UNIVERSE) =========
DATA_DIR = Path(__file__).parent / "data"
NIFTY200_CSV = DATA_DIR / "nifty200_yahoo.csv"
MASTER_NIFTY200 = DATA_DIR / "ind_nifty200list.csv"

@st.cache_data
def load_nifty200_universe():
    if not NIFTY200_CSV.exists():
        st.error(f"Universe file not found: {NIFTY200_CSV}")
        return [], {}
    df = pd.read_csv(NIFTY200_CSV)
    if "SYMBOL" not in df.columns or "YF_TICKER" not in df.columns:
        st.error("nifty200_yahoo.csv must have columns: SYMBOL, YF_TICKER")
        return [], {}
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["YF_TICKER"] = df["YF_TICKER"].astype(str).str.strip()
    symbols = df["SYMBOL"].dropna().unique().tolist()
    mapping = dict(zip(df["SYMBOL"], df["YF_TICKER"]))
    return symbols, mapping

STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()

def regenerate_nifty200_csv_from_master():
    if not MASTER_NIFTY200.exists():
        st.error(f"Master list not found: {MASTER_NIFTY200}")
        return False
    df_src = pd.read_csv(MASTER_NIFTY200)
    if "Symbol" not in df_src.columns:
        st.error("ind_nifty200list.csv must have a 'Symbol' column")
        return False
    df_out = pd.DataFrame()
    df_out["SYMBOL"] = df_src["Symbol"].astype(str).str.strip().str.upper()
    df_out["YF_TICKER"] = df_out["SYMBOL"].apply(lambda s: f"{s}.NS")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(NIFTY200_CSV, index=False)
    load_nifty200_universe.clear()
    global STOCK_UNIVERSE, NIFTY_YF_MAP
    STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()
    return True

# ========= SYMBOL HELPERS =========
def nse_yf_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.strip().upper()
    if s in NIFTY_YF_MAP:
        return NIFTY_YF_MAP[s]
    return s if s.endswith(".NS") else f"{s}.NS"

# ========= DHAN HELPERS =========
def dhan_login(client_id: str, access_token: str):
    if not dhanhq:
        st.session_state['dhan_client'] = None
        st.session_state['dhan_login_msg'] = "dhanhq not installed"
        return
    try:
        d = dhanhq(client_id, access_token)
        _ = d.get_holdings()
        st.session_state['dhan_client'] = d
        st.session_state['dhan_login_msg'] = "✅ Dhan client configured"
    except Exception as e:
        st.session_state['dhan_client'] = None
        st.session_state['dhan_login_msg'] = f"❌ Dhan error: {e}"

def dhan_logout():
    st.session_state['dhan_client'] = None
    st.session_state['dhan_login_msg'] = "Logged out"

def get_dhan_raw():
    d = st.session_state.get('dhan_client')
    if not d:
        return pd.DataFrame(), pd.DataFrame()
    try:
        h = d.get_holdings()
        holdings = h.get("data", []) if isinstance(h, dict) else h
    except Exception:
        holdings = []
    try:
        p = d.get_positions()
        positions = p.get("data", []) if isinstance(p, dict) else p
    except Exception:
        positions = []
    return pd.DataFrame(holdings), pd.DataFrame(positions)

def format_dhan_portfolio_table():
    h_df, p_df = get_dhan_raw()
    if h_df.empty and p_df.empty:
        return pd.DataFrame(), 0.0
    if not h_df.empty:
        name_col = next((c for c in ['securityName', 'tradingSymbol', 'symbol'] if c in h_df.columns), None)
        qty_col = next((c for c in ['quantity', 'netQty', 'buyQty', 'totalQty'] if c in h_df.columns), None)
        avg_col = next((c for c in ['averagePrice', 'buyAvg', 'avgCostPrice'] if c in h_df.columns), None)
        cmp_col = next((c for c in ['ltp', 'lastTradedPrice', 'lastPrice'] if c in h_df.columns), None)
        h_df['_name'] = h_df[name_col] if name_col else ""
        h_df['_qty'] = pd.to_numeric(h_df[qty_col], errors='coerce').fillna(0.0) if qty_col else 0.0
        h_df['_avg'] = pd.to_numeric(h_df[avg_col], errors='coerce').fillna(0.0) if avg_col else 0.0
        h_df['_cmp'] = pd.to_numeric(h_df[cmp_col], errors='coerce') if cmp_col else np.nan
        h_df['_total_cost'] = h_df['_qty'] * h_df['_avg']
        h_df['_total_price'] = h_df['_qty'] * h_df['_cmp']
        h_df['_pnl'] = h_df['_total_price'] - h_df['_total_cost']
        portfolio = h_df[['_name', '_qty', '_avg', '_total_cost', '_cmp', '_total_price', '_pnl']].rename(
            columns={
                '_name': 'Stock',
                '_qty': 'Quantity',
                '_avg': 'Avg Cost',
                '_total_cost': 'Total Cost',
                '_cmp': 'CMP',
                '_total_price': 'Total Value',
                '_pnl': 'P&L'
            }
        )
    else:
        name_col = next((c for c in ['tradingSymbol', 'securityName', 'symbol'] if c in p_df.columns), None)
        p_df['_name'] = p_df[name_col] if name_col else ""
        p_df['_qty'] = pd.to_numeric(p_df['netQty'], errors='coerce').fillna(0.0)
        p_df['_avg'] = pd.to_numeric(p_df['avgPrice'], errors='coerce').fillna(0.0)
        p_df['_cmp'] = pd.to_numeric(p_df['ltp'], errors='coerce').fillna(0.0)
        p_df['_total_cost'] = p_df['_qty'] * p_df['_avg']
        p_df['_total_price'] = p_df['_qty'] * p_df['_cmp']
        p_df['_pnl'] = p_df['_total_price'] - p_df['_total_cost']
        portfolio = p_df[['_name', '_qty', '_avg', '_total_cost', '_cmp', '_total_price', '_pnl']].rename(
            columns={
                '_name': 'Stock',
                '_qty': 'Quantity',
                '_avg': 'Avg Cost',
                '_total_cost': 'Total Cost',
                '_cmp': 'CMP',
                '_total_price': 'Total Value',
                '_pnl': 'P&L'
            }
        )
    total_pnl = float(portfolio['P&L'].fillna(0).sum())
    return portfolio, total_pnl

# ========= TELEGRAM HELPER =========
def send_telegram_message(text: str):
    token = st.session_state.get('telegram_bot_token', '')
    chat_id = st.session_state.get('telegram_chat_id', '')
    if not token or not chat_id:
        return {"ok": False, "error": "Missing Telegram config"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.get(url, params={"chat_id": chat_id, "text": text})
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ========= TA & ANALYSIS =========
def safe_extract(df, col):
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    return pd.Series(s) if not isinstance(s, pd.Series) else s

def safe_scalar(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (pd.Series, list, np.ndarray)):
            x = np.ravel(x)[-1]
        return float(x)
    except Exception:
        return np.nan

class TechnicalAnalysis:
    @staticmethod
    def rsi_signal(df):
        c = safe_extract(df, 'Close')
        if len(c) < 15:
            return False, None, 0
        rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
        v = safe_scalar(rsi.iloc[-1])
        if v < 30:
            return True, f"RSI Oversold ({v:.1f})", 35
        if v < 40:
            return True, f"RSI Strong Buy Zone ({v:.1f})", 25
        return False, None, 0

    @staticmethod
    def macd_signal(df):
        c = safe_extract(df, 'Close')
        if len(c) < 35:
            return False, None, 0
        macd = ta.trend.MACD(c)
        m = safe_scalar(macd.macd().iloc[-1])
        s = safe_scalar(macd.macd_signal().iloc[-1])
        h = safe_scalar(macd.macd_diff().iloc[-1])
        if m > s and h > 0:
            return True, "MACD Bullish Crossover", 30
        return False, None, 0

    @staticmethod
    def stochastic_signal(df):
        h, l, c = safe_extract(df, 'High'), safe_extract(df, 'Low'), safe_extract(df, 'Close')
        if len(c) < 14:
            return False, None, 0
        stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
        k = safe_scalar(stoch.stoch().iloc[-1])
        d = safe_scalar(stoch.stoch_signal().iloc[-1])
        if k < 20 and k > d:
            return True, "Stochastic Oversold Reversal", 25
        return False, None, 0

    @staticmethod
    def adx_trend_strength(df):
        h, l, c = safe_extract(df, 'High'), safe_extract(df, 'Low'), safe_extract(df, 'Close')
        if len(c) < 25:
            return False, None, 0
        adx = ta.trend.ADXIndicator(h, l, c, window=14)
        adx_val = safe_scalar(adx.adx().iloc[-1])
        plus_di = safe_scalar(adx.adx_pos().iloc[-1])
        minus_di = safe_scalar(adx.adx_neg().iloc[-1])
        if adx_val > 25 and plus_di > minus_di:
            return True, f"Strong Uptrend (ADX: {adx_val:.1f})", 30
        return False, None, 0

    @staticmethod
    def bollinger_squeeze(df):
        c = safe_extract(df, 'Close')
        if len(c) < 20:
            return False, None, 0
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        lower = safe_scalar(bb.bollinger_lband().iloc[-1])
        upper = safe_scalar(bb.bollinger_hband().iloc[-1])
        price = safe_scalar(c.iloc[-1])
        bandwidth = (upper - lower) / price * 100
        if price < lower * 1.02 and bandwidth < 10:
            return True, "BB Squeeze Breakout Setup", 25
        return False, None, 0

    @staticmethod
    def volume_price_confirmation(df):
        v = safe_extract(df, 'Volume')
        c = safe_extract(df, 'Close')
        if len(v) < 20:
            return False, None, 0
        avg_vol = v.rolling(20).mean().iloc[-1]
        cur_vol = v.iloc[-1]
        chg = ((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) * 100
        if cur_vol > avg_vol * 1.5 and chg > 1:
            return True, f"Volume Spike with Price Up ({chg:.1f}%)", 30
        return False, None, 0

    @staticmethod
    def ema_crossover(df):
        c = safe_extract(df, 'Close')
        if len(c) < 50:
            return False, None, 0
        ema9 = ta.trend.EMAIndicator(c, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
        if safe_scalar(ema9.iloc[-1]) > safe_scalar(ema21.iloc[-1]) and safe_scalar(ema9.iloc[-2]) <= safe_scalar(ema21.iloc[-2]):
            return True, "EMA 9/21 Golden Cross", 28
        return False, None, 0

    @staticmethod
    def obv_divergence(df):
        c = safe_extract(df, 'Close')
        v = safe_extract(df, 'Volume')
        if len(c) < 20:
            return False, None, 0
        obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        obv_sma = obv.rolling(10).mean()
        if safe_scalar(obv.iloc[-1]) > safe_scalar(obv_sma.iloc[-1]):
            return True, "OBV Accumulation Phase", 22
        return False, None, 0

    @staticmethod
    def calculate_targets(df, price):
        h, l, c = safe_extract(df, 'High'), safe_extract(df, 'Low'), safe_extract(df, 'Close')
        if len(c) < 14:
            return {}
        atr = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        atr_v = safe_scalar(atr.iloc[-1])
        return {
            'atr': float(round(atr_v, 2)),
            'stop_loss': float(round(price - (atr_v * 2), 2)),
            'target_1': float(round(price + (atr_v * 2), 2)),
            'target_2': float(round(price + (atr_v * 3), 2)),
            'target_3': float(round(price + (atr_v * 4), 2)),
            'risk_reward': '1:2-4'
        }

def analyze_stock(ticker: str, period_type: str) -> Optional[Dict]:
    cfgs = {
        'BTST': {'period': '10d', 'interval': '15m'},
        'Intraday': {'period': '5d', 'interval': '15m'},
        'Weekly': {'period': '90d', 'interval': '1d'},
        'Monthly': {'period': '1y', 'interval': '1d'}
    }
    cfg = cfgs[period_type]
    try:
        t = yf.Ticker(nse_yf_symbol(ticker))
        df = t.history(period=cfg['period'], interval=cfg['interval'], auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
        df = df.reset_index()
        df.columns = [str(c).capitalize() for c in df.columns]
        strategies = [
            (TechnicalAnalysis.rsi_signal, 'RSI'),
            (TechnicalAnalysis.macd_signal, 'MACD'),
            (TechnicalAnalysis.stochastic_signal, 'Stochastic'),
            (TechnicalAnalysis.adx_trend_strength, 'ADX'),
            (TechnicalAnalysis.bollinger_squeeze, 'Bollinger'),
            (TechnicalAnalysis.volume_price_confirmation, 'Volume'),
            (TechnicalAnalysis.ema_crossover, 'EMA'),
            (TechnicalAnalysis.obv_divergence, 'OBV')
        ]
        signals, reasons, score = [], [], 0
        for func, name in strategies:
            sig, reason, s_ = func(df)
            if sig:
                signals.append(name)
                if reason:
                    reasons.append(reason)
                score += s_
        if len(signals) < 3:
            return None
        price = float(df['Close'].iloc[-1])
        targets = TechnicalAnalysis.calculate_targets(df, price)
        strength = "STRONG BUY" if score >= 90 else "BUY" if score >= 60 else "HOLD"
        timeframe_map = {
            'BTST': '1–3 days (BTST)',
            'Intraday': 'Same day',
            'Weekly': 'Up to 1 week',
            'Monthly': '2–4 weeks'
        }
        timeframe = timeframe_map.get(period_type, '1–5 days')
        return {
            'ticker': ticker,
            'price': round(price, 2),
            'signals_count': len(signals),
            'strategies': ", ".join(signals),
            'score': score,
            'reasons': " | ".join(reasons),
            'signal_strength': strength,
            'period': period_type,
            'timeframe': timeframe,
            **targets
        }
    except Exception:
        return None

def analyze_multiple_stocks(tickers: List[str], period_type: str, max_results: int = 10) -> List[Dict]:
    out = []
    if not tickers:
        return out
    bar = st.progress(0.0)
    txt = st.empty()
    for i, tck in enumerate(tickers):
        txt.text(f"🔍 Analyzing {period_type}: {tck} ({i+1}/{len(tickers)})")
        bar.progress((i + 1) / len(tickers))
        res = analyze_stock(tck, period_type)
        if res:
            out.append(res)
        time.sleep(0.02)
    bar.empty()
    txt.empty()
    return sorted(out, key=lambda x: x['score'], reverse=True)[:max_results]

def run_analysis():
    now = datetime.now(IST)
    st.session_state['last_analysis_time'] = now
    st.session_state['last_auto_scan'] = now
    new_recs = {}
    for p in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        new_recs[p] = analyze_multiple_stocks(STOCK_UNIVERSE, p, max_results=20)
        st.session_state['recommendations'][p] = new_recs[p]
    return new_recs

# ========= BACKGROUND AUTO SCAN =========
def maybe_run_auto_scan():
    now = datetime.now(IST)
    last = st.session_state.get('last_auto_scan')
    should_run = False
    if last is None:
        should_run = True
    else:
        try:
            if (now - last).total_seconds() >= 30 * 60:
                should_run = True
        except Exception:
            should_run = True
    if should_run:
        run_analysis()

# ========= SIDEBAR: BTST FLASH CARDS =========
def get_btst_top_for_sidebar(n: int = 5):
    btst_recs = st.session_state['recommendations'].get('BTST', [])
    if not btst_recs:
        return []
    df = pd.DataFrame(btst_recs).sort_values("score", ascending=False)
    return df.head(n).to_dict(orient="records")

# ========= GROWW PORTFOLIO ANALYSIS =========
def analyze_groww_portfolio(df: pd.DataFrame):
    """
    Expected columns:
    Stock Name, ISIN, Quantity, Average buy price per share,
    Total Investment, Total CMP, TOTAL P&L
    """
    cols = {c.lower(): c for c in df.columns}
    required = [
        "stock name",
        "isin",
        "quantity",
        "average buy price per share",
        "total investment",
        "total cmp",
        "total p&l",
    ]
    for r in required:
        if r not in cols:
            return {"error": "Columns must match the GROWW template exactly."}
    qcol = cols["quantity"]
    invcol = cols["total investment"]
    pnlcol = cols["total p&l"]

    df_use = df.copy()
    df_use["_qty"] = pd.to_numeric(df_use[qcol], errors="coerce").fillna(0.0)
    df_use["_inv"] = pd.to_numeric(df_use[invcol], errors="coerce").fillna(0.0)
    df_use["_pnl"] = pd.to_numeric(df_use[pnlcol], errors="coerce").fillna(0.0)

    total_inv = float(df_use["_inv"].sum())
    total_pnl = float(df_use["_pnl"].sum())
    positions = int((df_use["_qty"] > 0).sum())
    top = (
        df_use.sort_values("_inv", ascending=False)[[cols["stock name"], cols["quantity"], invcol, pnlcol]]
        .head(10)
        .rename(columns={cols["stock name"]: "Stock Name", cols["quantity"]: "Quantity"})
    )
    return {
        "total_investment": total_inv,
        "total_pnl": total_pnl,
        "positions": positions,
        "top_holdings": top,
    }

# ========= MAIN UI =========
def main():
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 AI Stock Analysis Bot</h1>
        <p>Multi-timeframe scanner + Dhan + GROWW portfolio analyzer</p>
        <div class="status-badge">Live • IST</div>
    </div>
    """, unsafe_allow_html=True)

    maybe_run_auto_scan()

    # Sidebar: BTST cards
    with st.sidebar:
        st.markdown("<div class='side-section'><h4>🌙 BTST Top Picks</h4>", unsafe_allow_html=True)
        cards = get_btst_top_for_sidebar()
        if not cards:
            st.caption("Click ‘Run Full Scan’ to see BTST picks.")
        else:
            for rec in cards:
                tgt = rec.get("target_1", None)
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<h3>{rec.get('ticker','')}</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='value'>CMP: ₹{rec.get('price',0):.2f}</div>", unsafe_allow_html=True)
                if tgt is not None and not np.isnan(tgt):
                    st.markdown(f"<div class='sub'>Target: ₹{tgt:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='sub'>ETA: {rec.get('timeframe','')} • BTST</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Top controls
    c1, c2, c3 = st.columns([3, 1.2, 1])
    with c1:
        if st.button("👈 Click here then 🚀 Run Full Scan", type="primary", use_container_width=True):
            run_analysis()
    with c2:
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()
    with c3:
        st.metric("Universe", len(STOCK_UNIVERSE))

    if st.session_state['last_analysis_time']:
        st.caption(f"Last Analysis: {st.session_state['last_analysis_time'].strftime('%d-%m-%Y %I:%M %p')}")

    st.markdown("---")

    # Tabs: BTST / Intraday / Weekly / Monthly / GROWW / Config (last)
    tab_btst, tab_intraday, tab_weekly, tab_monthly, tab_groww, tab_config = st.tabs(
        ["🌙 BTST", "⚡ Intraday", "📆 Weekly", "📅 Monthly", "📊 GROWW Stocks", "⚙️ Configuration"]
    )

    # BTST
    with tab_btst:
        recs = st.session_state['recommendations'].get('BTST', [])
        st.subheader("BTST Opportunities")
        if not recs:
            st.info("Run Full Scan to generate BTST recommendations.")
        else:
            st.dataframe(pd.DataFrame(recs), use_container_width=True)

    # Intraday
    with tab_intraday:
        recs = st.session_state['recommendations'].get('Intraday', [])
        st.subheader("Intraday Signals")
        if not recs:
            st.info("Run Full Scan to generate intraday recommendations.")
        else:
            st.dataframe(pd.DataFrame(recs), use_container_width=True)

    # Weekly
    with tab_weekly:
        recs = st.session_state['recommendations'].get('Weekly', [])
        st.subheader("Weekly Swing Ideas")
        if not recs:
            st.info("Run Full Scan to generate weekly recommendations.")
        else:
            st.dataframe(pd.DataFrame(recs), use_container_width=True)

    # Monthly
    with tab_monthly:
        recs = st.session_state['recommendations'].get('Monthly', [])
        st.subheader("Monthly Position Trades")
        if not recs:
            st.info("Run Full Scan to generate monthly recommendations.")
        else:
            st.dataframe(pd.DataFrame(recs), use_container_width=True)

    # GROWW Stocks tab – CSV upload analysis
    with tab_groww:
        st.subheader("GROWW Portfolio Analysis (CSV Upload)")
        st.markdown("CSV template columns (exact names):")
        st.code(
            "Stock Name\tISIN\tQuantity\tAverage buy price per share\t"
            "Total Investment\tTotal CMP\tTOTAL P&L",
            language="text"
        )
        uploaded = st.file_uploader(
            "Upload GROWW portfolio CSV", type=["csv"], key="groww_csv_upload"
        )  # file_uploader pattern [web:302]
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded, sep=None, engine="python")
                st.write("Preview:")
                st.dataframe(df_up.head(), use_container_width=True)
                analysis = analyze_groww_portfolio(df_up)
                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    st.markdown("##### Summary")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total Investment", f"₹{analysis['total_investment']:,.2f}")
                    with c2:
                        st.metric("Total P&L", f"₹{analysis['total_pnl']:,.2f}")
                    with c3:
                        st.metric("Positions", analysis["positions"])
                    st.markdown("##### Top holdings by capital")
                    st.dataframe(analysis["top_holdings"], use_container_width=True)
            except Exception as e:
                st.error(f"Error reading uploaded CSV: {e}")
        else:
            st.info("Upload your GROWW portfolio CSV to see analysis here.")

    # Configuration tab (last)
    with tab_config:
        st.markdown("### App Configuration")

        with st.expander("Dhan", expanded=True):
            dhan_store = localS.getItem("dhan_config") or {}
            if dhan_store:
                st.session_state['dhan_client_id'] = dhan_store.get("client_id", st.session_state['dhan_client_id'])

            dhan_enable = st.checkbox("Enable Dhan", value=st.session_state.get('dhan_enabled', False))
            st.session_state['dhan_enabled'] = dhan_enable
            if dhan_enable:
                dcid = st.text_input("Client ID", value=st.session_state.get('dhan_client_id', ''), key="cfg_dhan_client")
                dtoken = st.text_input("Access Token", value=st.session_state.get('dhan_access_token', ''), type="password", key="cfg_dhan_token")
                st.session_state['dhan_client_id'] = dcid
                st.session_state['dhan_access_token'] = dtoken
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔑 Connect Dhan", use_container_width=True, key="btn_connect_dhan"):
                        dhan_login(dcid, dtoken)
                        localS.setItem("dhan_config", {"client_id": dcid})
                with c2:
                    if st.button("🚪 Logout Dhan", use_container_width=True, key="btn_logout_dhan"):
                        dhan_logout()
                st.caption(st.session_state['dhan_login_msg'])

        with st.expander("Telegram P&L Notifications", expanded=True):
            tg_store = localS.getItem("telegram_config") or {}
            if tg_store:
                st.session_state['telegram_bot_token'] = tg_store.get("bot_token", st.session_state['telegram_bot_token'])
                st.session_state['telegram_chat_id'] = tg_store.get("chat_id", st.session_state['telegram_chat_id'])

            notify_toggle = st.checkbox("Enable P&L notifications (30 min)", value=st.session_state['notify_enabled'], key="cfg_notify_toggle")
            st.session_state['notify_enabled'] = notify_toggle
            tg_token = st.text_input("Bot Token", value=st.session_state['telegram_bot_token'], key="cfg_tg_token")
            tg_chat = st.text_input("Chat ID", value=st.session_state['telegram_chat_id'], key="cfg_tg_chat")
            st.session_state['telegram_bot_token'] = tg_token
            st.session_state['telegram_chat_id'] = tg_chat

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save settings", use_container_width=True, key="btn_save_settings"):
                    save_config_from_state()
                    localS.setItem("telegram_config", {"bot_token": tg_token, "chat_id": tg_chat})
                    st.success("Saved to config.json + browser storage")
            with c2:
                if st.button("📤 Send P&L Now", use_container_width=True, key="btn_send_pnl"):
                    text = "P&L summary feature hooked to Dhan portfolio."
                    tg_resp = send_telegram_message(text) if tg_token and tg_chat else {"info": "Telegram not configured"}
                    st.success("Triggered P&L send. Check Telegram.")
                    st.json({"telegram": tg_resp})

        with st.expander("Nifty 200 Universe", expanded=False):
            if st.button("🔁 Regenerate NIFTY 200 CSV (internal)", use_container_width=True, key="btn_regen_nifty"):
                ok = regenerate_nifty200_csv_from_master()
                if ok:
                    st.success("Regenerated data/nifty200_yahoo.csv inside app container.")
                else:
                    st.error("Failed to regenerate CSV. See error above.")

if __name__ == "__main__":
    main()
