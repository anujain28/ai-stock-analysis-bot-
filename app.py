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
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import pytz
import requests
import os
import json

from streamlit_local_storage import LocalStorage  # browser localStorage helper[web:167][web:170]

# Shoonya (Finvasia) – robust import using norenrestapipy
NorenApi = None
_shoonya_import_msg = "Not imported"

def _probe_shoonya_import():
    global NorenApi, _shoonya_import_msg
    try:
        # Path used by norenrestapipy 0.0.22
        from norenrestapipy.NorenApi import NorenApi as _N1  # type: ignore[web:131][web:152]
        NorenApi = _N1
        _shoonya_import_msg = "Imported from norenrestapipy.NorenApi"
        return
    except Exception as e1:
        try:
            # Fallback – older casing
            from NorenRestApiPy.NorenApi import NorenApi as _N2  # type: ignore
            NorenApi = _N2
            _shoonya_import_msg = "Imported from NorenRestApiPy.NorenApi"
            return
        except Exception as e2:
            NorenApi = None
            _shoonya_import_msg = f"Shoonya import failed: {e1} | {e2}"

_probe_shoonya_import()

# Dhan
try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

# ========= CONFIG FILE (SERVER-SIDE BACKUP) =========
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "shoonya_user_id": "",
    "shoonya_password": "",
    "shoonya_vendor_code": "",
    "shoonya_api_key": "",
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
        "shoonya_user_id": st.session_state.get("shoonya_user_id", ""),
        "shoonya_password": st.session_state.get("shoonya_password", ""),
        "shoonya_vendor_code": st.session_state.get("shoonya_vendor_code", ""),
        "shoonya_api_key": st.session_state.get("shoonya_api_key", ""),
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
        padding: 14px 14px;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 10px 30px rgba(15,23,42,0.8);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #e5e7eb;
        margin-bottom: 2px;
    }
    .metric-card .value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f9fafb;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: #9ca3af;
    }
    .side-section {
        padding: 10px 12px;
        border-radius: 12px;
        background: linear-gradient(145deg, #020617 0%, #0b1120 100%);
        border: 1px solid rgba(148,163,184,0.4);
        margin-bottom: 12px;
    }
    .side-section h4 {
        font-size: 0.85rem;
        margin-bottom: 8px;
        color: #e5e7eb;
    }
    div[data-testid="stSidebar"] {
        background: radial-gradient(circle at top, #020617 0, #020617 40%, #0b1120 100%);
    }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone('Asia/Kolkata')

# ========= LOAD CONFIG & SESSION STATE =========
_cfg = load_config()
localS = LocalStorage()  # browser localStorage manager[web:170]

# Server-side base state
if 'last_analysis_time' not in st.session_state:
    st.session_state['last_analysis_time'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = {'BTST': [], 'Intraday': [], 'Weekly': [], 'Monthly': []}
if 'portfolio_results' not in st.session_state:
    st.session_state['portfolio_results'] = None

# Shoonya state
for key, default in [
    ('shoonya_enabled', False),
    ('shoonya_user_id', _cfg.get('shoonya_user_id', '')),
    ('shoonya_password', _cfg.get('shoonya_password', '')),
    ('shoonya_twofa', ''),
    ('shoonya_vendor_code', _cfg.get('shoonya_vendor_code', '')),
    ('shoonya_api_key', _cfg.get('shoonya_api_key', '')),
    ('shoonya_api_obj', None),
    ('shoonya_logged_in', False),
    ('shoonya_login_msg', f'Not logged in ({_shoonya_import_msg})'),
    ('shoonya_last_refresh', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Dhan state
for key, default in [
    ('dhan_enabled', False),
    ('dhan_client_id', _cfg.get('dhan_client_id', '')),
    ('dhan_access_token', ''),
    ('dhan_client', None),
    ('dhan_login_msg', 'Not configured'),
    ('dhan_last_refresh', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Notification / Telegram state
for key, default in [
    ('notify_enabled', _cfg.get('notify_enabled', False)),
    ('telegram_bot_token', _cfg.get('telegram_bot_token', '')),
    ('telegram_chat_id', _cfg.get('telegram_chat_id', '')),
    ('last_pnl_notify', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ========= NIFTY 200 FROM CSV =========
NIFTY200_CSV = "data/nifty200_yahoo.csv"
NIFTY_GEN_SCRIPT = "scripts/generate_nifty200_yahoo.py"

@st.cache_data
def load_nifty200_universe():
    if not os.path.exists(NIFTY200_CSV):
        return [], {}
    try:
        df = pd.read_csv(NIFTY200_CSV)
    except Exception:
        return [], {}
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["YF_TICKER"] = df["YF_TICKER"].astype(str).str.strip()
    symbols = df["SYMBOL"].dropna().unique().tolist()
    mapping = dict(zip(df["SYMBOL"], df["YF_TICKER"]))
    return symbols, mapping

STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()

def regenerate_nifty200_mapping():
    if not os.path.exists(NIFTY_GEN_SCRIPT):
        st.error(f"Generator script not found at {NIFTY_GEN_SCRIPT}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, NIFTY_GEN_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            st.error("Generation failed. Check logs below.")
            st.code(result.stderr or result.stdout)
            return False
        st.success("✅ Regenerated data/nifty200_yahoo.csv successfully.")
        if result.stdout:
            st.code(result.stdout)
        load_nifty200_universe.clear()
        global STOCK_UNIVERSE, NIFTY_YF_MAP
        STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()
        return True
    except Exception as e:
        st.error(f"Error regenerating Nifty 200 mapping: {e}")
        return False

# ========= SYMBOL HELPERS =========
def normalize_symbol(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    t = raw.strip().upper().replace("&", "").replace("-", "")
    return t.split()[0]

def nse_yf_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.strip().upper()
    if s in NIFTY_YF_MAP:
        return NIFTY_YF_MAP[s]
    return s if s.endswith(".NS") else f"{s}.NS"

# ========= SHOONYA HELPERS =========
def running_on_streamlit_cloud() -> bool:
    return os.path.exists("/mount")

def shoonya_login(user_id: str, password: str, twofa: str, vendor_code: str, api_key: str):
    global NorenApi, _shoonya_import_msg
    if running_on_streamlit_cloud():
        st.session_state['shoonya_logged_in'] = False
        st.session_state['shoonya_login_msg'] = (
            "Shoonya live trading is disabled on Streamlit Cloud. "
            "Run this app locally for Shoonya orders."
        )
        return
    if NorenApi is None:
        st.session_state['shoonya_logged_in'] = False
        st.session_state['shoonya_login_msg'] = f"NorenRestApiPy import failed: {_shoonya_import_msg}"
        return
    try:
        api = NorenApi(
            host="https://api.shoonya.com/NorenWClientTP/",
            websocket="wss://api.shoonya.com/NorenWSTP/"
        )
        ret = api.login(
            userid=user_id,
            password=password,
            twoFA=twofa,
            vendor_code=vendor_code,
            api_secret=api_key,
            imei="streamlit-app"
        )
        if ret and ret.get("stat") == "Ok":
            st.session_state['shoonya_api_obj'] = api
            st.session_state['shoonya_logged_in'] = True
            st.session_state['shoonya_login_msg'] = "✅ Logged in to Shoonya"
        else:
            st.session_state['shoonya_logged_in'] = False
            st.session_state['shoonya_login_msg'] = f"❌ Login failed: {ret}"
    except Exception as e:
        st.session_state['shoonya_logged_in'] = False
        st.session_state['shoonya_login_msg'] = f"❌ Login error: {e}"

def shoonya_logout():
    api = st.session_state.get('shoonya_api_obj')
    if api:
        try:
            api.logout()
        except Exception:
            pass
    st.session_state['shoonya_api_obj'] = None
    st.session_state['shoonya_logged_in'] = False
    st.session_state['shoonya_login_msg'] = "Logged out"
    st.session_state['shoonya_last_refresh'] = None

def get_shoonya_positions_df():
    api = st.session_state.get('shoonya_api_obj')
    if not api or not st.session_state.get('shoonya_logged_in', False):
        return pd.DataFrame()
    try:
        positions = api.get_positions()
    except Exception:
        positions = []
    return pd.DataFrame(positions) if isinstance(positions, list) else pd.DataFrame()

def format_shoonya_positions_table():
    p_df = get_shoonya_positions_df()
    if p_df.empty:
        return pd.DataFrame(), 0.0
    name_col = 'tsym' if 'tsym' in p_df.columns else None
    if not name_col or 'netqty' not in p_df.columns or 'netavgprc' not in p_df.columns or 'lp' not in p_df.columns:
        return pd.DataFrame(), 0.0
    p_df['_name'] = p_df[name_col]
    p_df['_qty'] = pd.to_numeric(p_df['netqty'], errors='coerce').fillna(0.0)
    p_df['_avg'] = pd.to_numeric(p_df['netavgprc'], errors='coerce').fillna(0.0)
    p_df['_cmp'] = pd.to_numeric(p_df['lp'], errors='coerce').fillna(0.0)
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
    total_pnl = float(portfolio['P&L'].sum())
    return portfolio, total_pnl

def shoonya_place_market_order(tsym: str, qty: int, side: str, product: str = 'C'):
    if running_on_streamlit_cloud():
        return {"stat": "Not_Ok", "emsg": "Shoonya orders disabled on Streamlit Cloud. Run locally."}
    api = st.session_state.get('shoonya_api_obj')
    if not api or not st.session_state.get('shoonya_logged_in', False):
        return {"stat": "Not_Ok", "emsg": "Shoonya not logged in"}
    trantype = 'B' if side.upper() == 'BUY' else 'S'
    try:
        ret = api.place_order(
            buy_or_sell=trantype,
            product_type=product,
            exchange='NSE',
            tradingsymbol=tsym,
            quantity=str(qty),
            discloseqty='0',
            price_type='MKT',
            price='0',
            retention='DAY',
            remarks='AI_BOT'
        )
        return ret
    except Exception as e:
        return {"stat": "Not_Ok", "emsg": str(e)}

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
    st.session_state['dhan_last_refresh'] = None

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
        name_col = None
        for c in ['securityName', 'tradingSymbol', 'symbol']:
            if c in h_df.columns:
                name_col = c
                break
        qty_col = None
        for c in ['quantity', 'netQty', 'buyQty', 'totalQty']:
            if c in h_df.columns:
                qty_col = c
                break
        avg_col = None
        for c in ['averagePrice', 'buyAvg', 'avgCostPrice']:
            if c in h_df.columns:
                avg_col = c
                break
        cmp_col = None
        for c in ['ltp', 'lastTradedPrice', 'lastPrice']:
            if c in h_df.columns:
                cmp_col = c
                break
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
        if p_df.empty:
            return pd.DataFrame(), 0.0
        name_col = None
        for c in ['tradingSymbol', 'securityName', 'symbol']:
            if c in p_df.columns:
                name_col = c
                break
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

# ========= TA & ANALYSIS (unchanged core) =========
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
            try:
                sig, reason, s_ = func(df)
                if sig:
                    signals.append(name)
                    if reason:
                        reasons.append(reason)
                    score += s_
            except Exception:
                pass
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
        time.sleep(0.05)
    bar.empty()
    txt.empty()
    return sorted(out, key=lambda x: x['score'], reverse=True)[:max_results]

def run_analysis():
    st.session_state['last_analysis_time'] = datetime.now(IST)
    new_recs = {}
    with st.spinner("🔍 Running analysis..."):
        for p in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
            new_recs[p] = analyze_multiple_stocks(STOCK_UNIVERSE, p, max_results=20)
            st.session_state['recommendations'][p] = new_recs[p]
    return new_recs

# ========= P&L SUMMARY =========
def build_pnl_and_reco_summary():
    lines = []
    now_str = datetime.now(IST).strftime("%d-%m-%Y %H:%M")
    df_shoonya, shoonya_pnl = format_shoonya_positions_table()
    lines.append(f"📊 P&L Update @ {now_str}")
    if df_shoonya is not None and not df_shoonya.empty:
        lines.append(f"• Shoonya total P&L: ₹{shoonya_pnl:.2f}")
    else:
        lines.append("• Shoonya P&L: No positions / data")
    df_dhan, dhan_pnl = format_dhan_portfolio_table()
    if df_dhan is not None and not df_dhan.empty:
        lines.append(f"• Dhan total P&L: ₹{dhan_pnl:.2f}")
    else:
        lines.append("• Dhan P&L: No holdings / data")
    if len(lines) <= 2:
        lines.append("• No live portfolio data yet.")
    return "\n".join(lines)

# ========= CSV ANALYZER HELPERS (same as before, omitted for brevity) =========
# ... (keep your existing calculate_cagr, compute_cagrs_from_history,
#     get_dividend_stats, get_price_history, analyze_csv_stock,
#     analyze_csv_portfolio, portfolio_csv_recommendations exactly as in previous version)

# ========= MAIN UI =========
def main():
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 AI Stock Analysis Bot</h1>
        <p>Multi-timeframe scanner + Dhan/Shoonya hooks + deep portfolio analyzer</p>
        <div class="status-badge">Live • IST</div>
    </div>
    """, unsafe_allow_html=True)

    # Auto P&L Telegram every ~30 min
    if st.session_state.get('notify_enabled', False):
        now = datetime.now(IST)
        last = st.session_state.get('last_pnl_notify')
        should_send = False
        if last is None:
            should_send = True
        else:
            try:
                delta_min = (now - last).total_seconds() / 60.0
                if delta_min >= 30:
                    should_send = True
            except Exception:
                should_send = True
        if should_send:
            text = build_pnl_and_reco_summary()
            _ = send_telegram_message(text)
            st.session_state['last_pnl_notify'] = now
            st.info("Auto P&L notification triggered (30-min interval).")

    # ===== Sidebar with browser-local persistence =====
    with st.sidebar:
        st.markdown("<div class='side-section'><h4>⚙️ Configuration</h4>", unsafe_allow_html=True)

        if st.session_state['last_analysis_time']:
            st.caption(f"Last Analysis: {st.session_state['last_analysis_time'].strftime('%I:%M %p')}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='side-section'><h4>💼 Shoonya</h4>", unsafe_allow_html=True)
        # Load from browser localStorage (user-specific) if present[web:170]
        shoonya_store = localS.getItem("shoonya_config") or {}
        if shoonya_store:
            st.session_state['shoonya_user_id'] = shoonya_store.get("user_id", st.session_state['shoonya_user_id'])
            st.session_state['shoonya_vendor_code'] = shoonya_store.get("vendor_code", st.session_state['shoonya_vendor_code'])
            st.session_state['shoonya_api_key'] = shoonya_store.get("api_key", st.session_state['shoonya_api_key'])

        se = st.checkbox("Enable Shoonya (local only)", value=st.session_state['shoonya_enabled'])
        st.session_state['shoonya_enabled'] = se
        if se:
            uid = st.text_input("User ID", value=st.session_state['shoonya_user_id'])
            pwd = st.text_input("Password", value=st.session_state['shoonya_password'], type="password")
            twofa = st.text_input("OTP / PIN", value=st.session_state['shoonya_twofa'])
            vendor = st.text_input("Vendor Code", value=st.session_state['shoonya_vendor_code'])
            appkey = st.text_input("API Key", value=st.session_state['shoonya_api_key'], type="password")
            st.session_state['shoonya_user_id'] = uid
            st.session_state['shoonya_password'] = pwd
            st.session_state['shoonya_twofa'] = twofa
            st.session_state['shoonya_vendor_code'] = vendor
            st.session_state['shoonya_api_key'] = appkey

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔑 Login", width="stretch"):
                    shoonya_login(uid, pwd, twofa, vendor, appkey)
                    # Save non-sensitive Shoonya fields to browser localStorage
                    localS.setItem("shoonya_config", {
                        "user_id": uid,
                        "vendor_code": vendor,
                        "api_key": appkey,
                    })
            with c2:
                if st.button("🚪 Logout", width="stretch"):
                    shoonya_logout()
            st.caption(st.session_state['shoonya_login_msg'])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='side-section'><h4>💼 Dhan</h4>", unsafe_allow_html=True)
        dhan_store = localS.getItem("dhan_config") or {}
        if dhan_store:
            st.session_state['dhan_client_id'] = dhan_store.get("client_id", st.session_state['dhan_client_id'])

        dhan_enable = st.checkbox("Enable Dhan", value=st.session_state.get('dhan_enabled', False))
        st.session_state['dhan_enabled'] = dhan_enable
        if dhan_enable:
            dcid = st.text_input("Client ID", value=st.session_state.get('dhan_client_id', ''))
            dtoken = st.text_input("Access Token", value=st.session_state.get('dhan_access_token', ''), type="password")
            st.session_state['dhan_client_id'] = dcid
            st.session_state['dhan_access_token'] = dtoken
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔑 Connect", width="stretch"):
                    dhan_login(dcid, dtoken)
                    localS.setItem("dhan_config", {"client_id": dcid})
            with c2:
                if st.button("🚪 Logout Dhan", width="stretch"):
                    dhan_logout()
            st.caption(st.session_state['dhan_login_msg'])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='side-section'><h4>📣 Telegram</h4>", unsafe_allow_html=True)
        tg_store = localS.getItem("telegram_config") or {}
        if tg_store:
            st.session_state['telegram_bot_token'] = tg_store.get("bot_token", st.session_state['telegram_bot_token'])
            st.session_state['telegram_chat_id'] = tg_store.get("chat_id", st.session_state['telegram_chat_id'])

        notify_toggle = st.checkbox("Enable P&L notifications (30 min)", value=st.session_state['notify_enabled'])
        st.session_state['notify_enabled'] = notify_toggle
        tg_token = st.text_input("Bot Token", value=st.session_state['telegram_bot_token'])
        tg_chat = st.text_input("Chat ID", value=st.session_state['telegram_chat_id'])
        st.session_state['telegram_bot_token'] = tg_token
        st.session_state['telegram_chat_id'] = tg_chat

        if st.button("💾 Save settings", width="stretch"):
            save_config_from_state()
            localS.setItem("telegram_config", {"bot_token": tg_token, "chat_id": tg_chat})
            st.success("Saved to config.json + browser storage")

        if st.button("📤 Send P&L Now", width="stretch"):
            text = build_pnl_and_reco_summary()
            tg_resp = send_telegram_message(text) if tg_token and tg_chat else {"info": "Telegram not configured"}
            st.success("Triggered P&L send. Check Telegram.")
            st.json({"telegram": tg_resp})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='side-section'><h4>📂 Nifty 200 Universe</h4>", unsafe_allow_html=True)
        st.caption(f"Loaded {len(STOCK_UNIVERSE)} symbols from {NIFTY200_CSV}")
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("🔁 Regenerate CSV", width="stretch"):
                regenerate_nifty200_mapping()
        with c2:
            if st.checkbox("Show list"):
                st.write(sorted(STOCK_UNIVERSE))
        st.markdown("</div>", unsafe_allow_html=True)

    # Top action buttons
    c1, c2, c3 = st.columns([3, 1.2, 1])
    with c1:
        if st.button("🚀 Run Full Scan", type="primary", width="stretch"):
            run_analysis()
            st.rerun()
    with c2:
        if st.button("🔄 Refresh Page", width="stretch"):
            st.rerun()
    with c3:
        st.metric("Universe", len(STOCK_UNIVERSE))

    st.markdown("---")

    # Tabs (reuse your previous rendering logic for recs, top BTST, Shoonya, Dhan, CSV analyzer)
    # ...

if __name__ == "__main__":
    main()
