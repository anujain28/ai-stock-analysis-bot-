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

# Shoonya (Finvasia)
try:
    from NorenRestApiPy.NorenApi import NorenApi
except ImportError:
    NorenApi = None

# Dhan
try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

# ========= CONFIG FILE =========
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
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .status-live {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        animation: pulse 2s infinite;
        color: white;
        font-weight: bold;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone('Asia/Kolkata')

# ========= LOAD CONFIG & SESSION STATE =========
_cfg = load_config()

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
    ('shoonya_login_msg', 'Not logged in'),
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
    """
    Load Nifty 200 stock universe from CSV so it stays in sync with NSE/Yahoo symbols.
    Expected columns: SYMBOL, YF_TICKER.[web:62][web:50]
    """
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
    """
    Call scripts/generate_nifty200_yahoo.py to rebuild data/nifty200_yahoo.csv.[web:62]
    """
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
    """
    Convert NSE symbol to Yahoo ticker using CSV mapping first, then fall back to .NS suffix.[web:50]
    """
    if not sym:
        return ""
    s = sym.strip().upper()
    if s in NIFTY_YF_MAP:
        return NIFTY_YF_MAP[s]
    return s if s.endswith(".NS") else f"{s}.NS"

# ========= SHOONYA HELPERS =========
def shoonya_login(user_id: str, password: str, twofa: str, vendor_code: str, api_key: str):
    if not NorenApi:
        st.session_state['shoonya_logged_in'] = False
        st.session_state['shoonya_login_msg'] = "NorenRestApiPy not installed"
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

# ========= P&L + POSITONAL RECO SUMMARY =========
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
    rec_map = {}
    for period, lst in st.session_state.get('recommendations', {}).items():
        for r in lst:
            rec_map.setdefault(r['ticker'], []).append(r)
    lines.append("")
    lines.append("🎯 Positional Portfolio Signals:")
    if df_dhan is not None and not df_dhan.empty:
        for _, row in df_dhan.iterrows():
            name = str(row['Stock'])
            ticker_guess = name.split()[0].upper().replace('&', '').replace('-', '')
            if ticker_guess not in rec_map:
                continue
            best = sorted(rec_map[ticker_guess], key=lambda x: x.get('score', 0), reverse=True)[0]
            tgt = best.get('target_1')
            sig = best.get('signal_strength', 'HOLD')
            cmp_val = row.get('CMP', np.nan)
            try:
                cmp_val = float(cmp_val)
            except Exception:
                cmp_val = np.nan
            if not np.isnan(cmp_val) and tgt:
                dist = ((tgt - cmp_val) / cmp_val) * 100
                if abs(dist) <= 3:
                    advise = "Near target – consider immediate sell / booking profits"
                elif dist < -3:
                    advise = "Below target – trail SL or re‑evaluate"
                else:
                    advise = "HOLD for target"
                lines.append(
                    f"• {ticker_guess}: {sig}, CMP ~₹{cmp_val:.2f}, T1 ₹{tgt:.2f} → {advise}"
                )
            else:
                lines.append(f"• {ticker_guess}: {sig} (no clean CMP/target)")
    if len(lines) <= 4:
        lines.append("• No matching positional recommendations yet.")
    return "\n".join(lines)

# ========= PORTFOLIO CSV ANALYZER HELPERS =========
def calculate_cagr(begin_value: float, end_value: float, years: float) -> float:
    if begin_value <= 0 or years <= 0:
        return np.nan
    return (end_value / begin_value) ** (1 / years) - 1

def compute_cagrs_from_history(hist: pd.DataFrame, years_list=None):
    if years_list is None:
        years_list = [1, 3, 5, 10, 15, 20]
    if hist is None or hist.empty:
        return {f"{y}Y": np.nan for y in years_list}
    close = hist["Close"]
    last_price = float(close.iloc[-1])
    last_date = close.index[-1]
    out = {}
    for y in years_list:
        target_date = last_date - timedelta(days=365 * y)
        try:
            idx = close.index.get_indexer([target_date], method="nearest")[0]
            actual_date = close.index[idx]
            if abs((target_date - actual_date).days) > 30:
                out[f"{y}Y"] = np.nan
            else:
                start_price = float(close.iloc[idx])
                out[f"{y}Y"] = calculate_cagr(start_price, last_price, y) * 100.0
        except Exception:
            out[f"{y}Y"] = np.nan
    return out

def get_dividend_stats(ticker: str, years: int = 5):
    sym = nse_yf_symbol(ticker)
    t = yf.Ticker(sym)
    try:
        divs = t.dividends
    except Exception:
        divs = pd.Series(dtype=float)
    if divs is None or divs.empty:
        return {
            "Div Years": years,
            "Div Total ps": 0.0,
            "Div Annual ps": 0.0,
            "Div Yield %": 0.0,
            "Last Div Date": None,
            "Last Div Amt": 0.0,
        }

    last_date = divs.index.max()
    cutoff = last_date - pd.DateOffset(years=years)
    divs_n = divs[divs.index >= cutoff]
    total_ps = float(divs_n.sum()) if not divs_n.empty else 0.0
    annual_ps = total_ps / years if years > 0 else 0.0

    try:
        hist = t.history(period="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            cmp_price = float(hist["Close"].iloc[-1])
        else:
            cmp_price = np.nan
    except Exception:
        cmp_price = np.nan

    if not np.isnan(cmp_price) and cmp_price > 0:
        dy = (annual_ps / cmp_price) * 100.0
    else:
        dy = 0.0

    last_div_date = divs.index.max()
    last_div_amt = float(divs.iloc[-1])

    return {
        "Div Years": years,
        "Div Total ps": round(total_ps, 2),
        "Div Annual ps": round(annual_ps, 2),
        "Div Yield %": round(dy, 2),
        "Last Div Date": last_div_date,
        "Last Div Amt": round(last_div_amt, 2),
    }

def get_price_history(ticker: str):
    sym = nse_yf_symbol(ticker)
    t = yf.Ticker(sym)
    try:
        hist = t.history(period="max", auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()
    return hist

def analyze_csv_stock(row):
    ticker = str(row["Ticker"])
    qty = float(row["Quantity"])
    avg_price = float(row["Avg Price"])

    hist = get_price_history(ticker)
    if hist is None or hist.empty:
        return None

    cmp_price = float(hist["Close"].iloc[-1])
    cagr_map = compute_cagrs_from_history(hist)
    div_stats = get_dividend_stats(ticker, years=5)

    invested = qty * avg_price
    curr_val = qty * cmp_price
    pnl = curr_val - invested
    pnl_pct = (pnl / invested * 100.0) if invested > 0 else 0.0

    data = {
        "Ticker": ticker,
        "Qty": qty,
        "Avg Price": round(avg_price, 2),
        "CMP": round(cmp_price, 2),
        "Invested": round(invested, 2),
        "Current Value": round(curr_val, 2),
        "P&L": round(pnl, 2),
        "P&L %": round(pnl_pct, 2),
        "Div Yield %": div_stats["Div Yield %"],
        "Div Total (5Y) ps": div_stats["Div Total ps"],
        "Div Annual ps": div_stats["Div Annual ps"],
        "Last Div Date": div_stats["Last Div Date"],
        "Last Div Amt": div_stats["Last Div Amt"],
    }
    for k, v in cagr_map.items():
        data[f"{k}Y CAGR %"] = round(v, 2) if pd.notna(v) else np.nan
    return data

def analyze_csv_portfolio(base_df: pd.DataFrame):
    results = []
    if base_df is None or base_df.empty:
        return None, None
    progress = st.progress(0.0)
    for i, row in base_df.iterrows():
        res = analyze_csv_stock(row)
        if res:
            results.append(res)
        progress.progress((i + 1) / len(base_df))
    progress.empty()

    if not results:
        return None, None

    final_df = pd.DataFrame(results)

    total_inv = final_df["Invested"].sum()
    total_curr = final_df["Current Value"].sum()
    total_pnl = final_df["P&L"].sum()
    total_ret = (total_pnl / total_inv * 100.0) if total_inv > 0 else 0.0

    horizons = ["1Y", "3Y", "5Y", "10Y", "15Y", "20Y"]
    port_cagr = {}
    for h in horizons:
        col = f"{h}Y CAGR %"
        if col in final_df.columns:
            w = final_df["Current Value"]
            vals = final_df[col]
            mask = vals.notna() & w.notna()
            if mask.any():
                port_cagr[h] = (vals[mask] * w[mask]).sum() / w[mask].sum()
            else:
                port_cagr[h] = np.nan
        else:
            port_cagr[h] = np.nan

    summary = {
        "Total Invested": float(total_inv),
        "Current Value": float(total_curr),
        "Total P&L": float(total_pnl),
        "Total Return %": float(total_ret),
        "Portfolio CAGR": port_cagr,
    }
    return final_df, summary

def portfolio_csv_recommendations(df: pd.DataFrame, summary: dict):
    recs = []
    winners = df[df["P&L %"] > 100]
    if not winners.empty:
        recs.append(
            "🚀 Multi-baggers: "
            + ", ".join(winners["Ticker"].tolist())
            + " are up >100%. Consider partial profit booking to de-risk."
        )
    if "5Y CAGR %" in df.columns:
        laggards = df[df["5Y CAGR %"].notna() & (df["5Y CAGR %"] < 8)]
        if not laggards.empty:
            recs.append(
                "⚠️ Long-term underperformers (<8% 5Y CAGR): "
                + ", ".join(laggards["Ticker"].tolist())
                + ". Review thesis and consider re-allocation."
            )
    high_div = df[df["Div Yield %"] > 2.5]
    if not high_div.empty:
        recs.append(
            "💰 Dividend names: "
            + ", ".join(high_div["Ticker"].tolist())
            + " offer >2.5% trailing dividend yield, useful for income allocation."
        )
    if "10Y CAGR %" in df.columns:
        compounders = df[df["10Y CAGR %"].notna() & (df["10Y CAGR %"] > 15)]
        if not compounders.empty:
            recs.append(
                "💎 Compounders (>15% 10Y CAGR): "
                + ", ".join(compounders["Ticker"].tolist())
                + ". These can be held through cycles and added on dips."
            )
    port_5y = summary["Portfolio CAGR"].get("5Y")
    if pd.notna(port_5y) and port_5y < 10:
        recs.append(
            "📉 Portfolio 5Y CAGR <10%. To maximize profit, trim chronic laggards and rotate "
            "into consistent high-CAGR names with reasonable risk."
        )
    port_10y = summary["Portfolio CAGR"].get("10Y")
    if pd.notna(port_10y) and port_10y > 15:
        recs.append(
            "📈 Strong 10Y portfolio CAGR (>15%). Maintain discipline, avoid over-concentration, "
            "and keep some cash for corrections."
        )
    if not recs:
        recs.append("✅ Portfolio looks balanced. Focus on risk management and sizing.")
    return recs

# ========= MAIN UI =========
def main():
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 AI Stock Analysis Bot</h1>
        <p>Technical Analysis with Shoonya & Dhan Portfolios + Deep CSV Portfolio Analyzer</p>
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

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        st.markdown("#### 🤖 Analysis Info")
        if st.session_state['last_analysis_time']:
            st.info(f"Last Analysis:\n{st.session_state['last_analysis_time'].strftime('%I:%M %p')}")

        st.markdown("---")
        st.markdown("#### 💼 Shoonya")
        se = st.checkbox("Enable Shoonya", value=st.session_state['shoonya_enabled'])
        st.session_state['shoonya_enabled'] = se
        if se:
            uid = st.text_input("Shoonya User ID", value=st.session_state['shoonya_user_id'])
            pwd = st.text_input("Shoonya Password", value=st.session_state['shoonya_password'], type="password")
            twofa = st.text_input("Shoonya OTP / TOTP / PIN", value=st.session_state['shoonya_twofa'])
            vendor = st.text_input("Shoonya Vendor Code", value=st.session_state['shoonya_vendor_code'])
            appkey = st.text_input("Shoonya API Key", value=st.session_state['shoonya_api_key'], type="password")
            st.session_state['shoonya_user_id'] = uid
            st.session_state['shoonya_password'] = pwd
            st.session_state['shoonya_twofa'] = twofa
            st.session_state['shoonya_vendor_code'] = vendor
            st.session_state['shoonya_api_key'] = appkey
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("🔑 Login Shoonya", use_container_width=True):
                    shoonya_login(uid, pwd, twofa, vendor, appkey)
            with sc2:
                if st.button("🚪 Logout Shoonya", use_container_width=True):
                    shoonya_logout()
            col = "green" if "✅" in st.session_state['shoonya_login_msg'] else "red" if "❌" in st.session_state['shoonya_login_msg'] else "orange"
            st.markdown(
                f"<div style='background:{col};color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;'>{st.session_state['shoonya_login_msg']}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### 💼 Dhan")
        dhan_enable = st.checkbox("Enable Dhan", value=st.session_state.get('dhan_enabled', False))
        st.session_state['dhan_enabled'] = dhan_enable
        if dhan_enable:
            dcid = st.text_input("Dhan Client ID", value=st.session_state.get('dhan_client_id', ''))
            dtoken = st.text_input("Dhan Access Token", value=st.session_state.get('dhan_access_token', ''), type="password")
            st.session_state['dhan_client_id'] = dcid
            st.session_state['dhan_access_token'] = dtoken
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("🔑 Connect Dhan", use_container_width=True):
                    dhan_login(dcid, dtoken)
            with dc2:
                if st.button("🚪 Logout Dhan", use_container_width=True):
                    dhan_logout()
            col = "green" if "✅" in st.session_state['dhan_login_msg'] else "red" if "❌" in st.session_state['dhan_login_msg'] else "orange"
            st.markdown(
                f"<div style='background:{col};color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;'>{st.session_state['dhan_login_msg']}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### 📣 Telegram Notifications")

        notify_toggle = st.checkbox("Enable P&L notifications (30 min check)", value=st.session_state['notify_enabled'])
        st.session_state['notify_enabled'] = notify_toggle

        tg_token = st.text_input("Telegram Bot Token", value=st.session_state['telegram_bot_token'])
        tg_chat = st.text_input("Telegram Chat ID", value=st.session_state['telegram_chat_id'])
        st.session_state['telegram_bot_token'] = tg_token
        st.session_state['telegram_chat_id'] = tg_chat

        if st.button("💾 Save notification & broker settings", use_container_width=True):
            save_config_from_state()
            st.success("Settings saved to config.json")

        if st.button("📤 Send P&L Now", use_container_width=True):
            text = build_pnl_and_reco_summary()
            tg_resp = send_telegram_message(text) if tg_token and tg_chat else {"info": "Telegram not configured"}
            st.success("Triggered P&L send. Check Telegram.")
            st.json({"telegram": tg_resp})

        st.markdown("---")
        st.markdown("#### 📂 Nifty 200 Universe")
        st.caption(f"Loaded {len(STOCK_UNIVERSE)} symbols from {NIFTY200_CSV}")

        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            if st.button("🔁 Regenerate Nifty200 CSV", use_container_width=True):
                regenerate_nifty200_mapping()
        with gen_col2:
            if st.checkbox("Show universe list"):
                st.write(sorted(STOCK_UNIVERSE))

    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("🚀 Run Analysis Now", type="primary", use_container_width=True):
            run_analysis()
            st.rerun()
    with c2:
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()

    st.markdown("---")

    btst_tab, intraday_tab, weekly_tab, monthly_tab, top_btst_tab, shoonya_tab, dhan_tab, csv_tab = st.tabs(
        ["🌙 BTST", "⚡ Intraday", "📅 Weekly", "📆 Monthly", "⭐ Top 5 BTST",
         "📊 Shoonya Positions", "📊 Dhan Portfolio", "📂 Portfolio Analyzer (CSV)"]
    )

    def render_recs(period: str, tab_container):
        with tab_container:
            recs = st.session_state['recommendations'].get(period, [])
            if recs:
                st.markdown(f"### {period} Recommendations")
                df = pd.DataFrame(recs)
                if 'price' in df.columns:
                    df['price'] = df['price'].map(lambda x: f"{x:.2f}")
                for col in ['target_1','target_2','target_3','stop_loss']:
                    if col in df.columns:
                        df[col] = df[col].map(lambda x: f"{x:.2f}")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Click 'Run Analysis Now' to get recommendations")

    def render_top_btst(tab_container):
        with tab_container:
            recs = st.session_state['recommendations'].get('BTST', [])
            if not recs:
                st.info("Click 'Run Analysis Now' to get BTST picks.")
                return
            recs_sorted = sorted(recs, key=lambda x: x.get('score', 0), reverse=True)[:5]
            rows = []
            for r in recs_sorted:
                rows.append({
                    "Stock": r['ticker'],
                    "CMP (₹)": f"{r['price']:.2f}",
                    "Target 1 (₹)": f"{r.get('target_1', 0):.2f}",
                    "Target 2 (₹)": f"{r.get('target_2', 0):.2f}",
                    "Target 3 (₹)": f"{r.get('target_3', 0):.2f}",
                    "SL (₹)": f"{r.get('stop_loss', 0):.2f}",
                    "Est. Timeframe": r.get('timeframe', '1–3 days (BTST)'),
                    "Signal": r.get('signal_strength', ''),
                    "Score": r.get('score', 0),
                })
            df = pd.DataFrame(rows)
            st.markdown("### ⭐ Top 5 BTST Picks")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### 🛒 Trade Top BTST Picks via Shoonya (Manual)")
            if not st.session_state.get('shoonya_enabled', False) or not st.session_state.get('shoonya_logged_in', False):
                st.warning("Enable and login Shoonya from sidebar to place live orders.")
                return

            for r in recs_sorted:
                st.markdown(f"#### {r['ticker']} – {r['signal_strength']} (Score {r['score']})")
                st.write(
                    f"CMP: ₹{r['price']:.2f} | "
                    f"T1/T2/T3: ₹{r.get('target_1',0):.2f} / ₹{r.get('target_2',0):.2f} / ₹{r.get('target_3',0):.2f} | "
                    f"SL: ₹{r.get('stop_loss',0):.2f} | "
                    f"Timeframe: {r.get('timeframe','1–3 days (BTST)')}"
                )
                qty = st.number_input(
                    f"Qty for {r['ticker']}",
                    min_value=1, step=1, value=1,
                    key=f"btst_qty_{r['ticker']}"
                )
                product_choice = st.selectbox(
                    f"Product for {r['ticker']}",
                    ["CNC (delivery)", "INTRADAY"],
                    index=0,
                    key=f"btst_prd_{r['ticker']}"
                )
                prd_code = 'C' if product_choice.startswith("CNC") else 'I'
                tsym = f"{r['ticker']}-EQ"
                col_b, col_s = st.columns(2)
                resp_placeholder = st.empty()
                if col_b.button(f"✅ BUY {r['ticker']} via Shoonya", key=f"btst_buy_{r['ticker']}"):
                    resp = shoonya_place_market_order(tsym, int(qty), "BUY", product=prd_code)
                    resp_placeholder.json(resp)
                if col_s.button(f"❌ SELL {r['ticker']} via Shoonya", key=f"btst_sell_{r['ticker']}"):
                    resp = shoonya_place_market_order(tsym, int(qty), "SELL", product=prd_code)
                    resp_placeholder.json(resp)
                st.markdown("---")

    render_recs('BTST', btst_tab)
    render_recs('Intraday', intraday_tab)
    render_recs('Weekly', weekly_tab)
    render_recs('Monthly', monthly_tab)
    render_top_btst(top_btst_tab)

    with shoonya_tab:
        st.markdown("### 📊 Shoonya Positions P&L (Intraday + Carry Forward)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Shoonya Positions", use_container_width=True):
                st.session_state['shoonya_last_refresh'] = datetime.now(IST)
        with col2:
            if st.session_state['shoonya_last_refresh']:
                st.caption(f"Last refresh: {st.session_state['shoonya_last_refresh'].strftime('%I:%M:%S %p')}")
        if not st.session_state['shoonya_enabled']:
            st.warning("Enable Shoonya and login from sidebar.")
        elif not st.session_state.get('shoonya_logged_in', False):
            st.warning("Shoonya enabled but not logged in.")
        else:
            df_pos_raw, pnl_pos = format_shoonya_positions_table()
            if df_pos_raw is None or df_pos_raw.empty:
                st.info("No positions data from get_positions().")
            else:
                df_pos = df_pos_raw.copy()
                for col in ['Avg Cost', 'CMP', 'Total Cost', 'Total Value', 'P&L']:
                    df_pos[col] = df_pos[col].map(lambda x: f"{float(x):.2f}")
                st.markdown(f"**Total Positions P&L:** ₹{pnl_pos:.2f}")
                st.dataframe(df_pos, use_container_width=True)

    with dhan_tab:
        st.markdown("### 📊 Dhan Portfolio & P&L")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Dhan", use_container_width=True):
                st.session_state['dhan_last_refresh'] = datetime.now(IST)
        with col2:
            if st.session_state['dhan_last_refresh']:
                st.caption(f"Last refresh: {st.session_state['dhan_last_refresh'].strftime('%I:%M:%S %p')}")
        if not st.session_state.get('dhan_enabled', False):
            st.warning("Enable Dhan and configure Client ID + Access Token.")
        elif not st.session_state.get('dhan_client'):
            st.warning("Dhan is enabled but not connected yet.")
        else:
            df_pf, pnl = format_dhan_portfolio_table()
            if df_pf is None or df_pf.empty:
                st.info("No holdings/positions data from Dhan.")
            else:
                df_show = df_pf.copy()
                for col in ['Avg Cost','CMP','Total Cost','Total Value','P&L']:
                    if col in df_show.columns:
                        df_show[col] = df_show[col].map(lambda x: f"{float(x):.2f}")
                st.markdown(f"**Total P&L:** ₹{pnl:.2f}")
                st.dataframe(df_show, use_container_width=True)

    # ===== CSV Portfolio Analyzer Tab =====
    with csv_tab:
        st.markdown("### 📂 Portfolio Analyzer (CSV)")
        st.write("Upload CSV/Excel in format:")
        st.code("Stock Name | ISIN | Quantity | Average buy price per share | Total Investment | Total CMP | TOTAL P&L")

        uploaded = st.file_uploader("Upload portfolio file", type=["csv", "xlsx"])
        base_df = None

        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)

                cols_lower = {c.lower(): c for c in raw.columns}
                col_name = cols_lower.get("stock name")
                col_qty = cols_lower.get("quantity")
                col_avg = cols_lower.get("average buy price per share")

                if not (col_name and col_qty and col_avg):
                    st.error("Could not detect required columns. Please ensure exact headers.")
                    st.write("Detected columns:", list(raw.columns))
                else:
                    st.success(f"Mapped: Stock Name='{col_name}', Qty='{col_qty}', Avg Price='{col_avg}'")
                    tmp = pd.DataFrame({
                        "Ticker": raw[col_name].apply(normalize_symbol),
                        "Quantity": raw[col_qty],
                        "Avg Price": raw[col_avg],
                    })
                    base_df = tmp
            except Exception as e:
                st.error(f"Error reading file: {e}")

        if st.session_state["portfolio_results"] is not None and base_df is None:
            st.info("Using last analyzed CSV portfolio. Upload a file to replace it.")
            prev = st.session_state["portfolio_results"]
            base_df = prev[["Ticker", "Qty", "Avg Price"]].rename(columns={"Qty": "Quantity"})

        if base_df is not None:
            st.markdown("#### Edit / Clean Portfolio (Optional)")
            edit_df = st.data_editor(
                base_df,
                num_rows="dynamic",
                use_container_width=True,
                key="csv_portfolio_input_editor",
            )

            if st.button("⚡ Analyze CSV Portfolio", type="primary"):
                work_df = edit_df.rename(
                    columns={"Ticker": "Ticker", "Quantity": "Quantity", "Avg Price": "Avg Price"}
                )
                final_df, summary = analyze_csv_portfolio(work_df)
                if final_df is None:
                    st.warning("No valid stocks analyzed. Check symbols.")
                else:
                    st.session_state["portfolio_results"] = final_df.copy()

                    st.markdown("#### Portfolio Summary")
                    total_inv = summary["Total Invested"]
                    total_curr = summary["Current Value"]
                    total_pnl = summary["Total P&L"]
                    total_ret = summary["Total Return %"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Invested", f"₹{total_inv:,.0f}")
                    c2.metric("Current Value", f"₹{total_curr:,.0f}")
                    c3.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{total_ret:.2f}%")
                    port_5y = summary["Portfolio CAGR"].get("5Y", np.nan)
                    c4.metric("5Y Portfolio CAGR", f"{port_5y:.2f}%")

                    st.markdown("#### Stock-wise CAGR & Dividend Details")
                    st.dataframe(final_df, use_container_width=True)

                    st.markdown("#### Portfolio CAGR by Horizon")
                    rows = []
                    for h in ["1Y", "3Y", "5Y", "10Y", "15Y", "20Y"]:
                        rows.append({
                            "Horizon": h,
                            "Portfolio CAGR %": f"{summary['Portfolio CAGR'].get(h, np.nan):.2f}"
                        })
                    st.table(pd.DataFrame(rows))

                    st.markdown("#### Profit Maximization Recommendations")
                    for rec in portfolio_csv_recommendations(final_df, summary):
                        st.info(rec)

                    st.markdown("#### Save Analyzed Portfolio")
                    csv_data = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download analyzed_portfolio.csv",
                        data=csv_data,
                        file_name="analyzed_portfolio.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Upload a CSV/Excel file in the specified format to analyze your portfolio.")

if __name__ == "__main__":
    main()
