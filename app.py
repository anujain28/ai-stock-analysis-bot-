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
from datetime import datetime, time as dtime
from typing import Dict, List, Optional
import pytz
import requests
import os
import json
from pathlib import Path

from streamlit_local_storage import LocalStorage

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

# ========= PAGE CONFIG & ENHANCED CSS =========
st.set_page_config(
    page_title="🚀 AI Stock Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.95) 0%, rgba(168, 85, 247, 0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        position: relative;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-header h1 {
        font-size: clamp(1.8rem, 5vw, 3.5rem);
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 20px rgba(0, 0, 0, 0.3);
    }
    
    .hero-header .subtitle {
        font-size: clamp(0.9rem, 2.5vw, 1.2rem);
        opacity: 0.95;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .hero-header .emoji-float {
        font-size: clamp(2rem, 6vw, 4rem);
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.5rem;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.2);
        border: 2px solid rgba(16, 185, 129, 0.5);
        color: #d1fae5;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .status-pill::before {
        content: '●';
        color: #10b981;
        font-size: 1.2rem;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.5) inset;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    /* Stock Card */
    .stock-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.25rem;
        border: 2px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stock-card:hover {
        transform: scale(1.02);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3);
    }
    
    .stock-card .ticker {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stock-card .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-strong {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    .badge-buy {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .badge-hold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }
    
    .price-section {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .price-item {
        text-align: center;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .price-item .label {
        font-size: 0.75rem;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }
    
    .price-item .value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    .profit-highlight {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 0.75rem;
        border-radius: 12px;
        border: 2px solid #10b981;
        margin: 0.75rem 0;
        text-align: center;
    }
    
    .profit-highlight .amount {
        font-size: 1.3rem;
        font-weight: 800;
        color: #059669;
    }
    
    .profit-highlight .percentage {
        font-size: 1rem;
        font-weight: 600;
        color: #047857;
    }
    
    .reason-box {
        background: rgba(99, 102, 241, 0.05);
        border-left: 4px solid #6366f1;
        padding: 0.75rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #4b5563;
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    [data-testid=stSidebar] {
        background: linear-gradient(180deg, rgba(99, 102, 241, 0.95) 0%, rgba(139, 92, 246, 0.95) 100%);
        backdrop-filter: blur(20px);
    }
    
    [data-testid=stSidebar] .sidebar-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid=stSidebar] h4 {
        color: white;
        font-weight: 700;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    /* Metric Cards */
    .metric-showcase {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(243, 244, 246, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid rgba(99, 102, 241, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.2);
    }
    
    .metric-box .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-box .label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-box .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .hero-header {
            padding: 2rem 1rem;
        }
        
        .hero-header h1 {
            font-size: 1.8rem;
        }
        
        .glass-card {
            padding: 1rem;
        }
        
        .stock-card {
            padding: 1rem;
        }
        
        .price-section {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .metric-showcase {
            grid-template-columns: 1fr;
        }
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading-shimmer {
        background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 197, 253, 0.1) 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #1e40af;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(167, 243, 208, 0.1) 100%);
        border-left: 4px solid #10b981;
        color: #047857;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(252, 211, 77, 0.1) 100%);
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
</style>
""", unsafe_allow_html=True)

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

# ========= NIFTY 200 FROM CSV =========
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
    bar.empty()
    txt.empty()
    return sorted(out, key=lambda x: x['score'], reverse=True)[:max_results]

def run_analysis():
    now = datetime.now(IST)
    st.session_state['last_analysis_time'] = now
    st.session_state['last_auto_scan'] = now
    for p in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        st.session_state['recommendations'][p] = analyze_multiple_stocks(STOCK_UNIVERSE, p, max_results=20)

def market_hours_window():
    now = datetime.now(IST)
    start = now.replace(hour=9, minute=10, second=0, microsecond=0)
    end = now.replace(hour=15, minute=40, second=0, microsecond=0)
    return start <= now <= end

def maybe_run_auto_scan():
    now = datetime.now(IST)
    last = st.session_state.get('last_auto_scan')
    should_run = False
    if market_hours_window():
        if last is None:
            should_run = True
        else:
            try:
                if (now - last).total_seconds() >= 20 * 60:
                    should_run = True
            except Exception:
                should_run = True
    if should_run:
        run_analysis()

def get_top_stocks(limit: int = 10):
    all_recs = []
    for period in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        for r in st.session_state['recommendations'].get(period, []):
            rec = dict(r)
            rec['period'] = period
            all_recs.append(rec)
    if not all_recs:
        return []
    df_all = pd.DataFrame(all_recs).sort_values("score", ascending=False)
    seen = set()
    unique_rows = []
    for _, row in df_all.iterrows():
        t = row.get('ticker')
        if t not in seen:
            seen.add(t)
            unique_rows.append(row)
        if len(unique_rows) >= limit:
            break
    if not unique_rows:
        return []
    return pd.DataFrame(unique_rows).to_dict(orient="records")

PAGES = [
    "🔥 Top Stocks",
    "🌙 BTST",
    "⚡ Intraday",
    "📆 Weekly",
    "📅 Monthly",
    "📊 Groww",
    "⚙️ Configuration",
]

def sidebar_nav():
    with st.sidebar:
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("<h4>📂 Navigation</h4>", unsafe_allow_html=True)
        page = st.radio("Go to", PAGES, index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("💼 Dhan Portfolio", expanded=False):
            df_port, total_pnl = format_dhan_portfolio_table()
            if df_port is None or df_port.empty:
                st.caption("No Dhan holdings/positions yet")
            else:
                st.dataframe(df_port, use_container_width=True, hide_index=True)
                pnl_color = "🟢" if total_pnl >= 0 else "🔴"
                st.markdown(f"**{pnl_color} Total P&L: ₹{total_pnl:,.2f}**")
    return page

def analyze_groww_portfolio(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    required = ["stock name", "isin", "quantity", "average buy price per share", "total investment", "total cmp", "total p&l"]
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
    top = df_use.sort_values("_inv", ascending=False)[[cols["stock name"], cols["quantity"], invcol, pnlcol]].head(10).rename(
        columns={cols["stock name"]: "Stock Name", cols["quantity"]: "Quantity"}
    )
    return {"total_investment": total_inv, "total_pnl": total_pnl, "positions": positions, "top_holdings": top}

def render_reco_cards(recs: List[Dict], label: str):
    if not recs:
        st.markdown(f"<div class='info-box'>💡 Run Full Scan to generate {label} recommendations</div>", unsafe_allow_html=True)
        return
    df = pd.DataFrame(recs).sort_values("score", ascending=False).head(10)
    for _, rec in df.iterrows():
        cmp_ = rec.get('price', 0.0)
        tgt = rec.get('target_1', np.nan)
        diff = tgt - cmp_ if tgt is not None and not np.isnan(tgt) else np.nan
        profit_pct = (diff / cmp_ * 100) if cmp_ and not np.isnan(diff) else np.nan
        reason = rec.get('reasons', '')
        strength = rec.get('signal_strength', 'BUY')
        badge_class = 'badge-strong' if strength == 'STRONG BUY' else 'badge-buy' if strength == 'BUY' else 'badge-hold'
        
        st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='ticker'>
                <span>📈 {rec.get('ticker','')}</span>
                <span class='badge {badge_class}'>{strength}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='price-section'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='price-item'>
                <div class='label'>💰 Current</div>
                <div class='value'>₹{cmp_:.2f}</div>
            </div>
            <div class='price-item'>
                <div class='label'>🎯 Target</div>
                <div class='value'>₹{tgt:.2f}</div>
            </div>
            <div class='price-item'>
                <div class='label'>🛡️ Stop Loss</div>
                <div class='value'>₹{rec.get('stop_loss', 0):.2f}</div>
            </div>
            <div class='price-item'>
                <div class='label'>⏱️ Timeframe</div>
                <div class='value' style='font-size: 0.85rem;'>{rec.get('timeframe','')}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if not np.isnan(diff):
            st.markdown(f"""
                <div class='profit-highlight'>
                    <div class='amount'>💎 Potential Profit: ₹{diff:.2f}</div>
                    <div class='percentage'>📊 Expected Return: +{profit_pct:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        if reason:
            st.markdown(f"<div class='reason-box'>📌 <strong>Technical Setup:</strong> {reason}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.markdown("""
        <div class='hero-header'>
            <div class='emoji-float'>🚀📈💎</div>
            <h1>AI Stock Analysis Bot</h1>
            <p class='subtitle'>Multi-timeframe scanner with Dhan & Groww integration</p>
            <div class='status-pill'>Live Trading Dashboard • IST</div>
        </div>
    """, unsafe_allow_html=True)

    maybe_run_auto_scan()

    # Action buttons
    col1, col2, col3 = st.columns([3, 1.5, 1])
    with col1:
        if st.button("🚀 Run Full Market Scan", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing Nifty 200 stocks..."):
                run_analysis()
                st.success("✅ Analysis Complete!")
    with col2:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    with col3:
        st.metric("📊 Universe", len(STOCK_UNIVERSE))

    if st.session_state['last_analysis_time']:
        st.markdown(f"<div class='info-box'>⏰ Last scan: {st.session_state['last_analysis_time'].strftime('%d %b %Y, %I:%M %p IST')}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    page = sidebar_nav()

    if page == "🔥 Top Stocks":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🏆 Top 10 Stocks Across All Timeframes")
        st.markdown("Premium opportunities based on highest technical scores")
        st.markdown("</div>", unsafe_allow_html=True)
        top_recs = get_top_stocks(limit=10)
        render_reco_cards(top_recs, "Top")
        
    elif page == "🌙 BTST":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🌙 BTST Opportunities")
        st.markdown("Buy Today, Sell Tomorrow setups (1-3 day holds)")
        st.markdown("</div>", unsafe_allow_html=True)
        recs = st.session_state['recommendations'].get('BTST', [])
        for r in recs:
            r.setdefault("period", "BTST")
        render_reco_cards(recs, "BTST")
        
    elif page == "⚡ Intraday":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Intraday Trading Signals")
        st.markdown("Same-day trading opportunities with quick profit targets")
        st.markdown("</div>", unsafe_allow_html=True)
        recs = st.session_state['recommendations'].get('Intraday', [])
        for r in recs:
            r.setdefault("period", "Intraday")
        render_reco_cards(recs, "Intraday")
        
    elif page == "📆 Weekly":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📆 Weekly Swing Trades")
        st.markdown("Medium-term positions (up to 1 week)")
        st.markdown("</div>", unsafe_allow_html=True)
        recs = st.session_state['recommendations'].get('Weekly', [])
        for r in recs:
            r.setdefault("period", "Weekly")
        render_reco_cards(recs, "Weekly")
        
    elif page == "📅 Monthly":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📅 Monthly Position Trades")
        st.markdown("Long-term opportunities (2-4 weeks)")
        st.markdown("</div>", unsafe_allow_html=True)
        recs = st.session_state['recommendations'].get('Monthly', [])
        for r in recs:
            r.setdefault("period", "Monthly")
        render_reco_cards(recs, "Monthly")
        
    elif page == "📊 Groww":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Groww Portfolio Analyzer")
        st.markdown("Upload your Groww CSV to analyze your portfolio performance")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("**📋 CSV Template Format:**")
        st.code("Stock Name | ISIN | Quantity | Average buy price per share | Total Investment | Total CMP | TOTAL P&L", language="text")
        st.markdown("</div>", unsafe_allow_html=True)
        
        uploaded = st.file_uploader("📁 Upload Groww Portfolio CSV", type=["csv"], key="groww_csv_upload")
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded, sep=None, engine="python")
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("**📄 Data Preview:**")
                st.dataframe(df_up.head(), use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                analysis = analyze_groww_portfolio(df_up)
                if "error" in analysis:
                    st.markdown("<div class='success-box'>✅ Settings saved successfully!</div>", unsafe_allow_html=True)
            with c2:
                if st.button("📤 Send Test Message", use_container_width=True, key="btn_send_pnl"):
                    text = "🤖 Stock Bot Test: Telegram notifications are working perfectly!"
                    tg_resp = send_telegram_message(text) if tg_token and tg_chat else {"info": "Telegram not configured"}
                    if tg_resp.get("ok"):
                        st.markdown("<div class='success-box'>✅ Test message sent to Telegram!</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='warning-box'>⚠️ Failed: {tg_resp.get('error', 'Unknown error')}</div>", unsafe_allow_html=True)

        with st.expander("🔢 Nifty 200 Universe Management", expanded=False):
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown(f"**📊 Current Universe Size:** {len(STOCK_UNIVERSE)} stocks")
            st.markdown("**📁 Data Source:** `data/ind_nifty200list.csv`")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🔄 Regenerate NIFTY 200 CSV", use_container_width=True, key="btn_regen_nifty"):
                ok = regenerate_nifty200_csv_from_master()
                if ok:
                    st.markdown("<div class='success-box'>✅ Successfully regenerated universe data!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-box'>⚠️ Failed to regenerate CSV. Check error messages above.</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div class='glass-card' style='text-align: center; padding: 2rem;'>
            <h3 style='background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🚀 AI-Powered Stock Analysis Platform
            </h3>
            <p style='color: #6b7280; margin-top: 0.5rem;'>
                Powered by Technical Analysis • Real-time Market Data • Multi-timeframe Strategies
            </p>
            <p style='color: #9ca3af; font-size: 0.85rem; margin-top: 1rem;'>
                💡 Disclaimer: This tool is for educational purposes only. Always do your own research before investing.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()(f"<div class='warning-box'>⚠️ {analysis['error']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='metric-showcase'>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("""
                            <div class='metric-box'>
                                <div class='icon'>💰</div>
                                <div class='label'>Total Investment</div>
                                <div class='value'>₹{:,.0f}</div>
                            </div>
                        """.format(analysis['total_investment']), unsafe_allow_html=True)
                    with col2:
                        pnl_emoji = "📈" if analysis['total_pnl'] >= 0 else "📉"
                        st.markdown("""
                            <div class='metric-box'>
                                <div class='icon'>{}</div>
                                <div class='label'>Total P&L</div>
                                <div class='value'>₹{:,.0f}</div>
                            </div>
                        """.format(pnl_emoji, analysis['total_pnl']), unsafe_allow_html=True)
                    with col3:
                        st.markdown("""
                            <div class='metric-box'>
                                <div class='icon'>📊</div>
                                <div class='label'>Positions</div>
                                <div class='value'>{}</div>
                            </div>
                        """.format(analysis['positions']), unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("**🏆 Top Holdings by Capital**")
                    st.dataframe(analysis["top_holdings"], use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"<div class='warning-box'>⚠️ Error reading CSV: {e}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='info-box'>💡 Upload your Groww portfolio CSV to see detailed analysis</div>", unsafe_allow_html=True)
    
    elif page == "⚙️ Configuration":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Application Settings")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔐 Dhan Integration", expanded=True):
            dhan_store = localS.getItem("dhan_config") or {}
            if dhan_store:
                st.session_state['dhan_client_id'] = dhan_store.get("client_id", st.session_state['dhan_client_id'])

            dhan_enable = st.checkbox("✅ Enable Dhan Integration", value=st.session_state.get('dhan_enabled', False))
            st.session_state['dhan_enabled'] = dhan_enable
            if dhan_enable:
                dcid = st.text_input("🆔 Client ID", value=st.session_state.get('dhan_client_id', ''), key="cfg_dhan_client")
                dtoken = st.text_input("🔑 Access Token", value=st.session_state.get('dhan_access_token', ''), type="password", key="cfg_dhan_token")
                st.session_state['dhan_client_id'] = dcid
                st.session_state['dhan_access_token'] = dtoken
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔌 Connect Dhan", use_container_width=True, key="btn_connect_dhan"):
                        dhan_login(dcid, dtoken)
                        localS.setItem("dhan_config", {"client_id": dcid})
                with c2:
                    if st.button("🚪 Logout", use_container_width=True, key="btn_logout_dhan"):
                        dhan_logout()
                
                msg = st.session_state['dhan_login_msg']
                if "✅" in msg:
                    st.markdown(f"<div class='success-box'>{msg}</div>", unsafe_allow_html=True)
                elif "❌" in msg:
                    st.markdown(f"<div class='warning-box'>{msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='info-box'>{msg}</div>", unsafe_allow_html=True)

                df_port, total_pnl = format_dhan_portfolio_table()
                if df_port is not None and not df_port.empty:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("**💼 Your Portfolio**")
                    st.dataframe(df_port, use_container_width=True, hide_index=True)
                    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
                    st.markdown(f"**{pnl_emoji} Total P&L: ₹{total_pnl:,.2f}**")
                    st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📱 Telegram Notifications", expanded=False):
            tg_store = localS.getItem("telegram_config") or {}
            if tg_store:
                st.session_state['telegram_bot_token'] = tg_store.get("bot_token", st.session_state['telegram_bot_token'])
                st.session_state['telegram_chat_id'] = tg_store.get("chat_id", st.session_state['telegram_chat_id'])

            notify_toggle = st.checkbox("🔔 Enable P&L notifications (every 30 min)", value=st.session_state['notify_enabled'], key="cfg_notify_toggle")
            st.session_state['notify_enabled'] = notify_toggle
            tg_token = st.text_input("🤖 Bot Token", value=st.session_state['telegram_bot_token'], key="cfg_tg_token")
            tg_chat = st.text_input("💬 Chat ID", value=st.session_state['telegram_chat_id'], key="cfg_tg_chat")
            st.session_state['telegram_bot_token'] = tg_token
            st.session_state['telegram_chat_id'] = tg_chat

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save Settings", use_container_width=True, key="btn_save_settings"):
                    save_config_from_state()
                    localS.setItem("telegram_config", {"bot_token": tg_token, "chat_id": tg_chat})
                    st.markdown
