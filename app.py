# ========= AUTO-INSTALL (ONLY DHAN, OPTIONAL) =========
import subprocess
import sys

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
from typing import Dict, List, Optional
import pytz
import requests
import os
import json
from pathlib import Path
from difflib import SequenceMatcher

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
    :root {
        --bg-main: #f3f4f6;
        --bg-header-from: #4f46e5;
        --bg-header-to: #0ea5e9;
        --bg-card: #ffffff;
        --border-card: #1e293b;
        --accent: #ffffff;
        --accent-soft: #bbf7d0;
        --nav-bg: #e5e7eb;
        --nav-bg-hover: #4f46e5;
        --nav-bg-active: #4338ca;
        --nav-text: #111827;
        --nav-text-active: #0f172a;
        --btn-primary: #4f46e5;
        --btn-primary-hover: #4338ca;
        --btn-secondary: #f97316;
        --btn-secondary-hover: #ea580c;
        --groww-bg: #ffffff;
        --groww-text: #1f293b;
        --groww-header: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --groww-violet: #8b5cf6;
        --groww-violet-hover: #7c3aed;
    }

    .stApp {
        background-color: var(--bg-main);
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .main-header {
        background: linear-gradient(120deg, var(--bg-header-from) 0%, var(--bg-header-to) 100%);
        padding: 18px 18px;
        border-radius: 18px;
        text-align: left;
        color: white;
        margin-bottom: 12px;
        box-shadow: 0 12px 28px rgba(15,23,42,0.35);
        border: 1px solid rgba(255,255,255,0.14);
    }

    .groww-portfolio-snapshot {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 25px 50px rgba(139, 92, 246, 0.15);
    }

    .metric-card {
        padding: 24px;
        border-radius: 20px;
        background: var(--groww-bg);
        border: 2px solid #f1f5f9;
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.15);
        margin-bottom: 20px;
        color: var(--groww-text);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 60px rgba(139, 92, 246, 0.25);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: var(--groww-header);
    }
    .metric-card h3 {
        font-size: 1.1rem;
        color: #1e293b;
        margin-bottom: 12px;
        font-weight: 700;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f293b;
        margin-bottom: 8px;
    }
    .metric-card .emoji {
        font-size: 2.5rem;
        margin-right: 12px;
        display: inline-block;
    }
    .metric-card .sub {
        font-size: 0.95rem;
        color: #64748b;
    }

    /* Hide raw preview completely */
    [data-testid="stInfo"] {
        display: none !important;
    }

    /* File uploader styling - Violet background, white font */
    div[data-testid="stFileUploader"] > label div[data-testid="stFileUploadDropzone"] {
        border: 3px dashed var(--groww-violet) !important;
        background: linear-gradient(135deg, var(--groww-violet) 0%, var(--groww-violet-hover) 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 32px !important;
        font-weight: 600 !important;
        text-align: center;
    }
    div[data-testid="stFileUploader"] > label div[data-testid="stFileUploadDropzone"]:hover {
        background: linear-gradient(135deg, var(--groww-violet-hover) 0%, #6d28d9 100%) !important;
        box-shadow: 0 20px 40px rgba(139, 92, 246, 0.4) !important;
    }
    div[data-testid="stFileUploader"] > label > div > span {
        color: white !important;
        font-weight: 700 !important;
    }

    /* Portfolio Table */
    .groww-table {
        background: white;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-top: 24px;
    }
    .groww-table th {
        background: linear-gradient(135deg, var(--groww-violet), var(--groww-violet-hover)) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 16px !important;
        font-size: 1rem !important;
        text-align: center;
    }
    .groww-table td {
        padding: 16px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        vertical-align: middle !important;
        text-align: center;
    }
    .groww-table tr:hover {
        background-color: rgba(139,92,246,0.05) !important;
    }
    .totals-row {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }

    /* Other button styles */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, var(--btn-primary) 0%, var(--btn-primary-hover) 100%) !important;
        color: #ffffff !important;
        border: 2px solid var(--btn-primary) !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone('Asia/Kolkata')

# ========= LOAD CONFIG & SESSION STATE =========
_cfg = load_config()

# Initialize session state
if 'last_analysis_time' not in st.session_state:
    st.session_state['last_analysis_time'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = {'BTST': [], 'Intraday': [], 'Weekly': [], 'Monthly': []}
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "📊 Groww"

# Groww specific state
if 'groww_data' not in st.session_state:
    st.session_state.groww_data = None
if 'groww_totals' not in st.session_state:
    st.session_state.groww_totals = {}

# ========= NAVIGATION =========
NAV_PAGES = ["🔥 Top Stocks", "📊 Groww", "🤝 Dhan", "⚙️ Configuration"]

def sidebar_nav():
    with st.sidebar:
        st.markdown("### 📂 Views")
        page = st.radio(
            "Navigation",
            NAV_PAGES,
            index=NAV_PAGES.index(st.session_state.get("current_page", "📊 Groww")),
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = page

# ========= GROWW FUNCTIONS =========
def load_groww_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = excel_file.sheet_names[0]
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return pd.DataFrame()

def map_groww_columns(df):
    possible_stock_cols = ['Stock Name', 'stock name', 'Symbol', 'Name', 'Company']
    possible_qty_cols = ['Quantity', 'quantity', 'Qty', 'Units']
    possible_avg_cols = ['Average buy price per share', 'Avg Price', 'Average Price']
    possible_inv_cols = ['Total Investment', 'total investment', 'Investment']
    possible_cmp_cols = ['Total CMP', 'total cmp', 'Current Value', 'Market Value']
    possible_pnl_cols = ['Total P&L', 'total p&l', 'P&L']
    possible_isin_cols = ['ISIN', 'isin']

    col_map = {}
    for col_list, key in [
        (possible_stock_cols, 'stock'),
        (possible_qty_cols, 'qty'),
        (possible_avg_cols, 'avg'),
        (possible_inv_cols, 'inv'),
        (possible_cmp_cols, 'cmp'),
        (possible_pnl_cols, 'pnl'),
    ]:
        for col_name in col_list:
            if col_name in df.columns:
                col_map[key] = col_name
                break
    
    required = ['stock', 'qty', 'inv', 'cmp', 'pnl']
    if all(k in col_map for k in required):
        return col_map
    return None

# ========= MAIN UI =========
def main():
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 AI Stock Analysis Bot</h1>
        <p>Multi-timeframe scanner • 📈 NIFTY 200 • 📊 Groww Analytics</p>
        <div class="status-badge">Live • IST</div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_nav()

    # Top buttons
    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("🚀 Run Full Scan", type="primary", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    page = st.session_state['current_page']

    if page == "📊 Groww":
        st.markdown("## 💜 Groww Portfolio Analyzer")
        
        # File uploader with violet styling
        uploaded_file = st.file_uploader(
            "📁 Browse Files",
            type=["csv", "xls", "xlsx"],
            key="groww_upload",
            help="Upload your Groww portfolio CSV/Excel file"
        )

        if uploaded_file is not None:
            with st.spinner("🔄 Processing your portfolio..."):
                df_raw = load_groww_file(uploaded_file)
                col_map = map_groww_columns(df_raw)
                
                if col_map:
                    df = df_raw.copy()
                    df['_qty'] = pd.to_numeric(df[col_map['qty']], errors='coerce').fillna(0)
                    df['_inv'] = pd.to_numeric(df[col_map['inv']], errors='coerce').fillna(0)
                    df['_cmp_total'] = pd.to_numeric(df[col_map['cmp']], errors='coerce').fillna(0)
                    df['_pnl'] = pd.to_numeric(df[col_map['pnl']], errors='coerce').fillna(0)
                    df['_avg_price'] = np.where(df['_qty'] > 0, df['_inv'] / df['_qty'], 0)
                    df['_cmp_price'] = np.where(df['_qty'] > 0, df['_cmp_total'] / df['_qty'], 0)
                    
                    # Filter valid holdings
                    df = df[df['_qty'] > 0].copy()
                    
                    # Calculate totals
                    totals = {
                        'total_inv': df['_inv'].sum(),
                        'total_cmp': df['_cmp_total'].sum(),
                        'total_pnl': df['_pnl'].sum(),
                        'total_qty': df['_qty'].sum(),
                        'pnl_pct': ((df['_cmp_total'].sum() - df['_inv'].sum()) / df['_inv'].sum() * 100) if df['_inv'].sum() > 0 else 0
                    }
                    
                    st.session_state.groww_data = df
                    st.session_state.groww_totals = totals
                    
                    st.success(f"✅ Loaded {len(df)} holdings | Total Value: ₹{totals['total_cmp']:,.0f}")
                else:
                    st.error("❌ Could not find required columns in file")
                    st.dataframe(df_raw.head())

        # Beautiful Portfolio Snapshot
        if st.session_state.groww_data is not None:
            st.markdown('<div class="groww-portfolio-snapshot">', unsafe_allow_html=True)
            st.markdown("### 📊 Portfolio Snapshot")
            
            totals = st.session_state.groww_totals
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <span class='emoji'>💰</span><span class='value'>₹{totals['total_inv']:,.0f}</span>
                    <h3>Total Invested</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <span class='emoji'>📈</span><span class='value'>₹{totals['total_cmp']:,.0f}</span>
                    <h3>Current Value</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                pnl_emoji = "🟢" if totals['total_pnl'] >= 0 else "🔴"
                st.markdown(f"""
                <div class='metric-card'>
                    <span class='emoji'>{pnl_emoji}</span><span class='value'>₹{totals['total_pnl']:,.0f}</span>
                    <h3>Total P&L</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                pnl_pct_emoji = "🟢" if totals['pnl_pct'] >= 0 else "🔴"
                st.markdown(f"""
                <div class='metric-card'>
                    <span class='emoji'>{pnl_pct_emoji}</span><span class='value'>{totals['pnl_pct']:.1f}%</span>
                    <h3>Portfolio Return</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Portfolio Table WITHOUT ISIN + Totals
            st.markdown("### 📋 Holdings")
            df_display = st.session_state.groww_data[[
                col_map['stock'], '_qty', '_avg_price', '_inv', 
                '_cmp_price', '_cmp_total', '_pnl'
            ]].copy()
            
            df_display.columns = ['Stock', 'Qty', 'Avg Price', 'Total Invested', 
                                'CMP', 'Market Value', 'P&L']
            
            # Format numbers
            for col in ['Avg Price', 'Total Invested', 'CMP', 'Market Value', 'P&L']:
                df_display[col] = df_display[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "-")
            df_display['Qty'] = df_display['Qty'].astype(int)
            
            # Add totals row
            totals_row = pd.DataFrame([{
                'Stock': '🎯 TOTAL',
                'Qty': f"{int(totals['total_qty']):,d}",
                'Avg Price': '-',
                'Total Invested': f"₹{int(totals['total_inv']):,d}",
                'CMP': '-',
                'Market Value': f"₹{int(totals['total_cmp']):,d}",
                'P&L': f"₹{int(totals['total_pnl']):,d}"
            }])
            
            df_final = pd.concat([df_display, totals_row], ignore_index=True)
            
            st.markdown('<div class="groww-table">', unsafe_allow_html=True)
            st.dataframe(df_final, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("🌙 Groww page has all your requested features! Other pages coming soon.")

if __name__ == "__main__":
    main()
