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
from typing import Dict, List, Optional
import pytz
import requests
import os
import json
from pathlib import Path

from streamlit_local_storage import LocalStorage

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

# ========= CONFIG FILE =========
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {"dhan_client_id": "", "telegram_bot_token": "", "telegram_chat_id": "", "notify_enabled": False}

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
st.set_page_config(page_title="🤖 AI Stock Analysis Bot", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #f3f4f6; color: #111827; }
    body { background-color: #f3f4f6; color: #111827; }
    .main-header {
        background: linear-gradient(120deg, #4f46e5 0%, #0ea5e9 100%);
        padding: 18px; border-radius: 18px; color: white; margin-bottom: 12px;
        box-shadow: 0 12px 28px rgba(15,23,42,0.35);
    }
    .main-header h1 { margin-bottom: 4px; font-size: clamp(1.6rem, 3vw, 2.3rem); }
    .main-header p { margin: 0; font-size: 0.9rem; opacity: 0.96; }
    .status-badge {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em;
        background: rgba(15,23,42,0.35); border: 1px solid rgba(226,232,240,0.7); margin-top: 6px;
    }
    .metric-card {
        padding: 12px; border-radius: 14px; background: #ffffff;
        border: 1px solid #e5e7eb; box-shadow: 0 4px 12px rgba(15,23,42,0.18);
        margin-bottom: 10px; color: #111827;
    }
    .metric-card h3 { font-size: 0.95rem; color: #0f172a; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.05rem; font-weight: 600; color: #111827; }
    .metric-card .sub { font-size: 0.8rem; color: #4b5563; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .chip { padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; background: #ffffff; border: 1px solid #d1d5db; color: #111827; }
    .st-key-refresh_btn button { background-color: #f97316 !important; color: #ffffff !important; }
    div[data-testid="stFileUploader"] section button {
        background-color: #1e293b !important; color: #f8fafc !important;
        border: 1px solid #334155 !important; font-weight: 500 !important;
    }
    div[data-testid="stFileUploader"] section button:hover {
        background-color: #334155 !important; color: #ffffff !important;
    }
    .st-key-btn_auto_calc button {
        background-color: #1e293b !important; color: #f8fafc !important;
        border-color: #334155 !important; font-weight: 500 !important;
    }
    .st-key-btn_auto_calc button:hover {
        background-color: #334155 !important; color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone('Asia/Kolkata')
_cfg = load_config()
localS = LocalStorage()

# Session state initialization
if 'last_analysis_time' not in st.session_state:
    st.session_state['last_analysis_time'] = None
if 'last_auto_scan' not in st.session_state:
    st.session_state['last_auto_scan'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = {'BTST': [], 'Intraday': [], 'Weekly': [], 'Monthly': []}
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "🔥 Top Stocks"

for key, default in [('dhan_enabled', False), ('dhan_client_id', _cfg.get('dhan_client_id', '')),
                     ('dhan_access_token', ''), ('dhan_client', None), ('dhan_login_msg', 'Not configured'),
                     ('notify_enabled', _cfg.get('notify_enabled', False)),
                     ('telegram_bot_token', _cfg.get('telegram_bot_token', '')),
                     ('telegram_chat_id', _cfg.get('telegram_chat_id', '')), ('last_pnl_notify', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ========= NIFTY 200 =========
DATA_DIR = Path(__file__).parent / "data"
NIFTY200_CSV = DATA_DIR / "nifty200_yahoo.csv"
MASTER_NIFTY200 = DATA_DIR / "ind_nifty200list.csv"

@st.cache_data
def load_nifty200_universe():
    if not NIFTY200_CSV.exists():
        return [], {}
    df = pd.read_csv(NIFTY200_CSV)
    if "SYMBOL" not in df.columns or "YF_TICKER" not in df.columns:
        return [], {}
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["YF_TICKER"] = df["YF_TICKER"].astype(str).str.strip()
    symbols = df["SYMBOL"].dropna().unique().tolist()
    mapping = dict(zip(df["SYMBOL"], df["YF_TICKER"]))
    return symbols, mapping

STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()

def nse_yf_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.strip().upper()
    if s in NIFTY_YF_MAP:
        return NIFTY_YF_MAP[s]
    return s if s.endswith(".NS") else f"{s}.NS"

# ========= GROWW HELPERS =========
def load_groww_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Only CSV, XLS, or XLSX files are supported.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

def map_groww_columns(df: pd.DataFrame):
    norm_cols = {c.lower().strip(): c for c in df.columns}
    required_map = {
        "stock name": "stock name", "isin": "isin", "quantity": "quantity",
        "average buy price per share": "average buy price per share",
        "total investment": "total investment", "total cmp": "total cmp", "total p&l": "total p&l",
    }
    out, missing = {}, []
    for logical_key, norm_header in required_map.items():
        if norm_header in norm_cols:
            out[logical_key] = norm_cols[norm_header]
        else:
            missing.append(logical_key)
    if missing:
        msg = "Columns must match Groww template. Missing: " + ", ".join(missing)
        return None, msg
    return out, None

def fetch_dividend_and_cagr(stock_name: str, isin: str, cmp_value: float):
    sym = stock_name.split()[0].upper().strip() if stock_name else ""
    yf_ticker = NIFTY_YF_MAP.get(sym, None)
    if not yf_ticker and sym:
        yf_ticker = f"{sym}.NS"
    div_yield, div_rupees, cagr = 0.0, 0.0, 0.05
    if not yf_ticker:
        return div_yield, div_rupees, cagr
    try:
        t = yf.Ticker(yf_ticker)
        info = t.info or {}
        raw_yield = info.get("dividendYield")
        if raw_yield is not None:
            div_yield = float(raw_yield)
        if cmp_value and div_yield:
            div_rupees = div_yield * cmp_value
        hist = t.history(period="10y")
        if hist is not None and not hist.empty:
            hist = hist.dropna(subset=["Close"])
            first_price = float(hist["Close"].iloc[0])
            last_price = float(hist["Close"].iloc[-1])
            years = max((hist.index[-1] - hist.index[0]).days / 365.0, 1.0)
            if first_price > 0 and years > 0:
                cagr = (last_price / first_price) ** (1.0 / years) - 1.0
    except Exception:
        pass
    return float(div_yield), float(div_rupees), float(cagr)

def classify_strength(pct_pnl: float, cagr: float, price_zero: bool) -> str:
    if price_zero:
        return "Super Strong"
    if cagr >= 0.15 and pct_pnl >= 20:
        return "Super Strong"
    if cagr >= 0.10 and pct_pnl >= 0:
        return "Strong"
    if cagr >= 0.05 or pct_pnl > -15:
        return "Medium"
    return "Sell"

def get_recommendation(pct_pnl: float, cagr: float, price_zero: bool) -> str:
    strength = classify_strength(pct_pnl, cagr, price_zero)
    if strength in ["Super Strong", "Strong"]:
        return "BUY"
    elif strength == "Medium":
        return "HOLD"
    else:
        return "SELL"

def project_value(current_value: float, cagr: float, yearly_dividend: float, years: int) -> float:
    future = current_value * ((1 + cagr) ** years)
    future += yearly_dividend * years
    return float(future)

# ========= SIDEBAR NAV =========
NAV_PAGES = ["🔥 Top Stocks", "🌙 BTST", "⚡ Intraday", "📆 Weekly", "📅 Monthly", "📊 Groww", "🤝 Dhan", "⚙️ Configuration"]

def sidebar_nav():
    with st.sidebar:
        st.markdown("### 📂 Views")
        page = st.radio("Navigation", NAV_PAGES,
                       index=NAV_PAGES.index(st.session_state.get("current_page", "🔥 Top Stocks")),
                       label_visibility="collapsed")
        st.session_state["current_page"] = page

# ========= MAIN UI =========
def main():
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 AI Stock Analysis Bot</h1>
        <p>Multi-timeframe scanner • 📈 NIFTY 200 • 🤝 Dhan • 📊 Groww</p>
        <div class="status-badge">Live • IST</div>
    </div>
    """, unsafe_allow_html=True)

    sidebar_nav()
    page = st.session_state['current_page']

    if page == "📊 Groww":
        st.subheader("📊 Groww Portfolio Analysis (CSV / Excel Upload)")
        st.markdown("Upload your Groww holdings file (CSV/XLS/XLSX) to get advanced analytics with all values in ₹ INR.")
        st.code("Stock Name\tISIN\tQuantity\tAverage buy price per share\tTotal Investment\tTotal CMP\tTOTAL P&L", language="text")
        
        uploaded = st.file_uploader("Upload Groww portfolio file", type=["csv", "xls", "xlsx"], key="groww_file_upload")

        if uploaded is not None:
            df_up = load_groww_file(uploaded)
            if df_up.empty:
                st.stop()

            cols, err = map_groww_columns(df_up)
            if err:
                st.error(err)
                st.dataframe(df_up.head(), use_container_width=True, hide_index=True)
                st.stop()

            st.write("🔍 Raw preview:")
            st.dataframe(df_up.head(), use_container_width=True, hide_index=True)

            df = df_up.copy()
            df["_qty"] = pd.to_numeric(df[cols["quantity"]], errors="coerce").fillna(0.0)
            df["_inv"] = pd.to_numeric(df[cols["total investment"]], errors="coerce").fillna(0.0)
            df["_cmp_total"] = pd.to_numeric(df[cols["total cmp"]], errors="coerce").fillna(0.0)
            df["_pnl"] = pd.to_numeric(df[cols["total p&l"]], errors="coerce").fillna(0.0)
            df["_cmp_per_share"] = np.where(df["_qty"] > 0, df["_cmp_total"] / df["_qty"], 0.0)

            div_yields, div_rupees_list, cagr_list, strength_list = [], [], [], []

            st.info("Fetching dividend yield and CAGR for each stock from internet; defaults used if not found.")
            prog = st.progress(0.0)
            for i, row in df.iterrows():
                stock_name = str(row[cols["stock name"]])
                cmp_ps = float(row["_cmp_per_share"])
                is_zero_price = cmp_ps <= 0.0

                div_y, div_r, cagr = fetch_dividend_and_cagr(stock_name, str(row[cols["isin"]]), cmp_ps)
                div_yields.append(div_y)
                div_rupees_list.append(div_r)
                cagr_list.append(cagr)

                inv_val = float(row["_inv"])
                cur_val = float(row["_cmp_total"])
                pct_pnl = ((cur_val - inv_val) / inv_val * 100.0) if inv_val > 0 else 0.0
                strength = classify_strength(pct_pnl, cagr, is_zero_price)
                strength_list.append(strength)
                prog.progress((i + 1) / len(df))
            prog.empty()

            df["Dividend Yield"] = div_yields
            df["Dividend/Share (₹)"] = div_rupees_list
            df["CAGR (decimal)"] = cagr_list
            df["CAGR (%)"] = df["CAGR (decimal)"] * 100.0
            df["Strength"] = strength_list
            df["Yearly Dividend (₹)"] = df["Dividend/Share (₹)"] * df["_qty"]

            total_inv = float(df["_inv"].sum())
            total_cmp_val = float(df["_cmp_total"].sum())
            total_pnl = float(df["_pnl"].sum())
            total_yearly_div = float(df["Yearly Dividend (₹)"].sum())

            st.markdown("### 📈 Portfolio Snapshot (All values in ₹ INR)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Investment", f"₹{total_inv:,.2f}")
            with c2:
                st.metric("Current Value", f"₹{total_cmp_val:,.2f}")
            with c3:
                st.metric("Total P&L", f"₹{total_pnl:,.2f}")

            if total_cmp_val > 0:
                df["_weight"] = df["_cmp_total"] / total_cmp_val
                portfolio_cagr = float((df["CAGR (decimal)"] * df["_weight"]).sum())
            else:
                portfolio_cagr = 0.05

            st.markdown("#### 🔮 Portfolio Value Projections")
            st.markdown("*Based on weighted CAGR and total yearly dividend. All values in ₹ INR.*")
            years_list = [1, 5, 10, 15, 20]
            proj_data = []
            for y in years_list:
                v = project_value(total_cmp_val, portfolio_cagr, total_yearly_div, y)
                capital_gain = v - total_cmp_val - (total_yearly_div * y)
                proj_data.append({
                    "Years": y,
                    "Current Value (₹)": f"{total_cmp_val:,.2f}",
                    "Total Dividends (₹)": f"{total_yearly_div * y:,.2f}",
                    "Capital Gain (₹)": f"{capital_gain:,.2f}",
                    "Projected Value (₹)": f"{v:,.2f}"
                })
            st.table(pd.DataFrame(proj_data))

            st.markdown("### 🛠 Adjust Portfolio & Recalculate")
            st.write("Adjust quantities, CMP, CAGR, and dividend per share, then click **Auto Calculate**.")

            editable_cols = [cols["stock name"], cols["quantity"], cols["total investment"],
                           cols["total cmp"], "Dividend/Share (₹)", "CAGR (%)"]
            edit_df = df[editable_cols].copy()
            edit_df = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True,
                                    hide_index=True, key="groww_edit_table")

            if st.button("⚙️ Auto Calculate", use_container_width=True, key="btn_auto_calc"):
                df2 = edit_df.copy()
                df2["_qty"] = pd.to_numeric(df2[cols["quantity"]], errors="coerce").fillna(0.0)
                df2["_inv"] = pd.to_numeric(df2[cols["total investment"]], errors="coerce").fillna(0.0)
                df2["_cmp_total"] = pd.to_numeric(df2[cols["total cmp"]], errors="coerce").fillna(0.0)
                df2["Dividend/Share (₹)"] = pd.to_numeric(df2["Dividend/Share (₹)"], errors="coerce").fillna(0.0)
                df2["CAGR (%)"] = pd.to_numeric(df2["CAGR (%)"], errors="coerce").fillna(5.0)
                df2["CAGR (decimal)"] = df2["CAGR (%)"] / 100.0
                df2["_pnl"] = df2["_cmp_total"] - df2["_inv"]
                df2["_cmp_per_share"] = np.where(df2["_qty"] > 0, df2["_cmp_total"] / df2["_qty"], 0.0)

                strengths, recommendations = [], []
                for _, r in df2.iterrows():
                    inv2 = float(r["_inv"])
                    cur2 = float(r["_cmp_total"])
                    pct_pnl2 = ((cur2 - inv2) / inv2 * 100.0) if inv2 > 0 else 0.0
                    is_zero_price2 = float(r["_cmp_per_share"]) <= 0.0
                    strength = classify_strength(pct_pnl2, float(r["CAGR (decimal)"]), is_zero_price2)
                    recommendation = get_recommendation(pct_pnl2, float(r["CAGR (decimal)"]), is_zero_price2)
                    strengths.append(strength)
                    recommendations.append(recommendation)
                
                df2["Strength"] = strengths
                df2["Recommendation"] = recommendations
                df2["Yearly Dividend (₹)"] = df2["Dividend/Share (₹)"] * df2["_qty"]

                total_inv2 = float(df2["_inv"].sum())
                total_cmp_val2 = float(df2["_cmp_total"].sum())
                total_pnl2 = float(df2["_pnl"].sum())
                total_yearly_div2 = float(df2["Yearly Dividend (₹)"].sum())

                if total_cmp_val2 > 0:
                    df2["_weight"] = df2["_cmp_total"] / total_cmp_val2
                    portfolio_cagr2 = float((df2["CAGR (decimal)"] * df2["_weight"]).sum())
                else:
                    portfolio_cagr2 = 0.05

                st.markdown("#### 🔁 Recalculated Snapshot (All values in ₹ INR)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Investment", f"₹{total_inv2:,.2f}")
                with c2:
                    st.metric("Current Value", f"₹{total_cmp_val2:,.2f}")
                with c3:
                    st.metric("Total P&L", f"₹{total_pnl2:,.2f}")

                st.markdown("#### 🔮 Updated Projections")
                proj2 = []
                for y in years_list:
                    v2 = project_value(total_cmp_val2, portfolio_cagr2, total_yearly_div2, y)
                    capital_gain2 = v2 - total_cmp_val2 - (total_yearly_div2 * y)
                    proj2.append({
                        "Years": y,
                        "Current Value (₹)": f"{total_cmp_val2:,.2f}",
                        "Total Dividends (₹)": f"{total_yearly_div2 * y:,.2f}",
                        "Capital Gain (₹)": f"{capital_gain2:,.2f}",
                        "Projected Value (₹)": f"{v2:,.2f}"
                    })
                st.table(pd.DataFrame(proj2))

                st.markdown("#### 📋 Detailed Results with Recommendations (All values in ₹ INR)")
                show_cols = [cols["stock name"], cols["quantity"], cols["total investment"],
                           cols["total cmp"], "_pnl", "Dividend/Share (₹)", "Yearly Dividend (₹)",
                           "CAGR (%)", "Strength", "Recommendation"]
                df_show = df2[show_cols].rename(columns={
                    cols["stock name"]: "Stock Name", cols["quantity"]: "Quantity",
                    cols["total investment"]: "Total Investment (₹)", cols["total cmp"]: "Total CMP (₹)",
                    "_pnl": "Total P&L (₹)"
                })
                st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("Choose your Groww CSV/XLS/XLSX file to see advanced insights here.")
    else:
        st.info("This script only displays the Groww page functionality. Other pages remain unchanged.")

if __name__ == "__main__":
    main()
