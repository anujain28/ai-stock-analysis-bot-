# scan_helpers.py
import os
from pathlib import Path
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from app import run_analysis  # CHANGE if your main scan function has a different name

IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = Path("data")
SCANS_DIR = DATA_DIR / "scans"
SCANS_DIR.mkdir(parents=True, exist_ok=True)


def is_market_time(now: dt.datetime | None = None) -> bool:
    """Return True only on Mon–Fri between 9:30 and 16:30 IST."""
    if now is None:
        now = dt.datetime.now(IST)
    # Monday=0, Sunday=6
    if now.weekday() > 4:
        return False
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return start <= now <= end


def run_scan_and_save_csv() -> str | None:
    """
    Run the existing scan (using run_analysis) and save to a timestamped CSV.
    Returns the CSV path or None if not in market time.
    """
    now = dt.datetime.now(IST)
    if not is_market_time(now):
        return None

    # 1) Run your existing scan logic.
    #    IMPORTANT: run_analysis must return a pandas DataFrame.
    df = run_analysis()  # adapt if your function has parameters

    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_analysis() must return a pandas DataFrame")

    # 2) Save to CSV
    stamp = now.strftime("%Y-%m-%d_%H-%M")
    out_path = SCANS_DIR / f"scan_{stamp}.csv"
    df.to_csv(out_path, index=False)

    return str(out_path)
