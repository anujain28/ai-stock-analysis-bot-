# scan_helpers.py
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app import run_analysis  # your existing scan function

IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = Path("data")
SCANS_DIR = DATA_DIR / "scans"
SCANS_DIR.mkdir(parents=True, exist_ok=True)

def is_market_time(now: dt.datetime | None = None) -> bool:
    """Mon–Fri between 9:30 and 16:30 IST."""
    if now is None:
        now = dt.datetime.now(IST)
    if now.weekday() > 4:
        return False
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return start <= now <= end

def run_scan_and_save_csv() -> str | None:
    """
    Run existing scan and save to timestamped CSV.
    Returns CSV path or None if not market time.
    """
    now = dt.datetime.now(IST)
    if not is_market_time(now):
        return None

    df = run_analysis()  # must return pandas DataFrame

    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_analysis() must return a pandas DataFrame")

    stamp = now.strftime("%Y-%m-%d_%H-%M")
    out_path = SCANS_DIR / f"scan_{stamp}.csv"
    df.to_csv(out_path, index=False)
    return str(out_path)
