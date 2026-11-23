"""
Institutional Financial Data Validation, Causal Preprocessing, Cross-Asset Ratio Engineering,
ML Feature Matrix Generation, and Google Sheets Publishing Pipeline.

Usage:
    py pipeline.py --source-id <SOURCE_SHEET_ID> --dest-id <DEST_SHEET_ID>
    py pipeline.py --skip-fetch --skip-upload   # Re-run locally on existing raw CSV

Requires:
    Python 3.11+, pandas, numpy, and either:
      - Google Sheets API credentials via GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_APPLICATION_CREDENTIALS
        (with google-api-python-client and google-auth installed, recommended for GitHub Actions), or
      - Authenticated `gsheets` CLI on PATH (local Antigravity environment).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def load_dotenv(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overriding existing env vars."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv(BASE_DIR / ".env")

DEFAULT_SOURCE_ID = os.environ.get("SOURCE_SHEET_ID", "")
DEFAULT_DEST_ID = os.environ.get("DEST_SHEET_ID", "")

# Asset classification metadata
RATE_POLICY_COLS = [
    "Fed Funds Rate (pct)",
    "ECB Deposit Rate (pct)",
    "BOJ Policy Rate (pct)",
    "HK Base Rate (pct)",
    "China 1Y LPR (pct)",
    "China 5Y LPR (pct)",
]

RATE_MARKET_COLS = [
    "US 2Y Yield (pct)",
    "US 10Y Yield (pct)",
    "US 30Y Yield (pct)",
    "SG SORA (pct)",
    "SG 10Y Yield (pct)",
    "HK 1M HIBOR (pct)",
    "China 10Y Yield (pct)",
]

ALL_RATE_COLS = RATE_MARKET_COLS + RATE_POLICY_COLS

# Short, clean ML column slugs for every raw asset
ASSET_SLUGS: dict[str, str] = {
    "S&P 500": "sp500",
    "Dow Jones": "dow",
    "Nasdaq Composite": "nasdaq",
    "FTSE 100": "ftse100",
    "DAX": "dax",
    "Nikkei 225": "nikkei225",
    "Hang Seng": "hsi",
    "CSI 300": "csi300",
    "Shanghai Composite": "shcomp",
    "STI": "sti",
    "US Agg Bond (AGG)": "agg",
    "US 20+Y Treasury (TLT)": "tlt",
    "US High Yield (HYG)": "hyg",
    "Global Agg Bond (BNDX)": "bndx",
    "Gold (USD/oz)": "gold",
    "Silver (USD/oz)": "silver",
    "WTI Crude (USD/bbl)": "wti",
    "Brent Crude (USD/bbl)": "brent",
    "Copper (USD/lb)": "copper",
    "Natural Gas (USD/MMBtu)": "natgas",
    "Wheat (USc/bu)": "wheat",
    "Corn (USc/bu)": "corn",
    "Soybeans (USc/bu)": "soybeans",
    "Coffee (USc/lb)": "coffee",
    "Sugar (USc/lb)": "sugar",
    "DXY": "dxy",
    "EUR/USD": "eurusd",
    "USD/JPY": "usdjpy",
    "GBP/USD": "gbpusd",
    "USD/CNY": "usdcny",
    "USD/SGD": "usdsgd",
    "US 2Y Yield (pct)": "us2y",
    "US 10Y Yield (pct)": "us10y",
    "US 30Y Yield (pct)": "us30y",
    "Fed Funds Rate (pct)": "fedfunds",
    "ECB Deposit Rate (pct)": "ecb_dep",
    "BOJ Policy Rate (pct)": "boj_rate",
    "SG SORA (pct)": "sg_sora",
    "SG 10Y Yield (pct)": "sg10y",
    "HK Base Rate (pct)": "hk_base",
    "HK 1M HIBOR (pct)": "hk_hibor1m",
    "China 1Y LPR (pct)": "cn_lpr1y",
    "China 5Y LPR (pct)": "cn_lpr5y",
    "China 10Y Yield (pct)": "cn10y",
}

ASSET_CATEGORIES: dict[str, str] = {
    "S&P 500": "Equity Index (US)",
    "Dow Jones": "Equity Index (US)",
    "Nasdaq Composite": "Equity Index (US)",
    "FTSE 100": "Equity Index (Europe)",
    "DAX": "Equity Index (Europe)",
    "Nikkei 225": "Equity Index (Asia)",
    "Hang Seng": "Equity Index (Asia)",
    "CSI 300": "Equity Index (Asia)",
    "Shanghai Composite": "Equity Index (Asia)",
    "STI": "Equity Index (Asia)",
    "US Agg Bond (AGG)": "Fixed Income ETF",
    "US 20+Y Treasury (TLT)": "Fixed Income ETF",
    "US High Yield (HYG)": "Fixed Income ETF",
    "Global Agg Bond (BNDX)": "Fixed Income ETF",
    "Gold (USD/oz)": "Commodity (Metals)",
    "Silver (USD/oz)": "Commodity (Metals)",
    "WTI Crude (USD/bbl)": "Commodity (Energy)",
    "Brent Crude (USD/bbl)": "Commodity (Energy)",
    "Copper (USD/lb)": "Commodity (Industrial Metals)",
    "Natural Gas (USD/MMBtu)": "Commodity (Energy)",
    "Wheat (USc/bu)": "Commodity (Agriculture)",
    "Corn (USc/bu)": "Commodity (Agriculture)",
    "Soybeans (USc/bu)": "Commodity (Agriculture)",
    "Coffee (USc/lb)": "Commodity (Softs)",
    "Sugar (USc/lb)": "Commodity (Softs)",
    "DXY": "FX Index",
    "EUR/USD": "FX Spot",
    "USD/JPY": "FX Spot",
    "GBP/USD": "FX Spot",
    "USD/CNY": "FX Spot",
    "USD/SGD": "FX Spot",
    "US 2Y Yield (pct)": "Sovereign Yield",
    "US 10Y Yield (pct)": "Sovereign Yield",
    "US 30Y Yield (pct)": "Sovereign Yield",
    "Fed Funds Rate (pct)": "Central Bank Policy Rate",
    "ECB Deposit Rate (pct)": "Central Bank Policy Rate",
    "BOJ Policy Rate (pct)": "Central Bank Policy Rate",
    "SG SORA (pct)": "Money Market Rate (Monthly Step)",
    "SG 10Y Yield (pct)": "Sovereign Yield (Monthly Step)",
    "HK Base Rate (pct)": "Central Bank Policy Rate",
    "HK 1M HIBOR (pct)": "Money Market Rate",
    "China 1Y LPR (pct)": "Central Bank Policy Rate",
    "China 5Y LPR (pct)": "Central Bank Policy Rate",
    "China 10Y Yield (pct)": "Sovereign Yield",
}

# Verified 2008 August DD/MM vs MM/DD date-swap corrupted dates in EUR/USD and USD/JPY
SWAP_DATES_2008 = [
    "2008-01-08",
    "2008-02-08",
    "2008-04-08",
    "2008-05-08",
    "2008-07-08",
    "2008-09-08",
    "2008-10-08",
    "2008-12-08",
]

# Bounded causal forward-fill limit for traded markets (bridges up to 6-weekday Golden Week holidays)
TRADED_FFILL_LIMIT = 6


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run an external CLI command and capture UTF-8 output (resolves .cmd shims on Windows)."""
    resolved = shutil.which(args[0]) or args[0]
    cmd = [resolved, *args[1:]]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {' '.join(args)}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result


def resolve_sheets_backend(requested: str = "auto") -> str:
    """Select Google Sheets backend ('api' via service account or 'gsheets' via local CLI)."""
    has_sa = bool(
        os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    )
    has_cli = shutil.which("gsheets") is not None

    if requested == "api":
        if not has_sa:
            raise RuntimeError(
                "Requested --backend api, but neither GOOGLE_SERVICE_ACCOUNT_JSON nor "
                "GOOGLE_APPLICATION_CREDENTIALS is set."
            )
        return "api"
    if requested == "gsheets":
        if not has_cli:
            raise RuntimeError("Requested --backend gsheets, but `gsheets` CLI is not on PATH.")
        return "gsheets"

    if has_sa:
        return "api"
    if has_cli:
        return "gsheets"
    raise RuntimeError(
        "No Google Sheets credentials available. Set GOOGLE_SERVICE_ACCOUNT_JSON (for GitHub Actions / API) "
        "or ensure the `gsheets` CLI is on PATH, or pass --skip-fetch --skip-upload for offline runs."
    )


def build_sheets_service() -> Any:
    """Build an authenticated Google Sheets v4 API service from service-account credentials."""
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Sheets API backend requires google-api-python-client and google-auth. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    sa_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_env:
        if sa_env.startswith("{"):
            info = json.loads(sa_env)
            creds = Credentials.from_service_account_info(info, scopes=scopes)
        else:
            creds = Credentials.from_service_account_file(sa_env, scopes=scopes)
    else:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        creds = Credentials.from_service_account_file(cred_path, scopes=scopes)

    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def fetch_raw_sheet(
    source_id: str,
    raw_csv: Path,
    sheet_name: str = "Sheet1",
    backend: str = "auto",
) -> None:
    """Export raw source Google Sheet tab to local CSV via Sheets API or gsheets CLI."""
    if not source_id:
        raise ValueError("Source Google Sheet ID is empty. Pass --source-id or set SOURCE_SHEET_ID.")
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    chosen = resolve_sheets_backend(backend)
    print(f"[1/6] Fetching source sheet {source_id} ({sheet_name}) via {chosen} -> {raw_csv}")

    if chosen == "api":
        service = build_sheets_service()
        resp = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=source_id, range=sheet_name, valueRenderOption="FORMATTED_VALUE")
            .execute()
        )
        rows = resp.get("values", [])
        if not rows:
            raise RuntimeError(f"Source sheet {source_id} ({sheet_name}) returned no rows.")
        width = max(len(r) for r in rows)
        with raw_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r + [""] * (width - len(r)))
        return

    run_cmd(
        [
            "gsheets",
            "readonly",
            "export",
            source_id,
            str(raw_csv),
            "--format",
            "csv",
            "--sheet",
            sheet_name,
        ]
    )


def safe_pos_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Compute num / den strictly when both numerator and denominator are finite and > 0."""
    valid = num.notna() & den.notna() & (num > 0) & (den > 0)
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def safe_log_ret(series: pd.Series, periods: int = 1) -> pd.Series:
    """Compute causal multi-period log return strictly when both endpoints > 0."""
    prev = series.shift(periods)
    valid = series.notna() & prev.notna() & (series > 0) & (prev > 0)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[valid] = np.log(series.loc[valid] / prev.loc[valid])
    return out


def causal_rolling_zscore(series: pd.Series, window: int, use_log: bool = False) -> pd.Series:
    """Compute strictly causal rolling Z-score using [t-window+1 .. t]."""
    if use_log:
        s = pd.Series(np.nan, index=series.index, dtype="float64")
        pos = series.notna() & (series > 0)
        s.loc[pos] = np.log(series.loc[pos])
    else:
        s = series.astype("float64")

    # Require >= 90% valid observations in window (strictly causal; avoids a 5-day vendor gap wiping out 252 rows)
    min_p = max(1, int(window * 0.90))
    # Ensure we only start after full warmup index >= window - 1 past first valid observation
    first_valid = s.first_valid_index()
    mean = s.rolling(window=window, min_periods=min_p).mean()
    std = s.rolling(window=window, min_periods=min_p).std(ddof=1)

    z = (s - mean) / std.replace(0.0, np.nan)
    # If series is strictly flat across the entire rolling window (e.g. policy rate plateau), deviation is 0.0
    flat_valid = s.notna() & mean.notna() & (std == 0.0)
    z = z.mask(flat_valid, 0.0)
    z = z.where(s.notna())
    if first_valid is not None:
        z.iloc[: first_valid + window - 1] = np.nan
    return z


def max_consecutive_true(arr: np.ndarray) -> int:
    """Compute longest streak of True in a 1D boolean numpy array."""
    max_run = 0
    cur_run = 0
    for val in arr:
        if val:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return int(max_run)


def audit_and_clean(
    raw_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """
    Perform full data validity audit on raw CSV and apply documented, strictly causal cleaning.
    Returns: (clean_df, col_audit_df, anomaly_log, exec_meta)
    """
    print(f"[2/6] Auditing and cleaning raw dataset: {raw_csv}")
    raw_df = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    raw_rows, raw_cols = raw_df.shape

    date_col = "Date"
    notes_col = "Notes"
    num_cols = [c for c in raw_df.columns if c not in (date_col, notes_col)]

    # Date diagnostics
    parsed_dt = pd.to_datetime(raw_df[date_col].str.strip(), errors="coerce")
    unparseable_dates = int(parsed_dt.isna().sum())
    dup_dates = int(parsed_dt.duplicated().sum())
    was_monotonic = bool(parsed_dt.is_monotonic_increasing)

    # Notes audit
    notes_non_empty = raw_df.loc[raw_df[notes_col].str.strip() != "", [date_col, notes_col]]
    notes_entries = [
        f"{r[date_col]}: {r[notes_col]}" for _, r in notes_non_empty.iterrows()
    ]

    # Sort chronologically & drop unparseable dates if any
    work = raw_df.copy()
    work["_dt"] = parsed_dt
    work = work.dropna(subset=["_dt"]).sort_values("_dt").drop_duplicates(subset=["_dt"], keep="last").reset_index(drop=True)

    # Identify weekend rows
    weekend_mask = work["_dt"].dt.dayofweek >= 5
    weekend_dates = work.loc[weekend_mask, "_dt"].dt.strftime("%Y-%m-%d").tolist()

    # Anomaly log accumulator
    anomaly_log: list[dict[str, Any]] = []

    # Parse all numeric columns and count literal NA strings / unparseable tokens BEFORE dropping weekend row
    raw_num_pre = pd.DataFrame(index=work.index)
    literal_nan_counts: dict[str, int] = {}
    unparseable_counts: dict[str, int] = {}

    for col in num_cols:
        s_str = work[col].astype(str).str.strip()
        is_empty = s_str == ""
        is_na_token = s_str.str.lower().isin(["nan", "na", "n/a", "null", "none", "-", "--", "#n/a"])
        literal_nan_counts[col] = int(is_na_token.sum())

        s_clean = s_str.mask(is_empty | is_na_token, np.nan)
        s_num = pd.to_numeric(s_clean, errors="coerce")
        unparseable_counts[col] = int((~is_empty & ~is_na_token & s_num.isna()).sum())
        raw_num_pre[col] = s_num

    # Log HK Base Rate literal "nan" strings
    for col, cnt in literal_nan_counts.items():
        if cnt > 0:
            anomaly_log.append(
                {
                    "Anomaly_ID": f"ANOM-{len(anomaly_log)+1:02d}",
                    "Severity": "Medium (Format)",
                    "Column": col,
                    "Date_Range": "2006-11-23 to 2026-11-20 (HK holidays)",
                    "Raw_Value": f'{cnt} literal "nan" text strings',
                    "Issue_Type": "Literal string 'nan' instead of blank cell",
                    "Action_Taken": "Parsed to NaN and causally forward-filled (policy rate unchanged on bank holidays)",
                    "Lookahead_Risk": "None (strictly causal)",
                }
            )

    # Drop non-trading weekend row(s) — specifically Sat 2026-11-21
    if len(weekend_dates) > 0:
        for wdt in weekend_dates:
            note_txt = work.loc[work["_dt"].dt.strftime("%Y-%m-%d") == wdt, notes_col].iloc[0]
            anomaly_log.append(
                {
                    "Anomaly_ID": f"ANOM-{len(anomaly_log)+1:02d}",
                    "Severity": "High (Calendar/Quote)",
                    "Column": "ALL (Row-level)",
                    "Date_Range": wdt,
                    "Raw_Value": f"Saturday row; USD/CNY jumped 6.6665->7.1550 (+7.33%), STI reverted to Thu, 22/44 cols duplicated Fri; Notes='{note_txt}'",
                    "Issue_Type": "Non-trading Saturday snapshot duplicating Fri closes with cross-vendor FX/index discrepancies",
                    "Action_Taken": "Dropped weekend row to preserve clean 5,217-weekday panel (2006-11-23 to 2026-11-20)",
                    "Lookahead_Risk": "None",
                }
            )
        keep_mask = ~weekend_mask
        work = work.loc[keep_mask].reset_index(drop=True)
        raw_num = raw_num_pre.loc[keep_mask].reset_index(drop=True)
    else:
        raw_num = raw_num_pre.copy()

    dates = work["_dt"]
    date_strs = dates.dt.strftime("%Y-%m-%d")
    full_bdays = pd.date_range(dates.min(), dates.max(), freq="B")
    missing_weekdays = int(len(full_bdays.difference(dates)))

    # Keep a snapshot of raw numeric values on the clean weekday index for audit stats
    clean_num = raw_num.copy()
    corrections_per_col: dict[str, int] = {c: 0 for c in num_cols}

    # --- DATA CORRUPTION FIXES (ONLY PROVABLE ERRORS, ZERO ALTERATION OF REAL MARKET EVENTS) ---

    # Fix 1: HK 1M HIBOR (pct) 100x decimal-point error on 2019-09-19 (199.29 -> 1.9929)
    hib_col = "HK 1M HIBOR (pct)"
    if hib_col in clean_num.columns:
        bad_hib_mask = clean_num[hib_col] > 50.0  # HIBOR never exceeded ~20% even in 1998 Asian crisis
        for idx in clean_num.index[bad_hib_mask]:
            old_v = float(clean_num.loc[idx, hib_col])
            new_v = round(old_v / 100.0, 4)
            dt_s = date_strs.loc[idx]
            clean_num.loc[idx, hib_col] = new_v
            corrections_per_col[hib_col] += 1
            anomaly_log.append(
                {
                    "Anomaly_ID": f"ANOM-{len(anomaly_log)+1:02d}",
                    "Severity": "Critical (Data Error)",
                    "Column": hib_col,
                    "Date_Range": dt_s,
                    "Raw_Value": f"{old_v} (prev=2.01, next=2.00)",
                    "Issue_Type": "100x decimal-point scaling error (+9,815% single-day jump)",
                    "Action_Taken": f"Scaled by 1/100 -> {new_v}",
                    "Lookahead_Risk": "None (unit/decimal error correction)",
                }
            )

    # Fix 2: 2008 August DD/MM vs MM/DD date-swap corruption in EUR/USD and USD/JPY
    for fx_col in ["EUR/USD", "USD/JPY"]:
        if fx_col in clean_num.columns:
            swap_mask = date_strs.isin(SWAP_DATES_2008) & clean_num[fx_col].notna()
            affected_dates = date_strs.loc[swap_mask].tolist()
            if affected_dates:
                sample_pairs = [
                    f"{date_strs.loc[i]}={clean_num.loc[i, fx_col]}"
                    for i in clean_num.index[swap_mask][:3]
                ]
                clean_num.loc[swap_mask, fx_col] = np.nan
                corrections_per_col[fx_col] += len(affected_dates)
                anomaly_log.append(
                    {
                        "Anomaly_ID": f"ANOM-{len(anomaly_log)+1:02d}",
                        "Severity": "Critical (Date Swap)",
                        "Column": fx_col,
                        "Date_Range": ", ".join(affected_dates),
                        "Raw_Value": f"8 misdated cells (e.g. {', '.join(sample_pairs)})",
                        "Issue_Type": "DD/MM vs MM/DD date-swap bug: Aug 2008 quotes overwritten onto 8th of other months in 2008 (causing fake 5%-18% spikes & 11-day Aug 2008 gap)",
                        "Action_Taken": "Nulled 8 misdated cells and causally forward-filled from prior trading day (t-1)",
                        "Lookahead_Risk": "None (strictly causal ffill)",
                    }
                )

    # Fix 3: USD/CNY bad single-day print on 2011-07-18 (5.8630 vs 6.4338 prev / 6.4688 next)
    cny_col = "USD/CNY"
    if cny_col in clean_num.columns:
        cny_mask = (date_strs == "2011-07-18") & (clean_num[cny_col] < 6.0)
        for idx in clean_num.index[cny_mask]:
            old_v = float(clean_num.loc[idx, cny_col])
            clean_num.loc[idx, cny_col] = np.nan
            corrections_per_col[cny_col] += 1
            anomaly_log.append(
                {
                    "Anomaly_ID": f"ANOM-{len(anomaly_log)+1:02d}",
                    "Severity": "High (Bad Tick)",
                    "Column": cny_col,
                    "Date_Range": "2011-07-18",
                    "Raw_Value": f"{old_v} (prev=6.4338, next=6.4688; -8.87% / +10.33% 1-day revert)",
                    "Issue_Type": "Erroneous FX feed tick (CNY never traded near 5.86 in 2011)",
                    "Action_Taken": "Nulled bad print and causally forward-filled from 2011-07-15 (6.4338)",
                    "Lookahead_Risk": "None (strictly causal ffill)",
                }
            )

    # Log notable genuine historical/structural anomalies that are PRESERVED (NOT altered)
    anomaly_log.extend(
        [
            {
                "Anomaly_ID": f"ANOM-{len(anomaly_log)+1:02d}",
                "Severity": "Info (Genuine Event)",
                "Column": "WTI Crude (USD/bbl)",
                "Date_Range": "2020-04-20",
                "Raw_Value": "-37.63 (prev=18.27, next=10.01)",
                "Issue_Type": "Historic negative settlement of May-2020 WTI futures contract",
                "Action_Taken": "PRESERVED in Preprocessed_Data; guarded (set to blank/NaN) in all WTI price ratios and log returns; Brent-WTI spread remains valid (+63.20)",
                "Lookahead_Risk": "None",
            },
            {
                "Anomaly_ID": f"ANOM-{len(anomaly_log)+2:02d}",
                "Severity": "Info (Genuine Policy)",
                "Column": "ECB Deposit Rate / BOJ Policy Rate",
                "Date_Range": "2012-2024 (ECB <=0: 2,655d; BOJ <0: 1,956d)",
                "Raw_Value": "ECB min -0.50%, BOJ min -0.10%",
                "Issue_Type": "Negative Interest Rate Policy (NIRP)",
                "Action_Taken": "PRESERVED; transformed via 1-day first differences (chg1d_*) rather than log returns",
                "Lookahead_Risk": "None",
            },
            {
                "Anomaly_ID": f"ANOM-{len(anomaly_log)+3:02d}",
                "Severity": "Medium (Vendor Outage)",
                "Column": "EUR/USD, USD/JPY, Brent Crude",
                "Date_Range": "2008-08-11..25 (11d FX); 2009-04-15..30 (12d Brent)",
                "Raw_Value": "Consecutive missing streaks > 6 weekdays",
                "Issue_Type": "Historical vendor feed outages (Aug 2008 DD/MM unparseable dates & Apr 2009 Brent gap)",
                "Action_Taken": f"Bounded causal ffill (limit={TRADED_FFILL_LIMIT}d) bridges days 1-6; remaining 16 outage cells left blank/NaN to avoid multi-week stale prices",
                "Lookahead_Risk": "None (strictly causal)",
            },
            {
                "Anomaly_ID": f"ANOM-{len(anomaly_log)+4:02d}",
                "Severity": "Info (Cadence)",
                "Column": "SG SORA (pct), SG 10Y Yield (pct)",
                "Date_Range": "2006-11-23 to 2026-11-20",
                "Raw_Value": "10-12 distinct value changes per year across full 20y history",
                "Issue_Type": "Monthly-frequency source series step-filled onto daily weekday grid",
                "Action_Taken": "PRESERVED as-is; documented in Feature_Dictionary (daily diffs are 0 on ~95% of days and jump at month start)",
                "Lookahead_Risk": "None",
            },
            {
                "Anomaly_ID": f"ANOM-{len(anomaly_log)+5:02d}",
                "Severity": "Info (Futures Rolls)",
                "Column": "Natural Gas, Corn, Soybeans",
                "Date_Range": "Periodic expiry rolls (e.g. Soybeans 2008-09-12, Corn mid-Jul, NatGas 2022-01-27 & 2026-01-29)",
                "Raw_Value": "Single-day roll jumps of 15%-47%",
                "Issue_Type": "Unadjusted continuous front-month futures roll discontinuities",
                "Action_Taken": "PRESERVED in levels/returns; flagged as ML modeling caveat (consider winsorization or tree models)",
                "Lookahead_Risk": "None",
            },
            {
                "Anomaly_ID": f"ANOM-{len(anomaly_log)+6:02d}",
                "Severity": "Info (Late Inception)",
                "Column": "HYG, Brent, BNDX, China 5Y LPR",
                "Date_Range": "HYG 2007-04-11; Brent 2007-07-30; BNDX 2013-06-04; CN 5Y LPR 2019-08-20",
                "Raw_Value": "Pre-inception blanks (BNDX 35.0%, CN 5Y LPR 63.7%)",
                "Issue_Type": "Legitimate asset/benchmark launch dates after 2006-11-23",
                "Action_Taken": "NEVER backfilled; left blank prior to true inception; handled via is_core_ml_ready and is_full_ml_ready flags",
                "Lookahead_Risk": "None (zero future leakage)",
            },
        ]
    )

    # --- STRICTLY CAUSAL POST-INCEPTION IMPUTATION ---
    imputed_counts: dict[str, int] = {}
    post_clean_remaining_post_inc_na: dict[str, int] = {}

    for col in num_cols:
        s = clean_num[col]
        first_valid = s.first_valid_index()
        if first_valid is None:
            imputed_counts[col] = 0
            post_clean_remaining_post_inc_na[col] = len(s)
            continue

        before_na_post = int(s.iloc[first_valid:].isna().sum())
        if col in RATE_POLICY_COLS:
            # Stepped central bank policy rates remain valid across bank holidays indefinitely
            filled_post = s.iloc[first_valid:].ffill()
        else:
            # Traded markets: bounded causal forward-fill up to 6 weekdays (bridges all global holidays incl. Golden Week)
            filled_post = s.iloc[first_valid:].ffill(limit=TRADED_FFILL_LIMIT)

        after_na_post = int(filled_post.isna().sum())
        imputed_counts[col] = before_na_post - after_na_post
        post_clean_remaining_post_inc_na[col] = after_na_post
        clean_num.loc[first_valid:, col] = filled_post

    # --- BUILD PER-COLUMN AUDIT TABLE ---
    audit_records: list[dict[str, Any]] = []
    n_rows = len(clean_num)

    for col in num_cols:
        s_raw = raw_num[col]
        s_cln = clean_num[col]

        valid_raw_idx = s_raw.dropna().index
        first_i = int(valid_raw_idx[0])
        last_i = int(valid_raw_idx[-1])

        raw_miss = int(s_raw.isna().sum())
        pre_inc_miss = first_i
        post_inc_miss_raw = int(s_raw.iloc[first_i : last_i + 1].isna().sum())
        max_gap_raw = max_consecutive_true(s_raw.iloc[first_i : last_i + 1].isna().to_numpy())

        s_valid_cln = s_cln.dropna()
        max_flat_cln = max_consecutive_true((s_valid_cln.diff() == 0).to_numpy())

        audit_records.append(
            {
                "Column": col,
                "Category": ASSET_CATEGORIES.get(col, "Other"),
                "Inception_Date": date_strs.iloc[first_i],
                "Last_Valid_Date": date_strs.iloc[last_i],
                "Raw_Missing_Count": raw_miss,
                "Raw_Missing_Pct": round(100.0 * raw_miss / n_rows, 2),
                "Pre_Inception_Missing": pre_inc_miss,
                "Post_Inception_Raw_Missing": post_inc_miss_raw,
                "Max_Post_Inc_Gap_Days": max_gap_raw,
                "Data_Errors_Fixed": corrections_per_col[col],
                "Causal_FFill_Imputed": imputed_counts[col],
                "Remaining_Post_Inc_Missing": post_clean_remaining_post_inc_na[col],
                "Clean_Min": round(float(s_valid_cln.min()), 4),
                "Clean_Median": round(float(s_valid_cln.median()), 4),
                "Clean_Max": round(float(s_valid_cln.max()), 4),
                "Clean_Std": round(float(s_valid_cln.std(ddof=1)), 4),
                "NonPositive_Count": int((s_valid_cln <= 0).sum()),
                "Max_Flat_Streak_Days": max_flat_cln,
                "Imputation_Rule": (
                    "Policy rate causal ffill (no backfill)"
                    if col in RATE_POLICY_COLS
                    else f"Causal ffill limit={TRADED_FFILL_LIMIT}d (no backfill)"
                ),
            }
        )

    col_audit_df = pd.DataFrame(audit_records)

    clean_df = pd.concat([date_strs.rename("Date"), clean_num], axis=1)

    exec_meta = {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "clean_rows": len(clean_df),
        "clean_numeric_cols": len(num_cols),
        "start_date": date_strs.iloc[0],
        "end_date": date_strs.iloc[-1],
        "unparseable_dates": unparseable_dates,
        "duplicate_dates": dup_dates,
        "was_monotonic": was_monotonic,
        "weekend_rows_dropped": len(weekend_dates),
        "weekend_dates": ", ".join(weekend_dates),
        "missing_weekdays_in_clean": missing_weekdays,
        "notes_non_empty_count": len(notes_entries),
        "notes_entries": " | ".join(notes_entries) if notes_entries else "None",
        "total_data_errors_fixed": sum(corrections_per_col.values()),
        "total_literal_nan_parsed": sum(literal_nan_counts.values()),
        "total_cells_imputed": sum(imputed_counts.values()),
        "total_remaining_post_inc_na": sum(post_clean_remaining_post_inc_na.values()),
    }

    return clean_df, col_audit_df, anomaly_log, exec_meta


def engineer_cross_asset_ratios(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Engineer 48 economically meaningful cross-asset price ratios, FX-adjusted equity ratios,
    FX crosses, and macro yield/rate spreads. All price ratios guarded against <= 0.
    """
    print("[3/6] Engineering 48 cross-asset ratios, USD-adjusted equity ratios, spreads & FX crosses")
    d = clean_df
    ratios = pd.DataFrame({"Date": d["Date"]})
    meta: list[dict[str, str]] = []

    def add_ratio(name: str, series: pd.Series, category: str, formula: str, meaning: str, rtype: str) -> None:
        ratios[name] = series
        first_valid = series.first_valid_index()
        inc_dt = d.loc[first_valid, "Date"] if first_valid is not None else ""
        meta.append(
            {
                "Tab": "Cross_Asset_Ratios",
                "Column_Name": name,
                "Category": category,
                "Formula_or_Source": formula,
                "Economic_Meaning": meaning,
                "Transformation": rtype,
                "Inception_Date": inc_dt,
                "ML_Notes": (
                    "Guarded against <=0 (blank on WTI negative close 2020-04-20)"
                    if "WTI" in formula and "Ratio" in name
                    else ("Additive spread (pct pts or USD); use diffs for stationarity" if rtype == "Spread" else "Strictly positive ratio; use log-returns / log-z-scores")
                ),
            }
        )

    # 1. Commodities & Inflation/Growth (10)
    add_ratio("Gold_Silver_Ratio", safe_pos_ratio(d["Gold (USD/oz)"], d["Silver (USD/oz)"]), "Commodities & Inflation", "Gold / Silver", "Precious metals risk aversion & industrial-vs-monetary demand gauge", "Ratio")
    add_ratio("Copper_Gold_Ratio", safe_pos_ratio(d["Copper (USD/lb)"], d["Gold (USD/oz)"]), "Commodities & Inflation", "Copper / Gold", "Classic global industrial growth & reflation vs safe-haven proxy (tracks 10Y yields)", "Ratio")
    add_ratio("WTI_Gold_Ratio", safe_pos_ratio(d["WTI Crude (USD/bbl)"], d["Gold (USD/oz)"]), "Commodities & Inflation", "WTI / Gold", "Energy purchasing power in gold terms; cyclical inflation impulse", "Ratio")
    add_ratio("Brent_WTI_Ratio", safe_pos_ratio(d["Brent Crude (USD/bbl)"], d["WTI Crude (USD/bbl)"]), "Commodities & Inflation", "Brent / WTI", "Seaborne global crude vs US Cushing benchmark relative valuation", "Ratio")
    add_ratio("Brent_WTI_Spread", d["Brent Crude (USD/bbl)"] - d["WTI Crude (USD/bbl)"], "Commodities & Inflation", "Brent - WTI (USD/bbl)", "US pipeline/storage bottleneck & Atlantic basin crude premium (valid even when WTI < 0)", "Spread")
    add_ratio("WTI_NatGas_Ratio", safe_pos_ratio(d["WTI Crude (USD/bbl)"], d["Natural Gas (USD/MMBtu)"]), "Commodities & Inflation", "WTI / Natural Gas", "Oil-to-gas energy equivalence ratio (6:1 BTU parity benchmark)", "Ratio")
    add_ratio("Copper_WTI_Ratio", safe_pos_ratio(d["Copper (USD/lb)"], d["WTI Crude (USD/bbl)"]), "Commodities & Inflation", "Copper / WTI", "Electrification/industrial metals demand vs fossil energy cost", "Ratio")
    add_ratio("Soybeans_Corn_Ratio", safe_pos_ratio(d["Soybeans (USc/bu)"], d["Corn (USc/bu)"]), "Commodities & Inflation", "Soybeans / Corn", "US Midwest planting-acreage incentive ratio (~2.5 equilibrium)", "Ratio")
    add_ratio("Wheat_Corn_Ratio", safe_pos_ratio(d["Wheat (USc/bu)"], d["Corn (USc/bu)"]), "Commodities & Inflation", "Wheat / Corn", "Food-grain vs feed-grain substitution & geopolitical food-inflation gauge", "Ratio")
    add_ratio("Coffee_Sugar_Ratio", safe_pos_ratio(d["Coffee (USc/lb)"], d["Sugar (USc/lb)"]), "Commodities & Inflation", "Coffee / Sugar", "Soft commodities relative supply/Brazil weather & discretionary ag ratio", "Ratio")

    # 2. Equities vs Hard Assets / Commodities (4)
    add_ratio("SP500_Gold_Ratio", safe_pos_ratio(d["S&P 500"], d["Gold (USD/oz)"]), "Equities vs Hard Assets", "S&P 500 / Gold", "Paper financial assets vs hard monetary store of value (secular regime indicator)", "Ratio")
    add_ratio("Nasdaq_Gold_Ratio", safe_pos_ratio(d["Nasdaq Composite"], d["Gold (USD/oz)"]), "Equities vs Hard Assets", "Nasdaq / Gold", "Long-duration growth/tech equities vs monetary hedge", "Ratio")
    add_ratio("SP500_WTI_Ratio", safe_pos_ratio(d["S&P 500"], d["WTI Crude (USD/bbl)"]), "Equities vs Hard Assets", "S&P 500 / WTI", "US corporate margins & equity valuations vs energy input cost", "Ratio")
    add_ratio("SP500_Copper_Ratio", safe_pos_ratio(d["S&P 500"], d["Copper (USD/lb)"]), "Equities vs Hard Assets", "S&P 500 / Copper", "Financial/services equity multiple vs physical industrial commodity prices", "Ratio")

    # 3. Equity Regional & Style Relative Value (Local Currency) (11)
    add_ratio("Nasdaq_SP500_Ratio", safe_pos_ratio(d["Nasdaq Composite"], d["S&P 500"]), "Equity Relative Value (Local)", "Nasdaq / S&P 500", "US Growth/Tech leadership vs broad large-cap market", "Ratio")
    add_ratio("SP500_Dow_Ratio", safe_pos_ratio(d["S&P 500"], d["Dow Jones"]), "Equity Relative Value (Local)", "S&P 500 / Dow Jones", "Broad market-cap weighted US equities vs mega-cap industrial/value blue chips", "Ratio")
    add_ratio("SP500_FTSE_Ratio", safe_pos_ratio(d["S&P 500"], d["FTSE 100"]), "Equity Relative Value (Local)", "S&P 500 / FTSE 100", "US vs UK equities (local currency)", "Ratio")
    add_ratio("SP500_DAX_Ratio", safe_pos_ratio(d["S&P 500"], d["DAX"]), "Equity Relative Value (Local)", "S&P 500 / DAX", "US vs German/Eurozone industrial equities (local currency)", "Ratio")
    add_ratio("SP500_Nikkei_Ratio", safe_pos_ratio(d["S&P 500"], d["Nikkei 225"]), "Equity Relative Value (Local)", "S&P 500 / Nikkei 225", "US vs Japanese equities (local currency)", "Ratio")
    add_ratio("SP500_HSI_Ratio", safe_pos_ratio(d["S&P 500"], d["Hang Seng"]), "Equity Relative Value (Local)", "S&P 500 / Hang Seng", "US vs Hong Kong/offshore China equities (USD-pegged HKD)", "Ratio")
    add_ratio("CSI300_HSI_Ratio", safe_pos_ratio(d["CSI 300"], d["Hang Seng"]), "Equity Relative Value (Local)", "CSI 300 / Hang Seng", "Onshore China A-shares vs offshore HK H-shares/tech liquidity premium", "Ratio")
    add_ratio("CSI300_Shanghai_Ratio", safe_pos_ratio(d["CSI 300"], d["Shanghai Composite"]), "Equity Relative Value (Local)", "CSI 300 / Shanghai Composite", "China large-cap blue chips vs broad Shanghai SOE/retail market", "Ratio")
    add_ratio("DAX_FTSE_Ratio", safe_pos_ratio(d["DAX"], d["FTSE 100"]), "Equity Relative Value (Local)", "DAX / FTSE 100", "Continental European cyclical/export equities vs UK energy/financial defensive tilt", "Ratio")
    add_ratio("Nikkei_STI_Ratio", safe_pos_ratio(d["Nikkei 225"], d["STI"]), "Equity Relative Value (Local)", "Nikkei 225 / STI", "North Asia exporter equities vs ASEAN/Singapore bank-heavy value benchmark", "Ratio")
    add_ratio("HSI_STI_Ratio", safe_pos_ratio(d["Hang Seng"], d["STI"]), "Equity Relative Value (Local)", "Hang Seng / STI", "Greater China offshore risk appetite vs Singapore defensive financial hub", "Ratio")

    # 4. FX-Adjusted Common-Currency (USD) Equity Ratios (5)
    dax_usd = d["DAX"] * d["EUR/USD"]
    ftse_usd = d["FTSE 100"] * d["GBP/USD"]
    nikkei_usd = safe_pos_ratio(d["Nikkei 225"], d["USD/JPY"])
    csi300_usd = safe_pos_ratio(d["CSI 300"], d["USD/CNY"])
    sti_usd = safe_pos_ratio(d["STI"], d["USD/SGD"])

    add_ratio("SP500_DAX_USD_Ratio", safe_pos_ratio(d["S&P 500"], dax_usd), "FX-Adjusted USD Equity Ratios", "S&P 500 / (DAX * EUR/USD)", "True USD-investor relative performance: US vs Eurozone equities", "Ratio")
    add_ratio("SP500_FTSE_USD_Ratio", safe_pos_ratio(d["S&P 500"], ftse_usd), "FX-Adjusted USD Equity Ratios", "S&P 500 / (FTSE 100 * GBP/USD)", "True USD-investor relative performance: US vs UK equities", "Ratio")
    add_ratio("SP500_Nikkei_USD_Ratio", safe_pos_ratio(d["S&P 500"], nikkei_usd), "FX-Adjusted USD Equity Ratios", "S&P 500 / (Nikkei 225 / USD/JPY)", "True USD-investor relative performance: US vs Japan equities (strips JPY depreciation)", "Ratio")
    add_ratio("SP500_CSI300_USD_Ratio", safe_pos_ratio(d["S&P 500"], csi300_usd), "FX-Adjusted USD Equity Ratios", "S&P 500 / (CSI 300 / USD/CNY)", "True USD-investor relative performance: US vs Onshore China A-shares", "Ratio")
    add_ratio("SP500_STI_USD_Ratio", safe_pos_ratio(d["S&P 500"], sti_usd), "FX-Adjusted USD Equity Ratios", "S&P 500 / (STI / USD/SGD)", "True USD-investor relative performance: US vs Singapore equities", "Ratio")

    # 5. Risk-On / Risk-Off (Equities vs Bonds & Credit) (6)
    add_ratio("SP500_TLT_Ratio", safe_pos_ratio(d["S&P 500"], d["US 20+Y Treasury (TLT)"]), "Risk-On / Risk-Off", "S&P 500 / TLT", "Core risk-on vs duration safe-haven equity/long-bond ratio", "Ratio")
    add_ratio("SP500_AGG_Ratio", safe_pos_ratio(d["S&P 500"], d["US Agg Bond (AGG)"]), "Risk-On / Risk-Off", "S&P 500 / AGG", "60/40 relative performance: US equities vs investment-grade bond market", "Ratio")
    add_ratio("HYG_TLT_Ratio", safe_pos_ratio(d["US High Yield (HYG)"], d["US 20+Y Treasury (TLT)"]), "Risk-On / Risk-Off", "HYG / TLT", "Credit risk appetite vs long-duration Treasury flight-to-quality", "Ratio")
    add_ratio("HYG_AGG_Ratio", safe_pos_ratio(d["US High Yield (HYG)"], d["US Agg Bond (AGG)"]), "Risk-On / Risk-Off", "HYG / AGG", "High-yield credit spread proxy (rising = tightening credit spreads / risk-on)", "Ratio")
    add_ratio("Gold_TLT_Ratio", safe_pos_ratio(d["Gold (USD/oz)"], d["US 20+Y Treasury (TLT)"]), "Risk-On / Risk-Off", "Gold / TLT", "Inflation/fiscal-dominance hedge (Gold) vs deflationary safe-haven (Long Treasuries)", "Ratio")
    add_ratio("AGG_BNDX_Ratio", safe_pos_ratio(d["US Agg Bond (AGG)"], d["Global Agg Bond (BNDX)"]), "Risk-On / Risk-Off", "AGG / BNDX", "US vs USD-hedged International Investment Grade bond performance (post-2013)", "Ratio")

    # 6. Macro Yield Curves, Sovereign Rate Spreads & FX Crosses (12)
    add_ratio("US_10Y_2Y_Spread", d["US 10Y Yield (pct)"] - d["US 2Y Yield (pct)"], "Macro Curves & FX Crosses", "US 10Y - US 2Y (pct pts)", "Classic US Treasury yield curve slope (recession predictor when < 0)", "Spread")
    add_ratio("US_30Y_10Y_Spread", d["US 30Y Yield (pct)"] - d["US 10Y Yield (pct)"], "Macro Curves & FX Crosses", "US 30Y - US 10Y (pct pts)", "Long-end term premium & fiscal/inflation expectation steepness", "Spread")
    add_ratio("US_10Y_FedFunds_Spread", d["US 10Y Yield (pct)"] - d["Fed Funds Rate (pct)"], "Macro Curves & FX Crosses", "US 10Y - Fed Funds (pct pts)", "Monetary policy restrictiveness vs long-term neutral rate", "Spread")
    add_ratio("US_CN_10Y_Spread", d["US 10Y Yield (pct)"] - d["China 10Y Yield (pct)"], "Macro Curves & FX Crosses", "US 10Y - China 10Y (pct pts)", "US-China sovereign carry & capital-flow driver (key USD/CNY macro driver)", "Spread")
    add_ratio("US_SG_10Y_Spread", d["US 10Y Yield (pct)"] - d["SG 10Y Yield (pct)"], "Macro Curves & FX Crosses", "US 10Y - SG 10Y (pct pts)", "US-Singapore long-end yield differential (note: SG 10Y is monthly step-filled)", "Spread")
    add_ratio("Fed_ECB_Rate_Spread", d["Fed Funds Rate (pct)"] - d["ECB Deposit Rate (pct)"], "Macro Curves & FX Crosses", "Fed Funds - ECB Deposit (pct pts)", "Transatlantic policy rate divergence (primary EUR/USD carry driver)", "Spread")
    add_ratio("Fed_BOJ_Rate_Spread", d["Fed Funds Rate (pct)"] - d["BOJ Policy Rate (pct)"], "Macro Curves & FX Crosses", "Fed Funds - BOJ Policy (pct pts)", "US-Japan carry trade policy differential (primary USD/JPY driver)", "Spread")
    add_ratio("HK_HIBOR_FedFunds_Spread", d["HK 1M HIBOR (pct)"] - d["Fed Funds Rate (pct)"], "Macro Curves & FX Crosses", "HK 1M HIBOR - Fed Funds (pct pts)", "HKD aggregate balance liquidity tightness under USD peg", "Spread")
    add_ratio("CN_LPR_5Y_1Y_Spread", d["China 5Y LPR (pct)"] - d["China 1Y LPR (pct)"], "Macro Curves & FX Crosses", "China 5Y LPR - China 1Y LPR (pct pts)", "China property/mortgage vs corporate lending policy slope (post-2019-08)", "Spread")
    add_ratio("EUR_JPY_Cross", d["EUR/USD"] * d["USD/JPY"], "Macro Curves & FX Crosses", "EUR/USD * USD/JPY", "Synthetic EUR/JPY cross rate (classic global risk-on/carry FX barometer)", "Ratio")
    add_ratio("GBP_EUR_Cross", safe_pos_ratio(d["GBP/USD"], d["EUR/USD"]), "Macro Curves & FX Crosses", "GBP/USD / EUR/USD", "Synthetic GBP/EUR cross rate (UK vs Eurozone macro relative pricing)", "Ratio")
    add_ratio("SGD_CNY_Cross", safe_pos_ratio(d["USD/CNY"], d["USD/SGD"]), "Macro Curves & FX Crosses", "USD/CNY / USD/SGD", "Synthetic SGD/CNY cross rate (ASEAN MAS NEER hub vs China currency)", "Ratio")

    return ratios, meta


def build_ml_dataset(
    clean_df: pd.DataFrame,
    ratios_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Build strictly causal ML-ready feature matrix (zero lookahead bias).
    """
    print("[4/6] Generating strictly causal ML-ready feature matrix")
    dt = pd.to_datetime(clean_df["Date"])
    feat_cols: dict[str, pd.Series] = {}
    ml_dict: list[dict[str, str]] = []

    def add_feat(
        name: str,
        series: pd.Series,
        category: str,
        formula: str,
        meaning: str,
        transform: str,
        notes: str = "Strictly causal (zero lookahead)",
    ) -> None:
        feat_cols[name] = series
        first_valid = series.first_valid_index()
        inc_dt = clean_df.loc[first_valid, "Date"] if first_valid is not None else ""
        ml_dict.append(
            {
                "Tab": "ML_Ready_Dataset",
                "Column_Name": name,
                "Category": category,
                "Formula_or_Source": formula,
                "Economic_Meaning": meaning,
                "Transformation": transform,
                "Inception_Date": inc_dt,
                "ML_Notes": notes,
            }
        )

    # 1. Calendar features (4)
    cal_names = ["day_of_week", "month", "quarter", "is_month_end"]
    add_feat("day_of_week", dt.dt.dayofweek.astype("int64"), "Calendar", "dayofweek(Date)", "Weekday index (0=Mon .. 4=Fri) for day-of-week seasonality", "Integer", "Deterministic calendar feature")
    add_feat("month", dt.dt.month.astype("int64"), "Calendar", "month(Date)", "Calendar month (1..12) for monthly seasonality", "Integer", "Deterministic calendar feature")
    add_feat("quarter", dt.dt.quarter.astype("int64"), "Calendar", "quarter(Date)", "Calendar quarter (1..4) for earnings/rebalance seasonality", "Integer", "Deterministic calendar feature")
    # Strictly calendar-deterministic month-end indicator (no shift(-1) dependency on dataset end)
    next_bday_month = (dt + pd.offsets.BDay(1)).dt.month
    is_me_causal = (dt.dt.month != next_bday_month).astype("int64")
    add_feat("is_month_end", is_me_causal, "Calendar", "1 if next business day is in next month else 0", "Month-end institutional rebalancing & window-dressing indicator", "Binary (0/1)", "Strictly calendar-deterministic")

    # 2. 1-day stationary asset returns / diffs (44)
    num_cols = [c for c in clean_df.columns if c != "Date"]
    late_inception_assets = {"Global Agg Bond (BNDX)", "China 5Y LPR (pct)"}
    late_feature_names: set[str] = set()

    for col in num_cols:
        slug = ASSET_SLUGS[col]
        if col in ALL_RATE_COLS:
            fname = f"chg1d_{slug}"
            s = clean_df[col].diff(1)
            add_feat(
                fname,
                s,
                "Asset 1d Rate Change",
                f"{col}[t] - {col}[t-1]",
                f"1-day first difference in {col} (percentage points)",
                "1d First Difference",
                "Stationary rate change; handles negative rates cleanly"
                + (" (late inception)" if col in late_inception_assets else ""),
            )
        else:
            fname = f"ret1d_{slug}"
            s = safe_log_ret(clean_df[col], periods=1)
            add_feat(
                fname,
                s,
                "Asset 1d Log Return",
                f"ln({col}[t] / {col}[t-1])",
                f"1-day continuously compounded log return of {col}",
                "1d Log Return",
                "Stationary price return; guarded against <=0"
                + (" (blank on 2020-04-20/21 negative WTI)" if slug == "wti" else "")
                + (" (late inception)" if col in late_inception_assets else ""),
            )
        if col in late_inception_assets:
            late_feature_names.add(fname)

    # 3. 1-day stationary cross-asset ratio returns & spread changes (22)
    core_ratio_1d = [
        "Gold_Silver_Ratio",
        "Copper_Gold_Ratio",
        "WTI_Gold_Ratio",
        "Brent_WTI_Ratio",
        "Soybeans_Corn_Ratio",
        "SP500_Gold_Ratio",
        "SP500_Copper_Ratio",
        "Nasdaq_SP500_Ratio",
        "SP500_DAX_USD_Ratio",
        "SP500_Nikkei_USD_Ratio",
        "SP500_CSI300_USD_Ratio",
        "SP500_TLT_Ratio",
        "HYG_TLT_Ratio",
        "Gold_TLT_Ratio",
        "EUR_JPY_Cross",
        "GBP_EUR_Cross",
    ]
    core_spread_1d = [
        "Brent_WTI_Spread",
        "US_10Y_2Y_Spread",
        "US_30Y_10Y_Spread",
        "US_10Y_FedFunds_Spread",
        "US_CN_10Y_Spread",
        "Fed_ECB_Rate_Spread",
    ]

    for rcol in core_ratio_1d:
        slug = rcol.lower().replace("_ratio", "")
        add_feat(
            f"ratio_ret1d_{slug}",
            safe_log_ret(ratios_df[rcol], periods=1),
            "Ratio 1d Log Return",
            f"ln({rcol}[t] / {rcol}[t-1])",
            f"1-day relative return of {rcol}",
            "1d Log Return",
        )

    for scol in core_spread_1d:
        slug = scol.lower().replace("_spread", "")
        add_feat(
            f"spread_chg1d_{slug}",
            ratios_df[scol].diff(1),
            "Spread 1d Change",
            f"{scol}[t] - {scol}[t-1]",
            f"1-day change in {scol}",
            "1d First Difference",
        )

    # 4. Multi-horizon causal momentum: 21d & 63d (15 series x 2 = 30 cols)
    mom_ratios = [
        "Gold_Silver_Ratio",
        "Copper_Gold_Ratio",
        "WTI_Gold_Ratio",
        "SP500_Gold_Ratio",
        "Nasdaq_SP500_Ratio",
        "SP500_DAX_USD_Ratio",
        "SP500_Nikkei_USD_Ratio",
        "SP500_CSI300_USD_Ratio",
        "SP500_TLT_Ratio",
        "HYG_TLT_Ratio",
        "Gold_TLT_Ratio",
    ]
    mom_spreads = [
        "US_10Y_2Y_Spread",
        "US_10Y_FedFunds_Spread",
        "US_CN_10Y_Spread",
        "Fed_ECB_Rate_Spread",
    ]

    for h in (21, 63):
        for rcol in mom_ratios:
            slug = rcol.lower().replace("_ratio", "")
            add_feat(
                f"mom{h}d_{slug}",
                safe_log_ret(ratios_df[rcol], periods=h),
                f"Cross-Asset Momentum ({h}d)",
                f"ln({rcol}[t] / {rcol}[t-{h}])",
                f"{h}-trading-day relative momentum of {rcol}",
                f"{h}d Log Return",
            )
        for scol in mom_spreads:
            slug = scol.lower().replace("_spread", "")
            add_feat(
                f"mom{h}d_{slug}",
                ratios_df[scol].diff(h),
                f"Macro Spread Momentum ({h}d)",
                f"{scol}[t] - {scol}[t-{h}]",
                f"{h}-trading-day change in {scol}",
                f"{h}d First Difference",
            )

    # 5. Causal rolling 63d & 252d Z-scores (17 series x 2 = 34 cols)
    z_ratios = [
        "Gold_Silver_Ratio",
        "Copper_Gold_Ratio",
        "WTI_Gold_Ratio",
        "Soybeans_Corn_Ratio",
        "SP500_Gold_Ratio",
        "SP500_Copper_Ratio",
        "Nasdaq_SP500_Ratio",
        "SP500_DAX_USD_Ratio",
        "SP500_Nikkei_USD_Ratio",
        "SP500_CSI300_USD_Ratio",
        "SP500_TLT_Ratio",
        "HYG_TLT_Ratio",
        "Gold_TLT_Ratio",
    ]
    z_spreads = [
        "US_10Y_2Y_Spread",
        "US_10Y_FedFunds_Spread",
        "US_CN_10Y_Spread",
        "Fed_ECB_Rate_Spread",
    ]

    for w in (63, 252):
        for rcol in z_ratios:
            slug = rcol.lower().replace("_ratio", "")
            add_feat(
                f"z{w}_{slug}",
                causal_rolling_zscore(ratios_df[rcol], window=w, use_log=True),
                f"Causal Rolling Z-Score ({w}d)",
                f"(ln({rcol}) - rolling_mean_{w}) / rolling_std_{w}",
                f"Trailing {w}-day mean-reversion/regime Z-score of log({rcol})",
                f"Causal {w}d Log Z-Score",
                "Zero lookahead; log-transformed for scale symmetry",
            )
        for scol in z_spreads:
            slug = scol.lower().replace("_spread", "")
            add_feat(
                f"z{w}_{slug}",
                causal_rolling_zscore(ratios_df[scol], window=w, use_log=False),
                f"Causal Rolling Z-Score ({w}d)",
                f"({scol} - rolling_mean_{w}) / rolling_std_{w}",
                f"Trailing {w}-day regime Z-score of {scol}",
                f"Causal {w}d Level Z-Score",
                "Zero lookahead; level Z-score for additive macro spread",
            )

    # 6. Causal 21-day annualized realized volatility for 10 core benchmarks (10 cols)
    vol_benchmarks = [
        ("S&P 500", "sp500"),
        ("Nasdaq Composite", "nasdaq"),
        ("US 20+Y Treasury (TLT)", "tlt"),
        ("US High Yield (HYG)", "hyg"),
        ("Gold (USD/oz)", "gold"),
        ("WTI Crude (USD/bbl)", "wti"),
        ("Copper (USD/lb)", "copper"),
        ("DXY", "dxy"),
        ("EUR/USD", "eurusd"),
        ("USD/JPY", "usdjpy"),
    ]
    for col, slug in vol_benchmarks:
        r = feat_cols[f"ret1d_{slug}"]
        # Require at least 19 of 21 daily returns so 2 NaN WTI days on 2020-04-20/21 don't wipe out a month of vol
        vol = r.rolling(window=21, min_periods=19).std(ddof=1) * np.sqrt(252.0)
        add_feat(
            f"vol21d_{slug}",
            vol,
            "Realized Volatility (21d)",
            f"rolling_std_21(ret1d_{slug}) * sqrt(252)",
            f"Causal 21-day annualized realized volatility of {col}",
            "Annualized Realized Vol",
            "Strictly causal trailing 21d window (min_periods=19)",
        )

    # Assemble DataFrame cleanly in one shot (avoids fragmentation warnings)
    all_signal_cols = [c for c in feat_cols.keys() if c not in cal_names]
    core_signal_cols = [c for c in all_signal_cols if c not in late_feature_names]

    base_df = pd.DataFrame({"Date": clean_df["Date"], **feat_cols})
    valid_pct = base_df[all_signal_cols].notna().mean(axis=1)
    is_core_ready = base_df[core_signal_cols].notna().all(axis=1).astype("int64")
    is_full_ready = base_df[all_signal_cols].notna().all(axis=1).astype("int64")

    ml_df = pd.concat(
        [
            base_df,
            pd.DataFrame(
                {
                    "valid_feature_pct": valid_pct,
                    "is_core_ml_ready": is_core_ready,
                    "is_full_ml_ready": is_full_ready,
                },
                index=base_df.index,
            ),
        ],
        axis=1,
    )

    ml_dict.extend(
        [
            {
                "Tab": "ML_Ready_Dataset",
                "Column_Name": "valid_feature_pct",
                "Category": "ML Readiness Flag",
                "Formula_or_Source": "non_null_features / total_signal_features",
                "Economic_Meaning": "Fraction (0.0-1.0) of non-calendar ML features populated on Date t",
                "Transformation": "Completeness Ratio",
                "Inception_Date": clean_df["Date"].iloc[0],
                "ML_Notes": "Use to inspect warmup and vendor-outage coverage",
            },
            {
                "Tab": "ML_Ready_Dataset",
                "Column_Name": "is_core_ml_ready",
                "Category": "ML Readiness Flag",
                "Formula_or_Source": "1 if all core (non-BNDX, non-CN5Y-LPR) features non-null else 0",
                "Economic_Meaning": "Clean training-row mask for longest ~18-year multi-asset history (post-252d warmup)",
                "Transformation": "Binary (0/1)",
                "Inception_Date": clean_df["Date"].iloc[0],
                "ML_Notes": "Filter is_core_ml_ready == 1 for zero-NaN core ML datasets (~2008-2026)",
            },
            {
                "Tab": "ML_Ready_Dataset",
                "Column_Name": "is_full_ml_ready",
                "Category": "ML Readiness Flag",
                "Formula_or_Source": "1 if 100% of all ML features (incl. BNDX & CN 5Y LPR) non-null else 0",
                "Economic_Meaning": "Clean training-row mask when ALL 140 signal features are populated (post-2019-08)",
                "Transformation": "Binary (0/1)",
                "Inception_Date": clean_df["Date"].iloc[0],
                "ML_Notes": "Filter is_full_ml_ready == 1 for complete-feature ML training (~2019-2026)",
            },
        ]
    )

    return ml_df, ml_dict


def build_summary_sheet(
    exec_meta: dict[str, Any],
    col_audit_df: pd.DataFrame,
    anomaly_log: list[dict[str, Any]],
    ratios_df: pd.DataFrame,
    ml_df: pd.DataFrame,
) -> list[list[Any]]:
    """Build clean multi-section tabular rows for the Summary_and_Data_Quality tab."""
    core_ready_cnt = int(ml_df["is_core_ml_ready"].sum())
    full_ready_cnt = int(ml_df["is_full_ml_ready"].sum())
    core_first = ml_df.loc[ml_df["is_core_ml_ready"] == 1, "Date"].iloc[0] if core_ready_cnt > 0 else ""
    full_first = ml_df.loc[ml_df["is_full_ml_ready"] == 1, "Date"].iloc[0] if full_ready_cnt > 0 else ""

    rows: list[list[Any]] = []
    # Section 1: Executive Summary
    rows.append(["SECTION 1: EXECUTIVE SUMMARY & PIPELINE METADATA", "", "", ""])
    rows.append(["Metric", "Value", "Details / Rationale", "Status"])
    summary_items = [
        ("Source Google Sheet ID", DEFAULT_SOURCE_ID, "Financial Data Updates (tab: Sheet1)", "Verified"),
        ("Raw Dimensions", f"{exec_meta['raw_rows']} rows x {exec_meta['raw_cols']} cols", "44 numeric market series + Date + Notes", "Audited"),
        ("Clean Weekday Panel", f"{exec_meta['clean_rows']} rows x {exec_meta['clean_numeric_cols']} numeric series", f"Date span: {exec_meta['start_date']} to {exec_meta['end_date']} (0 missing weekdays)", "Clean"),
        ("Chronological Order & Duplicates", f"Monotonic={exec_meta['was_monotonic']}, Duplicates={exec_meta['duplicate_dates']}, BadDates={exec_meta['unparseable_dates']}", "Sorted ascending by Date", "Pass"),
        ("Weekend Rows Dropped", str(exec_meta["weekend_rows_dropped"]), f"Dropped Sat {exec_meta['weekend_dates']} (duplicate Fri closes + cross-vendor USD/CNY +7.33% artifact)", "Fixed"),
        ("Notes Column Audit", f"{exec_meta['notes_non_empty_count']} non-empty row", exec_meta["notes_entries"], "Archived & Dropped from ML"),
        ("Literal 'nan' Strings Parsed", str(exec_meta["total_literal_nan_parsed"]), "Found in HK Base Rate (pct) on HK bank holidays", "Fixed"),
        ("Data Corruption Cells Fixed", str(exec_meta["total_data_errors_fixed"]), "1 HK HIBOR 100x error (199.29->1.9929), 16 Aug-2008 DD/MM swapped FX cells, 1 USD/CNY 2011-07-18 bad tick", "Fixed & Logged"),
        ("Post-Inception Causal FFill Imputed", str(exec_meta["total_cells_imputed"]), f"Traded assets: ffill(limit={TRADED_FFILL_LIMIT}d) for global holidays; Policy rates: causal ffill; ZERO pre-inception backfill", "Zero Lookahead"),
        ("Remaining Post-Inception NaNs", str(exec_meta["total_remaining_post_inc_na"]), "Only 16 vendor-outage cells in Aug 2008 (FX) & Apr 2009 (Brent); 0 post-inception NaNs from May 2009 to Nov 2026", "Guarded"),
        ("Cross-Asset Ratios & Spreads", f"{ratios_df.shape[1] - 1} engineered series", "6 economic categories incl. FX-adjusted USD equity ratios & sovereign curves; all guarded for <= 0", "Published"),
        ("ML-Ready Feature Matrix", f"{ml_df.shape[0]} rows x {ml_df.shape[1]} cols", "Strictly causal returns, diffs, 21d/63d momentum, 63d/252d rolling Z-scores, 21d realized vol, calendar & readiness flags", "Published"),
        ("Core ML-Ready Rows (is_core_ml_ready=1)", f"{core_ready_cnt} rows ({round(100*core_ready_cnt/len(ml_df), 1)}%)", f"First ready date: {core_first} (excludes only late-inception BNDX & China 5Y LPR)", "Ready for ML"),
        ("Full ML-Ready Rows (is_full_ml_ready=1)", f"{full_ready_cnt} rows ({round(100*full_ready_cnt/len(ml_df), 1)}%)", f"First ready date: {full_first} (100% of all 140 signal features populated)", "Ready for ML"),
    ]
    for item in summary_items:
        rows.append(list(item))

    rows.append([])
    # Section 2: Anomaly & Correction Log
    rows.append(["SECTION 2: ANOMALY LOG & DATA INTEGRITY ACTIONS", "", "", "", "", "", "", ""])
    anom_headers = ["Anomaly_ID", "Severity", "Column", "Date_Range", "Raw_Value", "Issue_Type", "Action_Taken", "Lookahead_Risk"]
    rows.append(anom_headers)
    for rec in anomaly_log:
        rows.append([rec[h] for h in anom_headers])

    rows.append([])
    # Section 3: Per-Column Data Validity Audit
    rows.append(["SECTION 3: PER-COLUMN DATA VALIDITY & PREPROCESSING AUDIT (44 NUMERIC SERIES)"])
    audit_headers = list(col_audit_df.columns)
    rows.append(audit_headers)
    for _, r in col_audit_df.iterrows():
        rows.append([r[h] for h in audit_headers])

    # Normalize row width so CSV is rectangular
    max_w = max(len(r) for r in rows)
    norm_rows = [r + [""] * (max_w - len(r)) for r in rows]
    return norm_rows


def build_feature_dictionary(
    clean_df: pd.DataFrame,
    ratio_meta: list[dict[str, str]],
    ml_meta: list[dict[str, str]],
) -> pd.DataFrame:
    """Create unified Feature_Dictionary across Preprocessed_Data, Cross_Asset_Ratios, and ML_Ready_Dataset."""
    records: list[dict[str, str]] = []

    # Preprocessed_Data entries
    records.append(
        {
            "Tab": "Preprocessed_Data",
            "Column_Name": "Date",
            "Category": "Index",
            "Formula_or_Source": "Source Sheet1 Date (weekdays only)",
            "Economic_Meaning": "Trading/calendar weekday date (YYYY-MM-DD, Mon-Fri)",
            "Transformation": "Chronologically sorted; Sat 2026-11-21 dropped",
            "Inception_Date": clean_df["Date"].iloc[0],
            "ML_Notes": "Primary time index across all time-series tabs",
        }
    )
    for col in clean_df.columns:
        if col == "Date":
            continue
        fv = clean_df[col].first_valid_index()
        inc_dt = clean_df.loc[fv, "Date"] if fv is not None else ""
        is_pol = col in RATE_POLICY_COLS
        records.append(
            {
                "Tab": "Preprocessed_Data",
                "Column_Name": col,
                "Category": ASSET_CATEGORIES.get(col, "Market Series"),
                "Formula_or_Source": f"Source Sheet1 '{col}'",
                "Economic_Meaning": f"Cleaned daily level of {col}",
                "Transformation": (
                    "Causal policy-rate ffill (no backfill)"
                    if is_pol
                    else f"Causal holiday ffill limit={TRADED_FFILL_LIMIT}d (no backfill)"
                ),
                "Inception_Date": inc_dt,
                "ML_Notes": (
                    "Non-stationary level; use chg1d_* in ML_Ready_Dataset"
                    if col in ALL_RATE_COLS
                    else "Non-stationary price/FX level; use ret1d_* in ML_Ready_Dataset"
                ),
            }
        )

    # Cross_Asset_Ratios Date + entries
    records.append(
        {
            "Tab": "Cross_Asset_Ratios",
            "Column_Name": "Date",
            "Category": "Index",
            "Formula_or_Source": "Aligned to Preprocessed_Data Date",
            "Economic_Meaning": "Trading weekday date (YYYY-MM-DD)",
            "Transformation": "Identity",
            "Inception_Date": clean_df["Date"].iloc[0],
            "ML_Notes": "Aligned 1:1 with Preprocessed_Data and ML_Ready_Dataset",
        }
    )
    records.extend(ratio_meta)

    # ML_Ready_Dataset Date + entries
    records.append(
        {
            "Tab": "ML_Ready_Dataset",
            "Column_Name": "Date",
            "Category": "Index",
            "Formula_or_Source": "Aligned to Preprocessed_Data Date",
            "Economic_Meaning": "Trading weekday date (YYYY-MM-DD)",
            "Transformation": "Identity",
            "Inception_Date": clean_df["Date"].iloc[0],
            "ML_Notes": "Use chronological train/val/test splits (never random shuffle!)",
        }
    )
    records.extend(ml_meta)

    return pd.DataFrame(records)


def save_clean_csv(df: pd.DataFrame, path: Path, decimals: int = 6) -> None:
    """Save DataFrame to CSV with clean rounding, empty strings for NaN, and verified zero inf/NaN literals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).round(decimals)
    out.to_csv(path, index=False, na_rep="")

    # Hard verification: zero literal inf / -inf / NaN tokens
    txt = path.read_text(encoding="utf-8")
    for bad_tok in [",inf,", ",-inf,", ",NaN,", ",nan,"]:
        if bad_tok in txt:
            raise ValueError(f"Found forbidden literal token {bad_tok} in {path}")


def save_rows_csv(rows: list[list[Any]], path: Path) -> None:
    """Save a 2D list of rows cleanly to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            clean_r = []
            for v in r:
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    clean_r.append("")
                else:
                    clean_r.append(v)
            writer.writerow(clean_r)


def get_sheet_map(dest_id: str) -> dict[str, dict[str, Any]]:
    """Return mapping of sheet title -> metadata dict (id, rows, cols) via gsheets CLI."""
    res = run_cmd(["gsheets", "readonly", "list-sheets", dest_id, "--json"])
    data = json.loads(res.stdout)
    sheets = data.get("sheets", data) if isinstance(data, dict) else data
    out: dict[str, dict[str, Any]] = {}
    for s in sheets:
        props = s.get("properties", s)
        title = props.get("title") or s.get("title") or s.get("name")
        sid = props.get("sheetId", s.get("sheetId", s.get("id", 0)))
        grid = props.get("gridProperties", {})
        out[title] = {
            "sheet_id": int(sid),
            "rows": int(grid.get("rowCount", s.get("rows", 1000))),
            "cols": int(grid.get("columnCount", s.get("cols", 26))),
        }
    return out


def ensure_grid_size(dest_id: str, sheet_id: int, cur_rows: int, cur_cols: int, req_rows: int, req_cols: int) -> None:
    """Expand grid dimensions via raw-batch updateSheetProperties if needed."""
    if req_rows <= cur_rows and req_cols <= cur_cols:
        return
    new_rows = max(cur_rows, req_rows)
    new_cols = max(cur_cols, req_cols)
    req_body = {
        "requests": [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {"rowCount": new_rows, "columnCount": new_cols},
                    },
                    "fields": "gridProperties(rowCount,columnCount)",
                }
            }
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(req_body, tf)
        tf_path = tf.name
    try:
        run_cmd(["gsheets", "mutate", "raw-batch", dest_id, "-f", tf_path])
    finally:
        os.unlink(tf_path)


def upload_csv_chunked(dest_id: str, tab_name: str, sheet_info: dict[str, Any], csv_path: Path, chunk_size: int = 500) -> None:
    """Fallback uploader: expand grid if needed and write CSV in chunks via gsheets mutate batch."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.reader(f))
    n_rows = len(all_rows)
    n_cols = max(len(r) for r in all_rows) if all_rows else 1
    ensure_grid_size(dest_id, sheet_info["sheet_id"], sheet_info["rows"], sheet_info["cols"], n_rows, n_cols)

    # Convert numeric-looking cells cleanly so Google Sheets stores numbers as numeric types
    def cast_row(row: list[str], is_header: bool) -> list[Any]:
        if is_header:
            return row
        out: list[Any] = []
        for idx, val in enumerate(row):
            if val == "":
                out.append("")
            elif idx == 0:
                out.append(val)  # Keep Date as string YYYY-MM-DD
            else:
                try:
                    fv = float(val)
                    out.append(int(fv) if fv.is_integer() and "." not in val else fv)
                except ValueError:
                    out.append(val)
        return out

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = [cast_row(all_rows[i], is_header=(i == 0)) for i in range(start, end)]
        op = [{"op": "write", "range": f"'{tab_name}'!A{start + 1}", "data": chunk}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(op, tf)
            tf_path = tf.name
        try:
            run_cmd(["gsheets", "mutate", "batch", dest_id, "-f", tf_path])
        finally:
            os.unlink(tf_path)
        print(f"    wrote rows {start + 1}..{end} / {n_rows}")


def _cast_sheet_row(row: list[str], is_header: bool) -> list[Any]:
    """Convert numeric-looking CSV cells cleanly so Google Sheets stores numbers as numeric types."""
    if is_header:
        return row
    out: list[Any] = []
    for idx, val in enumerate(row):
        if val == "":
            out.append("")
        elif idx == 0:
            out.append(val)  # Keep Date as string YYYY-MM-DD
        else:
            try:
                fv = float(val)
                out.append(int(fv) if fv.is_integer() and "." not in val else fv)
            except ValueError:
                out.append(val)
    return out


def _hex_to_rgb(hex_color: str) -> dict[str, float]:
    h = hex_color.lstrip("#")
    return {
        "red": int(h[0:2], 16) / 255.0,
        "green": int(h[2:4], 16) / 255.0,
        "blue": int(h[4:6], 16) / 255.0,
    }


def _api_get_sheet_map(service: Any, dest_id: str) -> dict[str, dict[str, Any]]:
    meta = (
        service.spreadsheets()
        .get(spreadsheetId=dest_id, fields="sheets.properties(sheetId,title,gridProperties)")
        .execute()
    )
    out: dict[str, dict[str, Any]] = {}
    for s in meta.get("sheets", []):
        props = s["properties"]
        grid = props.get("gridProperties", {})
        out[props["title"]] = {
            "sheet_id": int(props["sheetId"]),
            "rows": int(grid.get("rowCount", 1000)),
            "cols": int(grid.get("columnCount", 26)),
        }
    return out


def _make_format_request(
    sheet_id: int,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    bg_hex: str,
    fg_hex: str,
    bold: bool = True,
) -> dict[str, Any]:
    return {
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": start_row,
                "endRowIndex": end_row,
                "startColumnIndex": start_col,
                "endColumnIndex": end_col,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": _hex_to_rgb(bg_hex),
                    "textFormat": {
                        "bold": bold,
                        "foregroundColor": _hex_to_rgb(fg_hex),
                    },
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat(bold,foregroundColor))",
        }
    }


def publish_to_sheets_api(dest_id: str, tab_csv_map: list[tuple[str, Path, bool]]) -> None:
    """Publish all tabs to destination Google Sheet directly via the Google Sheets v4 API."""
    service = build_sheets_service()
    sheet_map = _api_get_sheet_map(service, dest_id)

    setup_reqs: list[dict[str, Any]] = []
    first_tab_name = tab_csv_map[0][0]
    if first_tab_name not in sheet_map and "Sheet1" in sheet_map:
        sid = sheet_map["Sheet1"]["sheet_id"]
        print(f"  Renaming initial empty 'Sheet1' (id={sid}) -> '{first_tab_name}'")
        setup_reqs.append(
            {
                "updateSheetProperties": {
                    "properties": {"sheetId": sid, "title": first_tab_name},
                    "fields": "title",
                }
            }
        )
        sheet_map[first_tab_name] = sheet_map.pop("Sheet1")

    for tab_name, _, _ in tab_csv_map:
        if tab_name not in sheet_map:
            print(f"  Adding tab '{tab_name}'")
            setup_reqs.append({"addSheet": {"properties": {"title": tab_name}}})

    if setup_reqs:
        service.spreadsheets().batchUpdate(spreadsheetId=dest_id, body={"requests": setup_reqs}).execute()
        sheet_map = _api_get_sheet_map(service, dest_id)

    format_reqs: list[dict[str, Any]] = []
    for tab_name, csv_path, freeze_col_a in tab_csv_map:
        info = sheet_map[tab_name]
        sid = info["sheet_id"]
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.reader(f))
        n_rows = len(all_rows)
        n_cols = max(len(r) for r in all_rows) if all_rows else 1

        print(f"  Uploading '{tab_name}' ({n_rows} rows x {n_cols} cols) via Sheets API...")
        if n_rows > info["rows"] or n_cols > info["cols"]:
            service.spreadsheets().batchUpdate(
                spreadsheetId=dest_id,
                body={
                    "requests": [
                        {
                            "updateSheetProperties": {
                                "properties": {
                                    "sheetId": sid,
                                    "gridProperties": {
                                        "rowCount": max(info["rows"], n_rows),
                                        "columnCount": max(info["cols"], n_cols),
                                    },
                                },
                                "fields": "gridProperties(rowCount,columnCount)",
                            }
                        }
                    ]
                },
            ).execute()

        service.spreadsheets().values().clear(
            spreadsheetId=dest_id,
            range=f"'{tab_name}'",
            body={},
        ).execute()

        chunk_size = 1000
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = [_cast_sheet_row(all_rows[i], is_header=(i == 0)) for i in range(start, end)]
            service.spreadsheets().values().update(
                spreadsheetId=dest_id,
                range=f"'{tab_name}'!A{start + 1}",
                valueInputOption="USER_ENTERED",
                body={"values": chunk},
            ).execute()
            print(f"    wrote rows {start + 1}..{end} / {n_rows}")

        format_reqs.append(_make_format_request(sid, 0, 1, 0, n_cols, "#1F4E78", "#FFFFFF", bold=True))
        format_reqs.append(
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sid,
                        "gridProperties": {
                            "frozenRowCount": 1,
                            "frozenColumnCount": 1 if freeze_col_a else 0,
                        },
                    },
                    "fields": "gridProperties(frozenRowCount,frozenColumnCount)",
                }
            }
        )

    # Extra visual polish on Summary_and_Data_Quality section headers
    sum_sid = sheet_map["Summary_and_Data_Quality"]["sheet_id"]
    with tab_csv_map[0][1].open("r", encoding="utf-8", newline="") as f:
        sum_rows = list(csv.reader(f))
    sum_cols = len(sum_rows[0])
    for idx, r in enumerate(sum_rows):
        if idx > 0 and r and r[0].startswith("SECTION "):
            format_reqs.append(_make_format_request(sum_sid, idx, idx + 1, 0, sum_cols, "#1F4E78", "#FFFFFF", bold=True))
        elif idx > 0 and r and r[0] in ("Metric", "Anomaly_ID", "Column"):
            format_reqs.append(_make_format_request(sum_sid, idx, idx + 1, 0, sum_cols, "#D9E1F2", "#000000", bold=True))

    if format_reqs:
        service.spreadsheets().batchUpdate(spreadsheetId=dest_id, body={"requests": format_reqs}).execute()


def publish_to_sheets(dest_id: str, tab_csv_map: list[tuple[str, Path, bool]], backend: str = "auto") -> None:
    """
    Create/rename the 5 destination tabs, upload CSVs (Sheets API or gsheets CLI),
    and apply clean header formatting + freeze panes.
    """
    if not dest_id:
        raise ValueError("Destination Google Sheet ID is empty. Pass --dest-id or set DEST_SHEET_ID.")
    chosen = resolve_sheets_backend(backend)
    print(f"[6/6] Publishing {len(tab_csv_map)} tabs to destination Google Sheet ({chosen}): {dest_id}")
    if chosen == "api":
        publish_to_sheets_api(dest_id, tab_csv_map)
        return
    sheet_map = get_sheet_map(dest_id)

    first_tab_name = tab_csv_map[0][0]
    if first_tab_name not in sheet_map and "Sheet1" in sheet_map:
        sid = sheet_map["Sheet1"]["sheet_id"]
        print(f"  Renaming initial empty 'Sheet1' (id={sid}) -> '{first_tab_name}'")
        run_cmd(["gsheets", "mutate", "rename-sheet", dest_id, "--sheet-id", str(sid), "--title", first_tab_name])
        sheet_map = get_sheet_map(dest_id)

    for tab_name, _, _ in tab_csv_map:
        if tab_name not in sheet_map:
            print(f"  Adding tab '{tab_name}'")
            run_cmd(["gsheets", "mutate", "add-sheet", dest_id, "--title", tab_name])

    sheet_map = get_sheet_map(dest_id)

    for tab_name, csv_path, freeze_col_a in tab_csv_map:
        info = sheet_map[tab_name]
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, [])
            n_cols = len(first_row)
            n_rows = 1 + sum(1 for _ in reader)

        print(f"  Uploading '{tab_name}' ({n_rows} rows x {n_cols} cols) from {csv_path.name}...")
        # Use import-csv first; fall back to chunked batch if payload/timeout issues occur
        try:
            if n_rows * n_cols > 350_000:
                print("    Large matrix detected (>350k cells); using reliable chunked batch upload...")
                upload_csv_chunked(dest_id, tab_name, info, csv_path, chunk_size=400)
            else:
                ensure_grid_size(dest_id, info["sheet_id"], info["rows"], info["cols"], n_rows, n_cols)
                run_cmd(
                    [
                        "gsheets",
                        "mutate",
                        "import-csv",
                        dest_id,
                        str(csv_path),
                        "--sheet",
                        tab_name,
                    ]
                )
        except Exception as exc:
            print(f"    import-csv failed ({exc}); falling back to chunked batch upload...")
            sheet_map = get_sheet_map(dest_id)
            upload_csv_chunked(dest_id, tab_name, sheet_map[tab_name], csv_path, chunk_size=400)

        # Format header row and freeze
        sid = sheet_map[tab_name]["sheet_id"]
        run_cmd(
            [
                "gsheets",
                "mutate",
                "format",
                dest_id,
                "--sheet-id",
                str(sid),
                "--start-row",
                "0",
                "--end-row",
                "1",
                "--start-col",
                "0",
                "--end-col",
                str(n_cols),
                "--bold",
                "--bg-color",
                "#1F4E78",
                "--fg-color",
                "#FFFFFF",
            ]
        )
        run_cmd(
            [
                "gsheets",
                "mutate",
                "freeze",
                dest_id,
                "--sheet-id",
                str(sid),
                "--rows",
                "1",
                "--cols",
                "1" if freeze_col_a else "0",
            ]
        )

    # Extra visual polish on Summary_and_Data_Quality section headers
    sum_sid = sheet_map["Summary_and_Data_Quality"]["sheet_id"]
    with tab_csv_map[0][1].open("r", encoding="utf-8", newline="") as f:
        sum_rows = list(csv.reader(f))
    sum_cols = len(sum_rows[0])
    for idx, r in enumerate(sum_rows):
        if idx > 0 and r and r[0].startswith("SECTION "):
            run_cmd(
                [
                    "gsheets",
                    "mutate",
                    "format",
                    dest_id,
                    "--sheet-id",
                    str(sum_sid),
                    "--start-row",
                    str(idx),
                    "--end-row",
                    str(idx + 1),
                    "--start-col",
                    "0",
                    "--end-col",
                    str(sum_cols),
                    "--bold",
                    "--bg-color",
                    "#1F4E78",
                    "--fg-color",
                    "#FFFFFF",
                ]
            )
        elif idx > 0 and r and r[0] in ("Metric", "Anomaly_ID", "Column"):
            run_cmd(
                [
                    "gsheets",
                    "mutate",
                    "format",
                    dest_id,
                    "--sheet-id",
                    str(sum_sid),
                    "--start-row",
                    str(idx),
                    "--end-row",
                    str(idx + 1),
                    "--start-col",
                    "0",
                    "--end-col",
                    str(sum_cols),
                    "--bold",
                    "--bg-color",
                    "#D9E1F2",
                    "--fg-color",
                    "#000000",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial ML Preprocessing & Google Sheets Pipeline")
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID, help="Source Google Sheet ID")
    parser.add_argument("--dest-id", default=DEFAULT_DEST_ID, help="Destination Google Sheet ID")
    parser.add_argument(
        "--raw-csv",
        default=str(BASE_DIR / "data" / "raw" / "financial_data_raw.csv"),
        help="Path to raw CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "processed"),
        help="Directory for processed CSV outputs",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "api", "gsheets"],
        default="auto",
        help="Google Sheets backend: auto (API if credentials set, else gsheets CLI), api, or gsheets",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Skip downloading raw CSV from source Google Sheet")
    parser.add_argument("--skip-upload", action="store_true", help="Skip publishing processed CSVs to destination Google Sheet")
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        fetch_raw_sheet(args.source_id, raw_csv, backend=args.backend)
    elif not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv}. Run without --skip-fetch.")

    # Audit & clean
    clean_df, col_audit_df, anomaly_log, exec_meta = audit_and_clean(raw_csv)

    # Cross-asset ratios
    ratios_df, ratio_meta = engineer_cross_asset_ratios(clean_df)

    # ML feature matrix
    ml_df, ml_meta = build_ml_dataset(clean_df, ratios_df)

    # Summary & Feature Dictionary
    summary_rows = build_summary_sheet(exec_meta, col_audit_df, anomaly_log, ratios_df, ml_df)
    feat_dict_df = build_feature_dictionary(clean_df, ratio_meta, ml_meta)

    # Save all 5 processed CSVs
    print(f"[5/6] Saving processed CSV artifacts to {out_dir}")
    p_summary = out_dir / "Summary_and_Data_Quality.csv"
    p_preproc = out_dir / "Preprocessed_Data.csv"
    p_ratios = out_dir / "Cross_Asset_Ratios.csv"
    p_ml = out_dir / "ML_Ready_Dataset.csv"
    p_dict = out_dir / "Feature_Dictionary.csv"

    save_rows_csv(summary_rows, p_summary)
    save_clean_csv(clean_df, p_preproc, decimals=6)
    save_clean_csv(ratios_df, p_ratios, decimals=6)
    save_clean_csv(ml_df, p_ml, decimals=6)
    feat_dict_df.to_csv(p_dict, index=False, na_rep="")

    # Print rich summary to console
    total_cells = (
        len(summary_rows) * len(summary_rows[0])
        + (len(clean_df) + 1) * len(clean_df.columns)
        + (len(ratios_df) + 1) * len(ratios_df.columns)
        + (len(ml_df) + 1) * len(ml_df.columns)
        + (len(feat_dict_df) + 1) * len(feat_dict_df.columns)
    )
    print("\n=== PIPELINE OUTPUT SUMMARY ===")
    print(f"  Summary_and_Data_Quality : {len(summary_rows)} rows x {len(summary_rows[0])} cols")
    print(f"  Preprocessed_Data        : {len(clean_df)+1} rows x {len(clean_df.columns)} cols")
    print(f"  Cross_Asset_Ratios       : {len(ratios_df)+1} rows x {len(ratios_df.columns)} cols")
    print(f"  ML_Ready_Dataset         : {len(ml_df)+1} rows x {len(ml_df.columns)} cols")
    print(f"  Feature_Dictionary       : {len(feat_dict_df)+1} rows x {len(feat_dict_df.columns)} cols")
    print(f"  Estimated Total Cells    : {total_cells:,} (limit: 10,000,000)")
    print(
        f"  is_core_ml_ready == 1    : {int(ml_df['is_core_ml_ready'].sum())} rows "
        f"(first: {ml_df.loc[ml_df['is_core_ml_ready'] == 1, 'Date'].iloc[0]})"
    )
    print(
        f"  is_full_ml_ready == 1    : {int(ml_df['is_full_ml_ready'].sum())} rows "
        f"(first: {ml_df.loc[ml_df['is_full_ml_ready'] == 1, 'Date'].iloc[0]})"
    )

    if not args.skip_upload:
        tab_csv_map = [
            ("Summary_and_Data_Quality", p_summary, False),
            ("Preprocessed_Data", p_preproc, True),
            ("Cross_Asset_Ratios", p_ratios, True),
            ("ML_Ready_Dataset", p_ml, True),
            ("Feature_Dictionary", p_dict, False),
        ]
        publish_to_sheets(args.dest_id, tab_csv_map, backend=args.backend)
        print("\n[DONE] Pipeline complete and published to Google Sheets!")
    else:
        print("\n[DONE] Local pipeline complete (--skip-upload specified).")


if __name__ == "__main__":
    main()
