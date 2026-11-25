"""
Institutional Multi-Asset Market Dynamics & Macro Insight Agent (Agent 4):
Consumes Agent 3's PCA, Strictly Causal Rolling PCA, Maximum Likelihood Factor Analysis,
and Hierarchical Clustering (+ HRP) outputs, synthesizes easy-to-read deep insights on
market dynamics and current ongoing macro developments (via Gemini API + Google Search
grounding, with a deterministic quantitative fallback when offline), and publishes
native Google Docs into My Drive -> Agent Folder:
  - Master rolling doc: `4. Market Dynamics & Macro Insights` (updated in place)
  - Dated archive doc : `4. Market Insights Archive / YYYY-MM-DD - Market Dynamics & Macro Insights`

Usage:
    py market_insight_agent.py                               # Fetch from Agent 3 sheet & publish
    py market_insight_agent.py --skip-fetch --skip-upload    # Offline local run from data/exploration/
    py market_insight_agent.py --skip-fetch --backend webapp # Chained GitHub Actions run after Agent 3

Environment variables (optional in offline mode):
    EXPLORATION_SHEET_ID     Agent 3 Google Sheet ID (for standalone fetch)
    DRIVE_AGENT_FOLDER_ID    Google Drive 'Agent Folder' ID (default: searches by name)
    INSIGHT_MASTER_DOC_ID    Master Google Doc ID for in-place updates (auto-located if blank)
    INSIGHT_ARCHIVE_FOLDER_ID Archive subfolder ID (auto-located/created if blank)
    GEMINI_API_KEY           Google Gemini API key for live Search-grounded macro synthesis
    GEMINI_MODEL             Gemini model name (default: gemini-2.5-flash)
    APPS_SCRIPT_URL          Google Apps Script Web App /exec URL
    APPS_SCRIPT_TOKEN        Shared secret token for Apps Script Web App
    GOOGLE_SERVICE_ACCOUNT_JSON  Optional Google Cloud Service Account JSON (path or raw string)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SGT_TZ = ZoneInfo("Asia/Singapore")


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

DEFAULT_EXPLORATION_ID = os.environ.get("EXPLORATION_SHEET_ID", "")
DEFAULT_AGENT_FOLDER_ID = os.environ.get("DRIVE_AGENT_FOLDER_ID", "")
DEFAULT_MASTER_DOC_ID = os.environ.get("INSIGHT_MASTER_DOC_ID", "")
DEFAULT_ARCHIVE_FOLDER_ID = os.environ.get("INSIGHT_ARCHIVE_FOLDER_ID", "")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

MASTER_DOC_TITLE = "4. Market Dynamics & Macro Insights"
ARCHIVE_FOLDER_NAME = "4. Market Insights Archive"
AGENT_FOLDER_NAME = "Agent Folder"

REQUIRED_TABS = [
    "Executive_Summary",
    "PCA_Overview_and_Loadings",
    "PCA_Scores_TimeSeries",
    "Rolling_PCA_Dynamics",
    "Rolling_PCA_Loadings_Snapshots",
    "Factor_Analysis_Results",
    "Factor_Scores_TimeSeries",
    "Hierarchical_Clustering",
]

TAIL_ONLY_TABS = {
    "PCA_Scores_TimeSeries": 35,
    "Rolling_PCA_Dynamics": 35,
    "Factor_Scores_TimeSeries": 35,
}


# =============================================================================
# 1. NETWORK & BACKEND HELPERS
# =============================================================================


def webapp_post(payload: dict[str, Any], max_retries: int = 4, timeout: int = 180) -> dict[str, Any]:
    """POST JSON payload to the Google Apps Script Web App bridge with exponential backoff."""
    url = os.environ.get("APPS_SCRIPT_URL", "").strip()
    token = os.environ.get("APPS_SCRIPT_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("APPS_SCRIPT_URL and APPS_SCRIPT_TOKEN must be set for --backend webapp.")
    body_dict = dict(payload)
    body_dict["token"] = token
    data = json.dumps(body_dict).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as je:
                snippet = raw[:300].replace("\n", " ")
                raise RuntimeError(f"Non-JSON response from Apps Script: {snippet}") from je
            if not parsed.get("ok"):
                raise RuntimeError(f"Apps Script error ({payload.get('action')}): {parsed.get('error', parsed)}")
            return parsed
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, RuntimeError) as exc:
            last_err = exc
            if attempt < max_retries:
                wait_s = 3 * (2 ** (attempt - 1))
                print(f"  [retry {attempt}/{max_retries - 1}] Apps Script ({payload.get('action')}) failed: {exc} — retrying in {wait_s}s...")
                time.sleep(wait_s)
    raise RuntimeError(f"Apps Script request failed after {max_retries} attempts: {last_err}")


def run_cmd(cmd: list[str], max_retries: int = 3) -> str:
    """Execute a CLI command (gsheets / gdrive) with retry for transient API rate limits."""
    last_err = ""
    for attempt in range(1, max_retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode == 0:
            return proc.stdout
        combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
        last_err = combined.strip()
        if ("rateLimitExceeded" in combined or "429" in combined or "Quota exceeded" in combined) and attempt < max_retries:
            wait_s = 15 * attempt
            print(f"  [rate-limit retry {attempt}/{max_retries - 1}] waiting {wait_s}s before retrying {cmd[0]}...")
            time.sleep(wait_s)
            continue
        break
    raise RuntimeError(f"Command failed ({' '.join(cmd[:4])}): {last_err}")


def resolve_fetch_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("APPS_SCRIPT_URL") and os.environ.get("APPS_SCRIPT_TOKEN"):
        return "webapp"
    if os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return "api"
    if shutil.which("gsheets"):
        return "gsheets"
    return "webapp"


def resolve_drive_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("APPS_SCRIPT_URL") and os.environ.get("APPS_SCRIPT_TOKEN"):
        return "webapp"
    if os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return "api"
    if shutil.which("gdrive"):
        return "gdrive"
    return "webapp"


def load_service_account_creds(scopes: list[str]):
    from google.oauth2 import service_account  # type: ignore

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        if raw.startswith("{"):
            info = json.loads(raw)
            return service_account.Credentials.from_service_account_info(info, scopes=scopes)
        return service_account.Credentials.from_service_account_file(raw, scopes=scopes)
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if path:
        return service_account.Credentials.from_service_account_file(path, scopes=scopes)
    raise RuntimeError("No Google Service Account credentials found.")


# =============================================================================
# 2. AGENT 3 DATA FETCHING & PARSING
# =============================================================================


def save_rows_csv(rows: list[list[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)


def fetch_agent3_tabs(source_id: str, out_dir: Path, backend: str = "auto") -> None:
    """Fetch Agent 3 analytical tabs (and recent tails of time-series tabs) to local CSVs."""
    if not source_id:
        raise ValueError("Agent 3 spreadsheet ID is empty. Pass --source-id or set EXPLORATION_SHEET_ID.")
    chosen = resolve_fetch_backend(backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Fetching Agent 3 outputs from Google Sheet ({chosen}): {source_id}")

    if chosen == "webapp":
        for tab in REQUIRED_TABS:
            if tab in TAIL_ONLY_TABS:
                res = webapp_post(
                    {
                        "action": "get_tail",
                        "source_id": source_id,
                        "sheet_name": tab,
                        "tail_rows": TAIL_ONLY_TABS[tab],
                    }
                )
                rows = [res.get("headers", [])] + res.get("tail_rows", [])
            else:
                res = webapp_post(
                    {
                        "action": "fetch",
                        "source_id": source_id,
                        "sheet_name": tab,
                        "start_row": 1,
                        "num_rows": 500,
                    }
                )
                rows = res.get("rows", [])
            save_rows_csv(rows, out_dir / f"{tab}.csv")
            print(f"  Fetched {tab:32s}: {len(rows):4d} rows")
        return

    if chosen == "api":
        from googleapiclient.discovery import build  # type: ignore

        creds = load_service_account_creds(["https://www.googleapis.com/auth/spreadsheets.readonly"])
        service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        meta = service.spreadsheets().get(spreadsheetId=source_id).execute()
        row_counts = {
            s["properties"]["title"]: int(s["properties"]["gridProperties"]["rowCount"])
            for s in meta.get("sheets", [])
        }
        for tab in REQUIRED_TABS:
            if tab in TAIL_ONLY_TABS:
                n_rows = row_counts.get(tab, 5000)
                hdr = (
                    service.spreadsheets()
                    .values()
                    .get(spreadsheetId=source_id, range=f"'{tab}'!1:1")
                    .execute()
                    .get("values", [])
                )
                start_r = max(2, n_rows - TAIL_ONLY_TABS[tab] - 5)
                tail = (
                    service.spreadsheets()
                    .values()
                    .get(spreadsheetId=source_id, range=f"'{tab}'!A{start_r}:AF{n_rows}")
                    .execute()
                    .get("values", [])
                )
                rows = hdr + [r for r in tail if r and str(r[0]).strip()]
            else:
                rows = (
                    service.spreadsheets()
                    .values()
                    .get(spreadsheetId=source_id, range=f"'{tab}'!A1:Z300")
                    .execute()
                    .get("values", [])
                )
            save_rows_csv(rows, out_dir / f"{tab}.csv")
            print(f"  Fetched {tab:32s}: {len(rows):4d} rows")
        return

    # Local gsheets CLI fallback
    info_raw = run_cmd(["gsheets", "readonly", "info", source_id, "--json"])
    info_json = json.loads(info_raw)
    row_counts = {}
    for sh in info_json.get("sheets", []):
        props = sh.get("properties", sh)
        title = props.get("title", "")
        grid = props.get("gridProperties", {})
        if title:
            row_counts[title] = int(grid.get("rowCount", 5000))

    for tab in REQUIRED_TABS:
        if tab in TAIL_ONLY_TABS:
            n_rows = row_counts.get(tab, 5000)
            hdr_raw = run_cmd(["gsheets", "readonly", "read", source_id, f"{tab}!1:1", "--json"])
            start_r = max(2, n_rows - TAIL_ONLY_TABS[tab] - 5)
            tail_raw = run_cmd(["gsheets", "readonly", "read", source_id, f"{tab}!A{start_r}:AF{n_rows}", "--json"])
            hdr = json.loads(hdr_raw)
            tail = json.loads(tail_raw)
            rows = (hdr if isinstance(hdr, list) else []) + [
                r for r in (tail if isinstance(tail, list) else []) if r and str(r[0]).strip()
            ]
        else:
            raw = run_cmd(["gsheets", "readonly", "read", source_id, f"{tab}!A1:Z250", "--json"])
            rows = json.loads(raw)
        save_rows_csv(rows if isinstance(rows, list) else [], out_dir / f"{tab}.csv")
        print(f"  Fetched {tab:32s}: {len(rows):4d} rows")


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def parse_sectioned_csv(rows: list[list[str]]) -> dict[int, tuple[str, list[dict[str, str]]]]:
    """
    Parse multi-section Agent 3 CSVs where each section starts with a cell matching
    'SECTION <N>: ...', followed by a column header row and data rows until a blank row.
    """
    sections: dict[int, tuple[str, list[dict[str, str]]]] = {}
    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]
        first = str(row[0]).strip() if row else ""
        m = re.match(r"^SECTION\s+(\d+)\s*:\s*(.*)$", first, flags=re.IGNORECASE)
        if not m:
            i += 1
            continue
        sec_num = int(m.group(1))
        sec_title = m.group(2).strip()
        i += 1
        if i >= n:
            break
        raw_headers = [str(c).strip() for c in rows[i]]
        # Trim trailing empty headers
        while raw_headers and raw_headers[-1] == "":
            raw_headers.pop()
        headers = raw_headers
        i += 1
        records: list[dict[str, str]] = []
        while i < n:
            cur = rows[i]
            non_empty = [str(c).strip() for c in cur if str(c).strip() != ""]
            if not non_empty:
                i += 1
                break
            if re.match(r"^SECTION\s+\d+\s*:", str(cur[0]).strip(), flags=re.IGNORECASE):
                break
            rec = {headers[col_i]: (str(cur[col_i]).strip() if col_i < len(cur) else "") for col_i in range(len(headers))}
            records.append(rec)
            i += 1
        sections[sec_num] = (sec_title, records)
    return sections


def to_float(val: Any, default: float = float("nan")) -> float:
    if val is None:
        return default
    s = str(val).strip().replace(",", "").replace("%", "")
    if not s or s.lower() in {"nan", "none", "n/a"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


# =============================================================================
# 3. DETERMINISTIC QUANTITATIVE SIGNAL EXTRACTION
# =============================================================================


def build_quantitative_brief(exploration_dir: Path, source_id: str) -> dict[str, Any]:
    """Extract all verified quantitative diagnostics from Agent 3 CSV files."""
    exec_sec = parse_sectioned_csv(read_csv_rows(exploration_dir / "Executive_Summary.csv"))
    pca_ov_sec = parse_sectioned_csv(read_csv_rows(exploration_dir / "PCA_Overview_and_Loadings.csv"))
    snap_sec = parse_sectioned_csv(read_csv_rows(exploration_dir / "Rolling_PCA_Loadings_Snapshots.csv"))
    fa_sec = parse_sectioned_csv(read_csv_rows(exploration_dir / "Factor_Analysis_Results.csv"))
    hc_sec = parse_sectioned_csv(read_csv_rows(exploration_dir / "Hierarchical_Clustering.csv"))

    roll_df = pd.read_csv(exploration_dir / "Rolling_PCA_Dynamics.csv")
    roll_df["Date"] = roll_df["Date"].astype(str).str.strip()
    roll_df = roll_df[roll_df["Date"] != ""].reset_index(drop=True)

    pca_sc_df = pd.read_csv(exploration_dir / "PCA_Scores_TimeSeries.csv")
    pca_sc_df["Date"] = pca_sc_df["Date"].astype(str).str.strip()
    pca_sc_df = pca_sc_df[pca_sc_df["Date"] != ""].reset_index(drop=True)

    fa_sc_df = pd.read_csv(exploration_dir / "Factor_Scores_TimeSeries.csv")
    fa_sc_df["Date"] = fa_sc_df["Date"].astype(str).str.strip()
    fa_sc_df = fa_sc_df[fa_sc_df["Date"] != ""].reset_index(drop=True)

    # Metadata from Executive Summary
    meta_map = {r.get("Metric", ""): r for r in exec_sec.get(1, ("", []))[1]}
    findings_map = {r.get("Metric", ""): r for r in exec_sec.get(2, ("", []))[1]}
    sample_window = meta_map.get("Sample Window (is_core_ml_ready == 1)", {}).get("Value", "")
    universe_desc = meta_map.get("Primary Analytical Universe", {}).get("Value", "35 Daily-Traded Multi-Asset Series")

    # Rolling PCA latest vs 1W (5d ago) vs 1M (21d ago)
    latest = roll_df.iloc[-1]
    latest_date = str(latest["Date"])
    idx_1m = max(0, len(roll_df) - 21)
    row_1m = roll_df.iloc[idx_1m]
    date_1m = str(row_1m["Date"])

    # Consecutive sessions in current Systemic_Regime
    cur_regime = str(latest["Systemic_Regime"])
    regime_streak = 0
    for r_val in reversed(roll_df["Systemic_Regime"].astype(str).tolist()):
        if r_val == cur_regime:
            regime_streak += 1
        else:
            break

    # Recent 21-day peak AR5
    tail_21 = roll_df.iloc[idx_1m:]
    peak_21_idx = int(tail_21["Absorption_Ratio_Top5"].astype(float).idxmax())
    peak_21_row = roll_df.loc[peak_21_idx]
    fast_peak_idx = int(tail_21["Fast63d_AR5"].astype(float).idxmax())
    fast_peak_row = roll_df.loc[fast_peak_idx]

    # Historical AR5 summary from Executive_Summary / Annual Snapshots
    annual_rows = snap_sec.get(1, ("", []))[1]
    latest_year_row = annual_rows[-1] if annual_rows else {}
    prior_year_row = annual_rows[-2] if len(annual_rows) >= 2 else {}
    hist_ar5_summary = findings_map.get("2. Rolling PCA (252d) — Historical Absorption Ratio (AR5)", {}).get("Value", "")
    stress_peaks = snap_sec.get(3, ("", []))[1][:6]

    # Pre-2022 Stock-Bond Product historical average
    pre22_sb = [
        to_float(r.get("Avg_Stock_Bond_Product"))
        for r in annual_rows
        if r.get("Year", "").isdigit() and int(r["Year"]) <= 2021
    ]
    pre22_sb_mean = float(np.nanmean(pre22_sb)) if pre22_sb else -0.50

    # Full-sample PCA summary
    pca_scree = pca_ov_sec.get(1, ("", []))[1]
    full_pca_summary = {
        "pc1_var_pct": to_float(pca_scree[0]["Variance_Explained_Pct"]) if len(pca_scree) > 0 else float("nan"),
        "top3_cum_pct": to_float(pca_scree[2]["Cumulative_Variance_Pct"]) if len(pca_scree) > 2 else float("nan"),
        "top5_cum_pct": to_float(pca_scree[4]["Cumulative_Variance_Pct"]) if len(pca_scree) > 4 else float("nan"),
        "top10_cum_pct": to_float(pca_scree[9]["Cumulative_Variance_Pct"]) if len(pca_scree) > 9 else float("nan"),
        "pa_signal_count": sum(1 for r in pca_scree if r.get("PA_Keep") == "Yes"),
        "mp_signal_count": sum(1 for r in pca_scree if r.get("MP_Signal") == "Signal"),
        "kaiser_count": sum(1 for r in pca_scree if r.get("Kaiser_Keep_GT1") == "Yes"),
        "eff_rank_detail": findings_map.get("1. Full-Sample PCA — PC1 / Top-3 / Top-5 Variance", {}).get("Details / Methodology", ""),
    }

    # Bellwether Asset PC1 Loadings from Rolling_PCA_Loadings_Snapshots Section 2
    snap_assets = snap_sec.get(2, ("", []))[1]
    ye_cols = [k for k in (snap_assets[0].keys() if snap_assets else []) if k.startswith("YE_")]
    latest_ye_col = ye_cols[-1] if ye_cols else "Latest_Date_Val"
    prior_ye_col = ye_cols[-2] if len(ye_cols) >= 2 else latest_ye_col
    prior_ye_label = prior_ye_col.replace("YE_", "YE ")

    asset_beta_map: dict[str, dict[str, Any]] = {}
    for r in snap_assets:
        name = r.get("Asset", "")
        if not name:
            continue
        asset_beta_map[name] = {
            "asset": name,
            "asset_class": r.get("Asset_Class", ""),
            "latest": to_float(r.get("Latest_Date_Val", r.get(latest_ye_col))),
            "prior_ye": to_float(r.get(prior_ye_col)),
            "hist_mean": to_float(r.get("Hist_Mean")),
            "hist_std": to_float(r.get("Hist_Std")),
            "hist_min": to_float(r.get("Hist_Min")),
            "hist_max": to_float(r.get("Hist_Max")),
        }

    # Factor Analysis diagnostics & 8-factor scorecard (1D, 1W=5d, 1M=21d)
    fa_diag_map = {r.get("Diagnostic", ""): r.get("Value", "") for r in fa_sec.get(1, ("", []))[1]}
    fa_themes = fa_sec.get(2, ("", []))[1]
    promax_phi = fa_sec.get(5, ("", []))[1]

    fa_tail_5 = fa_sc_df.tail(5)
    fa_tail_21 = fa_sc_df.tail(21)
    fa_scorecard: list[dict[str, Any]] = []
    for row in fa_themes:
        fid = row.get("Factor", "")
        if not fid or fid not in fa_sc_df.columns:
            continue
        s1 = float(fa_sc_df[fid].astype(float).iloc[-1])
        s5 = float(fa_tail_5[fid].astype(float).sum())
        s21 = float(fa_tail_21[fid].astype(float).sum())
        # Largest single-day swing in last 5 sessions
        tail5_vals = fa_tail_5[fid].astype(float).to_numpy()
        tail5_dates = fa_tail_5["Date"].astype(str).tolist()
        max_abs_i = int(np.argmax(np.abs(tail5_vals)))
        fa_scorecard.append(
            {
                "factor": fid,
                "theme": row.get("Economic_Theme", ""),
                "var_pct": to_float(row.get("Variance_Explained_Pct")),
                "top_pos": row.get("Top_Positive_Loadings", ""),
                "top_neg": row.get("Top_Negative_Loadings", ""),
                "move_1d": round(s1, 2),
                "move_1w": round(s5, 2),
                "move_1m": round(s21, 2),
                "max_5d_swing_val": round(float(tail5_vals[max_abs_i]), 2),
                "max_5d_swing_date": tail5_dates[max_abs_i],
            }
        )

    # Top Promax oblique cross-factor pairs
    promax_pairs: list[dict[str, Any]] = []
    factor_ids = [r.get("Factor", "") for r in promax_phi if r.get("Factor")]
    theme_lookup = {r.get("Factor", ""): r.get("Economic_Theme", "") for r in promax_phi}
    for i_idx, r in enumerate(promax_phi):
        fi = r.get("Factor", "")
        for j_idx in range(i_idx + 1, len(factor_ids)):
            fj = factor_ids[j_idx]
            val = to_float(r.get(fj))
            if not np.isnan(val):
                promax_pairs.append(
                    {
                        "pair": f"{fi} ({theme_lookup.get(fi, '')}) <-> {fj} ({theme_lookup.get(fj, '')})",
                        "corr": round(val, 3),
                    }
                )
    promax_pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)

    # Recent PCA scores (PC1..PC5)
    pca_tail_5 = pca_sc_df.tail(5)
    pca_tail_21 = pca_sc_df.tail(21)
    pca_moves = {}
    for pc in ["PC1", "PC2", "PC3", "PC4", "PC5"]:
        if pc in pca_sc_df.columns:
            pca_moves[pc] = {
                "1d": round(float(pca_sc_df[pc].astype(float).iloc[-1]), 2),
                "1w": round(float(pca_tail_5[pc].astype(float).sum()), 2),
                "1m": round(float(pca_tail_21[pc].astype(float).sum()), 2),
            }

    # Clustering & HRP profiles
    hc_assets = hc_sec.get(2, ("", []))[1]
    hc_clusters = hc_sec.get(3, ("", []))[1]
    clusters_clean = []
    for c in hc_clusters:
        clusters_clean.append(
            {
                "cluster_id": int(to_float(c.get("Cluster_ID"), 0)),
                "label": c.get("Cluster_Label", ""),
                "asset_count": int(to_float(c.get("Asset_Count"), 0)),
                "mean_intra_corr": round(to_float(c.get("Mean_Intra_Corr"), 0.0), 2),
                "mean_inter_corr": round(to_float(c.get("Mean_Inter_Corr"), 0.0), 2),
                "hrp_weight_pct": round(to_float(c.get("Total_HRP_Weight_Pct"), 0.0), 2),
                "members": c.get("Member_Assets", ""),
            }
        )
    clusters_clean.sort(key=lambda x: x["hrp_weight_pct"], reverse=True)

    top_hrp_assets = sorted(
        [
            {
                "asset": r.get("Asset", ""),
                "cluster": r.get("Cluster_Label", ""),
                "hrp_weight_pct": round(to_float(r.get("HRP_Corr_Weight_Pct"), 0.0), 2),
            }
            for r in hc_assets
            if r.get("Asset")
        ],
        key=lambda x: x["hrp_weight_pct"],
        reverse=True,
    )[:6]

    ar5_latest = float(latest["Absorption_Ratio_Top5"])
    ar5_z_latest = float(latest["AR5_Causal_Z252"])
    fast_ar5_latest = float(latest["Fast63d_AR5"])

    brief = {
        "generated_sgt": dt.datetime.now(SGT_TZ).strftime("%Y-%m-%d %H:%M SGT"),
        "report_date_sgt": dt.datetime.now(SGT_TZ).strftime("%Y-%m-%d"),
        "data_as_of": latest_date,
        "date_1m_ago": date_1m,
        "sample_window": sample_window,
        "universe_desc": universe_desc,
        "source_sheet_id": source_id or DEFAULT_EXPLORATION_ID,
        "prior_ye_label": prior_ye_label,
        "rolling_pca": {
            "regime": cur_regime,
            "regime_1m_ago": str(row_1m["Systemic_Regime"]),
            "regime_streak_days": regime_streak,
            "ar5_pct": round(ar5_latest, 2),
            "ar5_1m_ago_pct": round(float(row_1m["Absorption_Ratio_Top5"]), 2),
            "ar5_z252": round(ar5_z_latest, 2),
            "ar5_z252_1m_ago": round(float(row_1m["AR5_Causal_Z252"]), 2),
            "ar5_21d_peak_pct": round(float(peak_21_row["Absorption_Ratio_Top5"]), 2),
            "ar5_21d_peak_z": round(float(peak_21_row["AR5_Causal_Z252"]), 2),
            "ar5_21d_peak_date": str(peak_21_row["Date"]),
            "fast63d_ar5_pct": round(fast_ar5_latest, 2),
            "fast63d_ar5_1m_ago_pct": round(float(row_1m["Fast63d_AR5"]), 2),
            "fast_minus_slow_pp": round(fast_ar5_latest - ar5_latest, 2),
            "fast63d_21d_peak_pct": round(float(fast_peak_row["Fast63d_AR5"]), 2),
            "fast63d_21d_peak_date": str(fast_peak_row["Date"]),
            "pc1_var_pct": round(float(latest["PC1_Var_Pct"]), 2),
            "pc1_var_1m_ago_pct": round(float(row_1m["PC1_Var_Pct"]), 2),
            "pc2_var_pct": round(float(latest["PC2_Var_Pct"]), 2),
            "pc3_var_pct": round(float(latest["PC3_Var_Pct"]), 2),
            "pc3_var_1m_ago_pct": round(float(row_1m["PC3_Var_Pct"]), 2),
            "eff_rank": round(float(latest["Effective_Rank_Entropy"]), 2),
            "eff_rank_1m_ago": round(float(row_1m["Effective_Rank_Entropy"]), 2),
            "pc1_drift_21d": round(float(latest["PC1_Subspace_Drift_21d"]), 4),
            "stock_bond_product": round(float(latest["Stock_Bond_PC1_Product"]), 4),
            "stock_bond_1m_ago": round(float(row_1m["Stock_Bond_PC1_Product"]), 4),
            "stock_bond_pre2022_avg": round(pre22_sb_mean, 2),
            "ytd_crisis_days": int(to_float(latest_year_row.get("Crisis_Days_Count"), 0)),
            "latest_year": str(latest_year_row.get("Year", latest_date[:4])),
            "prior_year_sb_avg": to_float(prior_year_row.get("Avg_Stock_Bond_Product"), float("nan")),
            "hist_ar5_summary": hist_ar5_summary,
            "top_stress_peaks": stress_peaks,
        },
        "full_pca": full_pca_summary,
        "pca_moves": pca_moves,
        "bellwether_betas": asset_beta_map,
        "factor_analysis": {
            "kmo": fa_diag_map.get("Overall KMO (Kaiser-Meyer-Olkin)", ""),
            "rmsr": fa_diag_map.get("RMSR (Root Mean Square Residual)", ""),
            "cfi": fa_diag_map.get("Comparative Fit Index (CFI)", ""),
            "tli": fa_diag_map.get("Tucker-Lewis Index (TLI)", ""),
            "var_explained_pct": fa_diag_map.get("Total Common Variance Explained (%)", ""),
            "scorecard": fa_scorecard,
            "top_promax_pairs": promax_pairs[:5],
        },
        "clustering_hrp": {
            "k_clusters": len(clusters_clean),
            "clusters": clusters_clean,
            "top_hrp_assets": top_hrp_assets,
        },
    }
    return brief


# =============================================================================
# 4. NARRATIVE SYNTHESIS (GEMINI API + SEARCH GROUNDING OR DETERMINISTIC FALLBACK)
# =============================================================================


def build_deterministic_narrative(brief: dict[str, Any]) -> dict[str, Any]:
    """High-craft deterministic narrative when GEMINI_API_KEY is absent or offline."""
    rp = brief["rolling_pca"]
    betas = brief["bellwether_betas"]
    fa_list = brief["factor_analysis"]["scorecard"]
    clusters = brief["clustering_hrp"]["clusters"]

    sb_prod = rp["stock_bond_product"]
    sb_status = (
        "positive (stocks and long-duration Treasuries are moving together, impairing traditional 60/40 hedging)"
        if sb_prod > 0.15
        else "negative (Treasuries are acting as a traditional risk-off hedge against equities)"
        if sb_prod < -0.15
        else "near-neutral (stocks and Treasuries are largely decoupled on the primary macro axis)"
    )

    wti = betas.get("WTI Crude", {})
    wti_lat = wti.get("latest", float("nan"))
    wti_mean = wti.get("hist_mean", float("nan"))
    oil_bullet = (
        f"Crude oil's PC1 loading stands at {wti_lat:+.3f} vs. an 18-year historical mean of {wti_mean:+.3f}, "
        + (
            "acting as an inverse inflation-tax drag where energy spikes pressure both equities and bonds."
            if wti_lat < -0.25
            else "moving broadly in sync with global cyclical growth expectations."
        )
    )

    sorted_1m = sorted(fa_list, key=lambda x: abs(x["move_1m"]), reverse=True)
    top_f1 = sorted_1m[0] if sorted_1m else {"factor": "F1", "theme": "Rates", "move_1m": 0.0, "move_1w": 0.0}
    top_f2 = sorted_1m[1] if len(sorted_1m) > 1 else top_f1

    top2_clusters = clusters[:2]
    top2_hrp_sum = sum(c["hrp_weight_pct"] for c in top2_clusters)
    top2_names = " and ".join(f"{c['label']} ({c['hrp_weight_pct']:.1f}%)" for c in top2_clusters)
    western = next((c for c in clusters if "Western Equities" in c["label"]), None)
    west_note = (
        f" versus just {western['hrp_weight_pct']:.1f}% across all {western['asset_count']} crowded {western['label']} assets combined."
        if western
        else "."
    )

    factor_notes: dict[str, str] = {}
    for f in fa_list:
        m1, m5, m21 = f["move_1d"], f["move_1w"], f["move_1m"]
        direction = "strong downward pressure" if m21 <= -3.0 else "strong upward momentum" if m21 >= 3.0 else "moderate drift" if abs(m21) >= 1.2 else "range-bound consolidation"
        factor_notes[f["factor"]] = (
            f"{direction.capitalize()} over 1M ({m21:+.2f}\u03c3); 1W cumulative {m5:+.2f}\u03c3 "
            f"(largest recent swing {f['max_5d_swing_val']:+.2f}\u03c3 on {f['max_5d_swing_date']})."
        )

    ongoing_items = [
        {
            "title": "Systemic Cross-Asset Coupling Trajectory",
            "detail": (
                f"Markets have spent {rp['regime_streak_days']} consecutive trading days in the {rp['regime']} regime "
                f"(252d AR5 = {rp['ar5_pct']:.2f}%, Z = {rp['ar5_z252']:+.2f}). Fast 63-day coupling ({rp['fast63d_ar5_pct']:.2f}%) "
                f"sits {rp['fast_minus_slow_pp']:+.2f} pp relative to the 1-year baseline, indicating "
                + ("elevated short-term shock transmission across asset classes." if rp["fast_minus_slow_pp"] > 0 else "gradually normalizing short-term cross-asset dispersion.")
            ),
        },
        {
            "title": "Primary Factor Leadership & Dominant Dislocations",
            "detail": (
                f"Over the trailing 21 sessions, the largest latent factor moves occurred in {top_f1['factor']} "
                f"({top_f1['theme']}: {top_f1['move_1m']:+.2f}\u03c3 1M, {top_f1['move_1w']:+.2f}\u03c3 1W) and {top_f2['factor']} "
                f"({top_f2['theme']}: {top_f2['move_1m']:+.2f}\u03c3 1M, {top_f2['move_1w']:+.2f}\u03c3 1W)."
            ),
        },
        {
            "title": "Stock–Bond & Commodity Correlation Structure",
            "detail": (
                f"Stock–Bond PC1 coupling stands at {sb_prod:+.4f} (1M ago: {rp['stock_bond_1m_ago']:+.4f}; "
                f"pre-2022 average: {rp['stock_bond_pre2022_avg']:+.2f}), confirming that macro inflation/rate expectations "
                f"continue to govern cross-asset co-movement. {oil_bullet}"
            ),
        },
        {
            "title": "Live News Grounding Note",
            "detail": (
                "This run executed in deterministic quantitative mode (no GEMINI_API_KEY configured). "
                "Add a GEMINI_API_KEY secret in GitHub Actions to automatically enrich this section with live "
                "Google Search-grounded central bank, inflation, geopolitical, and earnings catalysts."
            ),
        },
    ]

    watchpoints = [
        f"Coupling Escalation Trigger: Watch whether 252d AR5 Z-score ({rp['ar5_z252']:+.2f}) crosses +1.50 or Fast 63d AR5 ({rp['fast63d_ar5_pct']:.2f}%) breaks 60%, which historically signals Crisis / High-Coupling stress.",
        f"Stock–Bond Regime Shift: Track Stock–Bond PC1 Product ({sb_prod:+.4f} vs. {rp['stock_bond_1m_ago']:+.4f} 1M ago); a sustained move back toward zero would restore Treasuries' traditional safe-haven buffer.",
        f"Extreme Factor Mean-Reversion vs. Continuation: Monitor {top_f1['factor']} ({top_f1['theme']}, {top_f1['move_1m']:+.2f}\u03c3 1M) and {top_f2['factor']} ({top_f2['theme']}, {top_f2['move_1m']:+.2f}\u03c3 1M) for capitulation reversals or trend follow-through.",
        f"Crude Oil Inflation Feedback Loop: With WTI PC1 loading at {wti_lat:+.3f}, watch energy supply headlines as a leading driver of global rate and equity volatility.",
        f"HRP Diversification Integrity: Maintain diversification across low-correlation blocks ({top2_names}, {top2_hrp_sum:.1f}% combined HRP weight) while monitoring whether Asia-Pacific PC1 loadings remain decoupled.",
    ]

    return {
        "mode": "deterministic_quantitative",
        "model": "none (offline quantitative mode)",
        "bottom_line_bullets": [
            f"Systemic Regime ({rp['regime']}, {rp['regime_streak_days']} consecutive sessions): 252d AR5 sits at {rp['ar5_pct']:.2f}% (Z = {rp['ar5_z252']:+.2f}), with fast 63d AR5 at {rp['fast63d_ar5_pct']:.2f}% ({rp['fast_minus_slow_pp']:+.2f} pp vs. 252d) and Effective Rank at {rp['eff_rank']:.2f}/35.",
            f"Stock–Bond & Energy Structure: Stock–Bond PC1 coupling is {sb_prod:+.4f} — {sb_status}. {oil_bullet}",
            f"Largest 1-Month Latent Factor Moves: {top_f1['factor']} ({top_f1['theme']}) moved {top_f1['move_1m']:+.2f}\u03c3 over 21 sessions ({top_f1['move_1w']:+.2f}\u03c3 1W), followed by {top_f2['factor']} ({top_f2['theme']}) at {top_f2['move_1m']:+.2f}\u03c3.",
            f"Portfolio Diversification (HRP): Top two diversifying blocks — {top2_names} — command {top2_hrp_sum:.1f}% of HRP weight{west_note}",
        ],
        "regime_bullets": [
            f"Fast vs. Slow Coupling: 63-day AR5 ({rp['fast63d_ar5_pct']:.2f}%, 1M peak {rp['fast63d_21d_peak_pct']:.2f}% on {rp['fast63d_21d_peak_date']}) sits {rp['fast_minus_slow_pp']:+.2f} pp versus the 252-day baseline ({rp['ar5_pct']:.2f}%).",
            f"Multi-Factor Variance Breadth: PC1 variance moved from {rp['pc1_var_1m_ago_pct']:.2f}% to {rp['pc1_var_pct']:.2f}% over the last month, while PC2 ({rp['pc2_var_pct']:.2f}%) and PC3 ({rp['pc3_var_pct']:.2f}%, from {rp['pc3_var_1m_ago_pct']:.2f}%) absorb meaningful secondary risk.",
            f"Stability & Historical Benchmark: 21-day PC1 Subspace Drift is {rp['pc1_drift_21d']:.4f} (low structural rotation), Effective Rank is {rp['eff_rank']:.2f} (vs. full-sample ~19.65), and YTD crisis days stand at {rp['ytd_crisis_days']}.",
        ],
        "structural_bullets": [
            f"Stock–Bond Coupling ({sb_prod:+.4f}): Compared to the pre-2022 historical average of {rp['stock_bond_pre2022_avg']:+.2f}, stocks and Treasuries remain strongly co-moving on the primary macro axis.",
            oil_bullet,
            "Regional & Geographic Dispersion: Compare Asia-Pacific equity PC1 loadings against Western benchmarks in the table above to identify where equity beta has decoupled from US rate dynamics.",
        ],
        "factor_commentary": factor_notes,
        "ongoing_developments": ongoing_items,
        "watchpoints": watchpoints,
        "grounding_sources": [],
    }


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM text response (handling optional markdown fences)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def synthesize_with_gemini(brief: dict[str, Any], model: str = DEFAULT_GEMINI_MODEL) -> dict[str, Any]:
    """
    Call Google Gemini API with Google Search grounding enabled to synthesize live ongoing
    macro/market developments directly linked to Agent 3's quantitative signals.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [info] GEMINI_API_KEY not set — using deterministic quantitative synthesis.")
        return build_deterministic_narrative(brief)

    # Compact verified payload for the prompt
    compact_betas = {
        k: brief["bellwether_betas"][k]
        for k in [
            "S&P 500",
            "Dow Jones",
            "Nasdaq Composite",
            "DAX",
            "FTSE 100",
            "Nikkei 225",
            "Hang Seng",
            "CSI 300",
            "STI",
            "US Agg Bond (AGG)",
            "US 20+Y Treasury (TLT)",
            "US High Yield (HYG)",
            "US 10Y Yield",
            "Gold",
            "Silver",
            "WTI Crude",
            "Brent Crude",
            "Copper",
            "DXY",
        ]
        if k in brief["bellwether_betas"]
    }
    prompt_data = {
        "data_as_of": brief["data_as_of"],
        "report_date_sgt": brief["report_date_sgt"],
        "rolling_pca": brief["rolling_pca"],
        "full_pca": brief["full_pca"],
        "pca_moves": brief["pca_moves"],
        "bellwether_betas": compact_betas,
        "factor_analysis": brief["factor_analysis"],
        "clustering_hrp": brief["clustering_hrp"],
    }

    prompt = (
        "You are Agent 4, an institutional cross-asset macro strategist writing an executive, easy-to-read "
        "Market Dynamics & Macro Insights briefing based on Agent 3's verified statistical data (PCA, strictly causal "
        "Rolling PCA, Maximum Likelihood Factor Analysis, and Hierarchical Clustering/HRP) PLUS live Google Search "
        f"research on current ongoing global market & macro developments around {brief['data_as_of']} / {brief['report_date_sgt']}.\n\n"
        "Use Google Search to gather the latest real-world macro developments (Fed, ECB, BOJ, PBOC policy & yields, "
        "US/Europe/China/Asia equity drivers, credit/AI/tech moves, crude oil WTI/Brent & geopolitical supply risks, "
        "gold/silver/copper moves, major FX, and trade/tariff developments) and EXPLICITLY connect those ongoing "
        "developments to Agent 3's exact quantitative signals below.\n\n"
        "Rules:\n"
        "- Never invent or alter any number from Agent 3's JSON below; cite exact figures.\n"
        "- Write in crisp, plain English so a busy decision-maker can scan and understand every takeaway in 2 minutes.\n"
        "- Return ONLY valid JSON (inside a ```json ... ``` block) with these exact keys:\n"
        "  {\n"
        '    "bottom_line_bullets": [4 concise executive bullets with bold lead-ins as plain text like "Lead-In: detail"],\n'
        '    "regime_bullets": [3 concise bullets diagnosing Rolling PCA coupling, 63d vs 252d spread, and historical comparison],\n'
        '    "structural_bullets": [3 concise bullets explaining Stock-Bond coupling, Commodity/Oil loading flips, and Asia/regional shifts],\n'
        '    "factor_commentary": {"F1": "...", "F2": "...", "F3": "...", "F4": "...", "F5": "...", "F6": "...", "F7": "...", "F8": "..."},\n'
        '    "ongoing_developments": [5 to 6 objects {"title": "...", "detail": "..."} tying live current news to Agent 3 signals],\n'
        '    "watchpoints": [5 concrete, actionable watchpoints for the next 1-2 weeks with trigger levels/events]\n'
        "  }\n\n"
        f"AGENT 3 VERIFIED QUANTITATIVE DATA:\n{json.dumps(prompt_data, indent=2)}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 4096},
    }
    req_bytes = json.dumps(payload).encode("utf-8")

    print(f"[3/5] Synthesizing deep insights + live ongoing developments via Gemini ({model} + Google Search)...")
    last_err: Exception | None = None
    for attempt in range(1, 4):
        req = urllib.request.Request(
            url,
            data=req_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            cand = (body.get("candidates") or [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            full_text = "\n".join(p.get("text", "") for p in parts if "text" in p)
            parsed = extract_json_object(full_text)

            # Extract Google Search grounding sources if present
            sources: list[dict[str, str]] = []
            seen_urls: set[str] = set()
            gm = cand.get("groundingMetadata", {})
            for ch in gm.get("groundingChunks", []):
                web = ch.get("web", {})
                uri = str(web.get("uri", "")).strip()
                title = str(web.get("title", "")).strip()
                if uri and uri not in seen_urls:
                    seen_urls.add(uri)
                    sources.append({"title": title or uri, "uri": uri})

            fallback = build_deterministic_narrative(brief)
            return {
                "mode": "gemini_search_grounded",
                "model": model,
                "bottom_line_bullets": parsed.get("bottom_line_bullets") or fallback["bottom_line_bullets"],
                "regime_bullets": parsed.get("regime_bullets") or fallback["regime_bullets"],
                "structural_bullets": parsed.get("structural_bullets") or fallback["structural_bullets"],
                "factor_commentary": parsed.get("factor_commentary") or fallback["factor_commentary"],
                "ongoing_developments": parsed.get("ongoing_developments") or fallback["ongoing_developments"],
                "watchpoints": parsed.get("watchpoints") or fallback["watchpoints"],
                "grounding_sources": sources[:10],
            }
        except Exception as exc:
            last_err = exc
            if attempt < 3:
                wait_s = 4 * attempt
                print(f"  [retry {attempt}/2] Gemini API call failed: {exc} — retrying in {wait_s}s...")
                time.sleep(wait_s)

    print(f"  [warn] Gemini API failed after retries ({last_err}); falling back to deterministic quantitative report.")
    fb = build_deterministic_narrative(brief)
    fb["mode"] = "deterministic_fallback_after_api_error"
    return fb


# =============================================================================
# 5. EXECUTIVE HTML & MARKDOWN RENDERERS (GOOGLE DOCS COMPATIBLE)
# =============================================================================


def fmt_lead_bullet(text: str) -> str:
    """Convert 'Lead-In: body text' into '<strong>Lead-In:</strong> body text' safely."""
    safe = html.escape(str(text))
    if ":" in safe and safe.index(":") < 85:
        lead, rest = safe.split(":", 1)
        return f"<strong>{lead}:</strong>{rest}"
    return safe


def fmt_signed(val: float, decimals: int = 2, suffix: str = "") -> str:
    if np.isnan(val):
        return "N/A"
    return f"{val:+.{decimals}f}{suffix}"


def render_html_report(brief: dict[str, Any], narrative: dict[str, Any]) -> str:
    """Render a clean, table-based executive HTML document optimized for Google Docs conversion."""
    rp = brief["rolling_pca"]
    betas = brief["bellwether_betas"]
    fa = brief["factor_analysis"]
    hc = brief["clustering_hrp"]

    regime_bg = "#FEF3F2" if "Crisis" in rp["regime"] else "#FFFAEB" if "Elevated" in rp["regime"] else "#ECFDF3"
    regime_fg = "#B42318" if "Crisis" in rp["regime"] else "#92400E" if "Elevated" in rp["regime"] else "#067647"
    sb_color = "#B42318" if rp["stock_bond_product"] > 0.15 else "#067647" if rp["stock_bond_product"] < -0.15 else "#1F2937"

    wti = betas.get("WTI Crude", {})
    wti_lat = wti.get("latest", float("nan"))
    wti_mean = wti.get("hist_mean", float("nan"))
    wti_min = wti.get("hist_min", float("nan"))

    # Bellwether comparison rows
    bellwether_groups = [
        ("US High Yield (HYG) / AGG", "US High Yield (HYG)", "US Agg Bond (AGG)"),
        ("US 10Y Yield / TLT", "US 10Y Yield", "US 20+Y Treasury (TLT)"),
        ("S&P 500 / Dow / DAX", "S&P 500", "Dow Jones"),
        ("WTI / Brent Crude", "WTI Crude", "Brent Crude"),
        ("Gold / Silver / Copper", "Gold", "Copper"),
        ("Nikkei / Hang Seng / CSI 300", "Nikkei 225", "CSI 300"),
    ]
    beta_rows_html = []
    for idx_b, (label, a1, a2) in enumerate(bellwether_groups):
        b1 = betas.get(a1, {})
        b2 = betas.get(a2, {})
        lat_str = f"{fmt_signed(b1.get('latest', float('nan')), 3)} / {fmt_signed(b2.get('latest', float('nan')), 3)}"
        mean_str = f"{fmt_signed(b1.get('hist_mean', float('nan')), 3)} / {fmt_signed(b2.get('hist_mean', float('nan')), 3)}"
        pye_str = f"{fmt_signed(b1.get('prior_ye', float('nan')), 3)} / {fmt_signed(b2.get('prior_ye', float('nan')), 3)}"
        bg = "#F8FAFC" if idx_b % 2 == 1 else "#FFFFFF"
        if "WTI" in label and not np.isnan(wti_lat) and wti_lat < -0.25:
            bg = "#FEF3F2"
        elif "Nikkei" in label and not np.isnan(b1.get("latest", float("nan"))) and abs(b1["latest"]) < 0.25:
            bg = "#ECFDF3"
        beta_rows_html.append(
            f'<tr style="background-color: {bg};">'
            f"<td><strong>{html.escape(label)}</strong></td>"
            f'<td align="center"><strong>{html.escape(lat_str)}</strong></td>'
            f'<td align="center">{html.escape(mean_str)}</td>'
            f'<td align="center">{html.escape(pye_str)}</td>'
            f"</tr>"
        )

    # 8-Factor scorecard rows
    fa_rows_html = []
    f_notes = narrative.get("factor_commentary", {})
    for idx_f, f in enumerate(fa["scorecard"]):
        bg = "#FEF3F2" if f["move_1m"] <= -4.0 else "#ECFDF3" if f["move_1m"] >= 4.0 else ("#F8FAFC" if idx_f % 2 == 1 else "#FFFFFF")
        c1m = "#B42318" if f["move_1m"] <= -2.5 else "#067647" if f["move_1m"] >= 2.5 else "#1F2937"
        c1w = "#B42318" if f["move_1w"] <= -2.0 else "#067647" if f["move_1w"] >= 2.0 else "#1F2937"
        note = f_notes.get(f["factor"], f"Largest 5d swing: {f['max_5d_swing_val']:+.2f}\u03c3 ({f['max_5d_swing_date']}).")
        fa_rows_html.append(
            f'<tr style="background-color: {bg};">'
            f'<td><strong>{html.escape(f["factor"])} &middot; {html.escape(f["theme"])}</strong><br>'
            f'<span style="font-size: 8.5pt; color: #6B7280;">Var: {f["var_pct"]:.1f}% | {html.escape(f["top_pos"][:60])}</span></td>'
            f'<td align="center">{f["move_1d"]:+.2f}&sigma;</td>'
            f'<td align="center" style="color: {c1w};"><strong>{f["move_1w"]:+.2f}&sigma;</strong></td>'
            f'<td align="center" style="color: {c1m};"><strong>{f["move_1m"]:+.2f}&sigma;</strong></td>'
            f"<td>{html.escape(str(note))}</td>"
            f"</tr>"
        )

    # Clustering / HRP rows
    hc_rows_html = []
    for idx_c, c in enumerate(hc["clusters"]):
        bg = "#ECFDF3" if idx_c < 3 else ("#FEF3F2" if "Western Equities" in c["label"] else ("#F8FAFC" if idx_c % 2 == 1 else "#FFFFFF"))
        hc_rows_html.append(
            f'<tr style="background-color: {bg};">'
            f'<td><strong>C{c["cluster_id"]} &middot; {html.escape(c["label"])}</strong></td>'
            f'<td align="center">{c["asset_count"]}</td>'
            f'<td align="center">{c["mean_intra_corr"]:.2f}</td>'
            f'<td align="center"><strong>{c["hrp_weight_pct"]:.2f}%</strong></td>'
            f'<td style="font-size: 8.8pt;">{html.escape(c["members"])}</td>'
            f"</tr>"
        )

    bl_bullets_html = "\n".join(
        f'<li style="margin-bottom: 5px;">{fmt_lead_bullet(b)}</li>'
        for b in narrative.get("bottom_line_bullets", [])
    )
    reg_bullets_html = "\n".join(
        f'<li style="margin-bottom: 6px;">{fmt_lead_bullet(b)}</li>'
        for b in narrative.get("regime_bullets", [])
    )
    str_bullets_html = "\n".join(
        f'<li style="margin-bottom: 5px;">{fmt_lead_bullet(b)}</li>'
        for b in narrative.get("structural_bullets", [])
    )
    ongoing_html = "\n".join(
        f'<li style="margin-bottom: 8px;"><strong>{html.escape(str(item.get("title", "")))}:</strong> '
        f'{html.escape(str(item.get("detail", "")))}</li>'
        for item in narrative.get("ongoing_developments", [])
    )
    watch_html = "\n".join(
        f'<li style="margin-bottom: 5px;">{fmt_lead_bullet(w)}</li>'
        for w in narrative.get("watchpoints", [])
    )

    promax_summary = "; ".join(
        f"{p['pair']} ({p['corr']:+.3f})" for p in fa.get("top_promax_pairs", [])[:3]
    )

    sources_block = ""
    if narrative.get("grounding_sources"):
        src_items = " &nbsp;|&nbsp; ".join(
            f'<a href="{html.escape(s["uri"])}" style="color: #1F4E78;">{html.escape(s["title"][:50])}</a>'
            for s in narrative["grounding_sources"][:8]
        )
        sources_block = f'<p style="font-size: 8.5pt; color: #6B7280; margin-top: 6px;"><strong>Live Search Grounding Sources:</strong> {src_items}</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(MASTER_DOC_TITLE)} — {html.escape(brief["report_date_sgt"])}</title>
</head>
<body style="font-family: Arial, Helvetica, sans-serif; color: #1F2937; line-height: 1.55; max-width: 820px; margin: 0 auto;">

<!-- HEADER BANNER -->
<table width="100%" cellpadding="14" cellspacing="0" style="background-color: #1F4E78; border-collapse: collapse; margin-bottom: 14px;">
  <tr>
    <td>
      <div style="font-size: 11pt; color: #D9E1F2; letter-spacing: 0.6px; text-transform: uppercase; font-weight: bold;">Agent 4 &middot; Cross-Asset Intelligence Brief</div>
      <div style="font-size: 20pt; color: #FFFFFF; font-weight: bold; margin-top: 4px;">Market Dynamics &amp; Macro Insights</div>
      <div style="font-size: 9.8pt; color: #E5EDF7; margin-top: 6px;">
        <strong>Data As Of:</strong> {html.escape(brief["data_as_of"])} ({html.escape(brief["sample_window"] or brief["universe_desc"])}) &nbsp;|&nbsp;
        <strong>Generated:</strong> {html.escape(brief["generated_sgt"])} &nbsp;|&nbsp;
        <strong>Engine:</strong> Agent 3 Stats + {html.escape(narrative.get("model", ""))}
      </div>
    </td>
  </tr>
</table>

<!-- EXECUTIVE CALLOUT / BOTTOM LINE -->
<table width="100%" cellpadding="12" cellspacing="0" style="background-color: {regime_bg}; border: 1px solid #F59E0B; border-left: 5px solid {regime_fg}; border-collapse: collapse; margin-bottom: 16px;">
  <tr>
    <td>
      <div style="font-size: 11.5pt; font-weight: bold; color: {regime_fg}; margin-bottom: 6px;">
        BOTTOM LINE &mdash; REGIME: {html.escape(rp["regime"].upper())} ({rp["regime_streak_days"]} CONSECUTIVE SESSIONS)
      </div>
      <ul style="margin: 0; padding-left: 18px; font-size: 10.2pt; color: #1F2937;">
        {bl_bullets_html}
      </ul>
    </td>
  </tr>
</table>

<!-- KPI SCORECARD TABLE -->
<table width="100%" cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; border-color: #CBD5E1; font-size: 9.3pt; margin-bottom: 18px;">
  <thead>
    <tr style="background-color: #1F4E78; color: #FFFFFF;">
      <th align="left">Headline Risk Metric</th>
      <th align="center">Latest ({html.escape(brief["data_as_of"])})</th>
      <th align="center">1M Ago ({html.escape(brief["date_1m_ago"])})</th>
      <th align="center">Historical Context</th>
      <th align="left">Plain-English Signal</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #FFFFFF;">
      <td><strong>Systemic Regime</strong></td>
      <td align="center" style="background-color: {regime_bg}; color: {regime_fg};"><strong>{html.escape(rp["regime"])}</strong></td>
      <td align="center">{html.escape(rp["regime_1m_ago"])}</td>
      <td align="center">{rp["ytd_crisis_days"]} crisis days in {html.escape(rp["latest_year"])}</td>
      <td>Active streak: {rp["regime_streak_days"]} trading days.</td>
    </tr>
    <tr style="background-color: #F8FAFC;">
      <td><strong>252d Absorption Ratio (AR5)</strong></td>
      <td align="center"><strong>{rp["ar5_pct"]:.2f}%</strong> (Z = {rp["ar5_z252"]:+.2f})</td>
      <td align="center">{rp["ar5_1m_ago_pct"]:.2f}% (Z = {rp["ar5_z252_1m_ago"]:+.2f})</td>
      <td align="center">{html.escape(rp["hist_ar5_summary"] or "18Y benchmark")}</td>
      <td>1M peak: {rp["ar5_21d_peak_pct"]:.2f}% (Z = {rp["ar5_21d_peak_z"]:+.2f}) on {html.escape(rp["ar5_21d_peak_date"])}.</td>
    </tr>
    <tr style="background-color: #FFFFFF;">
      <td><strong>Fast 63d Absorption Ratio</strong></td>
      <td align="center"><strong>{rp["fast63d_ar5_pct"]:.2f}%</strong></td>
      <td align="center">{rp["fast63d_ar5_1m_ago_pct"]:.2f}%</td>
      <td align="center">{rp["fast_minus_slow_pp"]:+.2f} pp vs. 252d</td>
      <td>Short-term coupling 1M peak: {rp["fast63d_21d_peak_pct"]:.2f}% ({html.escape(rp["fast63d_21d_peak_date"])}).</td>
    </tr>
    <tr style="background-color: #F8FAFC;">
      <td><strong>Effective Rank (Indep. Bets)</strong></td>
      <td align="center"><strong>{rp["eff_rank"]:.2f}</strong> / 35</td>
      <td align="center">{rp["eff_rank_1m_ago"]:.2f} / 35</td>
      <td align="center">21d PC1 Drift: {rp["pc1_drift_21d"]:.4f}</td>
      <td>Measures how many independent macro bets exist across all 35 assets.</td>
    </tr>
    <tr style="background-color: #FFFFFF;">
      <td><strong>Stock&ndash;Bond PC1 Product</strong></td>
      <td align="center" style="color: {sb_color};"><strong>{rp["stock_bond_product"]:+.4f}</strong></td>
      <td align="center">{rp["stock_bond_1m_ago"]:+.4f}</td>
      <td align="center">Pre-2022 avg: {rp["stock_bond_pre2022_avg"]:+.2f}</td>
      <td>{"Positive = stocks & bonds move together (impaired hedge)." if rp["stock_bond_product"] > 0.15 else "Negative = traditional safe-haven bond hedge."}</td>
    </tr>
    <tr style="background-color: #F8FAFC;">
      <td><strong>WTI Crude PC1 Loading</strong></td>
      <td align="center"><strong>{fmt_signed(wti_lat, 3)}</strong></td>
      <td align="center">&mdash;</td>
      <td align="center">Mean {fmt_signed(wti_mean, 3)} | Min {fmt_signed(wti_min, 3)}</td>
      <td>{"Inverse inflation-tax signal: oil shocks weigh on risk & duration." if not np.isnan(wti_lat) and wti_lat < -0.25 else "Pro-cyclical growth alignment."}</td>
    </tr>
  </tbody>
</table>

<!-- SECTION 1 -->
<h2 style="color: #1F4E78; border-bottom: 2px solid #1F4E78; padding-bottom: 4px; font-size: 13.5pt; margin-top: 20px;">
  1. Market Regime &amp; Systemic Risk Dynamics (Rolling PCA)
</h2>
<ul style="font-size: 10.2pt; padding-left: 20px;">
  {reg_bullets_html}
</ul>

<!-- SECTION 2 -->
<h2 style="color: #1F4E78; border-bottom: 2px solid #1F4E78; padding-bottom: 4px; font-size: 13.5pt; margin-top: 20px;">
  2. Cross-Asset Leadership &amp; Structural Shifts
</h2>
<table width="100%" cellpadding="7" cellspacing="0" border="1" style="border-collapse: collapse; border-color: #CBD5E1; font-size: 9.3pt; margin-bottom: 12px;">
  <thead>
    <tr style="background-color: #D9E1F2; color: #1F2937;">
      <th align="left">Bellwether Asset Pair</th>
      <th align="center">Latest 252d PC1</th>
      <th align="center">18Y Hist. Mean</th>
      <th align="center">{html.escape(brief["prior_ye_label"])}</th>
    </tr>
  </thead>
  <tbody>
    {"".join(beta_rows_html)}
  </tbody>
</table>
<ul style="font-size: 10.2pt; padding-left: 20px;">
  {str_bullets_html}
</ul>

<!-- SECTION 3 -->
<h2 style="color: #1F4E78; border-bottom: 2px solid #1F4E78; padding-bottom: 4px; font-size: 13.5pt; margin-top: 20px;">
  3. Latent Factor Scorecard (8 MLFA Macro Themes)
</h2>
<p style="font-size: 9.8pt; color: #374151;">
  MLFA Fit: KMO = <strong>{html.escape(fa["kmo"])}</strong> | RMSR = <strong>{html.escape(fa["rmsr"])}</strong> |
  CFI = <strong>{html.escape(fa["cfi"])}</strong> | Common Variance Explained = <strong>{html.escape(fa["var_explained_pct"])}%</strong>.
  Scores show cumulative standardized Thomson factor moves over 1 trading day, 1 week (5d), and 1 month (21d).
</p>
<table width="100%" cellpadding="7" cellspacing="0" border="1" style="border-collapse: collapse; border-color: #CBD5E1; font-size: 9pt; margin-bottom: 10px;">
  <thead>
    <tr style="background-color: #1F4E78; color: #FFFFFF;">
      <th align="left">Factor &amp; Economic Theme</th>
      <th align="center">1D</th>
      <th align="center">1W (5d)</th>
      <th align="center">1M (21d)</th>
      <th align="left">Momentum &amp; Key Driver</th>
    </tr>
  </thead>
  <tbody>
    {"".join(fa_rows_html)}
  </tbody>
</table>
<p style="font-size: 9.5pt; color: #4B5563;">
  <em>Top Promax Structural Cross-Factor Links (&Phi;):</em> {html.escape(promax_summary)}
</p>

<!-- SECTION 4 -->
<h2 style="color: #1F4E78; border-bottom: 2px solid #1F4E78; padding-bottom: 4px; font-size: 13.5pt; margin-top: 20px;">
  4. Current Ongoing Developments &amp; Macro Catalysts
</h2>
<ul style="font-size: 10.2pt; padding-left: 20px;">
  {ongoing_html}
</ul>

<!-- SECTION 5 -->
<h2 style="color: #1F4E78; border-bottom: 2px solid #1F4E78; padding-bottom: 4px; font-size: 13.5pt; margin-top: 20px;">
  5. Portfolio Diversification (Ward Clustering &amp; HRP) &amp; Watchpoints
</h2>
<table width="100%" cellpadding="6" cellspacing="0" border="1" style="border-collapse: collapse; border-color: #CBD5E1; font-size: 9pt; margin-bottom: 14px;">
  <thead>
    <tr style="background-color: #D9E1F2; color: #1F2937;">
      <th align="left">Cluster (Ward k={hc["k_clusters"]})</th>
      <th align="center">Assets</th>
      <th align="center">Intra-Corr</th>
      <th align="center">HRP Weight</th>
      <th align="left">Member Assets</th>
    </tr>
  </thead>
  <tbody>
    {"".join(hc_rows_html)}
  </tbody>
</table>

<!-- WATCHPOINTS BOX -->
<table width="100%" cellpadding="12" cellspacing="0" style="background-color: #F4F7FA; border: 1px solid #1F4E78; border-left: 5px solid #1F4E78; border-collapse: collapse; margin-bottom: 16px;">
  <tr>
    <td>
      <div style="font-size: 11.2pt; font-weight: bold; color: #1F4E78; margin-bottom: 6px;">
        KEY WATCHPOINTS FOR THE COMING 1&ndash;2 WEEKS
      </div>
      <ol style="margin: 0; padding-left: 20px; font-size: 9.8pt; color: #1F2937;">
        {watch_html}
      </ol>
    </td>
  </tr>
</table>

<!-- FOOTER -->
<p style="font-size: 8.5pt; color: #6B7280; border-top: 1px solid #E5E7EB; padding-top: 8px;">
  <strong>Methodology &amp; Source Audit:</strong> Quantitative signals extracted from Agent 3 workbook
  <em>3. Data Exploration - PCA, FA, Clustering</em> (ID: <code>{html.escape(brief["source_sheet_id"] or "local CSV")}</code>).
  Rolling PCA indicators (252d &amp; 63d) are strictly causal (zero lookahead); Full-Sample PCA, MLFA, and Ward/HRP provide structural benchmarks.
  Synthesis mode: <code>{html.escape(narrative.get("mode", ""))}</code>.
</p>
{sources_block}
</body>
</html>
"""


def render_markdown_report(brief: dict[str, Any], narrative: dict[str, Any]) -> str:
    """Render a clean Markdown archive alongside HTML for GitHub Actions artifacts & diffing."""
    rp = brief["rolling_pca"]
    lines = [
        f"# {MASTER_DOC_TITLE}",
        f"**Data As Of:** {brief['data_as_of']} | **Generated:** {brief['generated_sgt']} | **Regime:** {rp['regime']} ({rp['regime_streak_days']} sessions)",
        "",
        "## Bottom Line",
        *[f"- {b}" for b in narrative.get("bottom_line_bullets", [])],
        "",
        "## 1. Market Regime & Systemic Risk Dynamics",
        *[f"- {b}" for b in narrative.get("regime_bullets", [])],
        "",
        "## 2. Structural Shifts",
        *[f"- {b}" for b in narrative.get("structural_bullets", [])],
        "",
        "## 3. Latent Factor Scorecard (1D / 1W / 1M)",
        "| Factor | Theme | 1D | 1W (5d) | 1M (21d) | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    f_notes = narrative.get("factor_commentary", {})
    for f in brief["factor_analysis"]["scorecard"]:
        note = str(f_notes.get(f["factor"], "")).replace("|", "/")
        lines.append(
            f"| {f['factor']} | {f['theme']} | {f['move_1d']:+.2f}σ | {f['move_1w']:+.2f}σ | {f['move_1m']:+.2f}σ | {note} |"
        )
    lines += [
        "",
        "## 4. Current Ongoing Developments & Macro Catalysts",
        *[f"- **{x.get('title', '')}:** {x.get('detail', '')}" for x in narrative.get("ongoing_developments", [])],
        "",
        "## 5. Key Watchpoints (Next 1–2 Weeks)",
        *[f"{i}. {w}" for i, w in enumerate(narrative.get("watchpoints", []), start=1)],
        "",
    ]
    return "\n".join(lines)


# =============================================================================
# 6. GOOGLE DRIVE PUBLISHING (APPS SCRIPT WEBAPP, DRIVE API, OR GDRIVE CLI)
# =============================================================================


def publish_to_drive_webapp(
    html_content: str,
    report_date: str,
    agent_folder_id: str,
    master_doc_id: str,
    archive_folder_id: str,
) -> dict[str, Any]:
    archive_title = f"{report_date} - Market Dynamics & Macro Insights"
    try:
        res = webapp_post(
            {
                "action": "publish_doc",
                "folder_id": agent_folder_id,
                "folder_name": AGENT_FOLDER_NAME,
                "master_doc_id": master_doc_id,
                "master_title": MASTER_DOC_TITLE,
                "archive_folder_id": archive_folder_id,
                "archive_folder_name": ARCHIVE_FOLDER_NAME,
                "archive_title": archive_title,
                "html_content": html_content,
            }
        )
    except RuntimeError as exc:
        if "Unknown action: publish_doc" in str(exc):
            print(
                "::warning::Your deployed Google Apps Script Web App does not include 'publish_doc' yet. "
                "Paste the latest apps_script/Code.gs into Apps Script and deploy a new Web App version to enable "
                "automatic Google Drive publishing from GitHub Actions. Local HTML/MD artifacts were still generated."
            )
            return {"ok": False, "skipped_redeploy_needed": True}
        raise
    print(f"  Master Doc : {res.get('master_doc_name')} ({res.get('master_doc_id')}) -> {res.get('master_doc_url')}")
    print(f"  Archive Doc: {res.get('archive_doc_name')} ({res.get('archive_doc_id')}) -> {res.get('archive_doc_url')}")
    return res


def publish_to_drive_cli(
    latest_html: Path,
    dated_html: Path,
    agent_folder_id: str,
    master_doc_id: str,
    archive_folder_id: str,
) -> None:
    if not agent_folder_id:
        search_out = run_cmd(["gdrive", "readonly", "search", "--name-contains", AGENT_FOLDER_NAME, "--json"])
        matches = json.loads(search_out)
        for item in matches if isinstance(matches, list) else matches.get("files", []):
            if item.get("name") == AGENT_FOLDER_NAME:
                agent_folder_id = item.get("id", "")
                break
    if not agent_folder_id:
        raise RuntimeError("Could not locate 'Agent Folder' in Google Drive. Set DRIVE_AGENT_FOLDER_ID.")

    # Locate master doc & archive folder if IDs not pinned
    if not master_doc_id or not archive_folder_id:
        ls_raw = run_cmd(["gdrive", "readonly", "ls", agent_folder_id, "--max", "50", "--json"])
        items = json.loads(ls_raw)
        entries = items if isinstance(items, list) else items.get("files", [])
        for it in entries:
            if not master_doc_id and it.get("name") == MASTER_DOC_TITLE:
                master_doc_id = it.get("id", "")
            if not archive_folder_id and it.get("name") == ARCHIVE_FOLDER_NAME:
                archive_folder_id = it.get("id", "")

    if master_doc_id:
        run_cmd(
            [
                "gdrive",
                "mutate",
                "update",
                master_doc_id,
                str(latest_html),
                "--mime-type",
                "application/vnd.google-apps.document",
            ]
        )
        print(f"  Updated master Google Doc in place: {master_doc_id}")
    else:
        up = run_cmd(
            [
                "gdrive",
                "mutate",
                "upload",
                str(latest_html),
                "--parent",
                agent_folder_id,
                "--mime-type",
                "application/vnd.google-apps.document",
            ]
        )
        m = re.search(r"ID:\s*([A-Za-z0-9_-]+)", up)
        if m:
            master_doc_id = m.group(1)
            run_cmd(["gdrive", "mutate", "rename", master_doc_id, MASTER_DOC_TITLE])
        print(f"  Created master Google Doc: {master_doc_id}")

    if not archive_folder_id:
        mk = run_cmd(["gdrive", "mutate", "mkdir", ARCHIVE_FOLDER_NAME, "--parent", agent_folder_id])
        m = re.search(r"ID:\s*([A-Za-z0-9_-]+)", mk)
        if m:
            archive_folder_id = m.group(1)

    dated_title = dated_html.stem
    existing_dated_id = ""
    if archive_folder_id:
        ls_arch = run_cmd(["gdrive", "readonly", "ls", archive_folder_id, "--max", "100", "--json"])
        arch_items = json.loads(ls_arch)
        for it in (arch_items if isinstance(arch_items, list) else arch_items.get("files", [])):
            if it.get("name") == dated_title:
                existing_dated_id = it.get("id", "")
                break

    if existing_dated_id:
        run_cmd(
            [
                "gdrive",
                "mutate",
                "update",
                existing_dated_id,
                str(dated_html),
                "--mime-type",
                "application/vnd.google-apps.document",
            ]
        )
        print(f"  Updated dated archive Google Doc in place: {existing_dated_id}")
    elif archive_folder_id:
        run_cmd(
            [
                "gdrive",
                "mutate",
                "upload",
                str(dated_html),
                "--parent",
                archive_folder_id,
                "--mime-type",
                "application/vnd.google-apps.document",
            ]
        )
        print(f"  Uploaded dated archive Google Doc: {dated_title}")


def publish_to_drive_api(
    html_content: str,
    report_date: str,
    agent_folder_id: str,
    master_doc_id: str,
    archive_folder_id: str,
) -> None:
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaInMemoryUpload  # type: ignore

    creds = load_service_account_creds(["https://www.googleapis.com/auth/drive"])
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    if not agent_folder_id:
        res = (
            drive.files()
            .list(
                q=f"name='{AGENT_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)",
            )
            .execute()
        )
        files = res.get("files", [])
        if not files:
            raise RuntimeError("Could not find 'Agent Folder' via Drive API. Set DRIVE_AGENT_FOLDER_ID.")
        agent_folder_id = files[0]["id"]

    media = MediaInMemoryUpload(html_content.encode("utf-8"), mimetype="text/html", resumable=False)
    if master_doc_id:
        drive.files().update(fileId=master_doc_id, media_body=media, fields="id, webViewLink").execute()
        print(f"  Updated master Google Doc via Drive API: {master_doc_id}")
    else:
        meta = {
            "name": MASTER_DOC_TITLE,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [agent_folder_id],
        }
        created = drive.files().create(body=meta, media_body=media, fields="id, webViewLink").execute()
        print(f"  Created master Google Doc via Drive API: {created.get('id')} -> {created.get('webViewLink')}")

    if not archive_folder_id:
        res = (
            drive.files()
            .list(
                q=f"'{agent_folder_id}' in parents and name='{ARCHIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id)",
            )
            .execute()
        )
        if res.get("files"):
            archive_folder_id = res["files"][0]["id"]
        else:
            f_meta = {
                "name": ARCHIVE_FOLDER_NAME,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [agent_folder_id],
            }
            archive_folder_id = drive.files().create(body=f_meta, fields="id").execute()["id"]

    dated_title = f"{report_date} - Market Dynamics & Macro Insights"
    media2 = MediaInMemoryUpload(html_content.encode("utf-8"), mimetype="text/html", resumable=False)
    res_d = (
        drive.files()
        .list(
            q=f"'{archive_folder_id}' in parents and name='{dated_title}' and trashed=false",
            fields="files(id)",
        )
        .execute()
    )
    if res_d.get("files"):
        did = res_d["files"][0]["id"]
        drive.files().update(fileId=did, media_body=media2, fields="id").execute()
        print(f"  Updated archive Google Doc via Drive API: {did}")
    else:
        d_meta = {
            "name": dated_title,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [archive_folder_id],
        }
        created_d = drive.files().create(body=d_meta, media_body=media2, fields="id").execute()
        print(f"  Created archive Google Doc via Drive API: {created_d.get('id')}")


def publish_to_drive(
    html_content: str,
    latest_html: Path,
    dated_html: Path,
    report_date: str,
    agent_folder_id: str,
    master_doc_id: str,
    archive_folder_id: str,
    backend: str = "auto",
) -> None:
    chosen = resolve_drive_backend(backend)
    print(f"[5/5] Publishing native Google Docs to My Drive -> {AGENT_FOLDER_NAME} ({chosen})...")
    if chosen == "webapp":
        publish_to_drive_webapp(html_content, report_date, agent_folder_id, master_doc_id, archive_folder_id)
    elif chosen == "api":
        publish_to_drive_api(html_content, report_date, agent_folder_id, master_doc_id, archive_folder_id)
    else:
        publish_to_drive_cli(latest_html, dated_html, agent_folder_id, master_doc_id, archive_folder_id)


# =============================================================================
# 7. MAIN ORCHESTRATOR
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent 4: Market Dynamics & Macro Insight Synthesizer and Google Drive Publisher"
    )
    parser.add_argument(
        "--source-id",
        default=DEFAULT_EXPLORATION_ID,
        help="Agent 3 Google Sheet ID (default: EXPLORATION_SHEET_ID)",
    )
    parser.add_argument(
        "--exploration-dir",
        default=str(BASE_DIR / "data" / "exploration"),
        help="Directory containing Agent 3 CSV outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "insights"),
        help="Directory for Agent 4 HTML, Markdown, and JSON outputs",
    )
    parser.add_argument(
        "--agent-folder-id",
        default=DEFAULT_AGENT_FOLDER_ID,
        help="Google Drive 'Agent Folder' ID (optional; located by name if omitted)",
    )
    parser.add_argument(
        "--master-doc-id",
        default=DEFAULT_MASTER_DOC_ID,
        help="Master Google Doc ID for in-place updates (optional; located by name if omitted)",
    )
    parser.add_argument(
        "--archive-folder-id",
        default=DEFAULT_ARCHIVE_FOLDER_ID,
        help="Archive subfolder ID inside Agent Folder (optional; located/created if omitted)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "webapp", "api", "gdrive"],
        default="auto",
        help="Google backend: auto, webapp (Apps Script), api (Service Account), or gdrive (local CLI)",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model for Search-grounded synthesis (default: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip Gemini API call and use deterministic synthesis")
    parser.add_argument("--skip-fetch", action="store_true", help="Read existing local data/exploration/*.csv")
    parser.add_argument("--skip-upload", action="store_true", help="Write local report files only without uploading")
    args = parser.parse_args()

    exp_dir = Path(args.exploration_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        fetch_backend = "gsheets" if args.backend == "gdrive" else args.backend
        fetch_agent3_tabs(args.source_id, exp_dir, backend=fetch_backend)
    else:
        print(f"[1/5] Reading existing Agent 3 CSV outputs from {exp_dir}")

    missing = [t for t in REQUIRED_TABS if not (exp_dir / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Agent 3 CSV files in {exp_dir}: {missing}. Run Agent 3 first or omit --skip-fetch."
        )

    print("[2/5] Extracting quantitative regime, loading shift, latent factor & HRP signals...")
    brief = build_quantitative_brief(exp_dir, args.source_id)
    rp = brief["rolling_pca"]
    print(
        f"  Data as of {brief['data_as_of']} | Regime: {rp['regime']} ({rp['regime_streak_days']}d) | "
        f"AR5={rp['ar5_pct']:.2f}% (Z={rp['ar5_z252']:+.2f}) | Fast63d={rp['fast63d_ar5_pct']:.2f}% | "
        f"Stock-Bond={rp['stock_bond_product']:+.4f}"
    )

    if args.no_llm:
        print("[3/5] Running deterministic quantitative synthesis (--no-llm specified)...")
        narrative = build_deterministic_narrative(brief)
    else:
        narrative = synthesize_with_gemini(brief, model=args.gemini_model)

    print("[4/5] Rendering executive HTML & Markdown insight reports...")
    html_content = render_html_report(brief, narrative)
    md_content = render_markdown_report(brief, narrative)

    report_date = brief["report_date_sgt"]
    latest_html = out_dir / "latest_report.html"
    dated_html = out_dir / f"{report_date} - Market Dynamics & Macro Insights.html"
    latest_md = out_dir / "latest_report.md"
    brief_json = out_dir / "quantitative_brief.json"

    latest_html.write_text(html_content, encoding="utf-8")
    dated_html.write_text(html_content, encoding="utf-8")
    latest_md.write_text(md_content, encoding="utf-8")
    brief_json.write_text(json.dumps({"brief": brief, "narrative": narrative}, indent=2), encoding="utf-8")
    print(f"  Saved HTML : {latest_html} ({len(html_content):,} bytes)")
    print(f"  Saved Dated: {dated_html}")
    print(f"  Saved MD   : {latest_md}")

    if not args.skip_upload:
        publish_to_drive(
            html_content=html_content,
            latest_html=latest_html,
            dated_html=dated_html,
            report_date=report_date,
            agent_folder_id=args.agent_folder_id,
            master_doc_id=args.master_doc_id,
            archive_folder_id=args.archive_folder_id,
            backend=args.backend,
        )
        print("\n[DONE] Agent 4 Market Dynamics & Macro Insights published to Google Drive!")
    else:
        print("\n[DONE] Agent 4 local insight generation complete (--skip-upload specified).")


if __name__ == "__main__":
    main()
