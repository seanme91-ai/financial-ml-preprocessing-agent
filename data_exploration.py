"""
Institutional Multi-Asset Data Exploration Pipeline (Agent 3):
Full-Sample PCA, Strictly Causal Rolling PCA, Maximum Likelihood Factor Analysis
(Varimax & Promax), and Hierarchical Agglomerative Clustering (+ HRP).

Reads the ML-ready feature dataset prepared by Agent 2 (Preprocessed Data Google Sheet,
tab `ML_Ready_Dataset`, or local `data/processed/ML_Ready_Dataset.csv`) and publishes
9 richly structured, formatted analytical tabs to the destination Google Sheet
(`Data Exploration - PCA, FA, Clustering`).

Usage:
    py data_exploration.py --source-id <AGENT2_SHEET_ID> --dest-id <AGENT3_SHEET_ID>
    py data_exploration.py --skip-fetch --skip-upload   # Offline local run from data/processed/
    py data_exploration.py --skip-fetch --backend webapp  # Chained run right after Agent 2

Requires:
    Python 3.11+, pandas, numpy, scipy, and either:
      - Google Apps Script Web App bridge (APPS_SCRIPT_URL + APPS_SCRIPT_TOKEN),
      - Google Sheets API service account (GOOGLE_SERVICE_ACCOUNT_JSON), or
      - Authenticated `gsheets` CLI on PATH (local Antigravity environment).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    leaves_list,
    linkage,
    optimal_leaf_ordering,
)
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import chi2

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

DEFAULT_SOURCE_ID = os.environ.get("DEST_SHEET_ID", "")
DEFAULT_DEST_ID = os.environ.get("EXPLORATION_SHEET_ID", "")

# Metadata for all 44 stationary asset columns in Agent 2's ML_Ready_Dataset
ASSET_META: dict[str, tuple[str, str, bool]] = {
    # col_name: (Display Name, Asset Class, is_daily_market_traded)
    "ret1d_sp500": ("S&P 500", "Equity (US)", True),
    "ret1d_dow": ("Dow Jones", "Equity (US)", True),
    "ret1d_nasdaq": ("Nasdaq Composite", "Equity (US)", True),
    "ret1d_ftse100": ("FTSE 100", "Equity (Europe)", True),
    "ret1d_dax": ("DAX", "Equity (Europe)", True),
    "ret1d_nikkei225": ("Nikkei 225", "Equity (Asia)", True),
    "ret1d_hsi": ("Hang Seng", "Equity (Asia)", True),
    "ret1d_csi300": ("CSI 300", "Equity (Asia)", True),
    "ret1d_shcomp": ("Shanghai Composite", "Equity (Asia)", True),
    "ret1d_sti": ("STI", "Equity (Asia)", True),
    "ret1d_agg": ("US Agg Bond (AGG)", "Fixed Income ETF", True),
    "ret1d_tlt": ("US 20+Y Treasury (TLT)", "Fixed Income ETF", True),
    "ret1d_hyg": ("US High Yield (HYG)", "Fixed Income ETF", True),
    "ret1d_bndx": ("Global Agg Bond (BNDX)", "Fixed Income ETF", True),
    "ret1d_gold": ("Gold", "Commodity (Metals)", True),
    "ret1d_silver": ("Silver", "Commodity (Metals)", True),
    "ret1d_wti": ("WTI Crude", "Commodity (Energy)", True),
    "ret1d_brent": ("Brent Crude", "Commodity (Energy)", True),
    "ret1d_copper": ("Copper", "Commodity (Industrial)", True),
    "ret1d_natgas": ("Natural Gas", "Commodity (Energy)", True),
    "ret1d_wheat": ("Wheat", "Commodity (Agriculture)", True),
    "ret1d_corn": ("Corn", "Commodity (Agriculture)", True),
    "ret1d_soybeans": ("Soybeans", "Commodity (Agriculture)", True),
    "ret1d_coffee": ("Coffee", "Commodity (Softs)", True),
    "ret1d_sugar": ("Sugar", "Commodity (Softs)", True),
    "ret1d_dxy": ("DXY", "FX Index", True),
    "ret1d_eurusd": ("EUR/USD", "FX Spot", True),
    "ret1d_usdjpy": ("USD/JPY", "FX Spot", True),
    "ret1d_gbpusd": ("GBP/USD", "FX Spot", True),
    "ret1d_usdcny": ("USD/CNY", "FX Spot", True),
    "ret1d_usdsgd": ("USD/SGD", "FX Spot", True),
    "chg1d_us2y": ("US 2Y Yield", "Sovereign Yield", True),
    "chg1d_us10y": ("US 10Y Yield", "Sovereign Yield", True),
    "chg1d_us30y": ("US 30Y Yield", "Sovereign Yield", True),
    "chg1d_hk_hibor1m": ("HK 1M HIBOR", "Money Market Rate", True),
    "chg1d_cn10y": ("China 10Y Yield", "Sovereign Yield", True),
    "chg1d_fedfunds": ("Fed Funds Rate", "Policy Rate (Stepped)", False),
    "chg1d_ecb_dep": ("ECB Deposit Rate", "Policy Rate (Stepped)", False),
    "chg1d_boj_rate": ("BOJ Policy Rate", "Policy Rate (Stepped)", False),
    "chg1d_sg_sora": ("SG SORA", "Rate (Monthly Step)", False),
    "chg1d_sg10y": ("SG 10Y Yield", "Yield (Monthly Step)", False),
    "chg1d_hk_base": ("HK Base Rate", "Policy Rate (Stepped)", False),
    "chg1d_cn_lpr1y": ("China 1Y LPR", "Policy Rate (Stepped)", False),
    "chg1d_cn_lpr5y": ("China 5Y LPR", "Policy Rate (Stepped)", False),
}

LATE_INCEPTION_COLS = {"ret1d_bndx", "chg1d_cn_lpr5y"}

# Bellwether assets tracked daily in Rolling_PCA_Dynamics
ROLLING_PC1_TRACKED: list[tuple[str, str]] = [
    ("ret1d_sp500", "PC1_Load_SP500"),
    ("ret1d_tlt", "PC1_Load_TLT"),
    ("chg1d_us10y", "PC1_Load_US10Y"),
    ("ret1d_hyg", "PC1_Load_HYG"),
    ("ret1d_gold", "PC1_Load_Gold"),
    ("ret1d_wti", "PC1_Load_WTI"),
    ("ret1d_copper", "PC1_Load_Copper"),
    ("ret1d_dxy", "PC1_Load_DXY"),
    ("ret1d_usdjpy", "PC1_Load_USDJPY"),
    ("ret1d_csi300", "PC1_Load_CSI300"),
]

ROLLING_PC2_TRACKED: list[tuple[str, str]] = [
    ("ret1d_sp500", "PC2_Load_SP500"),
    ("ret1d_tlt", "PC2_Load_TLT"),
    ("chg1d_us10y", "PC2_Load_US10Y"),
    ("ret1d_gold", "PC2_Load_Gold"),
]

SUBHEADER_TOKENS = {
    "Metric",
    "Universe",
    "Tab_Name",
    "Component",
    "Asset",
    "Year",
    "Rank",
    "Diagnostic",
    "Factor",
    "Linkage_Method",
    "Cluster_ID",
    "Step",
    "Dendrogram_Order",
    "Asset_Order",
    "Leaf_No",
}


# =============================================================================
# 1. GOOGLE SHEETS I/O BACKENDS (Apps Script Web App / Sheets API / gsheets CLI)
# =============================================================================


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
    """Select Google Sheets backend ('webapp' via Apps Script, 'api' via service account, or 'gsheets' CLI)."""
    has_webapp = bool(
        os.environ.get("APPS_SCRIPT_URL", "").strip()
        and os.environ.get("APPS_SCRIPT_TOKEN", "").strip()
    )
    has_sa = bool(
        os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    )
    has_cli = shutil.which("gsheets") is not None

    if requested == "webapp":
        if not has_webapp:
            raise RuntimeError("Requested --backend webapp, but APPS_SCRIPT_URL/TOKEN not set.")
        return "webapp"
    if requested == "api":
        if not has_sa:
            raise RuntimeError("Requested --backend api, but service account credentials not set.")
        return "api"
    if requested == "gsheets":
        if not has_cli:
            raise RuntimeError("Requested --backend gsheets, but `gsheets` CLI is not on PATH.")
        return "gsheets"

    if has_webapp:
        return "webapp"
    if has_sa:
        return "api"
    if has_cli:
        return "gsheets"
    raise RuntimeError(
        "No Google Sheets credentials available. Set APPS_SCRIPT_URL + APPS_SCRIPT_TOKEN, "
        "GOOGLE_SERVICE_ACCOUNT_JSON, ensure `gsheets` CLI is on PATH, or pass --skip-fetch --skip-upload."
    )


def call_apps_script(payload: dict[str, Any], timeout: int = 90, max_retries: int = 4) -> dict[str, Any]:
    """POST a JSON action to the deployed Google Apps Script Web App with exponential backoff retries."""
    url = os.environ.get("APPS_SCRIPT_URL", "").strip()
    token = os.environ.get("APPS_SCRIPT_TOKEN", "").strip()
    if not url or not token:
        raise RuntimeError("APPS_SCRIPT_URL and APPS_SCRIPT_TOKEN must both be set.")

    body = json.dumps({**payload, "token": token}).encode("utf-8")
    action = payload.get("action", "unknown")
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                snippet = raw_text[:300].replace("\n", " ")
                raise RuntimeError(
                    "Apps Script returned non-JSON. Response preview: " + snippet
                ) from exc
            if not data.get("ok"):
                raise RuntimeError(f"Apps Script error: {data.get('error', data)}")
            return data
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
            last_exc = RuntimeError(f"Apps Script HTTP {exc.code}: {err_body}")
            retryable = exc.code in (429, 500, 502, 503, 504)
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
            last_exc = exc
            retryable = True

        if not retryable or attempt == max_retries:
            break
        wait_s = 5 * (2 ** (attempt - 1))
        print(
            f"    [WARN] Apps Script '{action}' attempt {attempt}/{max_retries} failed ({last_exc}); "
            f"retrying in {wait_s}s...",
            flush=True,
        )
        time.sleep(wait_s)

    raise RuntimeError(f"Apps Script '{action}' failed after {max_retries} attempts: {last_exc}") from last_exc


def build_sheets_service() -> Any:
    """Build an authenticated Google Sheets v4 API service from service-account credentials."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    sa_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_env:
        if sa_env.startswith("{"):
            creds = Credentials.from_service_account_info(json.loads(sa_env), scopes=scopes)
        else:
            creds = Credentials.from_service_account_file(sa_env, scopes=scopes)
    else:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        creds = Credentials.from_service_account_file(cred_path, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def fetch_agent2_ml_dataset(
    source_id: str,
    ml_csv: Path,
    sheet_name: str = "ML_Ready_Dataset",
    backend: str = "auto",
) -> None:
    """Fetch Agent 2's ML_Ready_Dataset tab from Google Sheets to local CSV."""
    if not source_id:
        raise ValueError("Agent 2 source sheet ID is empty. Pass --source-id or set DEST_SHEET_ID.")
    ml_csv.parent.mkdir(parents=True, exist_ok=True)
    chosen = resolve_sheets_backend(backend)
    print(f"[1/7] Fetching Agent 2 '{sheet_name}' from {source_id} via {chosen} -> {ml_csv}")

    if chosen == "webapp":
        all_rows: list[list[str]] = []
        start_row = 1
        chunk_rows = 800  # 148 cols wide -> keep payload modest for Apps Script
        while True:
            resp = call_apps_script(
                {
                    "action": "fetch",
                    "source_id": source_id,
                    "sheet_name": sheet_name,
                    "start_row": start_row,
                    "num_rows": chunk_rows,
                }
            )
            batch = resp.get("rows", [])
            if not batch:
                break
            all_rows.extend(batch)
            total_rows = int(resp.get("total_rows", len(all_rows)))
            print(f"    fetched rows {start_row}..{start_row + len(batch) - 1} / {total_rows}")
            if start_row + len(batch) - 1 >= total_rows or len(batch) < chunk_rows:
                break
            start_row += len(batch)
        if not all_rows:
            raise RuntimeError(f"Sheet {source_id} ({sheet_name}) returned no rows.")
        width = max(len(r) for r in all_rows)
        with ml_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in all_rows:
                writer.writerow(r + [""] * (width - len(r)))
        return

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
            raise RuntimeError(f"Sheet {source_id} ({sheet_name}) returned no rows.")
        width = max(len(r) for r in rows)
        with ml_csv.open("w", newline="", encoding="utf-8") as f:
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
            str(ml_csv),
            "--format",
            "csv",
            "--sheet",
            sheet_name,
        ]
    )


# =============================================================================
# 2. DATA PREPARATION & HELPER UTILITIES
# =============================================================================


def asset_name(col: str) -> str:
    return ASSET_META.get(col, (col, "Other", True))[0]


def asset_class(col: str) -> str:
    return ASSET_META.get(col, (col, "Other", True))[1]


def asset_slug(col: str) -> str:
    for prefix in ("ret1d_", "chg1d_"):
        if col.startswith(prefix):
            return col[len(prefix) :]
    return col


def standardize_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise standardization using sample std (ddof=1)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std_safe = np.where(std > 1e-12, std, 1.0)
    Z = (X - mean) / std_safe
    return Z, mean, std_safe


def winsorize_z(Z: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Winsorize standardized matrix at +/- clip standard deviations and re-standardize."""
    Zc = np.clip(Z, -clip, clip)
    Zc, _, _ = standardize_matrix(Zc)
    return Zc


def infer_factor_theme(loadings_col: np.ndarray, cols: list[str], threshold: float = 0.30) -> str:
    """Generate a concise, economically meaningful label weighted by squared loading magnitude."""
    order = np.argsort(np.abs(loadings_col))[::-1]
    top_idx = [i for i in order[:6] if abs(loadings_col[i]) >= threshold]
    if not top_idx:
        top_idx = list(order[:3])
    top_names = [asset_name(cols[i]) for i in top_idx]
    weights = np.array([loadings_col[i] ** 2 for i in top_idx], dtype=np.float64)
    w_tot = float(np.sum(weights)) + 1e-12

    def share(pred) -> float:
        return float(sum(w for i, w in zip(top_idx, weights) if pred(cols[i]))) / w_tot

    w_us_eq = share(lambda c: asset_class(c) == "Equity (US)")
    w_eu_eq = share(lambda c: asset_class(c) == "Equity (Europe)")
    w_cn_eq = share(lambda c: c in ("ret1d_csi300", "ret1d_shcomp", "ret1d_hsi"))
    w_asia_eq = share(lambda c: asset_class(c) == "Equity (Asia)")
    w_eq = w_us_eq + w_eu_eq + w_asia_eq
    w_rates = share(lambda c: any(k in asset_class(c) for k in ("Yield", "Fixed Income", "Rate")) and c != "ret1d_hyg")
    w_fx = share(lambda c: "FX" in asset_class(c))
    w_metals = share(lambda c: asset_name(c) in ("Gold", "Silver", "Copper"))
    w_energy = share(lambda c: asset_name(c) in ("WTI Crude", "Brent Crude", "Natural Gas"))
    w_ag = share(lambda c: "Agriculture" in asset_class(c) or "Softs" in asset_class(c))

    if w_rates >= 0.50:
        if share(lambda c: c in ("chg1d_hk_hibor1m", "chg1d_cn10y")) >= 0.40:
            return "Asia Rates & Liquidity"
        return "US Rates & Sovereign Duration"
    if w_cn_eq >= 0.55:
        return "China Onshore / Greater China Equities"
    if w_eu_eq >= 0.45 and w_us_eq < 0.25:
        return "European & Developed ex-US Equities"
    if w_us_eq >= 0.45 and w_eq >= 0.65:
        return "US Equities & Credit Risk-On" if "ret1d_hyg" in [cols[i] for i in top_idx] else "Global / US Risk-On Equities"
    if w_eq >= 0.50 and w_asia_eq >= 0.35:
        return "Asia-Pacific Equities"
    if w_eq >= 0.50:
        return "Global Risk-On Equities"
    if w_fx >= 0.50:
        return "Major FX vs US Dollar"
    if w_energy >= 0.50:
        if asset_name(cols[top_idx[0]]) == "Natural Gas" and len(top_idx) <= 2:
            return "Natural Gas Idiosyncratic"
        return "Energy / Crude Oil Complex"
    if w_metals >= 0.50:
        return "Precious & Industrial Metals"
    if w_ag >= 0.50:
        return "Agriculture / Grains Complex"
    if share(lambda c: c in ("chg1d_hk_hibor1m", "chg1d_cn10y")) >= 0.35:
        return "Asia Rates & Liquidity"
    return f"Cross-Asset ({', '.join(top_names[:3])})"


def top_loading_summary(loadings_col: np.ndarray, cols: list[str], sign: int = 1, k: int = 4) -> str:
    """Format top k positive (sign=1) or negative (sign=-1) asset loadings as readable text."""
    if sign > 0:
        idx = [i for i in np.argsort(loadings_col)[::-1] if loadings_col[i] > 0.15][:k]
    else:
        idx = [i for i in np.argsort(loadings_col) if loadings_col[i] < -0.15][:k]
    if not idx:
        return "None (|r| < 0.15)"
    return "; ".join(f"{asset_name(cols[i])} ({loadings_col[i]:+.2f})" for i in idx)


def fmt_num(x: float, dec: int = 4) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    return round(float(x), dec)


# =============================================================================
# 3. ANALYSIS 1: FULL-SAMPLE PCA + PARALLEL ANALYSIS + MARCHENKO-PASTUR
# =============================================================================


def run_parallel_analysis(n: int, p: int, n_sims: int = 200, seed: int = 42) -> np.ndarray:
    """Horn's Parallel Analysis: 95th percentile eigenvalues of uncorrelated N(0,1) correlation matrices."""
    rng = np.random.default_rng(seed)
    sim_evals = np.empty((n_sims, p), dtype=np.float64)
    for s in range(n_sims):
        Z_rand = rng.standard_normal((n, p))
        R_rand = np.corrcoef(Z_rand, rowvar=False)
        sim_evals[s] = np.linalg.eigvalsh(R_rand)[::-1]
    return np.percentile(sim_evals, 95, axis=0)


def run_full_pca(
    Z: np.ndarray,
    dates: pd.Series,
    cols: list[str],
    pa_sims: int = 200,
) -> dict[str, Any]:
    """Perform standardized correlation-matrix PCA with deterministic sign orientation & RMT diagnostics."""
    n, p = Z.shape
    R = np.corrcoef(Z, rowvar=False)
    R = 0.5 * (R + R.T)

    evals_asc, evecs_asc = np.linalg.eigh(R)
    evals = np.clip(evals_asc[::-1], 0.0, None)
    evecs = evecs_asc[:, ::-1].copy()

    # Deterministic economic sign orientation:
    # PC1 oriented so S&P 500 has positive loading (Global Risk-On);
    # PC2..p oriented so largest absolute loading is positive.
    sp_idx = cols.index("ret1d_sp500") if "ret1d_sp500" in cols else 0
    if evecs[sp_idx, 0] < 0:
        evecs[:, 0] *= -1.0
    for k in range(1, p):
        dom = int(np.argmax(np.abs(evecs[:, k])))
        if evecs[dom, k] < 0:
            evecs[:, k] *= -1.0

    var_exp = evals / np.sum(evals)
    cum_var = np.cumsum(var_exp)

    # Correlation loadings L_ij = corr(X_i, PC_j) = v_ij * sqrt(lambda_j)
    loadings = evecs * np.sqrt(evals)[None, :]
    # Percentage contribution of each asset to each PC: 100 * v_ij^2
    contrib_pct = 100.0 * (evecs ** 2)
    # Effective participation ratio per PC: 1 / sum(v_ij^4) (1 = single asset, p = equal weight)
    eff_part = 1.0 / np.sum(evecs ** 4, axis=0)

    # Horn's Parallel Analysis & Marchenko-Pastur RMT upper bound
    pa95 = run_parallel_analysis(n, p, n_sims=pa_sims, seed=42)
    mp_upper = (1.0 + math.sqrt(p / n)) ** 2
    mp_lower = (1.0 - math.sqrt(p / n)) ** 2

    kaiser_k = int(np.sum(evals > 1.0))
    pa_k = int(np.sum(evals > pa95))
    mp_k = int(np.sum(evals > mp_upper))

    # Effective rank via exponential Shannon entropy of normalized eigenvalues
    nz_p = var_exp[var_exp > 1e-12]
    eff_rank = float(np.exp(-np.sum(nz_p * np.log(nz_p))))

    # Standardized PC scores: S_raw = Z @ V -> unit-variance scores = S_raw / sqrt(lambda)
    raw_scores = Z @ evecs
    std_scores = raw_scores / np.sqrt(np.maximum(evals, 1e-12))[None, :]

    n_scores = min(10, p)
    scores_df = pd.DataFrame({"Date": dates.values})
    for k in range(n_scores):
        scores_df[f"PC{k + 1}"] = std_scores[:, k]

    themes = [infer_factor_theme(loadings[:, k], cols) for k in range(p)]

    return {
        "n": n,
        "p": p,
        "corr": R,
        "evals": evals,
        "evecs": evecs,
        "loadings": loadings,
        "contrib_pct": contrib_pct,
        "eff_part": eff_part,
        "var_exp_pct": 100.0 * var_exp,
        "cum_var_pct": 100.0 * cum_var,
        "pa95": pa95,
        "mp_upper": mp_upper,
        "mp_lower": mp_lower,
        "kaiser_k": kaiser_k,
        "pa_k": pa_k,
        "mp_k": mp_k,
        "eff_rank": eff_rank,
        "ar1": float(100.0 * cum_var[0]),
        "ar3": float(100.0 * cum_var[min(2, p - 1)]),
        "ar5": float(100.0 * cum_var[min(4, p - 1)]),
        "themes": themes,
        "scores_df": scores_df,
    }


# =============================================================================
# 4. ANALYSIS 2: STRICTLY CAUSAL ROLLING PCA (252d & 63d WINDOWS)
# =============================================================================


def run_rolling_pca(
    X_raw: np.ndarray,
    dates: pd.Series,
    cols: list[str],
    anchor_evecs: np.ndarray,
    window: int = 252,
    fast_window: int = 63,
) -> dict[str, Any]:
    """
    Strictly causal trailing Rolling PCA (zero lookahead).
    At every date t >= window - 1, standardizes X[t-window+1 : t+1] and computes
    correlation eigen-decomposition with day-by-day eigenvector sign alignment.
    """
    n, p = X_raw.shape
    n_roll = n - window + 1
    if n_roll <= 0:
        raise ValueError(f"Insufficient rows ({n}) for rolling window {window}.")

    roll_dates = dates.iloc[window - 1 :].reset_index(drop=True)
    mp_upper_slow = (1.0 + math.sqrt(p / window)) ** 2

    # Pre-allocate arrays for speed
    pc_var_pct = np.zeros((n_roll, 5), dtype=np.float64)
    ar1 = np.zeros(n_roll, dtype=np.float64)
    ar3 = np.zeros(n_roll, dtype=np.float64)
    ar5 = np.zeros(n_roll, dtype=np.float64)
    eff_rank = np.zeros(n_roll, dtype=np.float64)
    kaiser_cnt = np.zeros(n_roll, dtype=np.int32)
    mp_cnt = np.zeros(n_roll, dtype=np.int32)
    pc1_drift_21d = np.full(n_roll, np.nan, dtype=np.float64)

    # Track full PC1 loadings (for snapshots) and PC2 loadings (for bellwethers)
    pc1_loadings_all = np.zeros((n_roll, p), dtype=np.float64)
    pc1_evecs_all = np.zeros((n_roll, p), dtype=np.float64)
    pc2_loadings_all = np.zeros((n_roll, p), dtype=np.float64)

    # Fast (63d) rolling metrics aligned to the same output dates
    pc1_var_fast = np.zeros(n_roll, dtype=np.float64)
    ar5_fast = np.zeros(n_roll, dtype=np.float64)

    prev_v1 = anchor_evecs[:, 0].copy()
    prev_v2 = anchor_evecs[:, 1].copy()
    eq_anchor_idxs = [
        cols.index(c)
        for c in ("ret1d_sp500", "ret1d_dow", "ret1d_nasdaq", "ret1d_ftse100", "ret1d_dax")
        if c in cols
    ]

    for idx in range(n_roll):
        end = idx + window
        # Slow (252d) strictly causal window [end-window .. end-1]
        W_slow = X_raw[end - window : end]
        std_s = np.std(W_slow, axis=0, ddof=1)
        mean_s = np.mean(W_slow, axis=0)
        std_safe = np.where(std_s > 1e-12, std_s, 1.0)
        Zs = (W_slow - mean_s) / std_safe
        Rs = (Zs.T @ Zs) / (window - 1)
        Rs = 0.5 * (Rs + Rs.T)
        np.fill_diagonal(Rs, 1.0)

        ev_asc, ec_asc = np.linalg.eigh(Rs)
        ev = np.clip(ev_asc[::-1], 0.0, None)
        ec = ec_asc[:, ::-1]

        total_ev = float(np.sum(ev))
        vexp = ev / total_ev if total_ev > 0 else np.zeros(p)

        pc_var_pct[idx] = 100.0 * vexp[:5]
        ar1[idx] = 100.0 * vexp[0]
        ar3[idx] = 100.0 * np.sum(vexp[:3])
        ar5[idx] = 100.0 * np.sum(vexp[:5])
        nz = vexp[vexp > 1e-12]
        eff_rank[idx] = float(np.exp(-np.sum(nz * np.log(nz))))
        kaiser_cnt[idx] = int(np.sum(ev > 1.0))
        mp_cnt[idx] = int(np.sum(ev > mp_upper_slow))

        # Polarity alignment: orient PC1 so Global Developed Equities basket loads positively
        # (prevents post-rotation sign inversions), falling back to prior-day dot continuity
        v1 = ec[:, 0].copy()
        if eq_anchor_idxs:
            eq_sum = float(np.sum(v1[eq_anchor_idxs]))
            if abs(eq_sum) >= 0.05:
                if eq_sum < 0.0:
                    v1 *= -1.0
            elif float(np.dot(v1, prev_v1)) < 0.0:
                v1 *= -1.0
        elif float(np.dot(v1, prev_v1)) < 0.0:
            v1 *= -1.0

        v2 = ec[:, 1].copy()
        if float(np.dot(v2, prev_v2)) < 0.0:
            v2 *= -1.0

        prev_v1 = v1
        prev_v2 = v2
        pc1_evecs_all[idx] = v1
        pc1_loadings_all[idx] = v1 * math.sqrt(ev[0])
        pc2_loadings_all[idx] = v2 * math.sqrt(ev[1])

        if idx >= 21:
            cos_sim = abs(float(np.dot(v1, pc1_evecs_all[idx - 21])))
            pc1_drift_21d[idx] = max(0.0, 1.0 - min(1.0, cos_sim))

        # Fast (63d) strictly causal window [end-fast_window .. end-1]
        W_fast = X_raw[end - fast_window : end]
        std_f = np.std(W_fast, axis=0, ddof=1)
        mean_f = np.mean(W_fast, axis=0)
        std_fs = np.where(std_f > 1e-12, std_f, 1.0)
        Zf = (W_fast - mean_f) / std_fs
        Rf = (Zf.T @ Zf) / (fast_window - 1)
        Rf = 0.5 * (Rf + Rf.T)
        np.fill_diagonal(Rf, 1.0)
        ev_f = np.clip(np.linalg.eigvalsh(Rf)[::-1], 0.0, None)
        tot_f = float(np.sum(ev_f))
        if tot_f > 0:
            pc1_var_fast[idx] = 100.0 * ev_f[0] / tot_f
            ar5_fast[idx] = 100.0 * float(np.sum(ev_f[:5])) / tot_f

    # Strictly causal 252d rolling Z-score of Absorption Ratio (Top 5)
    ar5_series = pd.Series(ar5)
    ar5_roll_mean = ar5_series.rolling(window=252, min_periods=126).mean()
    ar5_roll_std = ar5_series.rolling(window=252, min_periods=126).std(ddof=1)
    ar5_z252 = (ar5_series - ar5_roll_mean) / ar5_roll_std.replace(0.0, np.nan)

    # Regime label from strictly causal AR5 Z-score
    def label_regime(z_val: float) -> str:
        if pd.isna(z_val):
            return "Warmup"
        if z_val >= 1.5:
            return "Crisis / High-Coupling"
        if z_val >= 0.75:
            return "Elevated Coupling"
        if z_val <= -1.0:
            return "Dispersed / Low-Coupling"
        return "Normal"

    regimes = [label_regime(v) for v in ar5_z252]

    # Build Rolling_PCA_Dynamics DataFrame
    dyn_data: dict[str, Any] = {
        "Date": roll_dates.values,
        "PC1_Var_Pct": pc_var_pct[:, 0],
        "PC2_Var_Pct": pc_var_pct[:, 1],
        "PC3_Var_Pct": pc_var_pct[:, 2],
        "PC4_Var_Pct": pc_var_pct[:, 3],
        "PC5_Var_Pct": pc_var_pct[:, 4],
        "Absorption_Ratio_Top1": ar1,
        "Absorption_Ratio_Top3": ar3,
        "Absorption_Ratio_Top5": ar5,
        "AR5_Causal_Z252": ar5_z252.values,
        "Effective_Rank_Entropy": eff_rank,
        "Kaiser_Factors_Count": kaiser_cnt,
        "MP_Signal_Factors_Count": mp_cnt,
        "PC1_Subspace_Drift_21d": pc1_drift_21d,
        "Fast63d_PC1_Var_Pct": pc1_var_fast,
        "Fast63d_AR5": ar5_fast,
    }
    for col_key, out_col in ROLLING_PC1_TRACKED:
        if col_key in cols:
            dyn_data[out_col] = pc1_loadings_all[:, cols.index(col_key)]
    for col_key, out_col in ROLLING_PC2_TRACKED:
        if col_key in cols:
            dyn_data[out_col] = pc2_loadings_all[:, cols.index(col_key)]

    # Stock-Bond co-movement indicator on PC1 (negative = classic hedge; positive = 2022-style inflation shock)
    if "ret1d_sp500" in cols and "ret1d_tlt" in cols:
        dyn_data["Stock_Bond_PC1_Product"] = (
            pc1_loadings_all[:, cols.index("ret1d_sp500")]
            * pc1_loadings_all[:, cols.index("ret1d_tlt")]
        )
    dyn_data["Systemic_Regime"] = regimes

    dyn_df = pd.DataFrame(dyn_data)

    # Identify top non-overlapping historical systemic stress peaks (min separation 126 trading days)
    stress_peaks: list[dict[str, Any]] = []
    ar5_work = ar5.copy()
    for _ in range(12):
        best_i = int(np.argmax(ar5_work))
        if ar5_work[best_i] <= 0:
            break
        stress_peaks.append(
            {
                "Date": str(roll_dates.iloc[best_i]),
                "AR5_252d": float(ar5[best_i]),
                "PC1_Var_252d": float(pc_var_pct[best_i, 0]),
                "AR3_252d": float(ar3[best_i]),
                "AR5_Z252": float(ar5_z252.iloc[best_i]) if pd.notna(ar5_z252.iloc[best_i]) else np.nan,
                "Eff_Rank": float(eff_rank[best_i]),
                "Fast63d_AR5": float(ar5_fast[best_i]),
                "Stock_Bond_Product": float(dyn_data["Stock_Bond_PC1_Product"][best_i])
                if "Stock_Bond_PC1_Product" in dyn_data
                else np.nan,
                "Regime": regimes[best_i],
            }
        )
        lo = max(0, best_i - 126)
        hi = min(n_roll, best_i + 127)
        ar5_work[lo:hi] = -1.0
    stress_peaks.sort(key=lambda d: d["AR5_252d"], reverse=True)

    return {
        "window": window,
        "fast_window": fast_window,
        "mp_upper_slow": mp_upper_slow,
        "dynamics_df": dyn_df,
        "roll_dates": roll_dates,
        "pc1_loadings_all": pc1_loadings_all,
        "stress_peaks": stress_peaks,
    }


# =============================================================================
# 5. ANALYSIS 3: FACTOR ANALYSIS (KMO, BARTLETT, MLFA, VARIMAX & PROMAX)
# =============================================================================


def compute_kmo_and_bartlett(R: np.ndarray, n: int) -> dict[str, Any]:
    """Compute Kaiser-Meyer-Olkin (overall & per-variable) and Bartlett's Sphericity Test."""
    p = R.shape[0]
    R_inv = np.linalg.pinv(R)
    diag_inv = np.clip(np.diag(R_inv), 1e-12, None)
    D = np.diag(1.0 / np.sqrt(diag_inv))
    P_part = -D @ R_inv @ D
    np.fill_diagonal(P_part, 0.0)

    R_off = R.copy()
    np.fill_diagonal(R_off, 0.0)

    r2_sum = np.sum(R_off ** 2, axis=0)
    p2_sum = np.sum(P_part ** 2, axis=0)
    kmo_per_var = r2_sum / np.maximum(r2_sum + p2_sum, 1e-12)
    kmo_overall = float(np.sum(r2_sum) / max(float(np.sum(r2_sum + p2_sum)), 1e-12))

    if kmo_overall >= 0.90:
        rating = "Marvelous (>=0.90)"
    elif kmo_overall >= 0.80:
        rating = "Meritorious (0.80-0.89)"
    elif kmo_overall >= 0.70:
        rating = "Middling (0.70-0.79)"
    elif kmo_overall >= 0.60:
        rating = "Mediocre (0.60-0.69)"
    elif kmo_overall >= 0.50:
        rating = "Miserable (0.50-0.59)"
    else:
        rating = "Unacceptable (<0.50)"

    sign, logdet = np.linalg.slogdet(R)
    bart_chi2 = float(-(n - 1.0 - (2.0 * p + 5.0) / 6.0) * logdet) if sign > 0 else float("nan")
    bart_df = int(p * (p - 1) // 2)
    bart_p = float(chi2.sf(bart_chi2, bart_df)) if not math.isnan(bart_chi2) else float("nan")
    smc = np.clip(1.0 - 1.0 / diag_inv, 0.0, 0.995)

    return {
        "kmo_overall": kmo_overall,
        "kmo_rating": rating,
        "kmo_per_var": kmo_per_var,
        "bart_chi2": bart_chi2,
        "bart_df": bart_df,
        "bart_p": bart_p,
        "smc": smc,
    }


def varimax_rotation(
    Phi: np.ndarray,
    kaiser_normalize: bool = True,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Kaiser-normalized orthogonal Varimax rotation using SVD algorithm."""
    p, m = Phi.shape
    if m <= 1:
        return Phi.copy(), np.eye(m)

    if kaiser_normalize:
        row_norms = np.sqrt(np.sum(Phi ** 2, axis=1, keepdims=True))
        row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
        A = Phi / row_norms
    else:
        row_norms = np.ones((p, 1))
        A = Phi.copy()

    R_rot = np.eye(m)
    d_prev = 0.0
    for _ in range(max_iter):
        L = A @ R_rot
        col_means = np.mean(L ** 2, axis=0, keepdims=True)
        B = A.T @ (L ** 3 - L * col_means)
        U, s, Vt = np.linalg.svd(B)
        R_rot = U @ Vt
        d_cur = float(np.sum(s))
        if d_prev > 0 and abs(d_cur - d_prev) < tol * max(1.0, d_cur):
            break
        d_prev = d_cur

    rotated = (A @ R_rot) * row_norms
    return rotated, R_rot


def promax_rotation(
    varimax_loadings: np.ndarray,
    kappa: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Oblique Promax rotation (Hendrickson & White 1964) starting from Varimax loadings."""
    p, m = varimax_loadings.shape
    if m <= 1:
        return varimax_loadings.copy(), np.eye(m)

    # Row-normalize Varimax loadings before raising to power kappa
    row_norms = np.sqrt(np.sum(varimax_loadings ** 2, axis=1, keepdims=True))
    row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
    H = varimax_loadings / row_norms

    # Target matrix preserving sign
    P_target = np.sign(H) * (np.abs(H) ** kappa)
    # Least-squares transformation: H @ Q ~= P_target
    Q, _, _, _ = np.linalg.lstsq(varimax_loadings, P_target, rcond=None)
    # Normalize columns of Q so implied factor variances = 1
    Q_inv = np.linalg.inv(Q)
    D_diag = np.sqrt(np.diag(Q_inv @ Q_inv.T))
    T_mat = Q @ np.diag(D_diag)
    T_inv = np.linalg.inv(T_mat)

    pattern = varimax_loadings @ T_mat
    factor_corr = T_inv @ T_inv.T
    # Normalize factor_corr to exact correlation matrix (unit diagonal)
    fc_diag = np.sqrt(np.clip(np.diag(factor_corr), 1e-12, None))
    factor_corr = factor_corr / np.outer(fc_diag, fc_diag)
    np.fill_diagonal(factor_corr, 1.0)
    return pattern, factor_corr


def run_factor_analysis(
    Z: np.ndarray,
    dates: pd.Series,
    cols: list[str],
    n_factors: int,
) -> dict[str, Any]:
    """
    Maximum Likelihood Factor Analysis (Jöreskog profile likelihood via L-BFGS-B with
    exact analytic gradient & Heywood bound psi >= 0.005), followed by Varimax
    (orthogonal) and Promax (oblique, kappa=4) rotations, full fit diagnostics, and
    daily regression factor scores.
    """
    n, p = Z.shape
    R = np.corrcoef(Z, rowvar=False)
    R = 0.5 * (R + R.T)
    m = max(1, min(n_factors, p - 2))

    adequacy = compute_kmo_and_bartlett(R, n)
    psi0 = np.clip(1.0 - adequacy["smc"], 0.005, 0.995)

    def mlfa_obj_grad(psi: np.ndarray) -> tuple[float, np.ndarray]:
        inv_sqrt = 1.0 / np.sqrt(psi)
        S_star = (R * inv_sqrt[:, None]) * inv_sqrt[None, :]
        S_star = 0.5 * (S_star + S_star.T)
        evals, evecs = np.linalg.eigh(S_star)
        rem_evals = np.clip(evals[: p - m], 1e-12, None)
        rem_vecs = evecs[:, : p - m]
        fval = float(np.sum(rem_evals - np.log(rem_evals) - 1.0))
        grad = ((rem_vecs ** 2) @ (1.0 - rem_evals)) / psi
        return fval, grad

    res = minimize(
        mlfa_obj_grad,
        psi0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.005, 0.999)] * p,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    psi = np.clip(res.x, 0.005, 0.999)
    heywood_count = int(np.sum(psi <= 0.0055))

    # Recover unrotated ML loadings: L = Psi^(1/2) @ U_m @ diag(sqrt(max(theta_m - 1, 0)))
    sqrt_psi = np.sqrt(psi)
    inv_sqrt = 1.0 / sqrt_psi
    S_star = (R * inv_sqrt[:, None]) * inv_sqrt[None, :]
    S_star = 0.5 * (S_star + S_star.T)
    ev_asc, ec_asc = np.linalg.eigh(S_star)
    top_ev = ev_asc[::-1][:m]
    top_ec = ec_asc[:, ::-1][:, :m]
    L_unrot = sqrt_psi[:, None] * (top_ec * np.sqrt(np.maximum(top_ev - 1.0, 0.0))[None, :])

    # Varimax orthogonal rotation
    L_varimax, _ = varimax_rotation(L_unrot, kaiser_normalize=True)

    # Sort Varimax factors by descending SS loadings and orient dominant loading positive
    ss_loadings = np.sum(L_varimax ** 2, axis=0)
    order = np.argsort(ss_loadings)[::-1]
    L_varimax = L_varimax[:, order]
    ss_loadings = ss_loadings[order]
    for k in range(m):
        dom = int(np.argmax(np.abs(L_varimax[:, k])))
        if L_varimax[dom, k] < 0:
            L_varimax[:, k] *= -1.0

    # Promax oblique rotation from ordered Varimax
    L_promax, factor_corr = promax_rotation(L_varimax, kappa=4.0)
    for k in range(m):
        if float(np.dot(L_promax[:, k], L_varimax[:, k])) < 0:
            L_promax[:, k] *= -1.0
            factor_corr[k, :] *= -1.0
            factor_corr[:, k] *= -1.0
            factor_corr[k, k] = 1.0

    # Communalities, uniqueness, Hofmann complexity
    communalities = np.sum(L_varimax ** 2, axis=1)
    uniquenesses = np.clip(1.0 - communalities, 0.0, 1.0)
    l2_sum = np.sum(L_varimax ** 2, axis=1)
    l4_sum = np.sum(L_varimax ** 4, axis=1)
    complexity = np.where(l4_sum > 1e-12, (l2_sum ** 2) / l4_sum, 1.0)

    # Model-implied correlation matrix & fit diagnostics
    R_hat = L_varimax @ L_varimax.T + np.diag(uniquenesses)
    resid = R - R_hat
    off_mask = ~np.eye(p, dtype=bool)
    rmsr = float(np.sqrt(np.mean(resid[off_mask] ** 2)))

    # ML Goodness-of-fit chi-square, df, TLI, RMSEA
    df_model = int(((p - m) ** 2 - (p + m)) // 2)
    f_ml = float(res.fun)
    ml_chi2 = float((n - 1.0 - (2.0 * p + 5.0) / 6.0 - 2.0 * m / 3.0) * f_ml)
    ml_pval = float(chi2.sf(ml_chi2, df_model)) if df_model > 0 else float("nan")
    rmsea = (
        float(math.sqrt(max(0.0, (ml_chi2 - df_model) / (df_model * (n - 1.0)))))
        if df_model > 0
        else float("nan")
    )
    # Null model (independence) chi2 is Bartlett's chi2
    null_chi2 = adequacy["bart_chi2"]
    null_df = adequacy["bart_df"]
    if df_model > 0 and null_df > 0 and (null_chi2 / null_df - 1.0) > 0:
        tli = float(((null_chi2 / null_df) - (ml_chi2 / df_model)) / ((null_chi2 / null_df) - 1.0))
        cfi = float(
            1.0
            - max(0.0, ml_chi2 - df_model) / max(1e-12, max(0.0, null_chi2 - null_df))
        )
    else:
        tli, cfi = float("nan"), float("nan")

    # Daily Thomson Regression Factor Scores: F = Z @ R^{-1} @ L_varimax
    R_inv = np.linalg.pinv(R)
    W_reg = R_inv @ L_varimax
    scores_raw = Z @ W_reg
    # Standardize scores to exact unit variance for clean ML consumption
    scores_std, _, _ = standardize_matrix(scores_raw)

    themes = [infer_factor_theme(L_varimax[:, k], cols, threshold=0.35) for k in range(m)]
    scores_df = pd.DataFrame({"Date": dates.values})
    for k in range(m):
        scores_df[f"F{k + 1}"] = scores_std[:, k]

    return {
        "m": m,
        "converged": bool(res.success),
        "n_iter": int(res.nit),
        "f_ml": f_ml,
        "heywood_count": heywood_count,
        "adequacy": adequacy,
        "L_unrot": L_unrot,
        "L_varimax": L_varimax,
        "L_promax": L_promax,
        "factor_corr": factor_corr,
        "ss_loadings": ss_loadings,
        "var_pct": 100.0 * ss_loadings / p,
        "cum_var_pct": 100.0 * np.cumsum(ss_loadings) / p,
        "communalities": communalities,
        "uniquenesses": uniquenesses,
        "complexity": complexity,
        "rmsr": rmsr,
        "df_model": df_model,
        "ml_chi2": ml_chi2,
        "ml_pval": ml_pval,
        "rmsea": rmsea,
        "tli": tli,
        "cfi": cfi,
        "themes": themes,
        "scores_df": scores_df,
    }


# =============================================================================
# 6. ANALYSIS 4: HIERARCHICAL CLUSTERING + CLUSTER VALIDATION + HRP WEIGHTS
# =============================================================================


def silhouette_from_dist(dist_mat: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute exact Silhouette coefficient from a precomputed symmetric distance matrix."""
    p = len(labels)
    unique_labs = np.unique(labels)
    if len(unique_labs) < 2 or len(unique_labs) >= p:
        return float("nan"), np.zeros(p)

    sil = np.zeros(p, dtype=np.float64)
    for i in range(p):
        ci = labels[i]
        same = np.where((labels == ci) & (np.arange(p) != i))[0]
        if len(same) == 0:
            sil[i] = 0.0
            continue
        a_i = float(np.mean(dist_mat[i, same]))
        b_i = float("inf")
        for cj in unique_labs:
            if cj == ci:
                continue
            other = np.where(labels == cj)[0]
            d_other = float(np.mean(dist_mat[i, other]))
            if d_other < b_i:
                b_i = d_other
        denom = max(a_i, b_i)
        sil[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0
    return float(np.mean(sil)), sil


def calinski_harabasz_and_db(Z_t: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute Calinski-Harabasz and Davies-Bouldin indices on asset feature vectors (p x n)."""
    p = Z_t.shape[0]
    unique_labs = np.unique(labels)
    k = len(unique_labs)
    if k < 2 or k >= p:
        return float("nan"), float("nan")

    global_mean = np.mean(Z_t, axis=0)
    centroids = []
    scatters = []
    bgss = 0.0
    wgss = 0.0
    for c in unique_labs:
        pts = Z_t[labels == c]
        mu = np.mean(pts, axis=0)
        centroids.append(mu)
        diffs = pts - mu
        sq_dists = np.sum(diffs ** 2, axis=1)
        wgss += float(np.sum(sq_dists))
        bgss += len(pts) * float(np.sum((mu - global_mean) ** 2))
        scatters.append(float(np.mean(np.sqrt(sq_dists))))

    ch = (bgss / (k - 1.0)) / (wgss / (p - k)) if wgss > 1e-12 else float("nan")

    centroids_arr = np.vstack(centroids)
    db_vals = []
    for i in range(k):
        max_r = 0.0
        for j in range(k):
            if i == j:
                continue
            m_ij = float(np.linalg.norm(centroids_arr[i] - centroids_arr[j]))
            if m_ij > 1e-12:
                r_ij = (scatters[i] + scatters[j]) / m_ij
                if r_ij > max_r:
                    max_r = r_ij
        db_vals.append(max_r)
    db = float(np.mean(db_vals))
    return float(ch), db


def intra_inter_corr(R: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute mean pairwise correlation within clusters vs between clusters."""
    p = len(labels)
    intra, inter = [], []
    for i in range(p):
        for j in range(i + 1, p):
            if labels[i] == labels[j]:
                intra.append(R[i, j])
            else:
                inter.append(R[i, j])
    m_intra = float(np.mean(intra)) if intra else float("nan")
    m_inter = float(np.mean(inter)) if inter else float("nan")
    return m_intra, m_inter


def compute_hrp_weights(cov: np.ndarray, ordered_indices: list[int]) -> np.ndarray:
    """
    López de Prado (2016) Hierarchical Risk Parity (HRP) recursive bisection weights
    on the quasi-diagonalized covariance matrix.
    """
    p = cov.shape[0]
    w = np.ones(p, dtype=np.float64)
    clusters: list[list[int]] = [list(ordered_indices)]

    def cluster_var(idxs: list[int]) -> float:
        sub = cov[np.ix_(idxs, idxs)]
        inv_diag = 1.0 / np.clip(np.diag(sub), 1e-12, None)
        ivp = inv_diag / np.sum(inv_diag)
        return float(ivp @ sub @ ivp)

    while clusters:
        next_clusters: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left = cl[:mid]
            right = cl[mid:]
            v_left = cluster_var(left)
            v_right = cluster_var(right)
            alpha = 1.0 - v_left / max(v_left + v_right, 1e-15)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    return w / np.sum(w)


def infer_cluster_label(member_cols: list[str], mean_intra_corr: float | None = None) -> str:
    """Generate a crisp, unique economic cluster label from member assets and cohesion."""
    classes = [asset_class(c) for c in member_cols]
    names = [asset_name(c) for c in member_cols]
    col_set = set(member_cols)
    if len(names) == 1:
        return f"Singleton: {names[0]}"
    if mean_intra_corr is not None and mean_intra_corr < 0.10:
        return "Idiosyncratic / Low-Corr Diversifiers (Softs, NatGas & Asia Rates)"
    if all(c == "Sovereign Yield" for c in classes):
        return "US Treasury Yields (2Y/10Y/30Y)"
    if all(c == "Fixed Income ETF" for c in classes):
        return "US Sovereign & Agg Bond ETFs (TLT/AGG)"
    if any(c == "Equity (US)" for c in classes) and any(c == "Equity (Europe)" for c in classes):
        if "ret1d_copper" in col_set:
            return "Western Equities, HY Credit & Copper"
        return "Western Equities & Credit"
    if all(c.startswith("Equity") for c in classes):
        if all("Asia" in c for c in classes):
            return "Asia-Pacific Equities"
        return "Global Equities"
    if all("FX" in c for c in classes):
        if col_set <= {"ret1d_eurusd", "ret1d_gbpusd"}:
            return "European FX (EUR/GBP vs USD)"
        if "ret1d_dxy" in col_set or "ret1d_usdjpy" in col_set:
            return "US Dollar & USD/Asia FX"
        return "G10 & Asia FX"
    if all("Agriculture" in c for c in classes):
        return "Agriculture Grains (Corn/Wheat/Soy)"
    if all("Agriculture" in c or "Softs" in c for c in classes):
        return "Agriculture & Softs"
    if all("Energy" in c for c in classes):
        return "Crude Oil Energy (WTI/Brent)"
    if all("Metals" in c for c in classes):
        return "Precious Metals (Gold/Silver)"
    if all("Metals" in c or "Industrial" in c for c in classes):
        return "Precious & Base Metals"
    counts: dict[str, int] = {}
    for c in classes:
        base = c.split(" (")[0]
        counts[base] = counts.get(base, 0) + 1
    top_cat = max(counts, key=counts.get)
    sample = ", ".join(names[:3])
    return f"{top_cat}-Led ({sample})"


def run_hierarchical_clustering(
    X_raw: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    cols: list[str],
) -> dict[str, Any]:
    """
    Hierarchical Agglomerative Clustering using López de Prado correlation distance
    d_ij = sqrt(0.5 * (1 - rho_ij)), comparing Ward, Average, Complete, and Single linkage
    over k=2..12, plus Optimal Leaf Ordering, Dendrogram Merge History, and HRP weights.
    """
    p = len(cols)
    dist_mat = np.sqrt(np.clip(0.5 * (1.0 - R), 0.0, 1.0))
    np.fill_diagonal(dist_mat, 0.0)
    cond_dist = squareform(dist_mat, checks=False)

    # Normalized asset-time vectors for Euclidean CH/DB indices (each asset row has unit norm)
    Z_assets = Z.T / math.sqrt(Z.shape[0] - 1.0)

    methods = ["ward", "average", "complete", "single"]
    linkages: dict[str, np.ndarray] = {}
    cophenetics: dict[str, float] = {}
    benchmark_rows: list[dict[str, Any]] = []

    best_k_by_method: dict[str, int] = {}
    for meth in methods:
        Z_link = linkage(cond_dist, method=meth)
        linkages[meth] = Z_link
        coph_r, _ = cophenet(Z_link, cond_dist)
        cophenetics[meth] = float(coph_r)

        best_sil = -1.0
        best_k = 6
        for k in range(2, 13):
            labs = fcluster(Z_link, t=k, criterion="maxclust")
            sil_mean, _ = silhouette_from_dist(dist_mat, labs)
            ch_idx, db_idx = calinski_harabasz_and_db(Z_assets, labs)
            m_intra, m_inter = intra_inter_corr(R, labs)
            sep_gap = m_intra - m_inter if not (math.isnan(m_intra) or math.isnan(m_inter)) else float("nan")
            sizes = pd.Series(labs).value_counts()
            n_singletons = int((sizes == 1).sum())

            benchmark_rows.append(
                {
                    "Linkage_Method": meth.capitalize(),
                    "Cophenetic_Corr": float(coph_r),
                    "k_Clusters": k,
                    "Silhouette_Score": sil_mean,
                    "Calinski_Harabasz": ch_idx,
                    "Davies_Bouldin": db_idx,
                    "Mean_Intra_Corr": m_intra,
                    "Mean_Inter_Corr": m_inter,
                    "Corr_Separation_Gap": sep_gap,
                    "Largest_Cluster_Size": int(sizes.max()),
                    "Singleton_Clusters": n_singletons,
                }
            )
            # Pick optimal k in practical range 4..10 by Silhouette
            if 4 <= k <= 10 and not math.isnan(sil_mean) and sil_mean > best_sil:
                best_sil = sil_mean
                best_k = k
        best_k_by_method[meth] = best_k

    # Primary linkage for compact portfolio clusters & HRP: Ward (with Optimal Leaf Ordering)
    Z_ward_olo = optimal_leaf_ordering(linkages["ward"], cond_dist)
    leaf_order = list(leaves_list(Z_ward_olo))

    # Extract exact U-shaped dendrogram branch coordinates (scaled to Leaf Index 1..p)
    dendro_raw = dendrogram(Z_ward_olo, no_plot=True)
    # Cut height midway between (p - opt_k_ward - 1)th and (p - opt_k_ward)th merge
    opt_k_ward = best_k_by_method["ward"]
    merge_dists_sorted = np.sort(Z_ward_olo[:, 2])
    cut_idx = max(0, p - opt_k_ward - 1)
    cut_height = (
        0.5 * (float(merge_dists_sorted[cut_idx]) + float(merge_dists_sorted[min(cut_idx + 1, p - 2)]))
        if p > 2
        else float(merge_dists_sorted[0])
    )
    dendro_branches: list[dict[str, Any]] = []
    for xs, ys in zip(dendro_raw["icoord"], dendro_raw["dcoord"]):
        # SciPy leaf positions are 5, 15, 25, ... -> map to 1.0, 2.0, 3.0, ...
        x_leaf = [round((float(v) - 5.0) / 10.0 + 1.0, 4) for v in xs]
        y_dist = [round(float(v), 4) for v in ys]
        is_intra = max(y_dist) <= cut_height + 1e-9
        dendro_branches.append({"x": x_leaf, "y": y_dist, "is_intra": is_intra})

    labs_ward_opt = fcluster(linkages["ward"], t=opt_k_ward, criterion="maxclust")
    labs_ward_k5 = fcluster(linkages["ward"], t=5, criterion="maxclust")
    labs_ward_k8 = fcluster(linkages["ward"], t=8, criterion="maxclust")
    labs_avg_opt = fcluster(linkages["average"], t=best_k_by_method["average"], criterion="maxclust")
    labs_comp_opt = fcluster(linkages["complete"], t=best_k_by_method["complete"], criterion="maxclust")

    # Relabel Ward optimal clusters sequentially 1..opt_k in dendrogram appearance order
    seen_order: list[int] = []
    for idx in leaf_order:
        c = int(labs_ward_opt[idx])
        if c not in seen_order:
            seen_order.append(c)
    remap = {old: new_i + 1 for new_i, old in enumerate(seen_order)}
    labs_ward_opt = np.array([remap[int(c)] for c in labs_ward_opt], dtype=int)

    ward_sil_mean, ward_sil_assets = silhouette_from_dist(dist_mat, labs_ward_opt)

    # Cluster labels for Ward optimal k
    cluster_labels_map: dict[int, str] = {}
    for cid in range(1, opt_k_ward + 1):
        m_idxs = [i for i in leaf_order if labs_ward_opt[i] == cid]
        m_cols = [cols[i] for i in m_idxs]
        intra_vals = [R[m_idxs[a], m_idxs[b]] for a in range(len(m_idxs)) for b in range(a + 1, len(m_idxs))]
        m_intra = float(np.mean(intra_vals)) if intra_vals else 1.0
        cluster_labels_map[cid] = infer_cluster_label(m_cols, mean_intra_corr=m_intra)

    # Covariance matrix (on raw returns/rate changes — annualized vol for reporting, standardized for cross-asset HRP)
    # Note: Mixing equity % returns (~0.01) with yield bp changes (~0.05 pct pts) in raw covariance
    # would bias naive variance by unit scale. Computing HRP on correlation-normalized (equal-unit-vol)
    # covariance gives pure correlation-structure diversification weights, while also reporting raw annualized vol!
    raw_ann_vol = np.std(X_raw, axis=0, ddof=1) * math.sqrt(252.0)
    # For pure price/FX assets (ret1d_*), raw_ann_vol * 100 is annualized % volatility;
    # for yields (chg1d_*), raw_ann_vol is annualized percentage-point volatility (bps/100).
    hrp_weights = compute_hrp_weights(R, leaf_order)  # Correlation-space HRP (unit-scale invariant across assets+yields)
    # Also compute pure-price HRP/IVP comparison (using equal-vol baseline = 1/p vs correlation-HRP)
    inv_vol_raw = 1.0 / np.clip(raw_ann_vol, 1e-12, None)
    ivp_weights = inv_vol_raw / np.sum(inv_vol_raw)

    # Nearest neighbor per asset
    nn_asset = []
    nn_corr = []
    nn_dist = []
    for i in range(p):
        d_row = dist_mat[i].copy()
        d_row[i] = np.inf
        j = int(np.argmin(d_row))
        nn_asset.append(asset_name(cols[j]))
        nn_corr.append(float(R[i, j]))
        nn_dist.append(float(dist_mat[i, j]))

    # Build Dendrogram Merge History for Ward linkage (34 merge steps)
    node_members: dict[int, list[str]] = {i: [asset_name(cols[i])] for i in range(p)}
    merge_history: list[dict[str, Any]] = []
    Z_w = linkages["ward"]
    for step_idx in range(p - 1):
        left_id = int(Z_w[step_idx, 0])
        right_id = int(Z_w[step_idx, 1])
        merge_d = float(Z_w[step_idx, 2])
        new_size = int(Z_w[step_idx, 3])
        new_id = p + step_idx
        left_m = node_members[left_id]
        right_m = node_members[right_id]
        merged_m = left_m + right_m
        node_members[new_id] = merged_m

        def short_node(nid: int, mems: list[str]) -> str:
            if nid < p:
                return mems[0]
            return f"Node_{nid - p + 1} ({len(mems)} assets: {', '.join(mems[:2])}{'...' if len(mems) > 2 else ''})"

        # Compute actual average pairwise correlation between left and right sub-clusters
        left_idxs = [i for i in range(p) if asset_name(cols[i]) in left_m]
        right_idxs = [i for i in range(p) if asset_name(cols[i]) in right_m]
        cross_corr = float(np.mean(R[np.ix_(left_idxs, right_idxs)]))

        merge_history.append(
            {
                "Step": step_idx + 1,
                "New_Node_ID": f"Node_{step_idx + 1}",
                "Left_Branch": short_node(left_id, left_m),
                "Right_Branch": short_node(right_id, right_m),
                "Ward_Merge_Distance": merge_d,
                "Avg_Cross_Branch_Corr": cross_corr,
                "New_Cluster_Size": new_size,
                "All_Merged_Members": "; ".join(merged_m),
            }
        )

    return {
        "dist_mat": dist_mat,
        "linkages": linkages,
        "cophenetics": cophenetics,
        "best_linkage_coph": max(cophenetics, key=cophenetics.get),
        "benchmark_rows": benchmark_rows,
        "best_k_by_method": best_k_by_method,
        "opt_k_ward": opt_k_ward,
        "leaf_order": leaf_order,
        "labs_ward_opt": labs_ward_opt,
        "labs_ward_k5": labs_ward_k5,
        "labs_ward_k8": labs_ward_k8,
        "labs_avg_opt": labs_avg_opt,
        "labs_comp_opt": labs_comp_opt,
        "cluster_labels_map": cluster_labels_map,
        "ward_sil_mean": ward_sil_mean,
        "ward_sil_assets": ward_sil_assets,
        "raw_ann_vol": raw_ann_vol,
        "hrp_weights": hrp_weights,
        "ivp_weights": ivp_weights,
        "nn_asset": nn_asset,
        "nn_corr": nn_corr,
        "nn_dist": nn_dist,
        "merge_history": merge_history,
        "dendro_branches": dendro_branches,
        "cut_height": cut_height,
    }


# =============================================================================
# 7. SENSITIVITY BENCHMARK ACROSS ALTERNATIVE UNIVERSES
# =============================================================================


def run_universe_sensitivities(ml_df: pd.DataFrame, daily_35_cols: list[str]) -> list[dict[str, Any]]:
    """Compare PCA dimensionality & factorability across 5 alternative universes/preprocessing specs."""
    results: list[dict[str, Any]] = []

    def eval_universe(name: str, sub_df: pd.DataFrame, notes: str, winsor: bool = False) -> None:
        X = sub_df.to_numpy(dtype=np.float64)
        n, p = X.shape
        Z, _, _ = standardize_matrix(X)
        if winsor:
            Z = winsorize_z(Z, clip=5.0)
        R = np.corrcoef(Z, rowvar=False)
        R = 0.5 * (R + R.T)
        ev = np.clip(np.linalg.eigvalsh(R)[::-1], 0.0, None)
        vexp = ev / np.sum(ev)
        cum = np.cumsum(vexp)
        mp_up = (1.0 + math.sqrt(p / n)) ** 2
        nz = vexp[vexp > 1e-12]
        eff_r = float(np.exp(-np.sum(nz * np.log(nz))))
        # Guard KMO when correlation matrix is strictly rank-deficient (algebraic collinearity)
        min_ev = float(np.min(ev))
        if min_ev < 1e-6:
            kmo_display: Any = "N/A (Rank-Deficient)"
        else:
            adeq = compute_kmo_and_bartlett(R, n)
            kmo_display = fmt_num(adeq["kmo_overall"], 4)
        results.append(
            {
                "Universe": name,
                "Rows_N": n,
                "Features_P": p,
                "PC1_Var_Pct": fmt_num(100.0 * vexp[0], 2),
                "Top3_AR3_Pct": fmt_num(100.0 * cum[min(2, p - 1)], 2),
                "Top5_AR5_Pct": fmt_num(100.0 * cum[min(4, p - 1)], 2),
                "Top10_Cum_Pct": fmt_num(100.0 * cum[min(9, p - 1)], 2),
                "Kaiser_GT1": int(np.sum(ev > 1.0)),
                "MP_Signal_Count": int(np.sum(ev > mp_up)),
                "Effective_Rank": fmt_num(eff_r, 2),
                "KMO_Overall": kmo_display,
                "Notes": notes,
            }
        )

    core_mask = ml_df["is_core_ml_ready"] == 1
    full_mask = ml_df["is_full_ml_ready"] == 1

    eval_universe(
        "Primary: Daily_35 (Core 2008-2026)",
        ml_df.loc[core_mask, daily_35_cols],
        "Primary universe for all 4 methods: 30 daily prices/FX + 5 daily yields; zero constant-window degeneracy",
        winsor=False,
    )
    eval_universe(
        "Sensitivity A: Daily_35 Winsorized (+/-5 sigma)",
        ml_df.loc[core_mask, daily_35_cols],
        "Tests robustness against futures-roll jumps & Mar-2020 crisis outliers",
        winsor=True,
    )

    core_42_cols = [
        c for c in ml_df.columns
        if (c.startswith("ret1d_") or c.startswith("chg1d_")) and c not in LATE_INCEPTION_COLS
    ]
    eval_universe(
        "Sensitivity B: Core_42 (incl. 7 Stepped Policy/Monthly Rates)",
        ml_df.loc[core_mask, core_42_cols],
        "Includes stepped central bank rates (95%-99.8% zeros); dilutes PC1% and adds noise eigenvalues",
        winsor=False,
    )

    daily_36_cols = daily_35_cols + (["ret1d_bndx"] if "ret1d_bndx" in ml_df.columns else [])
    eval_universe(
        "Sensitivity C: Post-2019 Daily_36 (incl. BNDX, 2019-2026)",
        ml_df.loc[full_mask, daily_36_cols],
        "Modern post-2019 regime (COVID + inflation shock + AI era) including Global Agg Bond (BNDX)",
        winsor=False,
    )

    excl_meta = {
        "Date",
        "day_of_week",
        "month",
        "quarter",
        "is_month_end",
        "valid_feature_pct",
        "is_core_ml_ready",
        "is_full_ml_ready",
    } | LATE_INCEPTION_COLS
    core_ml_138 = [c for c in ml_df.columns if c not in excl_meta]
    eval_universe(
        "Sensitivity D: Full Core ML Matrix (138 Features, 2008-2026)",
        ml_df.loc[core_mask, core_ml_138],
        "All Agent 2 returns, cross-asset ratios, momentum, Z-scores & vol (high collinearity -> low eff rank)",
        winsor=False,
    )
    return results


# =============================================================================
# 8. SHEET TABLE BUILDERS (9 COMPREHENSIVE TABS)
# =============================================================================


def normalize_rect(rows: list[list[Any]]) -> list[list[Any]]:
    max_w = max((len(r) for r in rows), default=1)
    return [r + [""] * (max_w - len(r)) for r in rows]


def build_executive_summary(
    source_id: str,
    dest_id: str,
    dates: pd.Series,
    cols: list[str],
    pca_res: dict[str, Any],
    roll_res: dict[str, Any],
    fa_res: dict[str, Any],
    hc_res: dict[str, Any],
    sens_rows: list[dict[str, Any]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    dyn_df = roll_res["dynamics_df"]
    latest_row = dyn_df.iloc[-1]
    peak0 = roll_res["stress_peaks"][0] if roll_res["stress_peaks"] else {}

    # SECTION 1
    rows.append(["SECTION 1: AGENT 3 DATA EXPLORATION — RUN METADATA & UNIVERSE DESIGN"])
    rows.append(["Metric", "Value", "Details / Methodology", "Status"])
    meta_items = [
        ("Agent Role", "Agent 3: Data Exploration (PCA, Rolling PCA, Factor Analysis, Hierarchical Clustering)", "Consumes Agent 2 ML_Ready_Dataset and publishes 10 formatted analytical & visual tabs", "Active"),
        ("Source Google Sheet ID (Agent 2)", source_id or "Local CSV", "Tab: ML_Ready_Dataset (strictly causal stationary features)", "Verified"),
        ("Destination Google Sheet ID (Agent 3)", dest_id or "Local CSV", "Data Exploration - PCA, FA, Clustering (10 tabs incl. Visual_Dashboard, color scales & 10 charts)", "Published"),
        ("Primary Analytical Universe", f"{len(cols)} Daily-Traded Multi-Asset Series (daily_35)", "30 daily price/FX log-returns + 5 daily sovereign/money-market yield diffs (Equities 10, FI ETFs 3, Commodities 11, FX 6, Yields 5)", "Clean"),
        ("Excluded from Primary Rolling Panel", "7 Stepped/Monthly Rates + 2 Late-Inception Series", "Stepped policy rates (Fed/ECB/BOJ/HK/CN LPR/SG) are 0 on 95.5%-99.8% of days (degenerate in 252d rolling windows); BNDX/CN5Y-LPR analyzed in Sensitivity C", "Documented"),
        ("Sample Window (is_core_ml_ready == 1)", f"{len(dates):,} trading weekdays ({dates.iloc[0]} to {dates.iloc[-1]})", "Zero missing values; ~18.7 years spanning GFC, Euro crisis, China 2015, COVID-19, 2022 inflation shock, and 2023-2026 regime", "Zero NaNs"),
        ("Standardization & Lookahead Control", "Correlation-matrix (unit-variance Z-scores)", "Rolling PCA is 100% strictly causal (trailing [t-W+1..t]); Full-sample PCA/FA/HC are structural exploratory benchmarks (fit inside folds for predictive ML)", "Audited"),
    ]
    for item in meta_items:
        rows.append(list(item))

    rows.append([])
    # SECTION 2: Key Findings Across All 4 Methods
    rows.append(["SECTION 2: EXECUTIVE FINDINGS ACROSS ALL FOUR METHODS"])
    rows.append(["Metric", "Value", "Details / Methodology", "Status"])
    opt_k = hc_res["opt_k_ward"]
    best_coph_m = hc_res["best_linkage_coph"]
    top3_hrp = sorted(
        [(asset_name(cols[i]), 100.0 * hc_res["hrp_weights"][i]) for i in range(len(cols))],
        key=lambda t: t[1],
        reverse=True,
    )[:5]
    top3_hrp_str = "; ".join(f"{n} ({w:.2f}%)" for n, w in top3_hrp)

    findings = [
        ("1. Full-Sample PCA — PC1 / Top-3 / Top-5 Variance", f"{pca_res['ar1']:.2f}% / {pca_res['ar3']:.2f}% / {pca_res['ar5']:.2f}%", f"PC1 ({pca_res['themes'][0]}) captures {pca_res['ar1']:.2f}%; Top 10 capture {pca_res['cum_var_pct'][9]:.2f}%; Effective Rank = {pca_res['eff_rank']:.2f} independent bets", "PCA"),
        ("1. Full-Sample PCA — Statistical Dimensionality", f"Parallel Analysis={pca_res['pa_k']} | Marchenko-Pastur={pca_res['mp_k']} | Kaiser={pca_res['kaiser_k']}", f"MP RMT upper noise bound lambda+ = {pca_res['mp_upper']:.4f}; {pca_res['pa_k']} non-random macro factors exceed 95th-pct Horn Parallel Analysis", "PCA"),
        ("2. Rolling PCA (252d) — Historical Absorption Ratio (AR5)", f"Mean={dyn_df['Absorption_Ratio_Top5'].mean():.2f}% | Min={dyn_df['Absorption_Ratio_Top5'].min():.2f}% | Max={dyn_df['Absorption_Ratio_Top5'].max():.2f}%", f"Peak systemic coupling: {peak0.get('Date','')} (AR5={peak0.get('AR5_252d',0):.2f}%, PC1={peak0.get('PC1_Var_252d',0):.2f}%)", "Rolling PCA"),
        ("2. Rolling PCA (252d) — Latest Market Regime", f"{latest_row['Date']}: {latest_row['Systemic_Regime']}", f"AR5={latest_row['Absorption_Ratio_Top5']:.2f}% (Z252={latest_row['AR5_Causal_Z252']:+.2f}), PC1={latest_row['PC1_Var_Pct']:.2f}%, EffRank={latest_row['Effective_Rank_Entropy']:.2f}, Fast63d AR5={latest_row['Fast63d_AR5']:.2f}%", "Rolling PCA"),
        ("3. Factor Analysis — Sampling Adequacy", f"KMO = {fa_res['adequacy']['kmo_overall']:.4f} ({fa_res['adequacy']['kmo_rating']})", f"Bartlett Sphericity chi2 = {fa_res['adequacy']['bart_chi2']:,.1f} (df={fa_res['adequacy']['bart_df']}, p < 1e-300); dataset strongly suited for latent factor extraction", "Factor Analysis"),
        ("3. Factor Analysis — MLFA Fit & Rotated Factors", f"m = {fa_res['m']} Factors | Var Explained = {fa_res['cum_var_pct'][-1]:.2f}% | RMSR = {fa_res['rmsr']:.4f}", f"TLI={fa_res['tli']:.4f}, CFI={fa_res['cfi']:.4f}, RMSEA={fa_res['rmsea']:.4f}, Converged={fa_res['converged']} ({fa_res['n_iter']} iters), Heywood bounds hit={fa_res['heywood_count']}", "Factor Analysis"),
        ("3. Factor Analysis — Identified Latent Themes", f"{fa_res['m']} Varimax + Promax Factors", " | ".join(f"F{k+1}: {fa_res['themes'][k]}" for k in range(fa_res["m"])), "Factor Analysis"),
        ("4. Hierarchical Clustering — Linkage Cophenetic Fidelity", f"Best={best_coph_m.capitalize()} ({hc_res['cophenetics'][best_coph_m]:.4f}) | Ward={hc_res['cophenetics']['ward']:.4f}", f"Complete={hc_res['cophenetics']['complete']:.4f}, Single={hc_res['cophenetics']['single']:.4f} on metric correlation distance d=sqrt(0.5*(1-rho))", "Clustering"),
        ("4. Hierarchical Clustering — Optimal Ward Partition", f"k = {opt_k} Clusters | Mean Silhouette = {hc_res['ward_sil_mean']:.4f}", " | ".join(f"C{cid}: {lbl}" for cid, lbl in hc_res["cluster_labels_map"].items()), "Clustering"),
        ("4. Hierarchical Risk Parity (HRP) — Top Diversifiers", f"35-Asset Equal-Vol HRP (Sum = 100.0%)", f"Top 5 HRP allocations (least correlated diversifiers): {top3_hrp_str}", "Clustering/HRP"),
    ]
    for item in findings:
        rows.append(list(item))

    rows.append([])
    # SECTION 3: Sensitivity Comparison Across Universes
    rows.append(["SECTION 3: SENSITIVITY COMPARISON ACROSS ALTERNATIVE UNIVERSES & WINSORIZATION"])
    sens_headers = [
        "Universe",
        "Rows_N",
        "Features_P",
        "PC1_Var_Pct",
        "Top3_AR3_Pct",
        "Top5_AR5_Pct",
        "Top10_Cum_Pct",
        "Kaiser_GT1",
        "MP_Signal_Count",
        "Effective_Rank",
        "KMO_Overall",
        "Notes",
    ]
    rows.append(sens_headers)
    for sr in sens_rows:
        rows.append([sr[h] for h in sens_headers])

    rows.append([])
    # SECTION 4: Tab Guide
    rows.append(["SECTION 4: WORKBOOK TAB DIRECTORY & DOWNSTREAM ML USAGE GUIDE"])
    rows.append(["Tab_Name", "Granularity", "Primary Contents", "Recommended Usage"])
    tab_guide = [
        ("Executive_Summary", "Overview", "Run metadata, key findings across all 4 methods, 5-universe sensitivity table, tab dictionary", "Start here for executive diagnostics and methodological caveats"),
        ("Visual_Dashboard", "10 Interactive Charts + Leaf Key", "PCA Scree Plot (with PA95 & MP bounds), Variance Explained, Rolling Absorption Ratios, Stock-Bond Regime Flip, AR5 Z-Score & EffRank, Cumulative PC Trends, MLFA Factor Variance, Optimal-k Silhouette, Ward Dendrogram + Leaf Key, HRP Weights", "Visual overview of all 4 statistical methods in one gallery"),
        ("PCA_Overview_and_Loadings", "35 Components / 35 Assets", "Scree table (Eigenvalues, Horn PA 95th pct, Marchenko-Pastur RMT), heatmap-shaded Correlation Loadings (PC1-PC12), Unit-Norm Eigenvectors (PC1-PC10), Asset % Contributions", "Inspect orthogonal macro risk drivers and build eigenportfolios"),
        ("PCA_Scores_TimeSeries", "4,825 Days x 10 PCs", "Daily standardized PC1-PC10 time series (sign-oriented: PC1+ = Risk-On)", "Exploratory macro factor attribution & regime visualization"),
        ("Rolling_PCA_Dynamics", "4,574 Days x 31 Metrics", "Strictly causal trailing 252d & 63d Rolling PCA: PC1-PC5 var %, Absorption Ratios, causal AR5 Z-score, Effective Rank, 21d PC1 drift, bellwether loadings, Stock-Bond coupling, Systemic Regime", "Zero-lookahead systemic risk features safe for live ML / risk-off triggers"),
        ("Rolling_PCA_Loadings_Snapshots", "Annual + Stress Peaks", "Annual regime averages (2009-2026), year-end & latest PC1 loadings for all 35 assets (with stability stats), top 12 historical systemic stress peaks", "Track structural beta drift (e.g. Stock-Bond correlation flip in 2022+)"),
        ("Factor_Analysis_Results", "MLFA + Varimax + Promax", "KMO & Bartlett tests, MLFA fit indices (RMSR/RMSEA/TLI/CFI), per-asset KMO/communality/uniqueness, Varimax orthogonal loadings, Promax oblique pattern matrix & inter-factor correlations", "Identify clean interpretable economic factors with idiosyncratic variance separated"),
        ("Factor_Scores_TimeSeries", "4,825 Days x m Factors", "Daily standardized Thomson Regression Factor Scores from Varimax MLFA", "Clean interpretable latent macro factor time series"),
        ("Hierarchical_Clustering", "4 Linkages x k=2..12 + 35 Assets", "Linkage benchmark (Ward/Average/Complete/Single), Silhouette/CH/DB sweep, optimal leaf-ordered cluster assignments, HRP weights, cluster profiles, full 34-step dendrogram merge history", "Portfolio diversification (HRP), correlated feature pruning, regime-robust clustering"),
        ("Correlation_and_Distance_Matrix", "35x35 Quasi-Diagonalized", "Full 35x35 Pearson correlation matrix & López de Prado metric distance matrix ordered by Optimal Leaf Ordering", "Visual block-diagonal heatmap inspection and external dendrogram/graph plotting"),
    ]
    for tg in tab_guide:
        rows.append(list(tg))

    return normalize_rect(rows)


def build_pca_overview_sheet(cols: list[str], pca_res: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    p = len(cols)
    evals = pca_res["evals"]
    var_pct = pca_res["var_exp_pct"]
    cum_pct = pca_res["cum_var_pct"]
    pa95 = pca_res["pa95"]
    mp_up = pca_res["mp_upper"]
    eff_part = pca_res["eff_part"]
    loadings = pca_res["loadings"]
    evecs = pca_res["evecs"]
    contrib = pca_res["contrib_pct"]
    themes = pca_res["themes"]

    # SECTION 1: Scree & Eigenvalue Diagnostics
    rows.append(["SECTION 1: PCA SCREE TABLE, HORN PARALLEL ANALYSIS & MARCHENKO-PASTUR RMT DIAGNOSTICS (ALL 35 COMPONENTS)"])
    rows.append(
        [
            "Component",
            "Eigenvalue",
            "Variance_Explained_Pct",
            "Cumulative_Variance_Pct",
            "Kaiser_Keep_GT1",
            "Parallel_Analysis_95th",
            "PA_Keep",
            "MP_Upper_Bound",
            "MP_Signal",
            "Eff_Participation_Ratio",
            "Top_Positive_Loadings",
            "Top_Negative_Loadings",
            "Economic_Theme",
        ]
    )
    for k in range(p):
        rows.append(
            [
                f"PC{k + 1}",
                fmt_num(evals[k], 4),
                fmt_num(var_pct[k], 3),
                fmt_num(cum_pct[k], 3),
                "Yes" if evals[k] > 1.0 else "No",
                fmt_num(pa95[k], 4),
                "Yes" if evals[k] > pa95[k] else "No",
                fmt_num(mp_up, 4),
                "Signal" if evals[k] > mp_up else "Noise",
                fmt_num(eff_part[k], 2),
                top_loading_summary(loadings[:, k], cols, sign=1, k=4),
                top_loading_summary(loadings[:, k], cols, sign=-1, k=4),
                themes[k] if k < 12 else "Residual / Idiosyncratic",
            ]
        )

    rows.append([])
    # SECTION 2: Correlation Loadings Matrix (PC1..PC12)
    n_load = min(12, p)
    kaiser_k = pca_res["kaiser_k"]
    rows.append([f"SECTION 2: PCA CORRELATION LOADINGS MATRIX (L = V * sqrt(lambda), PC1..PC{n_load}) & RECONSTRUCTION R2"])
    rows.append(
        ["Asset", "Slug", "Asset_Class"]
        + [f"PC{k + 1}" for k in range(n_load)]
        + ["R2_Top5_PCs", f"R2_Kaiser_{kaiser_k}PCs", "Primary_PC", "Primary_Loading"]
    )
    for i, col in enumerate(cols):
        r2_5 = float(np.sum(loadings[i, :5] ** 2))
        r2_k = float(np.sum(loadings[i, :kaiser_k] ** 2))
        dom_k = int(np.argmax(np.abs(loadings[i, :n_load])))
        rows.append(
            [asset_name(col), asset_slug(col), asset_class(col)]
            + [fmt_num(loadings[i, k], 4) for k in range(n_load)]
            + [fmt_num(r2_5, 4), fmt_num(r2_k, 4), f"PC{dom_k + 1}", fmt_num(loadings[i, dom_k], 4)]
        )

    rows.append([])
    # SECTION 3: Unit-Norm Eigenvector Weights (PC1..PC10)
    n_ev = min(10, p)
    rows.append([f"SECTION 3: ORTHONORMAL EIGENVECTOR WEIGHTS (EIGENPORTFOLIO WEIGHTS, sum(v^2)=1, PC1..PC{n_ev})"])
    rows.append(["Asset", "Slug", "Asset_Class"] + [f"PC{k + 1}_Weight" for k in range(n_ev)])
    for i, col in enumerate(cols):
        rows.append(
            [asset_name(col), asset_slug(col), asset_class(col)]
            + [fmt_num(evecs[i, k], 4) for k in range(n_ev)]
        )

    rows.append([])
    # SECTION 4: Asset Percentage Contributions (PC1..PC10)
    rows.append([f"SECTION 4: ASSET VARIANCE CONTRIBUTION TO EACH PC (100 * v_ij^2, each column sums to 100%, PC1..PC{n_ev})"])
    rows.append(["Asset", "Slug", "Asset_Class"] + [f"PC{k + 1}_Contrib_Pct" for k in range(n_ev)])
    for i, col in enumerate(cols):
        rows.append(
            [asset_name(col), asset_slug(col), asset_class(col)]
            + [fmt_num(contrib[i, k], 2) for k in range(n_ev)]
        )

    return normalize_rect(rows)


def build_rolling_snapshots_sheet(cols: list[str], roll_res: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    dyn_df = roll_res["dynamics_df"].copy()
    roll_dates = roll_res["roll_dates"]
    pc1_all = roll_res["pc1_loadings_all"]
    dyn_df["_Year"] = pd.to_datetime(dyn_df["Date"]).dt.year

    # SECTION 1: Annual Rolling PCA Regime Summary (2009-2026)
    rows.append(["SECTION 1: ANNUAL ROLLING PCA (252-DAY CAUSAL) SYSTEMIC RISK & REGIME SUMMARY (2009-2026)"])
    rows.append(
        [
            "Year",
            "Trading_Days",
            "Avg_PC1_Var_Pct",
            "Avg_AR3_Pct",
            "Avg_AR5_Pct",
            "Max_AR5_Pct",
            "Avg_Eff_Rank",
            "Avg_Kaiser_K",
            "Avg_MP_Signals",
            "Avg_PC1_Drift_21d",
            "Avg_SP500_PC1_Load",
            "Avg_TLT_PC1_Load",
            "Avg_Stock_Bond_Product",
            "Crisis_Days_Count",
            "Dominant_Regime",
        ]
    )
    for yr, grp in dyn_df.groupby("_Year"):
        dom_reg = grp["Systemic_Regime"].value_counts().index[0]
        crisis_d = int((grp["Systemic_Regime"] == "Crisis / High-Coupling").sum())
        rows.append(
            [
                int(yr),
                int(len(grp)),
                fmt_num(grp["PC1_Var_Pct"].mean(), 2),
                fmt_num(grp["Absorption_Ratio_Top3"].mean(), 2),
                fmt_num(grp["Absorption_Ratio_Top5"].mean(), 2),
                fmt_num(grp["Absorption_Ratio_Top5"].max(), 2),
                fmt_num(grp["Effective_Rank_Entropy"].mean(), 2),
                fmt_num(grp["Kaiser_Factors_Count"].mean(), 2),
                fmt_num(grp["MP_Signal_Factors_Count"].mean(), 2),
                fmt_num(grp["PC1_Subspace_Drift_21d"].mean(), 4),
                fmt_num(grp["PC1_Load_SP500"].mean(), 3) if "PC1_Load_SP500" in grp else "",
                fmt_num(grp["PC1_Load_TLT"].mean(), 3) if "PC1_Load_TLT" in grp else "",
                fmt_num(grp["Stock_Bond_PC1_Product"].mean(), 3) if "Stock_Bond_PC1_Product" in grp else "",
                crisis_d,
                dom_reg,
            ]
        )

    rows.append([])
    # SECTION 2: Year-End & Latest PC1 Loadings Across All 35 Assets
    years = sorted(dyn_df["_Year"].unique().tolist())
    yr_last_idx: dict[int, int] = {}
    for yr in years:
        yr_last_idx[yr] = int(dyn_df.index[dyn_df["_Year"] == yr][-1])

    rows.append(["SECTION 2: YEAR-END & LATEST 252D ROLLING PC1 CORRELATION LOADINGS ACROSS ALL 35 ASSETS (BETA STABILITY AUDIT)"])
    rows.append(
        ["Asset", "Asset_Class"]
        + [f"YE_{yr}" for yr in years]
        + ["Hist_Mean", "Hist_Std", "Hist_Min", "Hist_Max", "Latest_Date_Val"]
    )
    for i, col in enumerate(cols):
        series_i = pc1_all[:, i]
        rows.append(
            [asset_name(col), asset_class(col)]
            + [fmt_num(pc1_all[yr_last_idx[yr], i], 3) for yr in years]
            + [
                fmt_num(float(np.mean(series_i)), 3),
                fmt_num(float(np.std(series_i, ddof=1)), 3),
                fmt_num(float(np.min(series_i)), 3),
                fmt_num(float(np.max(series_i)), 3),
                fmt_num(float(series_i[-1]), 3),
            ]
        )

    rows.append([])
    # SECTION 3: Top Historical Systemic Fragility Peaks
    rows.append(["SECTION 3: TOP 12 HISTORICAL SYSTEMIC COUPLING / ABSORPTION RATIO PEAKS (NON-OVERLAPPING >= 126 TRADING DAYS)"])
    rows.append(
        [
            "Rank",
            "Peak_Date",
            "AR5_252d_Pct",
            "PC1_Var_252d_Pct",
            "AR3_252d_Pct",
            "AR5_Causal_Z252",
            "Eff_Rank_Entropy",
            "Fast63d_AR5_Pct",
            "Stock_Bond_PC1_Product",
            "Systemic_Regime",
        ]
    )
    for rank, pk in enumerate(roll_res["stress_peaks"], start=1):
        rows.append(
            [
                rank,
                pk["Date"],
                fmt_num(pk["AR5_252d"], 2),
                fmt_num(pk["PC1_Var_252d"], 2),
                fmt_num(pk["AR3_252d"], 2),
                fmt_num(pk["AR5_Z252"], 2) if not math.isnan(pk["AR5_Z252"]) else "",
                fmt_num(pk["Eff_Rank"], 2),
                fmt_num(pk["Fast63d_AR5"], 2),
                fmt_num(pk["Stock_Bond_Product"], 3) if not math.isnan(pk["Stock_Bond_Product"]) else "",
                pk["Regime"],
            ]
        )

    return normalize_rect(rows)


def build_factor_analysis_sheet(cols: list[str], fa_res: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    m = fa_res["m"]
    adeq = fa_res["adequacy"]
    Lv = fa_res["L_varimax"]
    Lp = fa_res["L_promax"]
    Phi = fa_res["factor_corr"]

    # SECTION 1: Adequacy & Model Fit Diagnostics
    rows.append(["SECTION 1: FACTOR ANALYSIS ADEQUACY, SPHERICITY & MAXIMUM LIKELIHOOD FIT DIAGNOSTICS"])
    rows.append(["Diagnostic", "Value", "Interpretation / Benchmark"])
    diag_items = [
        ("Estimation Method", "Maximum Likelihood Factor Analysis (Jöreskog Profile ML, L-BFGS-B)", f"Converged={fa_res['converged']} in {fa_res['n_iter']} iterations; Heywood lower-bound (psi=0.005) hits = {fa_res['heywood_count']}"),
        ("Number of Extracted Factors (m)", m, "Auto-selected via Horn's Parallel Analysis (95th pct) & Marchenko-Pastur RMT signal count"),
        ("Overall KMO (Kaiser-Meyer-Olkin)", fmt_num(adeq["kmo_overall"], 4), f"{adeq['kmo_rating']} (>0.70 recommended for reliable common factors)"),
        ("Bartlett's Test of Sphericity (chi2)", fmt_num(adeq["bart_chi2"], 2), f"df = {adeq['bart_df']}, p-value = {adeq['bart_p']:.2e} (rejects identity correlation matrix)"),
        ("Total Common Variance Explained (%)", fmt_num(fa_res["cum_var_pct"][-1], 2), "Share of standardized variance explained by m common latent factors (remainder is asset-specific uniqueness)"),
        ("ML Discrepancy Function (F_ML)", fmt_num(fa_res["f_ml"], 4), f"Model chi2 = {fa_res['ml_chi2']:,.1f}, df = {fa_res['df_model']}"),
        ("RMSR (Root Mean Square Residual)", fmt_num(fa_res["rmsr"], 4), "Off-diagonal correlation residual RMSE (<0.05 = excellent fit; <0.08 = good fit)"),
        ("RMSEA", fmt_num(fa_res["rmsea"], 4), "Root Mean Square Error of Approximation (<0.06 = close fit; <0.08 = acceptable)"),
        ("Tucker-Lewis Index (TLI)", fmt_num(fa_res["tli"], 4), "Incremental fit index penalized for complexity (>0.90 = good; >0.95 = excellent)"),
        ("Comparative Fit Index (CFI)", fmt_num(fa_res["cfi"], 4), "Comparative fit vs independence null model (>0.90 = good; >0.95 = excellent)"),
    ]
    for item in diag_items:
        rows.append(list(item))

    rows.append([])
    # SECTION 2: Factor Variance Summary & Economic Interpretation
    rows.append(["SECTION 2: VARIMAX ROTATED FACTOR VARIANCE SUMMARY & ECONOMIC LABELS"])
    rows.append(
        [
            "Factor",
            "Economic_Theme",
            "SS_Loadings",
            "Variance_Explained_Pct",
            "Cumulative_Var_Pct",
            "Top_Positive_Loadings",
            "Top_Negative_Loadings",
        ]
    )
    for k in range(m):
        rows.append(
            [
                f"F{k + 1}",
                fa_res["themes"][k],
                fmt_num(fa_res["ss_loadings"][k], 4),
                fmt_num(fa_res["var_pct"][k], 2),
                fmt_num(fa_res["cum_var_pct"][k], 2),
                top_loading_summary(Lv[:, k], cols, sign=1, k=5),
                top_loading_summary(Lv[:, k], cols, sign=-1, k=4),
            ]
        )

    rows.append([])
    # SECTION 3: Varimax Rotated Loadings + Per-Asset KMO / Communality / Uniqueness
    rows.append(["SECTION 3: VARIMAX (ORTHOGONAL) ROTATED FACTOR LOADINGS, PER-ASSET KMO & COMMUNALITIES (35 ASSETS)"])
    rows.append(
        ["Asset", "Asset_Class", "Asset_KMO", "Init_SMC", "Communality_h2", "Uniqueness_u2", "Complexity"]
        + [f"F{k + 1}" for k in range(m)]
        + ["Dominant_Factor", "Max_Abs_Loading"]
    )
    for i, col in enumerate(cols):
        dom_k = int(np.argmax(np.abs(Lv[i])))
        rows.append(
            [
                asset_name(col),
                asset_class(col),
                fmt_num(adeq["kmo_per_var"][i], 4),
                fmt_num(adeq["smc"][i], 4),
                fmt_num(fa_res["communalities"][i], 4),
                fmt_num(fa_res["uniquenesses"][i], 4),
                fmt_num(fa_res["complexity"][i], 2),
            ]
            + [fmt_num(Lv[i, k], 4) for k in range(m)]
            + [f"F{dom_k + 1} ({fa_res['themes'][dom_k]})", fmt_num(abs(Lv[i, dom_k]), 4)]
        )

    rows.append([])
    # SECTION 4: Promax Oblique Pattern Matrix
    rows.append(["SECTION 4: PROMAX (OBLIQUE, kappa=4) ROTATED FACTOR PATTERN MATRIX (35 ASSETS)"])
    rows.append(["Asset", "Asset_Class"] + [f"F{k + 1}_Promax" for k in range(m)])
    for i, col in enumerate(cols):
        rows.append(
            [asset_name(col), asset_class(col)]
            + [fmt_num(Lp[i, k], 4) for k in range(m)]
        )

    rows.append([])
    # SECTION 5: Promax Inter-Factor Correlation Matrix
    rows.append(["SECTION 5: PROMAX OBLIQUE INTER-FACTOR CORRELATION MATRIX (Phi)"])
    rows.append(["Factor", "Economic_Theme"] + [f"F{k + 1}" for k in range(m)])
    for k in range(m):
        rows.append(
            [f"F{k + 1}", fa_res["themes"][k]]
            + [fmt_num(Phi[k, j], 4) for j in range(m)]
        )

    return normalize_rect(rows)


def build_clustering_sheet(cols: list[str], R: np.ndarray, hc_res: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    opt_k = hc_res["opt_k_ward"]
    leaf_order = hc_res["leaf_order"]

    # SECTION 1: Linkage Method & Optimal k Benchmark
    rows.append(["SECTION 1: HIERARCHICAL CLUSTERING LINKAGE & OPTIMAL-K VALIDATION BENCHMARK (d_ij = sqrt(0.5*(1 - rho_ij)))"])
    bench_headers = [
        "Linkage_Method",
        "Cophenetic_Corr",
        "k_Clusters",
        "Silhouette_Score",
        "Calinski_Harabasz",
        "Davies_Bouldin",
        "Mean_Intra_Corr",
        "Mean_Inter_Corr",
        "Corr_Separation_Gap",
        "Largest_Cluster_Size",
        "Singleton_Clusters",
    ]
    rows.append(bench_headers)
    for br in hc_res["benchmark_rows"]:
        rows.append(
            [
                br["Linkage_Method"],
                fmt_num(br["Cophenetic_Corr"], 4),
                br["k_Clusters"],
                fmt_num(br["Silhouette_Score"], 4),
                fmt_num(br["Calinski_Harabasz"], 2),
                fmt_num(br["Davies_Bouldin"], 4),
                fmt_num(br["Mean_Intra_Corr"], 4),
                fmt_num(br["Mean_Inter_Corr"], 4),
                fmt_num(br["Corr_Separation_Gap"], 4),
                br["Largest_Cluster_Size"],
                br["Singleton_Clusters"],
            ]
        )

    rows.append([])
    # SECTION 2: Asset Cluster Memberships & HRP Weights (in Optimal Leaf Order)
    rows.append([f"SECTION 2: ASSET CLUSTER ASSIGNMENTS (QUASI-DIAGONAL OPTIMAL LEAF ORDER) & HIERARCHICAL RISK PARITY (HRP) WEIGHTS"])
    rows.append(
        [
            "Dendrogram_Order",
            "Asset",
            "Slug",
            "Asset_Class",
            f"Ward_Cluster_k{opt_k}",
            "Cluster_Label",
            "Ward_k5",
            "Ward_k8",
            f"AvgLink_k{hc_res['best_k_by_method']['average']}",
            f"CompleteLink_k{hc_res['best_k_by_method']['complete']}",
            "Asset_Silhouette",
            "Nearest_Neighbor",
            "NN_Corr",
            "NN_Dist",
            "Raw_Ann_Vol",
            "HRP_Corr_Weight_Pct",
            "Equal_Weight_Pct",
        ]
    )
    eq_w = 100.0 / len(cols)
    for pos, idx in enumerate(leaf_order, start=1):
        col = cols[idx]
        cid = int(hc_res["labs_ward_opt"][idx])
        rows.append(
            [
                pos,
                asset_name(col),
                asset_slug(col),
                asset_class(col),
                cid,
                hc_res["cluster_labels_map"][cid],
                int(hc_res["labs_ward_k5"][idx]),
                int(hc_res["labs_ward_k8"][idx]),
                int(hc_res["labs_avg_opt"][idx]),
                int(hc_res["labs_comp_opt"][idx]),
                fmt_num(hc_res["ward_sil_assets"][idx], 4),
                hc_res["nn_asset"][idx],
                fmt_num(hc_res["nn_corr"][idx], 4),
                fmt_num(hc_res["nn_dist"][idx], 4),
                fmt_num(hc_res["raw_ann_vol"][idx], 4),
                fmt_num(100.0 * hc_res["hrp_weights"][idx], 2),
                fmt_num(eq_w, 2),
            ]
        )

    rows.append([])
    # SECTION 3: Cluster Profiles (Ward Optimal k)
    rows.append([f"SECTION 3: CLUSTER PROFILES & INTERNAL COHESION SUMMARY (WARD OPTIMAL k = {opt_k})"])
    rows.append(
        [
            "Cluster_ID",
            "Cluster_Label",
            "Asset_Count",
            "Mean_Intra_Corr",
            "Min_Intra_Corr",
            "Max_Intra_Corr",
            "Mean_Inter_Corr",
            "Mean_Silhouette",
            "Total_HRP_Weight_Pct",
            "Member_Assets",
        ]
    )
    labs = hc_res["labs_ward_opt"]
    for cid in range(1, opt_k + 1):
        m_idxs = [i for i in leaf_order if labs[i] == cid]
        o_idxs = [i for i in leaf_order if labs[i] != cid]
        intra_vals = [R[m_idxs[a], m_idxs[b]] for a in range(len(m_idxs)) for b in range(a + 1, len(m_idxs))]
        inter_vals = [R[a, b] for a in m_idxs for b in o_idxs]
        rows.append(
            [
                cid,
                hc_res["cluster_labels_map"][cid],
                len(m_idxs),
                fmt_num(float(np.mean(intra_vals)), 4) if intra_vals else 1.0,
                fmt_num(float(np.min(intra_vals)), 4) if intra_vals else 1.0,
                fmt_num(float(np.max(intra_vals)), 4) if intra_vals else 1.0,
                fmt_num(float(np.mean(inter_vals)), 4) if inter_vals else 0.0,
                fmt_num(float(np.mean(hc_res["ward_sil_assets"][m_idxs])), 4),
                fmt_num(100.0 * float(np.sum(hc_res["hrp_weights"][m_idxs])), 2),
                "; ".join(asset_name(cols[i]) for i in m_idxs),
            ]
        )

    rows.append([])
    # SECTION 4: Complete Dendrogram Merge History (34 steps)
    rows.append(["SECTION 4: COMPLETE WARD DENDROGRAM AGGLOMERATIVE MERGE HISTORY (ALL 34 MERGE STEPS)"])
    rows.append(
        [
            "Step",
            "New_Node_ID",
            "Left_Branch",
            "Right_Branch",
            "Ward_Merge_Distance",
            "Avg_Cross_Branch_Corr",
            "New_Cluster_Size",
            "All_Merged_Members",
        ]
    )
    for mh in hc_res["merge_history"]:
        rows.append(
            [
                mh["Step"],
                mh["New_Node_ID"],
                mh["Left_Branch"],
                mh["Right_Branch"],
                fmt_num(mh["Ward_Merge_Distance"], 4),
                fmt_num(mh["Avg_Cross_Branch_Corr"], 4),
                mh["New_Cluster_Size"],
                mh["All_Merged_Members"],
            ]
        )

    return normalize_rect(rows)


def build_corr_dist_sheet(cols: list[str], R: np.ndarray, hc_res: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    order = hc_res["leaf_order"]
    dist_mat = hc_res["dist_mat"]
    ordered_names = [asset_name(cols[i]) for i in order]

    # SECTION 1: Quasi-Diagonalized Pearson Correlation Matrix
    rows.append(["SECTION 1: QUASI-DIAGONALIZED PEARSON CORRELATION MATRIX (35x35, ORDERED BY WARD OPTIMAL LEAF ORDERING)"])
    rows.append(["Asset_Order", "Cluster_ID"] + ordered_names)
    for idx in order:
        cid = int(hc_res["labs_ward_opt"][idx])
        rows.append(
            [asset_name(cols[idx]), cid]
            + [fmt_num(float(R[idx, j]), 4) for j in order]
        )

    rows.append([])
    # SECTION 2: Quasi-Diagonalized Correlation Distance Matrix
    rows.append(["SECTION 2: QUASI-DIAGONALIZED CORRELATION DISTANCE MATRIX (d_ij = sqrt(0.5*(1 - rho_ij)), 35x35)"])
    rows.append(["Asset_Order", "Cluster_ID"] + ordered_names)
    for idx in order:
        cid = int(hc_res["labs_ward_opt"][idx])
        rows.append(
            [asset_name(cols[idx]), cid]
            + [fmt_num(float(dist_mat[idx, j]), 4) for j in order]
        )

    return normalize_rect(rows)


def build_visual_dashboard_sheet(
    cols: list[str],
    pca_res: dict[str, Any],
    roll_res: dict[str, Any],
    fa_res: dict[str, Any],
    hc_res: dict[str, Any],
) -> tuple[list[list[Any]], dict[str, Any]]:
    """
    Build the Visual_Dashboard tab:
      - Rows 1-2: Executive dashboard header banner
      - Rows 4-103 (Cols A-U): Open visual canvas hosting 10 native embedded Google Sheets charts
      - Rows 105-141 (Cols A-F): Dendrogram Leaf Index Reference Key (1..35) directly below the Dendrogram
      - Cols W-AM (0-based 22..38): Compact helper series powering Rolling PCA, Cumulative PC, and Dendrogram charts
    """
    dyn_df = roll_res["dynamics_df"].reset_index(drop=True)
    # Sample every 5 trading days (~weekly) while guaranteeing exact latest date is included
    w_idxs = list(range(0, len(dyn_df), 5))
    if w_idxs[-1] != len(dyn_df) - 1:
        w_idxs.append(len(dyn_df) - 1)
    w_dyn = dyn_df.iloc[w_idxs].reset_index(drop=True)

    # Weekly-sampled cumulative standardized PC1..PC3 scores (smooth macro trend lines)
    sc_df = pca_res["scores_df"].reset_index(drop=True)
    cum_pc1 = sc_df["PC1"].cumsum()
    cum_pc2 = sc_df["PC2"].cumsum()
    cum_pc3 = sc_df["PC3"].cumsum()
    s_idxs = list(range(0, len(sc_df), 5))
    if s_idxs[-1] != len(sc_df) - 1:
        s_idxs.append(len(sc_df) - 1)

    # Dendrogram branch coordinates (34 branches * 5 rows + 2 rows for cut-height horizontal threshold)
    # Use non-blank X with blank Y on separator rows so Google Sheets LINE charts cleanly break segments
    dendro_rows: list[list[Any]] = []
    for br in hc_res["dendro_branches"]:
        for xv, yv in zip(br["x"], br["y"]):
            if br["is_intra"]:
                dendro_rows.append([xv, yv, "", ""])
            else:
                dendro_rows.append([xv, "", yv, ""])
        # Non-blank X + blank Y breaks the polyline cleanly with interpolateNulls=false
        dendro_rows.append([br["x"][-1], "", "", ""])
    cut_h = round(float(hc_res["cut_height"]), 4)
    dendro_rows.append([1.0, "", "", cut_h])
    dendro_rows.append([float(len(cols)), "", "", cut_h])

    # Determine required grid height (covers dashboard canvas + leaf key + weekly helper rows)
    leaf_key_start = 104  # 0-based row 104 -> Sheet Row 105
    n_rows = max(leaf_key_start + 2 + len(cols), 2 + len(w_dyn), 2 + len(s_idxs), 2 + len(dendro_rows))
    n_cols = 39  # Col A (0) .. Col AM (38)
    grid: list[list[Any]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    # Header banner on Cols A..T
    grid[0][0] = "VISUAL DASHBOARD — PCA, ROLLING PCA, FACTOR ANALYSIS & HIERARCHICAL CLUSTERING"
    grid[1][0] = (
        f"10 Live Native Charts | Sample: {sc_df['Date'].iloc[0]} to {sc_df['Date'].iloc[-1]} "
        f"({len(cols)} Daily Assets) | Scroll right (Cols W-AM) for compact chart helper tables; "
        f"scroll down (Row 105) for Dendrogram Leaf Index Key (1-35)."
    )

    # Dendrogram Leaf Index Reference Key directly below Chart 9 (Rows 105..141, Cols A..G)
    grid[leaf_key_start][0] = "SECTION: WARD DENDROGRAM LEAF INDEX REFERENCE KEY (X-AXIS POSITIONS 1 TO 35)"
    grid[leaf_key_start + 1][0:7] = [
        "Leaf_No",
        "Asset",
        "Asset_Class",
        f"Ward_Cluster_k{hc_res['opt_k_ward']}",
        "Cluster_Label",
        "HRP_Weight_Pct",
        "Nearest_Neighbor",
    ]
    for pos, idx in enumerate(hc_res["leaf_order"], start=1):
        r = leaf_key_start + 1 + pos
        cid = int(hc_res["labs_ward_opt"][idx])
        grid[r][0:7] = [
            pos,
            asset_name(cols[idx]),
            asset_class(cols[idx]),
            cid,
            hc_res["cluster_labels_map"][cid],
            fmt_num(100.0 * hc_res["hrp_weights"][idx], 2),
            f"{hc_res['nn_asset'][idx]} ({hc_res['nn_corr'][idx]:+.2f})",
        ]

    # Helper Block 1: Weekly Rolling PCA Time Series (Cols W..AE -> indices 22..30)
    grid[0][22] = "CHART HELPER 1: WEEKLY ROLLING PCA (252D CAUSAL)"
    grid[1][22:31] = [
        "Date",
        "PC1_Var_Pct",
        "AR3_Pct",
        "AR5_Pct",
        "AR5_Causal_Z252",
        "Effective_Rank",
        "PC1_Load_SP500",
        "PC1_Load_TLT",
        "Stock_Bond_PC1_Product",
    ]
    for i, row in w_dyn.iterrows():
        r = 2 + i
        grid[r][22:31] = [
            str(row["Date"]),
            fmt_num(row["PC1_Var_Pct"], 2),
            fmt_num(row["Absorption_Ratio_Top3"], 2),
            fmt_num(row["Absorption_Ratio_Top5"], 2),
            fmt_num(row["AR5_Causal_Z252"], 2) if pd.notna(row["AR5_Causal_Z252"]) else "",
            fmt_num(row["Effective_Rank_Entropy"], 2),
            fmt_num(row["PC1_Load_SP500"], 3) if "PC1_Load_SP500" in row else "",
            fmt_num(row["PC1_Load_TLT"], 3) if "PC1_Load_TLT" in row else "",
            fmt_num(row["Stock_Bond_PC1_Product"], 3) if "Stock_Bond_PC1_Product" in row else "",
        ]

    # Helper Block 2: Weekly Cumulative Standardized PCA Scores (Cols AF..AI -> indices 31..34)
    grid[0][31] = "CHART HELPER 2: CUMULATIVE PC SCORES"
    grid[1][31:35] = [
        "Date",
        "Cum_PC1_GlobalRiskOn",
        "Cum_PC2_USRates",
        "Cum_PC3_AsiaChina",
    ]
    for k_i, orig_i in enumerate(s_idxs):
        r = 2 + k_i
        grid[r][31:35] = [
            str(sc_df["Date"].iloc[orig_i]),
            fmt_num(float(cum_pc1.iloc[orig_i]), 2),
            fmt_num(float(cum_pc2.iloc[orig_i]), 2),
            fmt_num(float(cum_pc3.iloc[orig_i]), 2),
        ]

    # Helper Block 3: Exact Ward Dendrogram Branch Coordinates (Cols AJ..AM -> indices 35..38)
    grid[0][35] = "CHART HELPER 3: WARD DENDROGRAM COORDS"
    grid[1][35:39] = [
        "Leaf_X",
        "Intra_Cluster_Branches",
        "Inter_Cluster_Branches",
        f"Optimal_Cut_k{hc_res['opt_k_ward']}",
    ]
    for k_i, drow in enumerate(dendro_rows):
        grid[2 + k_i][35:39] = drow

    dash_meta = {
        "n_weekly_roll": len(w_dyn),
        "n_weekly_scores": len(s_idxs),
        "n_dendro_rows": len(dendro_rows),
        "leaf_key_start": leaf_key_start,
        "cut_height": cut_h,
    }
    return grid, dash_meta


# =============================================================================
# 9. COLOR SCALES & NATIVE GOOGLE SHEETS CHART SPECIFICATIONS
# =============================================================================


def _find_row_index(all_rows: list[list[str]], prefix: str, start_after: int = 0) -> int | None:
    for idx in range(start_after, len(all_rows)):
        r = all_rows[idx]
        if r and str(r[0]).strip().startswith(prefix):
            return idx
    return None


def _count_table_rows(all_rows: list[list[str]], data_start: int) -> int:
    cnt = 0
    for idx in range(data_start, len(all_rows)):
        r = all_rows[idx]
        if not r or str(r[0]).strip() == "":
            break
        cnt += 1
    return cnt


def build_color_scale_specs(
    tab_rows_map: dict[str, list[list[str]]],
    fa_res: dict[str, Any],
    hc_res: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build declarative conditional-formatting color-scale rules dynamically from actual CSV row offsets.
    All row/col bounds are 0-based [start, end) (end exclusive).
    """
    specs: list[dict[str, Any]] = []

    def add_grad(
        tab: str,
        sr: int,
        er: int,
        sc: int,
        ec: int,
        min_v: float,
        max_v: float,
        min_hex: str,
        max_hex: str,
        mid_v: float | None = None,
        mid_hex: str | None = None,
    ) -> None:
        if er <= sr or ec <= sc:
            return
        specs.append(
            {
                "tab": tab,
                "start_row": sr,
                "end_row": er,
                "start_col": sc,
                "end_col": ec,
                "min_val": min_v,
                "max_val": max_v,
                "min_hex": min_hex,
                "max_hex": max_hex,
                "mid_val": mid_v,
                "mid_hex": mid_hex,
            }
        )

    # 1. Correlation_and_Distance_Matrix: 35x35 Pearson Corr (Blue-White-Red) + Distance (Green-Yellow-Red)
    corr_rows = tab_rows_map.get("Correlation_and_Distance_Matrix", [])
    s1 = _find_row_index(corr_rows, "SECTION 1:")
    if s1 is not None:
        n1 = _count_table_rows(corr_rows, s1 + 2)
        add_grad("Correlation_and_Distance_Matrix", s1 + 2, s1 + 2 + n1, 2, 2 + n1, -1.0, 1.0, "#5B9BD5", "#E67C73", 0.0, "#FFFFFF")
    s2 = _find_row_index(corr_rows, "SECTION 2:")
    if s2 is not None:
        n2 = _count_table_rows(corr_rows, s2 + 2)
        add_grad("Correlation_and_Distance_Matrix", s2 + 2, s2 + 2 + n2, 2, 2 + n2, 0.0, 1.0, "#63BE7B", "#F8696B", 0.5, "#FFEB84")

    # 2. PCA_Overview_and_Loadings: Scree Eigenvalues, Loadings PC1-12, R2, Eigenvectors, Contributions
    pca_rows = tab_rows_map.get("PCA_Overview_and_Loadings", [])
    ps1 = _find_row_index(pca_rows, "SECTION 1:")
    if ps1 is not None:
        n_pc = _count_table_rows(pca_rows, ps1 + 2)
        add_grad("PCA_Overview_and_Loadings", ps1 + 2, ps1 + 2 + n_pc, 1, 2, 0.0, 7.0, "#FFFFFF", "#6FA8DC")
    ps2 = _find_row_index(pca_rows, "SECTION 2:")
    if ps2 is not None:
        n_a = _count_table_rows(pca_rows, ps2 + 2)
        add_grad("PCA_Overview_and_Loadings", ps2 + 2, ps2 + 2 + n_a, 3, 15, -0.8, 0.8, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
        add_grad("PCA_Overview_and_Loadings", ps2 + 2, ps2 + 2 + n_a, 15, 17, 0.0, 1.0, "#FFFFFF", "#63BE7B")
    ps3 = _find_row_index(pca_rows, "SECTION 3:")
    if ps3 is not None:
        n_a = _count_table_rows(pca_rows, ps3 + 2)
        add_grad("PCA_Overview_and_Loadings", ps3 + 2, ps3 + 2 + n_a, 3, 13, -0.45, 0.45, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
    ps4 = _find_row_index(pca_rows, "SECTION 4:")
    if ps4 is not None:
        n_a = _count_table_rows(pca_rows, ps4 + 2)
        add_grad("PCA_Overview_and_Loadings", ps4 + 2, ps4 + 2 + n_a, 3, 13, 0.0, 25.0, "#FFFFFF", "#6FA8DC")

    # 3. Rolling_PCA_Loadings_Snapshots: Annual AR5 & Stock-Bond flip, YE PC1 Loadings heatmap, Stress peaks
    rs_rows = tab_rows_map.get("Rolling_PCA_Loadings_Snapshots", [])
    rs1 = _find_row_index(rs_rows, "SECTION 1:")
    if rs1 is not None:
        n_yr = _count_table_rows(rs_rows, rs1 + 2)
        add_grad("Rolling_PCA_Loadings_Snapshots", rs1 + 2, rs1 + 2 + n_yr, 4, 6, 48.0, 62.0, "#63BE7B", "#E67C73", 55.0, "#FFFFFF")
        add_grad("Rolling_PCA_Loadings_Snapshots", rs1 + 2, rs1 + 2 + n_yr, 12, 13, -0.6, 0.6, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
    rs2 = _find_row_index(rs_rows, "SECTION 2:")
    if rs2 is not None:
        n_a = _count_table_rows(rs_rows, rs2 + 2)
        hdr2 = rs_rows[rs2 + 1]
        ye_end = next((j for j, h in enumerate(hdr2) if h == "Hist_Mean"), len(hdr2))
        add_grad("Rolling_PCA_Loadings_Snapshots", rs2 + 2, rs2 + 2 + n_a, 2, ye_end, -0.8, 0.8, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
        latest_col = next((j for j, h in enumerate(hdr2) if h == "Latest_Date_Val"), None)
        if latest_col is not None:
            add_grad("Rolling_PCA_Loadings_Snapshots", rs2 + 2, rs2 + 2 + n_a, latest_col, latest_col + 1, -0.8, 0.8, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
    rs3 = _find_row_index(rs_rows, "SECTION 3:")
    if rs3 is not None:
        n_pk = _count_table_rows(rs_rows, rs3 + 2)
        add_grad("Rolling_PCA_Loadings_Snapshots", rs3 + 2, rs3 + 2 + n_pk, 2, 3, 55.0, 62.0, "#FFFFFF", "#E67C73")

    # 4. Rolling_PCA_Dynamics: AR5, Causal AR5 Z-Score, Stock-Bond Product
    rd_rows = tab_rows_map.get("Rolling_PCA_Dynamics", [])
    if len(rd_rows) > 2:
        hdr_rd = rd_rows[0]
        n_rd = len(rd_rows)
        if "Absorption_Ratio_Top5" in hdr_rd:
            c = hdr_rd.index("Absorption_Ratio_Top5")
            add_grad("Rolling_PCA_Dynamics", 1, n_rd, c, c + 1, 47.0, 62.0, "#63BE7B", "#E67C73", 55.0, "#FFFFFF")
        if "AR5_Causal_Z252" in hdr_rd:
            c = hdr_rd.index("AR5_Causal_Z252")
            add_grad("Rolling_PCA_Dynamics", 1, n_rd, c, c + 1, -2.0, 2.5, "#63BE7B", "#E67C73", 0.0, "#FFFFFF")
        if "Stock_Bond_PC1_Product" in hdr_rd:
            c = hdr_rd.index("Stock_Bond_PC1_Product")
            add_grad("Rolling_PCA_Dynamics", 1, n_rd, c, c + 1, -0.7, 0.7, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")

    # 5. Factor_Analysis_Results: Asset KMO, Communality h2, Varimax Loadings, Promax Pattern, Factor Corr Phi
    fa_rows = tab_rows_map.get("Factor_Analysis_Results", [])
    m = fa_res["m"]
    fs3 = _find_row_index(fa_rows, "SECTION 3:")
    if fs3 is not None:
        n_a = _count_table_rows(fa_rows, fs3 + 2)
        add_grad("Factor_Analysis_Results", fs3 + 2, fs3 + 2 + n_a, 2, 3, 0.55, 0.95, "#FCE4D6", "#63BE7B", 0.75, "#FFFFFF")
        add_grad("Factor_Analysis_Results", fs3 + 2, fs3 + 2 + n_a, 4, 5, 0.0, 1.0, "#FFFFFF", "#63BE7B")
        add_grad("Factor_Analysis_Results", fs3 + 2, fs3 + 2 + n_a, 7, 7 + m, -0.85, 0.85, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
    fs4 = _find_row_index(fa_rows, "SECTION 4:")
    if fs4 is not None:
        n_a = _count_table_rows(fa_rows, fs4 + 2)
        add_grad("Factor_Analysis_Results", fs4 + 2, fs4 + 2 + n_a, 2, 2 + m, -0.85, 0.85, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")
    fs5 = _find_row_index(fa_rows, "SECTION 5:")
    if fs5 is not None:
        n_f = _count_table_rows(fa_rows, fs5 + 2)
        add_grad("Factor_Analysis_Results", fs5 + 2, fs5 + 2 + n_f, 2, 2 + m, -0.6, 0.6, "#6BAED6", "#E67C73", 0.0, "#FFFFFF")

    # 6. Hierarchical_Clustering: Benchmark Silhouette & Gap, Asset Silhouette/NN_Corr/HRP, Cluster Profiles
    hc_rows = tab_rows_map.get("Hierarchical_Clustering", [])
    hs1 = _find_row_index(hc_rows, "SECTION 1:")
    if hs1 is not None:
        n_b = _count_table_rows(hc_rows, hs1 + 2)
        add_grad("Hierarchical_Clustering", hs1 + 2, hs1 + 2 + n_b, 3, 4, 0.0, 0.30, "#FFFFFF", "#63BE7B")
        add_grad("Hierarchical_Clustering", hs1 + 2, hs1 + 2 + n_b, 8, 9, 0.0, 0.45, "#FFFFFF", "#63BE7B")
    hs2 = _find_row_index(hc_rows, "SECTION 2:")
    if hs2 is not None:
        n_a = _count_table_rows(hc_rows, hs2 + 2)
        add_grad("Hierarchical_Clustering", hs2 + 2, hs2 + 2 + n_a, 10, 11, -0.15, 0.60, "#F8696B", "#63BE7B", 0.15, "#FFFFFF")
        add_grad("Hierarchical_Clustering", hs2 + 2, hs2 + 2 + n_a, 12, 13, 0.0, 1.0, "#FFFFFF", "#63BE7B")
        add_grad("Hierarchical_Clustering", hs2 + 2, hs2 + 2 + n_a, 15, 16, 1.0, 6.6, "#FFFFFF", "#6FA8DC")
    hs3 = _find_row_index(hc_rows, "SECTION 3:")
    if hs3 is not None:
        n_c = _count_table_rows(hc_rows, hs3 + 2)
        add_grad("Hierarchical_Clustering", hs3 + 2, hs3 + 2 + n_c, 3, 4, 0.0, 0.90, "#FFFFFF", "#63BE7B")
        add_grad("Hierarchical_Clustering", hs3 + 2, hs3 + 2 + n_c, 8, 9, 3.0, 23.0, "#FFFFFF", "#6FA8DC")

    return specs


def build_chart_specs(
    tab_rows_map: dict[str, list[list[str]]],
    dash_meta: dict[str, Any],
    fa_res: dict[str, Any],
    hc_res: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build declarative specifications for 10 native Google Sheets charts placed in a clean
    2-column x 5-row executive gallery on Visual_Dashboard.
    All ranges are 0-based [start, end) (end exclusive), including header row (headerCount=1).
    """
    charts: list[dict[str, Any]] = []
    n_wr = dash_meta["n_weekly_roll"]
    n_ws = dash_meta["n_weekly_scores"]
    n_dn = dash_meta["n_dendro_rows"]

    # Dynamic row lookup for Factor Analysis Section 2 (F1..Fm variance summary)
    fa_rows = tab_rows_map.get("Factor_Analysis_Results", [])
    fs2 = _find_row_index(fa_rows, "SECTION 2:") or 13
    fa_hdr_row = fs2 + 1
    fa_end_row = fa_hdr_row + 1 + fa_res["m"]

    # Dynamic row lookup for Hierarchical Clustering Section 1 (Ward k=2..12) and Section 2 (35 assets)
    hc_rows = tab_rows_map.get("Hierarchical_Clustering", [])
    hs1 = _find_row_index(hc_rows, "SECTION 1:") or 0
    ward_hdr_row = hs1 + 1
    ward_end_row = ward_hdr_row + 1 + 11  # k=2..12 -> 11 Ward rows
    hs2 = _find_row_index(hc_rows, "SECTION 2:") or 47
    hc_asset_hdr_row = hs2 + 1
    hc_asset_end_row = hc_asset_hdr_row + 1 + _count_table_rows(hc_rows, hs2 + 2)

    # Row 1 (Anchor Row 3): Full-Sample PCA Scree & Cumulative Variance
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 3,
            "anchor_col": 0,
            "width": 660,
            "height": 380,
            "title": "1. PCA Scree Plot — Eigenvalues vs Horn Parallel Analysis (95th) & Marchenko-Pastur",
            "chart_type": "COMBO",
            "x_title": "Principal Component (PC1–PC20)",
            "y_left_title": "Eigenvalue",
            "domain": {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 0, "ec": 1},
            "series": [
                {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 1, "ec": 2, "type": "COLUMN", "axis": "LEFT_AXIS", "color": "#1F4E78"},
                {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 5, "ec": 6, "type": "LINE", "axis": "LEFT_AXIS", "color": "#E67C73", "line_width": 2},
                {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 7, "ec": 8, "type": "LINE", "axis": "LEFT_AXIS", "color": "#F4B400", "line_width": 2},
            ],
        }
    )
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 3,
            "anchor_col": 10,
            "width": 660,
            "height": 380,
            "title": "2. PCA Variance Explained (%) & Cumulative Variance Explained (%)",
            "chart_type": "COMBO",
            "x_title": "Principal Component (PC1–PC20)",
            "y_left_title": "Individual Var Explained (%)",
            "y_right_title": "Cumulative Var Explained (%)",
            "domain": {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 0, "ec": 1},
            "series": [
                {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 2, "ec": 3, "type": "COLUMN", "axis": "LEFT_AXIS", "color": "#5B9BD5"},
                {"tab": "PCA_Overview_and_Loadings", "sr": 1, "er": 22, "sc": 3, "ec": 4, "type": "LINE", "axis": "RIGHT_AXIS", "color": "#1F4E78", "line_width": 3},
            ],
        }
    )

    # Row 2 (Anchor Row 23): Strictly Causal Rolling PCA Time Series
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 23,
            "anchor_col": 0,
            "width": 660,
            "height": 380,
            "title": "3. Rolling PCA (252d Causal) — Systemic Absorption Ratios & PC1 Variance (2009–2026)",
            "chart_type": "LINE",
            "x_title": "Date",
            "y_left_title": "Variance Explained (%)",
            "domain": {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 22, "ec": 23},
            "series": [
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 25, "ec": 26, "axis": "LEFT_AXIS", "color": "#D93025", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 24, "ec": 25, "axis": "LEFT_AXIS", "color": "#F4B400", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 23, "ec": 24, "axis": "LEFT_AXIS", "color": "#1F4E78", "line_width": 2},
            ],
        }
    )
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 23,
            "anchor_col": 10,
            "width": 660,
            "height": 380,
            "title": "4. Rolling PCA Stock-Bond Correlation Regime Flip (S&P 500 vs TLT on PC1, 2009–2026)",
            "chart_type": "LINE",
            "x_title": "Date",
            "y_left_title": "PC1 Correlation Loading / Product",
            "domain": {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 22, "ec": 23},
            "series": [
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 28, "ec": 29, "axis": "LEFT_AXIS", "color": "#1F4E78", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 29, "ec": 30, "axis": "LEFT_AXIS", "color": "#D93025", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 30, "ec": 31, "axis": "LEFT_AXIS", "color": "#F4B400", "line_width": 2},
            ],
        }
    )

    # Row 3 (Anchor Row 43): Systemic Fragility Z-Score & Cumulative PC Trends
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 43,
            "anchor_col": 0,
            "width": 660,
            "height": 380,
            "title": "5. Causal Systemic Fragility — 252d Absorption Ratio Z-Score vs Effective Rank",
            "chart_type": "LINE",
            "x_title": "Date",
            "y_left_title": "AR5 Causal Z-Score (sigma)",
            "y_right_title": "Effective Rank (Independent Bets)",
            "domain": {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 22, "ec": 23},
            "series": [
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 26, "ec": 27, "axis": "LEFT_AXIS", "color": "#D93025", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_wr, "sc": 27, "ec": 28, "axis": "RIGHT_AXIS", "color": "#1F4E78", "line_width": 2},
            ],
        }
    )
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 43,
            "anchor_col": 10,
            "width": 660,
            "height": 380,
            "title": "6. Cumulative Standardized Macro PCA Factor Trends (PC1–PC3, 2008–2026)",
            "chart_type": "LINE",
            "x_title": "Date",
            "y_left_title": "Cumulative Standardized Score",
            "domain": {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 31, "ec": 32},
            "series": [
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 32, "ec": 33, "axis": "LEFT_AXIS", "color": "#1F4E78", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 33, "ec": 34, "axis": "LEFT_AXIS", "color": "#D93025", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 34, "ec": 35, "axis": "LEFT_AXIS", "color": "#0F9D58", "line_width": 2},
            ],
        }
    )

    # Row 4 (Anchor Row 63): Factor Analysis Variance & Clustering Optimal-k Validation
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 63,
            "anchor_col": 0,
            "width": 660,
            "height": 380,
            "title": f"7. Factor Analysis (MLFA Varimax, m={fa_res['m']}) — Common Variance Explained by Latent Factor",
            "chart_type": "COMBO",
            "x_title": "Latent Macro Factor",
            "y_left_title": "Factor Variance Explained (%)",
            "y_right_title": "Cumulative Common Variance (%)",
            "domain": {"tab": "Factor_Analysis_Results", "sr": fa_hdr_row, "er": fa_end_row, "sc": 1, "ec": 2},
            "series": [
                {"tab": "Factor_Analysis_Results", "sr": fa_hdr_row, "er": fa_end_row, "sc": 3, "ec": 4, "type": "COLUMN", "axis": "LEFT_AXIS", "color": "#1F4E78"},
                {"tab": "Factor_Analysis_Results", "sr": fa_hdr_row, "er": fa_end_row, "sc": 4, "ec": 5, "type": "LINE", "axis": "RIGHT_AXIS", "color": "#D93025", "line_width": 3},
            ],
        }
    )
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 63,
            "anchor_col": 10,
            "width": 660,
            "height": 380,
            "title": "8. Hierarchical Clustering — Ward Optimal-k Validation (Silhouette & Separation Gap, k=2..12)",
            "chart_type": "LINE",
            "x_title": "Number of Clusters (k)",
            "y_left_title": "Score / Correlation Separation Gap",
            "domain": {"tab": "Hierarchical_Clustering", "sr": ward_hdr_row, "er": ward_end_row, "sc": 2, "ec": 3},
            "series": [
                {"tab": "Hierarchical_Clustering", "sr": ward_hdr_row, "er": ward_end_row, "sc": 3, "ec": 4, "axis": "LEFT_AXIS", "color": "#1F4E78", "line_width": 3},
                {"tab": "Hierarchical_Clustering", "sr": ward_hdr_row, "er": ward_end_row, "sc": 8, "ec": 9, "axis": "LEFT_AXIS", "color": "#0F9D58", "line_width": 3},
            ],
        }
    )

    # Row 5 (Anchor Row 83): Ward Dendrogram & HRP Portfolio Weights
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 83,
            "anchor_col": 0,
            "width": 660,
            "height": 400,
            "title": f"9. Ward Hierarchical Clustering Dendrogram (Optimal Leaf Order 1–35; Cut k={hc_res['opt_k_ward']} at d={dash_meta['cut_height']:.2f})",
            "chart_type": "LINE",
            "x_title": "Optimal Leaf Order Index (1–35 — see Leaf Reference Key at Row 105 below)",
            "y_left_title": "Ward Merge Distance",
            "interpolate_nulls": False,
            "domain": {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_dn, "sc": 35, "ec": 36},
            "series": [
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_dn, "sc": 36, "ec": 37, "axis": "LEFT_AXIS", "color": "#1F4E78", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_dn, "sc": 37, "ec": 38, "axis": "LEFT_AXIS", "color": "#94A3B8", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_dn, "sc": 38, "ec": 39, "axis": "LEFT_AXIS", "color": "#D93025", "line_width": 2, "dashed": True},
            ],
        }
    )
    charts.append(
        {
            "target_tab": "Visual_Dashboard",
            "anchor_row": 83,
            "anchor_col": 10,
            "width": 660,
            "height": 400,
            "title": "10. Hierarchical Risk Parity (HRP) Portfolio Weights vs Equal Weight (All 35 Assets)",
            "chart_type": "COLUMN",
            "x_title": "Assets (Ordered by Ward Optimal Leaf Ordering)",
            "y_left_title": "Portfolio Weight (%)",
            "domain": {"tab": "Hierarchical_Clustering", "sr": hc_asset_hdr_row, "er": hc_asset_end_row, "sc": 1, "ec": 2},
            "series": [
                {"tab": "Hierarchical_Clustering", "sr": hc_asset_hdr_row, "er": hc_asset_end_row, "sc": 15, "ec": 16, "axis": "LEFT_AXIS", "color": "#1F4E78"},
                {"tab": "Hierarchical_Clustering", "sr": hc_asset_hdr_row, "er": hc_asset_end_row, "sc": 16, "ec": 17, "axis": "LEFT_AXIS", "color": "#B0BEC5"},
            ],
        }
    )

    # Bonus contextual chart directly beside PCA_Scores_TimeSeries data (Cols M+)
    charts.append(
        {
            "target_tab": "PCA_Scores_TimeSeries",
            "anchor_row": 1,
            "anchor_col": 12,
            "width": 680,
            "height": 400,
            "title": "Cumulative Standardized Macro PCA Scores (PC1 Risk-On, PC2 US Rates, PC3 Asia/China)",
            "chart_type": "LINE",
            "x_title": "Date",
            "y_left_title": "Cumulative Standardized Score",
            "domain": {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 31, "ec": 32},
            "series": [
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 32, "ec": 33, "axis": "LEFT_AXIS", "color": "#1F4E78", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 33, "ec": 34, "axis": "LEFT_AXIS", "color": "#D93025", "line_width": 2},
                {"tab": "Visual_Dashboard", "sr": 1, "er": 2 + n_ws, "sc": 34, "ec": 35, "axis": "LEFT_AXIS", "color": "#0F9D58", "line_width": 2},
            ],
        }
    )
    return charts


# =============================================================================
# 10. CSV SAVING & MULTI-BACKEND GOOGLE SHEETS PUBLISHING
# =============================================================================


def save_clean_csv(df: pd.DataFrame, path: Path, decimals: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).round(decimals)
    out.to_csv(path, index=False, na_rep="")
    txt = path.read_text(encoding="utf-8")
    for bad_tok in [",inf,", ",-inf,", ",NaN,", ",nan,"]:
        if bad_tok in txt:
            raise ValueError(f"Found forbidden literal token {bad_tok} in {path}")


def save_rows_csv(rows: list[list[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            clean_r = []
            for v in r:
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    clean_r.append("")
                else:
                    clean_r.append(v)
            writer.writerow(clean_r)


def _cast_sheet_row(row: list[str], is_header: bool, keep_col0_str: bool = True) -> list[Any]:
    if is_header:
        return row
    out: list[Any] = []
    for idx, val in enumerate(row):
        if val == "":
            out.append("")
        elif idx == 0 and keep_col0_str and "-" in val and len(val) == 10:
            out.append(val)  # Keep YYYY-MM-DD Date as string
        else:
            try:
                fv = float(val)
                if math.isnan(fv) or math.isinf(fv):
                    out.append("")
                elif fv.is_integer() and "." not in val and "e" not in val.lower():
                    out.append(int(fv))
                else:
                    out.append(fv)
            except ValueError:
                out.append(val)
    return out


def _detect_section_rows(all_rows: list[list[str]]) -> tuple[list[int], list[int]]:
    """Return 0-based row indices for SECTION banners and table subheader rows."""
    sec_rows: list[int] = []
    sub_rows: list[int] = []
    for idx, r in enumerate(all_rows):
        if idx == 0 or not r:
            continue
        first = str(r[0]).strip()
        if first.startswith("SECTION "):
            sec_rows.append(idx)
        elif first in SUBHEADER_TOKENS:
            sub_rows.append(idx)
    return sec_rows, sub_rows


def _hex_to_rgb(hex_color: str) -> dict[str, float]:
    h = hex_color.lstrip("#")
    return {
        "red": int(h[0:2], 16) / 255.0,
        "green": int(h[2:4], 16) / 255.0,
        "blue": int(h[4:6], 16) / 255.0,
    }


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


def _make_gradient_rule_request(sheet_id: int, spec: dict[str, Any]) -> dict[str, Any]:
    """Build a Google Sheets v4 API addConditionalFormatRule (gradient color scale) request."""
    grad: dict[str, Any] = {
        "minpoint": {
            "colorStyle": {"rgbColor": _hex_to_rgb(spec["min_hex"])},
            "type": "NUMBER",
            "value": str(spec["min_val"]),
        },
        "maxpoint": {
            "colorStyle": {"rgbColor": _hex_to_rgb(spec["max_hex"])},
            "type": "NUMBER",
            "value": str(spec["max_val"]),
        },
    }
    if spec.get("mid_val") is not None and spec.get("mid_hex"):
        grad["midpoint"] = {
            "colorStyle": {"rgbColor": _hex_to_rgb(spec["mid_hex"])},
            "type": "NUMBER",
            "value": str(spec["mid_val"]),
        }
    return {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [
                    {
                        "sheetId": sheet_id,
                        "startRowIndex": spec["start_row"],
                        "endRowIndex": spec["end_row"],
                        "startColumnIndex": spec["start_col"],
                        "endColumnIndex": spec["end_col"],
                    }
                ],
                "gradientRule": grad,
            },
            "index": 0,
        }
    }


def _make_chart_request(sheet_map: dict[str, dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Build a Google Sheets v4 API addChart request from a declarative chart spec."""
    target_sid = sheet_map[spec["target_tab"]]["sheet_id"]
    dom = spec["domain"]
    dom_sid = sheet_map[dom["tab"]]["sheet_id"]

    axes: list[dict[str, Any]] = [
        {"position": "BOTTOM_AXIS", "title": spec.get("x_title", "")},
        {"position": "LEFT_AXIS", "title": spec.get("y_left_title", "")},
    ]
    if spec.get("y_right_title"):
        axes.append({"position": "RIGHT_AXIS", "title": spec["y_right_title"]})

    series_reqs: list[dict[str, Any]] = []
    for s in spec["series"]:
        s_sid = sheet_map[s["tab"]]["sheet_id"]
        s_obj: dict[str, Any] = {
            "series": {
                "sourceRange": {
                    "sources": [
                        {
                            "sheetId": s_sid,
                            "startRowIndex": s["sr"],
                            "endRowIndex": s["er"],
                            "startColumnIndex": s["sc"],
                            "endColumnIndex": s["ec"],
                        }
                    ]
                }
            },
            "targetAxis": s.get("axis", "LEFT_AXIS"),
            "colorStyle": {"rgbColor": _hex_to_rgb(s.get("color", "#1F4E78"))},
        }
        if spec["chart_type"] == "COMBO" and s.get("type"):
            s_obj["type"] = s["type"]
        if spec["chart_type"] == "LINE" or s.get("type") == "LINE":
            s_obj["lineStyle"] = {
                "width": int(s.get("line_width", 2)),
                "type": "MEDIUM_DASHED" if s.get("dashed") else "SOLID",
            }
        series_reqs.append(s_obj)

    basic_chart: dict[str, Any] = {
        "chartType": spec["chart_type"],
        "legendPosition": "BOTTOM_LEGEND",
        "headerCount": 1,
        "axis": axes,
        "domains": [
            {
                "domain": {
                    "sourceRange": {
                        "sources": [
                            {
                                "sheetId": dom_sid,
                                "startRowIndex": dom["sr"],
                                "endRowIndex": dom["er"],
                                "startColumnIndex": dom["sc"],
                                "endColumnIndex": dom["ec"],
                            }
                        ]
                    }
                }
            }
        ],
        "series": series_reqs,
    }
    if "interpolate_nulls" in spec:
        basic_chart["interpolateNulls"] = bool(spec["interpolate_nulls"])

    return {
        "addChart": {
            "chart": {
                "spec": {
                    "title": spec["title"],
                    "titleTextFormat": {"bold": True, "fontSize": 11},
                    "basicChart": basic_chart,
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": target_sid,
                            "rowIndex": int(spec["anchor_row"]),
                            "columnIndex": int(spec["anchor_col"]),
                        },
                        "offsetXPixels": 8,
                        "offsetYPixels": 6,
                        "widthPixels": int(spec.get("width", 660)),
                        "heightPixels": int(spec.get("height", 380)),
                    }
                },
            }
        }
    }


def _collect_existing_visuals_from_meta(meta: dict[str, Any]) -> tuple[list[int], dict[int, int]]:
    """Extract existing chart IDs and conditional format rule counts per sheetId from full spreadsheet JSON."""
    chart_ids: list[int] = []
    cf_counts: dict[int, int] = {}
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        sid = int(props.get("sheetId", 0))
        for ch in s.get("charts", []):
            if "chartId" in ch:
                chart_ids.append(int(ch["chartId"]))
        cf_list = s.get("conditionalFormats", [])
        if cf_list:
            cf_counts[sid] = len(cf_list)
    return chart_ids, cf_counts


def _build_visual_batch_requests(
    sheet_map: dict[str, dict[str, Any]],
    tab_specs: list[tuple[str, Path, bool, bool]],
    color_scales: list[dict[str, Any]],
    chart_specs: list[dict[str, Any]],
    existing_chart_ids: list[int],
    existing_cf_counts: dict[int, int],
) -> list[dict[str, Any]]:
    """Build cleanup + tab-ordering + conditional color scales + embedded charts batch requests."""
    reqs: list[dict[str, Any]] = []

    # 1. Remove prior charts and conditional format rules so repeated runs never duplicate visuals
    for cid in existing_chart_ids:
        reqs.append({"deleteEmbeddedObject": {"objectId": cid}})
    for sid, cnt in existing_cf_counts.items():
        for _ in range(cnt):
            reqs.append({"deleteConditionalFormatRule": {"sheetId": sid, "index": 0}})

    # 2. Enforce clean tab ordering (Executive_Summary #1, Visual_Dashboard #2, ...)
    for tab_idx, (tab_name, _, _, _) in enumerate(tab_specs):
        if tab_name in sheet_map:
            reqs.append(
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_map[tab_name]["sheet_id"],
                            "index": tab_idx,
                        },
                        "fields": "index",
                    }
                }
            )

    # 3. Conditional formatting color scales (heatmaps)
    for cs in color_scales:
        if cs["tab"] in sheet_map:
            reqs.append(_make_gradient_rule_request(sheet_map[cs["tab"]]["sheet_id"], cs))

    # 4. Clean column widths so labels don't truncate and 35x35 correlation heatmap reads like a compact matrix
    wide_col0_tabs = [
        ("Executive_Summary", 270),
        ("PCA_Overview_and_Loadings", 185),
        ("Rolling_PCA_Loadings_Snapshots", 185),
        ("Factor_Analysis_Results", 210),
        ("Hierarchical_Clustering", 185),
        ("Correlation_and_Distance_Matrix", 175),
    ]
    for tname, px in wide_col0_tabs:
        if tname in sheet_map:
            reqs.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_map[tname]["sheet_id"],
                            "dimension": "COLUMNS",
                            "startIndex": 0,
                            "endIndex": 1,
                        },
                        "properties": {"pixelSize": px},
                        "fields": "pixelSize",
                    }
                }
            )
    if "Correlation_and_Distance_Matrix" in sheet_map:
        corr_sid = sheet_map["Correlation_and_Distance_Matrix"]["sheet_id"]
        # Compact 58px square-ish cells for the 35x35 correlation & distance heatmaps
        reqs.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": corr_sid,
                        "dimension": "COLUMNS",
                        "startIndex": 2,
                        "endIndex": 37,
                    },
                    "properties": {"pixelSize": 58},
                    "fields": "pixelSize",
                }
            }
        )
        # Wrap header text on the two 35x35 subheader rows so column labels remain legible
        for hdr_r in (1, 39):
            reqs.append(
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": corr_sid,
                            "startRowIndex": hdr_r,
                            "endRowIndex": hdr_r + 1,
                            "startColumnIndex": 2,
                            "endColumnIndex": 37,
                        },
                        "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP"}},
                        "fields": "userEnteredFormat.wrapStrategy",
                    }
                }
            )

    # 5. Native embedded Google Sheets charts
    for ch in chart_specs:
        if ch["target_tab"] in sheet_map and ch["domain"]["tab"] in sheet_map:
            reqs.append(_make_chart_request(sheet_map, ch))

    return reqs


def get_sheet_map_cli(dest_id: str) -> dict[str, dict[str, Any]]:
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


def ensure_grid_size_cli(
    dest_id: str, sheet_id: int, cur_rows: int, cur_cols: int, req_rows: int, req_cols: int
) -> None:
    if req_rows <= cur_rows and req_cols <= cur_cols:
        return
    req_body = {
        "requests": [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {
                            "rowCount": max(cur_rows, req_rows),
                            "columnCount": max(cur_cols, req_cols),
                        },
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


def publish_to_sheets_webapp(
    dest_id: str,
    tab_specs: list[tuple[str, Path, bool, bool]],
    color_scales: list[dict[str, Any]],
    chart_specs: list[dict[str, Any]],
) -> None:
    """Publish all tabs, color scales, and charts via Google Apps Script Web App bridge."""
    for tab_idx, (tab_name, csv_path, freeze_col_a, is_multi_section) in enumerate(tab_specs):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.reader(f))
        n_rows = len(all_rows)
        n_cols = max((len(r) for r in all_rows), default=1)
        print(f"  Uploading '{tab_name}' ({n_rows} rows x {n_cols} cols) via Apps Script...")

        call_apps_script(
            {
                "action": "prepare_tab",
                "dest_id": dest_id,
                "tab_name": tab_name,
                "rename_sheet1": tab_idx == 0,
                "rows": n_rows,
                "cols": n_cols,
            }
        )

        chunk_size = 500 if n_cols >= 25 else 1000
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = []
            for i in range(start, end):
                casted = _cast_sheet_row(all_rows[i], is_header=(i == 0))
                if len(casted) < n_cols:
                    casted = casted + [""] * (n_cols - len(casted))
                chunk.append(casted)
            call_apps_script(
                {
                    "action": "write_chunk",
                    "dest_id": dest_id,
                    "tab_name": tab_name,
                    "start_row": start + 1,
                    "values": chunk,
                }
            )
            print(f"    wrote rows {start + 1}..{end} / {n_rows}")

        section_rows: list[int] = []
        subheader_rows: list[int] = []
        if is_multi_section:
            sec0, sub0 = _detect_section_rows(all_rows)
            section_rows = [i + 1 for i in sec0]
            subheader_rows = [i + 1 for i in sub0]

        call_apps_script(
            {
                "action": "format_tab",
                "dest_id": dest_id,
                "tab_name": tab_name,
                "n_cols": n_cols,
                "freeze_col_a": freeze_col_a,
                "section_rows": section_rows,
                "subheader_rows": subheader_rows,
            }
        )

    # Apply color scales and rebuild charts (gracefully skipped if user hasn't redeployed Code.gs yet)
    try:
        print(f"  Applying {len(color_scales)} conditional color scales via Apps Script...")
        call_apps_script(
            {
                "action": "apply_color_scales",
                "dest_id": dest_id,
                "rules": color_scales,
            }
        )
        print(f"  Building {len(chart_specs)} native charts via Apps Script...")
        call_apps_script(
            {
                "action": "rebuild_charts",
                "dest_id": dest_id,
                "charts": chart_specs,
            }
        )
    except RuntimeError as exc:
        if "Unknown action" in str(exc):
            print(
                "  [INFO] Deployed Apps Script bridge does not yet have apply_color_scales/rebuild_charts; "
                "redeploy apps_script/Code.gs to auto-refresh color scales & charts on GitHub Actions runs."
            )
        else:
            raise


def publish_to_sheets_api(
    dest_id: str,
    tab_specs: list[tuple[str, Path, bool, bool]],
    color_scales: list[dict[str, Any]],
    chart_specs: list[dict[str, Any]],
) -> None:
    """Publish all tabs, color scales, and charts directly via Google Sheets v4 API."""
    service = build_sheets_service()

    def get_full_meta() -> tuple[dict[str, dict[str, Any]], list[int], dict[int, int]]:
        meta = (
            service.spreadsheets()
            .get(
                spreadsheetId=dest_id,
                fields="sheets(properties(sheetId,title,gridProperties),charts.chartId,conditionalFormats)",
            )
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
        c_ids, cf_cnts = _collect_existing_visuals_from_meta(meta)
        return out, c_ids, cf_cnts

    sheet_map, _, _ = get_full_meta()
    setup_reqs: list[dict[str, Any]] = []
    first_tab = tab_specs[0][0]
    if first_tab not in sheet_map and "Sheet1" in sheet_map:
        sid = sheet_map["Sheet1"]["sheet_id"]
        setup_reqs.append(
            {
                "updateSheetProperties": {
                    "properties": {"sheetId": sid, "title": first_tab},
                    "fields": "title",
                }
            }
        )
        sheet_map[first_tab] = sheet_map.pop("Sheet1")

    for tab_name, _, _, _ in tab_specs:
        if tab_name not in sheet_map:
            setup_reqs.append({"addSheet": {"properties": {"title": tab_name}}})

    if setup_reqs:
        service.spreadsheets().batchUpdate(spreadsheetId=dest_id, body={"requests": setup_reqs}).execute()
    sheet_map, existing_charts, existing_cfs = get_full_meta()

    format_reqs: list[dict[str, Any]] = []
    for tab_name, csv_path, freeze_col_a, is_multi_section in tab_specs:
        info = sheet_map[tab_name]
        sid = info["sheet_id"]
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.reader(f))
        n_rows = len(all_rows)
        n_cols = max((len(r) for r in all_rows), default=1)
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

        service.spreadsheets().values().clear(spreadsheetId=dest_id, range=f"'{tab_name}'", body={}).execute()

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
        if is_multi_section:
            sec0, sub0 = _detect_section_rows(all_rows)
            for r_i in sec0:
                format_reqs.append(_make_format_request(sid, r_i, r_i + 1, 0, n_cols, "#1F4E78", "#FFFFFF", bold=True))
            for r_i in sub0:
                format_reqs.append(_make_format_request(sid, r_i, r_i + 1, 0, n_cols, "#D9E1F2", "#000000", bold=True))

    visual_reqs = _build_visual_batch_requests(
        sheet_map, tab_specs, color_scales, chart_specs, existing_charts, existing_cfs
    )
    all_reqs = format_reqs + visual_reqs
    if all_reqs:
        print(f"  Applying formatting, {len(color_scales)} color scales & {len(chart_specs)} charts via Sheets API...")
        service.spreadsheets().batchUpdate(spreadsheetId=dest_id, body={"requests": all_reqs}).execute()


def publish_to_sheets_cli(
    dest_id: str,
    tab_specs: list[tuple[str, Path, bool, bool]],
    color_scales: list[dict[str, Any]],
    chart_specs: list[dict[str, Any]],
) -> None:
    """Publish all tabs, conditional color scales, and embedded charts locally via `gsheets` CLI."""
    sheet_map = get_sheet_map_cli(dest_id)
    first_tab = tab_specs[0][0]
    if first_tab not in sheet_map and "Sheet1" in sheet_map:
        sid = sheet_map["Sheet1"]["sheet_id"]
        print(f"  Renaming initial 'Sheet1' (id={sid}) -> '{first_tab}'")
        run_cmd(["gsheets", "mutate", "rename-sheet", dest_id, "--sheet-id", str(sid), "--title", first_tab])
        sheet_map = get_sheet_map_cli(dest_id)

    for tab_name, _, _, _ in tab_specs:
        if tab_name not in sheet_map:
            print(f"  Adding tab '{tab_name}'")
            run_cmd(["gsheets", "mutate", "add-sheet", dest_id, "--title", tab_name])

    sheet_map = get_sheet_map_cli(dest_id)
    # Inspect existing charts & conditional format rules so re-runs replace them cleanly
    info_res = run_cmd(["gsheets", "readonly", "info", dest_id, "--json"])
    full_meta = json.loads(info_res.stdout)
    existing_charts, existing_cfs = _collect_existing_visuals_from_meta(full_meta)

    format_reqs: list[dict[str, Any]] = []

    for tab_name, csv_path, freeze_col_a, is_multi_section in tab_specs:
        info = sheet_map[tab_name]
        sid = info["sheet_id"]
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.reader(f))
        n_rows = len(all_rows)
        n_cols = max((len(r) for r in all_rows), default=1)

        print(f"  Uploading '{tab_name}' ({n_rows} rows x {n_cols} cols) via gsheets CLI...")
        ensure_grid_size_cli(dest_id, sid, info["rows"], info["cols"], n_rows, n_cols)

        # Upload via chunked batch (reliable & preserves exact numeric types)
        chunk_size = 600
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = [_cast_sheet_row(all_rows[i], is_header=(i == 0)) for i in range(start, end)]
            op = [{"op": "write", "range": f"'{tab_name}'!A{start + 1}", "data": chunk}]
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(op, tf)
                tf_path = tf.name
            try:
                run_cmd(["gsheets", "mutate", "batch", dest_id, "-f", tf_path])
            finally:
                os.unlink(tf_path)
            if n_rows > chunk_size:
                print(f"    wrote rows {start + 1}..{end} / {n_rows}")

        # Header & section formatting
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
        if is_multi_section:
            sec0, sub0 = _detect_section_rows(all_rows)
            for r_i in sec0:
                format_reqs.append(_make_format_request(sid, r_i, r_i + 1, 0, n_cols, "#1F4E78", "#FFFFFF", bold=True))
            for r_i in sub0:
                format_reqs.append(_make_format_request(sid, r_i, r_i + 1, 0, n_cols, "#D9E1F2", "#000000", bold=True))

    visual_reqs = _build_visual_batch_requests(
        sheet_map, tab_specs, color_scales, chart_specs, existing_charts, existing_cfs
    )
    all_reqs = format_reqs + visual_reqs
    if all_reqs:
        print(
            f"  Applying formatting, {len(color_scales)} heatmap color scales & "
            f"{len(chart_specs)} native charts across {len(tab_specs)} tabs..."
        )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump({"requests": all_reqs}, tf)
            tf_path = tf.name
        try:
            run_cmd(["gsheets", "mutate", "raw-batch", dest_id, "-f", tf_path])
        finally:
            os.unlink(tf_path)


def publish_to_sheets(
    dest_id: str,
    tab_specs: list[tuple[str, Path, bool, bool]],
    color_scales: list[dict[str, Any]],
    chart_specs: list[dict[str, Any]],
    backend: str = "auto",
) -> None:
    if not dest_id:
        raise ValueError("Destination Google Sheet ID is empty. Pass --dest-id or set EXPLORATION_SHEET_ID.")
    chosen = resolve_sheets_backend(backend)
    print(f"[7/7] Publishing {len(tab_specs)} tabs + visuals to destination Google Sheet ({chosen}): {dest_id}")
    if chosen == "webapp":
        publish_to_sheets_webapp(dest_id, tab_specs, color_scales, chart_specs)
    elif chosen == "api":
        publish_to_sheets_api(dest_id, tab_specs, color_scales, chart_specs)
    else:
        publish_to_sheets_cli(dest_id, tab_specs, color_scales, chart_specs)


# =============================================================================
# 10. MAIN ORCHESTRATOR
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent 3: Multi-Asset PCA, Rolling PCA, Factor Analysis & Hierarchical Clustering Pipeline"
    )
    parser.add_argument(
        "--source-id",
        default=DEFAULT_SOURCE_ID,
        help="Agent 2 Google Sheet ID containing ML_Ready_Dataset (default: DEST_SHEET_ID)",
    )
    parser.add_argument(
        "--dest-id",
        default=DEFAULT_DEST_ID,
        help="Agent 3 Destination Google Sheet ID (default: EXPLORATION_SHEET_ID)",
    )
    parser.add_argument(
        "--ml-csv",
        default=str(BASE_DIR / "data" / "processed" / "ML_Ready_Dataset.csv"),
        help="Local path to Agent 2 ML_Ready_Dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "data" / "exploration"),
        help="Directory for Agent 3 CSV outputs",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "webapp", "api", "gsheets"],
        default="auto",
        help="Google Sheets backend: auto, webapp, api, or gsheets",
    )
    parser.add_argument("--rolling-window", type=int, default=252, help="Primary rolling PCA window (trading days)")
    parser.add_argument("--fast-window", type=int, default=63, help="Fast rolling PCA window (trading days)")
    parser.add_argument("--n-factors", type=int, default=0, help="Override Factor Analysis m (0 = auto via Horn PA)")
    parser.add_argument("--pa-sims", type=int, default=200, help="Monte Carlo replications for Horn Parallel Analysis")
    parser.add_argument("--skip-fetch", action="store_true", help="Read existing local ML_Ready_Dataset.csv")
    parser.add_argument("--skip-upload", action="store_true", help="Save local CSVs only without uploading to Sheets")
    args = parser.parse_args()

    ml_csv = Path(args.ml_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        fetch_agent2_ml_dataset(args.source_id, ml_csv, backend=args.backend)
    elif not ml_csv.exists():
        raise FileNotFoundError(
            f"ML_Ready_Dataset.csv not found at {ml_csv}. Run Agent 2 first or omit --skip-fetch."
        )

    print(f"[2/7] Loading Agent 2 ML_Ready_Dataset from {ml_csv}")
    ml_df = pd.read_csv(ml_csv)
    ml_df["Date"] = ml_df["Date"].astype(str).str.strip()

    # Build primary clean daily-traded universe (35 core daily assets, zero NaNs, zero stepped-rate degeneracy)
    daily_35_cols = [
        c for c in ASSET_META
        if c in ml_df.columns and ASSET_META[c][2] and c not in LATE_INCEPTION_COLS
    ]
    core_df = ml_df.loc[ml_df["is_core_ml_ready"] == 1, ["Date"] + daily_35_cols].dropna().reset_index(drop=True)
    dates = core_df["Date"]
    X_raw = core_df[daily_35_cols].to_numpy(dtype=np.float64)
    Z, _, _ = standardize_matrix(X_raw)
    print(
        f"  Primary universe: {len(daily_35_cols)} daily multi-asset series x {len(core_df):,} trading days "
        f"({dates.iloc[0]} to {dates.iloc[-1]})"
    )

    # 1. Full-Sample PCA
    print(f"[3/7] Running Full-Sample PCA + Horn Parallel Analysis ({args.pa_sims} reps) + Marchenko-Pastur RMT...")
    pca_res = run_full_pca(Z, dates, daily_35_cols, pa_sims=args.pa_sims)
    print(
        f"  PC1={pca_res['ar1']:.2f}%, Top-3={pca_res['ar3']:.2f}%, Top-5={pca_res['ar5']:.2f}% | "
        f"PA95={pca_res['pa_k']}, MP={pca_res['mp_k']}, Kaiser={pca_res['kaiser_k']}, EffRank={pca_res['eff_rank']:.2f}"
    )

    # 2. Strictly Causal Rolling PCA
    print(f"[4/7] Running Strictly Causal Rolling PCA (window={args.rolling_window}d & fast={args.fast_window}d)...")
    roll_res = run_rolling_pca(
        X_raw,
        dates,
        daily_35_cols,
        anchor_evecs=pca_res["evecs"],
        window=args.rolling_window,
        fast_window=args.fast_window,
    )
    dyn_df = roll_res["dynamics_df"]
    print(
        f"  Rolling rows={len(dyn_df):,} ({dyn_df['Date'].iloc[0]} to {dyn_df['Date'].iloc[-1]}) | "
        f"Latest AR5={dyn_df['Absorption_Ratio_Top5'].iloc[-1]:.2f}% ({dyn_df['Systemic_Regime'].iloc[-1]})"
    )

    # 3. Maximum Likelihood Factor Analysis (Varimax + Promax)
    n_factors = args.n_factors if args.n_factors > 0 else max(3, min(10, pca_res["pa_k"]))
    print(f"[5/7] Running Maximum Likelihood Factor Analysis (m={n_factors} factors, Varimax + Promax)...")
    fa_res = run_factor_analysis(Z, dates, daily_35_cols, n_factors=n_factors)
    print(
        f"  KMO={fa_res['adequacy']['kmo_overall']:.4f}, Bartlett chi2={fa_res['adequacy']['bart_chi2']:,.1f} | "
        f"VarExp={fa_res['cum_var_pct'][-1]:.2f}%, RMSR={fa_res['rmsr']:.4f}, TLI={fa_res['tli']:.4f}, Converged={fa_res['converged']}"
    )

    # 4. Hierarchical Agglomerative Clustering & HRP
    print("[6/7] Running Hierarchical Clustering (Ward/Average/Complete/Single, k=2..12) + Optimal Leaf Ordering + HRP...")
    hc_res = run_hierarchical_clustering(X_raw, Z, pca_res["corr"], daily_35_cols)
    print(
        f"  Cophenetic: Avg={hc_res['cophenetics']['average']:.4f}, Ward={hc_res['cophenetics']['ward']:.4f}, "
        f"Comp={hc_res['cophenetics']['complete']:.4f}, Single={hc_res['cophenetics']['single']:.4f} | "
        f"Ward Optimal k={hc_res['opt_k_ward']} (Silhouette={hc_res['ward_sil_mean']:.4f})"
    )

    # 5. Cross-Universe Sensitivities
    sens_rows = run_universe_sensitivities(ml_df, daily_35_cols)

    # Build & save all 10 CSV artifacts (incl. Visual_Dashboard)
    p_exec = out_dir / "Executive_Summary.csv"
    p_dash = out_dir / "Visual_Dashboard.csv"
    p_pca_ov = out_dir / "PCA_Overview_and_Loadings.csv"
    p_pca_sc = out_dir / "PCA_Scores_TimeSeries.csv"
    p_roll_dyn = out_dir / "Rolling_PCA_Dynamics.csv"
    p_roll_snap = out_dir / "Rolling_PCA_Loadings_Snapshots.csv"
    p_fa_res = out_dir / "Factor_Analysis_Results.csv"
    p_fa_sc = out_dir / "Factor_Scores_TimeSeries.csv"
    p_hc = out_dir / "Hierarchical_Clustering.csv"
    p_corr = out_dir / "Correlation_and_Distance_Matrix.csv"

    exec_rows = build_executive_summary(
        args.source_id, args.dest_id, dates, daily_35_cols, pca_res, roll_res, fa_res, hc_res, sens_rows
    )
    dash_rows, dash_meta = build_visual_dashboard_sheet(daily_35_cols, pca_res, roll_res, fa_res, hc_res)
    pca_ov_rows = build_pca_overview_sheet(daily_35_cols, pca_res)
    roll_snap_rows = build_rolling_snapshots_sheet(daily_35_cols, roll_res)
    fa_rows = build_factor_analysis_sheet(daily_35_cols, fa_res)
    hc_rows = build_clustering_sheet(daily_35_cols, pca_res["corr"], hc_res)
    corr_rows = build_corr_dist_sheet(daily_35_cols, pca_res["corr"], hc_res)

    save_rows_csv(exec_rows, p_exec)
    save_rows_csv(dash_rows, p_dash)
    save_rows_csv(pca_ov_rows, p_pca_ov)
    save_clean_csv(pca_res["scores_df"], p_pca_sc, decimals=5)
    save_clean_csv(dyn_df, p_roll_dyn, decimals=4)
    save_rows_csv(roll_snap_rows, p_roll_snap)
    save_rows_csv(fa_rows, p_fa_res)
    save_clean_csv(fa_res["scores_df"], p_fa_sc, decimals=5)
    save_rows_csv(hc_rows, p_hc)
    save_rows_csv(corr_rows, p_corr)

    # (tab_name, csv_path, freeze_col_a, is_multi_section)
    tab_specs: list[tuple[str, Path, bool, bool]] = [
        ("Executive_Summary", p_exec, False, True),
        ("Visual_Dashboard", p_dash, False, True),
        ("PCA_Overview_and_Loadings", p_pca_ov, True, True),
        ("PCA_Scores_TimeSeries", p_pca_sc, True, False),
        ("Rolling_PCA_Dynamics", p_roll_dyn, True, False),
        ("Rolling_PCA_Loadings_Snapshots", p_roll_snap, True, True),
        ("Factor_Analysis_Results", p_fa_res, True, True),
        ("Factor_Scores_TimeSeries", p_fa_sc, True, False),
        ("Hierarchical_Clustering", p_hc, True, True),
        ("Correlation_and_Distance_Matrix", p_corr, True, True),
    ]

    print("\n=== AGENT 3 OUTPUT SUMMARY ===")
    total_cells = 0
    tab_rows_map: dict[str, list[list[str]]] = {}
    for tname, path, _, _ in tab_specs:
        with path.open("r", encoding="utf-8", newline="") as f:
            r_all = list(csv.reader(f))
        tab_rows_map[tname] = r_all
        nr = len(r_all)
        nc = max((len(r) for r in r_all), default=0)
        total_cells += nr * nc
        print(f"  {tname:34s}: {nr:5d} rows x {nc:2d} cols ({nr * nc:7,d} cells)")
    print(f"  Total Cells Across {len(tab_specs)} Tabs        : {total_cells:,} (limit: 10,000,000)")

    color_scales = build_color_scale_specs(tab_rows_map, fa_res, hc_res)
    chart_specs = build_chart_specs(tab_rows_map, dash_meta, fa_res, hc_res)
    print(f"  Planned Heatmap Color Scales      : {len(color_scales)} gradient rules across 6 tabs")
    print(f"  Planned Native Charts             : {len(chart_specs)} charts on Visual_Dashboard")

    if not args.skip_upload:
        publish_to_sheets(args.dest_id, tab_specs, color_scales, chart_specs, backend=args.backend)
        print("\n[DONE] Agent 3 Data Exploration + Color Scales + Charts published to Google Sheets!")
    else:
        print("\n[DONE] Agent 3 local exploration complete (--skip-upload specified).")


if __name__ == "__main__":
    main()
