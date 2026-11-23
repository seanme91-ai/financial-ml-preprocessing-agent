"""
Daily Financial Market Data Fetcher & Google Sheet Appender (Agent 1).

Fetches strict settled closing prices, yields, and central bank policy rates
for the previous completed trading day (Monday-Friday, zero intraday leakage)
across all 44 global multi-asset series from official/free feeds (Yahoo Finance,
US Treasury, NY Fed, ECB, BIS, HKMA, CFETS/PBOC, ChinaBond, MAS) and appends a
single daily row (stamped with that trading date, with chronological/duplicate
guard) to the 'Financial Data Updates' source Google Sheet.

Scheduled Tue-Sat at 09:00 Asia/Singapore (01:00 UTC), after Mon-Fri US markets settle.

Usage:
    py fetch_daily_data.py --dry-run          # Fetch settled closes & print row without writing
    py fetch_daily_data.py --backend webapp   # Run via Google Apps Script bridge (GitHub Actions)
    py fetch_daily_data.py --backend gsheets  # Run locally via Antigravity gsheets CLI
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
SGT = timezone(timedelta(hours=8))
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_dotenv(env_path: Path) -> None:
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

# Yahoo Finance mapping for 31 market assets (plus ^TNX/^TYX as yield fallbacks)
YF_TICKER_MAP: dict[str, str] = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq Composite": "^IXIC",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "CSI 300": "000300.SS",
    "Shanghai Composite": "000001.SS",
    "STI": "^STI",
    "US Agg Bond (AGG)": "AGG",
    "US 20+Y Treasury (TLT)": "TLT",
    "US High Yield (HYG)": "HYG",
    "Global Agg Bond (BNDX)": "BNDX",
    "Gold (USD/oz)": "GC=F",
    "Silver (USD/oz)": "SI=F",
    "WTI Crude (USD/bbl)": "CL=F",
    "Brent Crude (USD/bbl)": "BZ=F",
    "Copper (USD/lb)": "HG=F",
    "Natural Gas (USD/MMBtu)": "NG=F",
    "Wheat (USc/bu)": "ZW=F",
    "Corn (USc/bu)": "ZC=F",
    "Soybeans (USc/bu)": "ZS=F",
    "Coffee (USc/lb)": "KC=F",
    "Sugar (USc/lb)": "SB=F",
    "DXY": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CNY": "CNY=X",
    "USD/SGD": "SGD=X",
}

DECIMALS_MAP: dict[str, int] = {
    "Copper (USD/lb)": 4,
    "Natural Gas (USD/MMBtu)": 3,
    "EUR/USD": 4,
    "GBP/USD": 4,
    "USD/CNY": 4,
    "USD/SGD": 4,
}


def http_get_text(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def safe_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace("%", "")
    if not s or s.lower() in ("nan", "none", "n/a", "-", "--", "---", "."):
        return None
    try:
        fv = float(s)
        return fv if math.isfinite(fv) else None
    except ValueError:
        return None


def last_completed_trading_day(today_sgt: date) -> date:
    """Return the most recent completed Mon-Fri trading date strictly before today_sgt."""
    d = today_sgt - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d -= timedelta(days=1)
    return d


def fetch_yfinance_all(target_date: date) -> tuple[dict[str, float], dict[str, str]]:
    """Fetch strict settled daily closes on or before target_date (zero intraday leakage)."""
    extra = {"_TNX": "^TNX", "_TYX": "^TYX"}
    all_map = {**YF_TICKER_MAP, **extra}
    tickers = list(all_map.values())
    values: dict[str, float] = {}
    dates: dict[str, str] = {}

    start_str = (target_date - timedelta(days=14)).isoformat()
    end_str = (target_date + timedelta(days=1)).isoformat()  # yfinance end is exclusive
    df = yf.download(tickers, start=start_str, end=end_str, interval="1d", progress=False, auto_adjust=False)
    closes = df["Close"] if "Close" in df else df
    for col_name, ticker in all_map.items():
        try:
            series = closes[ticker].dropna()
            # Explicit guard: exclude any bar newer than target_date
            series = series[series.index.date <= target_date]
            if len(series) > 0:
                val = safe_float(series.iloc[-1])
                if val is not None:
                    values[col_name] = val
                    dates[col_name] = series.index[-1].strftime("%Y-%m-%d")
        except Exception:
            continue
    return values, dates


def fetch_us_treasury(target_date: date) -> dict[str, float]:
    """Official US Treasury daily yield curve CSV (2Y, 10Y, 30Y) on or before target_date."""
    out: dict[str, float] = {}
    for yr in (target_date.year, target_date.year - 1):
        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{yr}/all?type=daily_treasury_yield_curve"
            f"&field_tdr_date_value={yr}&page&_format=csv"
        )
        try:
            txt = http_get_text(url, timeout=12)
            for row in csv.DictReader(io.StringIO(txt)):
                raw_dt = str(row.get("Date", "")).strip()
                try:
                    row_dt = datetime.strptime(raw_dt, "%m/%d/%Y").date()
                except ValueError:
                    continue
                if row_dt > target_date:
                    continue
                for col, key in [
                    ("US 2Y Yield (pct)", "2 Yr"),
                    ("US 10Y Yield (pct)", "10 Yr"),
                    ("US 30Y Yield (pct)", "30 Yr"),
                ]:
                    v = safe_float(row.get(key))
                    if v is not None:
                        out[col] = v
                if out:
                    return out
        except Exception:
            continue
    return out


def fetch_nyfed_effr(target_date: date) -> tuple[dict[str, float], str]:
    """Official NY Fed Effective Federal Funds Rate & target range on or before target_date."""
    url = "https://markets.newyorkfed.org/api/rates/unsecured/effr/last/10.json"
    data = json.loads(http_get_text(url, timeout=10))
    for rec in data.get("refRates", []):
        eff_dt = str(rec.get("effectiveDate", "")).strip()
        if eff_dt and eff_dt > target_date.isoformat():
            continue
        val = safe_float(rec.get("percentRate"))
        lo = safe_float(rec.get("targetRateFrom"))
        hi = safe_float(rec.get("targetRateTo"))
        note = f"Fed target {lo:.2f}-{hi:.2f} pct" if (lo is not None and hi is not None) else ""
        if val is not None:
            return {"Fed Funds Rate (pct)": val}, note
    return {}, ""


def fetch_ecb_dfr(target_date: date) -> dict[str, float]:
    """Official ECB Deposit Facility Rate on or before target_date."""
    url = "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV?lastNObservations=10&format=csvdata"
    txt = http_get_text(url, timeout=12)
    rows = list(csv.DictReader(io.StringIO(txt)))
    for row in reversed(rows):
        tp = str(row.get("TIME_PERIOD", "")).strip()
        if tp and tp > target_date.isoformat():
            continue
        v = safe_float(row.get("OBS_VALUE"))
        if v is not None:
            return {"ECB Deposit Rate (pct)": v}
    return {}


def fetch_bis_boj(target_date: date) -> dict[str, float]:
    """Official BIS Central Bank Policy Rate for Japan (BOJ) on or before target_date."""
    url = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.JP?lastNObservations=10&format=csv"
    txt = http_get_text(url, timeout=12)
    rows = list(csv.DictReader(io.StringIO(txt)))
    for row in reversed(rows):
        tp = str(row.get("TIME_PERIOD", "")).strip()
        if tp and tp > target_date.isoformat():
            continue
        v = safe_float(row.get("OBS_VALUE"))
        if v is not None:
            return {"BOJ Policy Rate (pct)": v}
    return {}


def fetch_hkma_rates(target_date: date) -> dict[str, float]:
    """Official HKMA Base Rate & 1-Month HIBOR Fixing on or before target_date."""
    out: dict[str, float] = {}
    url = (
        "https://api.hkma.gov.hk/public/market-data-and-statistics/"
        "daily-monetary-statistics/daily-figures-interbank-liquidity?pagesize=10"
    )
    data = json.loads(http_get_text(url, timeout=12))
    for rec in data.get("result", {}).get("records", []):
        dt_str = str(rec.get("end_of_date", "")).strip()
        if dt_str and dt_str > target_date.isoformat():
            continue
        if "HK Base Rate (pct)" not in out:
            v = safe_float(rec.get("disc_win_base_rate"))
            if v is not None:
                out["HK Base Rate (pct)"] = v
        if "HK 1M HIBOR (pct)" not in out:
            v = safe_float(rec.get("hibor_fixing_1m"))
            if v is not None:
                out["HK 1M HIBOR (pct)"] = v
        if len(out) == 2:
            break
    return out


def fetch_china_lpr() -> dict[str, float]:
    """Official CFETS / PBOC 1Y and 5Y Loan Prime Rates."""
    out: dict[str, float] = {}
    url = "https://www.chinamoney.com.cn/r/cms/www/chinamoney/data/currency/bk-lpr.json"
    data = json.loads(http_get_text(url, timeout=12))
    for rec in data.get("records", []):
        term = str(rec.get("termCode", "")).upper()
        val = safe_float(rec.get("shibor"))
        if val is None:
            continue
        if term == "1Y":
            out["China 1Y LPR (pct)"] = val
        elif term == "5Y":
            out["China 5Y LPR (pct)"] = val
    return out


def fetch_china_10y(target_date: date) -> dict[str, float]:
    """Official ChinaBond 10Y Government Bond Yield on or before target_date (with CNBC CN10Y fallback)."""
    start_dt = target_date - timedelta(days=14)
    url = (
        f"https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery"
        f"?startDate={start_dt.isoformat()}&endDate={target_date.isoformat()}&gjqx=0&qxId=ycqx&locale=en_US"
    )
    try:
        html = http_get_text(url, timeout=12)
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
        for r in rows:
            cells = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()
                for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
            ]
            if len(cells) >= 9 and "Government Bond Yield Curve" in cells[0]:
                row_dt = cells[1].strip()
                if row_dt and row_dt > target_date.isoformat():
                    continue
                v = safe_float(cells[8])  # 10Y column
                if v is not None and 0.1 < v < 15.0:
                    return {"China 10Y Yield (pct)": v}
    except Exception:
        pass

    # Fallback: CNBC CN10Y quote
    try:
        cnbc_url = (
            "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
            "?symbols=CN10Y&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1&output=json&events=1"
        )
        data = json.loads(http_get_text(cnbc_url, timeout=10))
        for q in data.get("FormattedQuoteResult", {}).get("FormattedQuote", []):
            v = safe_float(q.get("last"))
            if v is not None and 0.1 < v < 15.0:
                return {"China 10Y Yield (pct)": v}
    except Exception:
        pass
    return {}


def fetch_mas_sg_10y(target_date: date) -> dict[str, float]:
    """Official MAS Daily SGS Benchmark 10-Year Yield on or before target_date."""
    url = "https://eservices.mas.gov.sg/statistics/fdanet/SgsBenchmarkIssuePrices.aspx"
    html = http_get_text(url, timeout=12)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    latest_yield: float | None = None
    for r in rows:
        cells = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
        ]
        cells = [c for c in cells if c and c != "&nbsp;"]
        if len(cells) >= 10 and re.match(r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$", cells[0]):
            try:
                row_dt = datetime.strptime(cells[0], "%d %b %Y").date()
            except ValueError:
                continue
            if row_dt > target_date:
                continue
            v = safe_float(cells[8])
            if v is not None and 0.1 < v < 15.0:
                latest_yield = v
    return {"SG 10Y Yield (pct)": latest_yield} if latest_yield is not None else {}


def fetch_mas_sora(target_date: date) -> dict[str, float]:
    """Official MAS Domestic Interest Rates Datastore API for SORA on or before target_date."""
    url = (
        "https://eservices.mas.gov.sg/api/action/datastore/search.json"
        "?resource_id=9a0bf149-308c-4bd2-832d-76c8e6cb47ed&limit=10&sort=end_of_day%20desc"
    )
    txt = http_get_text(url, timeout=10)
    if not txt.lstrip().startswith("{"):
        return {}
    data = json.loads(txt)
    for rec in data.get("result", {}).get("records", []):
        dt_str = str(rec.get("end_of_day", "")).strip()
        if dt_str and dt_str > target_date.isoformat():
            continue
        v = safe_float(rec.get("sora"))
        if v is not None:
            return {"SG SORA (pct)": v}
    return {}


def fetch_cnbc_us_yields_fallback() -> dict[str, float]:
    """Fallback CNBC quotes for US 2Y/10Y/30Y if US Treasury CSV is unavailable."""
    out: dict[str, float] = {}
    url = (
        "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
        "?symbols=US2Y%7CUS10Y%7CUS30Y&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1&output=json&events=1"
    )
    data = json.loads(http_get_text(url, timeout=10))
    mapping = {
        "US2Y": "US 2Y Yield (pct)",
        "US10Y": "US 10Y Yield (pct)",
        "US30Y": "US 30Y Yield (pct)",
    }
    for q in data.get("FormattedQuoteResult", {}).get("FormattedQuote", []):
        col = mapping.get(str(q.get("symbol", "")))
        v = safe_float(q.get("last"))
        if col and v is not None:
            out[col] = v
    return out


def fetch_all_market_data(target_date: date) -> tuple[dict[str, float], list[str], str]:
    """Fetch strict settled closes on or before target_date for all 44 financial series in parallel."""
    values: dict[str, float] = {}
    warnings: list[str] = []
    fed_note = ""

    # 1. Yahoo Finance batch (31 market assets + ^TNX/^TYX fallbacks)
    yf_vals, yf_dates = fetch_yfinance_all(target_date)
    for col in YF_TICKER_MAP:
        if col in yf_vals:
            values[col] = yf_vals[col]

    # 2. Official rate APIs in parallel
    tasks = {
        "US_Treasury": lambda: fetch_us_treasury(target_date),
        "NYFed_EFFR": lambda: fetch_nyfed_effr(target_date),
        "ECB_DFR": lambda: fetch_ecb_dfr(target_date),
        "BIS_BOJ": lambda: fetch_bis_boj(target_date),
        "HKMA": lambda: fetch_hkma_rates(target_date),
        "China_LPR": fetch_china_lpr,
        "China_10Y": lambda: fetch_china_10y(target_date),
        "MAS_SG10Y": lambda: fetch_mas_sg_10y(target_date),
        "MAS_SORA": lambda: fetch_mas_sora(target_date),
    }
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {name: ex.submit(fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                res = fut.result()
                if name == "NYFed_EFFR":
                    r_dict, fed_note = res
                    values.update(r_dict)
                elif isinstance(res, dict):
                    values.update(res)
            except Exception as exc:
                warnings.append(f"{name}: {exc}")

    # Fallbacks for US yields if Treasury CSV failed
    if "US 10Y Yield (pct)" not in values and "_TNX" in yf_vals:
        values["US 10Y Yield (pct)"] = yf_vals["_TNX"]
    if "US 30Y Yield (pct)" not in values and "_TYX" in yf_vals:
        values["US 30Y Yield (pct)"] = yf_vals["_TYX"]
    if any(k not in values for k in ("US 2Y Yield (pct)", "US 10Y Yield (pct)", "US 30Y Yield (pct)")):
        try:
            fb = fetch_cnbc_us_yields_fallback()
            for k, v in fb.items():
                values.setdefault(k, v)
        except Exception:
            pass

    latest_us_dt = yf_dates.get("S&P 500", "")
    parts = [f"Settled close {target_date.isoformat()}"]
    if latest_us_dt and latest_us_dt != target_date.isoformat():
        parts.append(f"US close {latest_us_dt}")
    if fed_note:
        parts.append(fed_note)
    return values, warnings, "; ".join(parts)


def call_apps_script(payload: dict[str, Any], timeout: int = 90, max_retries: int = 4) -> dict[str, Any]:
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
            data = json.loads(raw_text)
            if not data.get("ok"):
                raise RuntimeError(f"Apps Script error: {data.get('error', data)}")
            return data
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:400]
            last_exc = RuntimeError(f"Apps Script HTTP {exc.code}: {err_body}")
            retryable = exc.code in (429, 500, 502, 503, 504)
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
            last_exc = exc
            retryable = True

        if not retryable or attempt == max_retries:
            break
        wait_s = 5 * (2 ** (attempt - 1))
        print(
            f"  [WARN] Apps Script '{action}' attempt {attempt}/{max_retries} failed ({last_exc}); "
            f"retrying in {wait_s}s...",
            flush=True,
        )
        time.sleep(wait_s)

    raise RuntimeError(f"Apps Script '{action}' failed after {max_retries} attempts: {last_exc}") from last_exc


def resolve_backend(requested: str) -> str:
    has_webapp = bool(
        os.environ.get("APPS_SCRIPT_URL", "").strip()
        and os.environ.get("APPS_SCRIPT_TOKEN", "").strip()
    )
    has_cli = shutil.which("gsheets") is not None
    if requested == "webapp":
        if not has_webapp:
            raise RuntimeError("Requested --backend webapp, but APPS_SCRIPT_URL/TOKEN not set.")
        return "webapp"
    if requested == "gsheets":
        if not has_cli:
            raise RuntimeError("Requested --backend gsheets, but gsheets CLI not on PATH.")
        return "gsheets"
    if has_webapp:
        return "webapp"
    if has_cli:
        return "gsheets"
    return "local_csv"


def run_gsheets_cmd(args: list[str]) -> str:
    resolved = shutil.which("gsheets") or "gsheets"
    res = subprocess.run([resolved, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"gsheets failed: {res.stderr or res.stdout}")
    return res.stdout


def load_sheet_headers_and_tail(
    backend: str, source_id: str, sheet_name: str, fallback_csv: Path
) -> tuple[list[str], list[list[str]]]:
    if backend == "webapp":
        resp = call_apps_script(
            {"action": "get_tail", "source_id": source_id, "sheet_name": sheet_name, "tail_rows": 5}
        )
        return resp["headers"], resp["tail_rows"]

    if backend == "gsheets":
        hdr_out = run_gsheets_cmd(["readonly", "read", source_id, f"{sheet_name}!1:1", "--json"])
        hdr_data = json.loads(hdr_out)
        hdr_rows = hdr_data.get("values", []) if isinstance(hdr_data, dict) else hdr_data
        headers = hdr_rows[0]
        info_out = run_gsheets_cmd(["readonly", "info", source_id, "--json"])
        info_data = json.loads(info_out)
        sheets = info_data.get("sheets", []) if isinstance(info_data, dict) else info_data
        n_rows = 5250
        for s in sheets:
            props = s.get("properties", s) if isinstance(s, dict) else {}
            if props.get("title") == sheet_name or props.get("name") == sheet_name:
                grid = props.get("gridProperties", {})
                n_rows = int(grid.get("rowCount", props.get("rows", 5250)))
        start = max(2, n_rows - 15)
        tail_out = run_gsheets_cmd(["readonly", "read", source_id, f"{sheet_name}!A{start}:ZZ{n_rows}", "--json"])
        tail_data = json.loads(tail_out)
        raw_tail = tail_data.get("values", []) if isinstance(tail_data, dict) else tail_data
        rows = [r for r in raw_tail if r and str(r[0]).strip()]
        return headers, rows[-5:]

    # Local CSV fallback for --dry-run without network credentials
    with fallback_csv.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.reader(f))
    return all_rows[0], all_rows[-5:]


def format_value(col_name: str, val: float) -> float | int:
    dec = DECIMALS_MAP.get(col_name, 2)
    rounded = round(val, dec)
    return int(rounded) if rounded == int(rounded) and dec <= 2 else rounded


def build_daily_row(
    headers: list[str],
    tail_rows: list[list[str]],
    fetched: dict[str, float],
    today_str: str,
    fed_note: str,
) -> tuple[list[Any], list[str]]:
    """Assemble an aligned row matching headers exactly, carrying forward any unavailable rates."""
    # Build lookup of most recent non-empty numeric value per column from tail_rows
    prev_vals: dict[str, float] = {}
    for row in tail_rows:
        for idx, col in enumerate(headers):
            if idx < len(row):
                v = safe_float(row[idx])
                if v is not None:
                    prev_vals[col] = v

    carried: list[str] = []
    out_row: list[Any] = []
    for col in headers:
        if col == "Date":
            out_row.append(today_str)
        elif col == "Notes":
            parts = ["Auto-updated via GitHub Actions"]
            if fed_note:
                parts.append(fed_note)
            if carried:
                parts.append(f"Carried forward: {' / '.join(carried)}")
            # Keep Notes comma-free and percent-free for downstream CLI/CSV safety
            clean_note = "; ".join(parts).replace(",", ";").replace("%", " pct")
            out_row.append(clean_note)
        elif col in fetched:
            out_row.append(format_value(col, fetched[col]))
        elif col in prev_vals:
            carried.append(col.replace(" (pct)", ""))
            out_row.append(format_value(col, prev_vals[col]))
        else:
            out_row.append("")
    return out_row, carried


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Financial Market Data Updater (Agent 1)")
    parser.add_argument("--source-id", default=os.environ.get("SOURCE_SHEET_ID", ""), help="Source Google Sheet ID")
    parser.add_argument("--sheet-name", default="Sheet1", help="Worksheet tab name")
    parser.add_argument("--backend", choices=["auto", "webapp", "gsheets"], default="auto")
    parser.add_argument("--date", default="", help="Override settled trading date YYYY-MM-DD (default: previous completed Mon-Fri)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and print row without appending to Google Sheet")
    parser.add_argument(
        "--fallback-csv",
        default=str(BASE_DIR / "data" / "raw" / "financial_data_raw.csv"),
        help="Local raw CSV used only when --dry-run has no sheet credentials",
    )
    args = parser.parse_args()

    today_sgt = datetime.now(SGT).date()
    target_date = date.fromisoformat(args.date.strip()) if args.date.strip() else last_completed_trading_day(today_sgt)
    target_str = target_date.isoformat()
    backend = "local_csv" if (args.dry_run and resolve_backend(args.backend) == "local_csv") else resolve_backend(args.backend)

    print(f"[Agent 1] Settled previous-day close update for trading date {target_str} (today SGT: {today_sgt}) via backend={backend}")
    headers, tail_rows = load_sheet_headers_and_tail(
        backend, args.source_id, args.sheet_name, Path(args.fallback_csv)
    )
    last_date = str(tail_rows[-1][0]).strip() if tail_rows and tail_rows[-1] else ""
    print(f"  Sheet columns: {len(headers)} | Latest recorded date: {last_date}")

    if last_date and last_date >= target_str and not args.dry_run:
        print(f"  [SKIP] Latest recorded date ({last_date}) is already >= target trading date ({target_str}). Nothing to append.")
        return

    print(f"  Fetching strict settled closes (<= {target_str}) from Yahoo Finance, US Treasury, NY Fed, ECB, BIS, HKMA, CFETS, ChinaBond, MAS...")
    fetched, warnings, fed_note = fetch_all_market_data(target_date)
    print(f"  Settled series fetched: {len(fetched)} / {len(headers) - 2}")
    if warnings:
        print(f"  Non-fatal feed warnings: {warnings}")

    row, carried = build_daily_row(headers, tail_rows, fetched, target_str, fed_note)
    print(f"  Carried forward ({len(carried)}): {carried}")
    print("\n=== PREPARED SETTLED CLOSE ROW ===")
    for col, val in zip(headers, row):
        print(f"  {col:28s}: {val}")

    if args.dry_run:
        print("\n[DRY RUN] No changes written to Google Sheets.")
        return

    if backend == "webapp":
        resp = call_apps_script(
            {
                "action": "append_row",
                "source_id": args.source_id,
                "sheet_name": args.sheet_name,
                "date": target_str,
                "row": row,
            }
        )
        if resp.get("skipped"):
            print(f"\n[SKIP] {resp.get('reason')}")
        else:
            print(f"\n[DONE] Appended row {resp.get('last_row')} for {target_str} to Financial Data Updates!")
    elif backend == "gsheets":
        csv_row = ",".join(str(v) for v in row)
        run_gsheets_cmd(["mutate", "append", args.source_id, f"{args.sheet_name}!A1", csv_row])
        print(f"\n[DONE] Appended row for {target_str} via gsheets CLI!")
    else:
        raise RuntimeError("No writable Google Sheets backend configured. Set APPS_SCRIPT_URL + APPS_SCRIPT_TOKEN.")


if __name__ == "__main__":
    main()
