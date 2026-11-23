# Financial ML Preprocessing & Cross-Asset Ratio Pipeline

A reproducible, strictly causal (zero lookahead bias) Python pipeline that fetches multi-asset financial time-series data from Google Sheets, audits data validity, fixes verified data corruptions while preserving genuine historical market events, engineers 48 cross-asset ratios/spreads/FX crosses, generates a 148-column ML-ready feature matrix, and publishes all formatted outputs to a destination Google Sheet.

## Directory Structure

```text
financial-ml-preprocessing-agent/
├── .github/
│   └── workflows/
│       └── run-pipeline.yml              # GitHub Actions CI/CD runner (on push & manual)
├── GEMINI.md                             # Antigravity workspace rule for auto-pushing edits
├── pipeline.py                           # End-to-end reproducible CLI pipeline
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variable template
├── README.md                             # Usage & ML modeling guide
└── data/
    ├── raw/
    │   └── financial_data_raw.csv        # Raw baseline dataset
    └── processed/                        # Generated locally / uploaded as GitHub Actions artifacts (gitignored)
        ├── Summary_and_Data_Quality.csv
        ├── Preprocessed_Data.csv
        ├── Cross_Asset_Ratios.csv
        ├── ML_Ready_Dataset.csv
        └── Feature_Dictionary.csv
```

## Prerequisites

- Python 3.11+ (`py` on Windows)
- Install dependencies:
  ```powershell
  pip install -r requirements.txt
  ```
- Google Sheets authentication (either):
  - **GitHub Actions / Cloud**: A Google Cloud Service Account JSON key (`GOOGLE_SERVICE_ACCOUNT_JSON` or `GOOGLE_APPLICATION_CREDENTIALS`)
  - **Local Antigravity**: Authenticated `gsheets` CLI on `PATH`

## How to Run Locally

Copy `.env.example` to `.env` and fill in your `SOURCE_SHEET_ID` and `DEST_SHEET_ID` (never commit `.env`).

### Full end-to-end run (fetch from source sheet, process, and publish to destination sheet)
```powershell
py pipeline.py
```
Or pass IDs explicitly:
```powershell
py pipeline.py --source-id <SOURCE_SHEET_ID> --dest-id <DEST_SHEET_ID>
```

### Local re-run without fetching or uploading
```powershell
py pipeline.py --skip-fetch --skip-upload
```

### Upload only from existing local raw export
```powershell
py pipeline.py --skip-fetch
```

## Running on GitHub Actions

This repository includes [`.github/workflows/run-pipeline.yml`](.github/workflows/run-pipeline.yml), which runs automatically on every push to `main` (and manually from the **Actions** tab).

- **Without secrets configured**: The workflow runs in offline verification mode (`--skip-fetch --skip-upload`) using `data/raw/financial_data_raw.csv` and uploads all 5 processed CSVs as a downloadable workflow artifact (`processed-ml-datasets`).
- **With Google Sheets sync enabled**:
  1. In Google Cloud Console, enable the **Google Sheets API**, create a **Service Account**, and download its JSON key.
  2. Share your source Google Sheet (**Viewer**) and destination Google Sheet (**Editor**) with the service account's email address (`...@...iam.gserviceaccount.com`).
  3. In your GitHub repo go to **Settings → Secrets and variables → Actions** and add three repository secrets:
     - `GOOGLE_SERVICE_ACCOUNT_JSON`: full contents of the downloaded JSON key
     - `SOURCE_SHEET_ID`: your source Google Sheet ID
     - `DEST_SHEET_ID`: your destination Google Sheet ID

## Destination Google Sheet Tabs

1. **`Summary_and_Data_Quality`**: Executive summary, full `Anomaly_Log` (every data error fixed + every genuine historical anomaly preserved and flagged), and a 44-asset per-column audit table.
2. **`Preprocessed_Data`** (`5,217` weekday rows × `45` cols): Chronologically sorted, weekday-only (`2006-11-23` to `2026-11-20`) cleaned asset levels with zero pre-inception backfill and bounded causal holiday forward-fill (`limit=6` trading days for traded markets; unlimited post-inception causal ffill for stepped central bank policy rates).
3. **`Cross_Asset_Ratios`** (`5,217` rows × `49` cols): 48 economically grounded cross-asset ratios and macro spreads across 6 themes (Commodities/Inflation, Equities vs Hard Assets, Local-Currency Equity Relative Value, FX-Adjusted Common-USD Equity Ratios, Risk-On/Risk-Off Bonds & Credit, and Macro Yield Curves / FX Crosses). All price ratios are guarded against `<= 0` inputs.
4. **`ML_Ready_Dataset`** (`5,217` rows × `148` cols): Strictly causal stationary features designed for tabular ML (LightGBM/XGBoost), linear models, regime clustering, and sequence models:
   - Calendar features (`day_of_week`, `month`, `quarter`, `is_month_end`)
   - 1-day log returns (`ret1d_*`, 31 price/FX series) & 1-day first differences (`chg1d_*`, 13 yields/rates)
   - 1-day stationary returns/diffs for 22 core cross-asset ratios & spreads
   - Multi-horizon causal momentum (`mom21d_*`, `mom63d_*`) for 15 macro ratios/spreads
   - Strictly causal rolling Z-scores (`z63_*`, `z252_*`) for 17 macro ratios/spreads
   - Causal 21-day annualized realized volatility (`vol21d_*`) for 10 benchmarks
   - Completeness flags: `valid_feature_pct`, `is_core_ml_ready`, `is_full_ml_ready`
5. **`Feature_Dictionary`**: Full documentation of every column across all three time-series tabs (Tab, Column_Name, Category, Formula_or_Source, Economic_Meaning, Transformation, Inception_Date, ML_Notes).

## ML Usage Guidelines (Zero Lookahead)

- **Train/Validation/Test Splits**: Always split chronologically (or use purged walk-forward cross-validation). Never random-shuffle time-series rows.
- **Selecting Clean Training Sets**:
  - Longest ~18-year multi-asset history (`4,825` rows, starting `2008-03-27`): filter `is_core_ml_ready == 1` and drop `ret1d_bndx` and `chg1d_cn_lpr5y`.
  - Full 140-feature panel (`1,889` rows, starting `2019-08-21`): filter `is_full_ml_ready == 1`.
- **Target Alignment**: When predicting forward returns `y_{t+h}`, align features at date `t` with returns computed strictly over `(t, t+h]`.
- **Modeling Caveats**:
  - `SG SORA (pct)` and `SG 10Y Yield (pct)` are monthly-frequency step-filled series in the source data (changes occur ~10–12 times/year).
  - Front-month futures series (`Natural Gas`, `Corn`, `Soybeans`) contain unadjusted expiry-roll jumps; tree models or winsorized returns are recommended.
  - `WTI Crude` closed at `-37.63` on `2020-04-20`; WTI price ratios and log-returns are cleanly masked (`NaN`/blank) on `2020-04-20/21` while `Brent_WTI_Spread` remains valid.
