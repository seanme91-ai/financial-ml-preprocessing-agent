# Financial ML Preprocessing, Cross-Asset Ratio & Data Exploration Pipeline (Agents 1–3)

A reproducible, strictly causal (zero lookahead bias) three-agent Python pipeline that:
1. **Agent 1 (`fetch_daily_data.py`)**: Fetches strict settled Mon–Fri closes across 44 global multi-asset series and appends daily updates to the source Google Sheet.
2. **Agent 2 (`pipeline.py`)**: Audits data validity, fixes verified data corruptions while preserving genuine historical market events, engineers 48 cross-asset ratios/spreads/FX crosses, generates a 148-column ML-ready feature matrix, and publishes 5 formatted tabs to the `Preprocessed Data` Google Sheet.
3. **Agent 3 (`data_exploration.py`)**: Consumes Agent 2's `ML_Ready_Dataset`, runs **Full-Sample PCA**, **Strictly Causal Rolling PCA (252d & 63d)**, **Maximum Likelihood Factor Analysis (Varimax + Promax)**, and **Hierarchical Agglomerative Clustering (+ Hierarchical Risk Parity)**, and publishes 9 formatted analytical tabs to the `Data Exploration - PCA, FA, Clustering` Google Sheet.

## Directory Structure

```text
financial-ml-preprocessing-agent/
├── .github/
│   └── workflows/
│       └── run-pipeline.yml              # Tue-Sat 09:00 SGT (Mon-Fri settled closes) & push runner (Agents 1-3)
├── apps_script/
│   └── Code.gs                           # Google Apps Script Web App bridge (read/append/publish)
├── fetch_daily_data.py                   # Agent 1: Daily 44-asset market data fetcher & sheet appender
├── pipeline.py                           # Agent 2: End-to-end causal ML preprocessing & ratio pipeline
├── data_exploration.py                   # Agent 3: PCA, Rolling PCA, Factor Analysis & Hierarchical Clustering
├── GEMINI.md                             # Antigravity workspace rule for auto-pushing edits
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variable template
├── README.md                             # Usage & ML modeling guide
└── data/
    ├── raw/
    │   └── financial_data_raw.csv        # Raw baseline dataset
    ├── processed/                        # Agent 2 outputs (gitignored; uploaded as Actions artifacts)
    │   ├── Summary_and_Data_Quality.csv
    │   ├── Preprocessed_Data.csv
    │   ├── Cross_Asset_Ratios.csv
    │   ├── ML_Ready_Dataset.csv
    │   └── Feature_Dictionary.csv
    └── exploration/                      # Agent 3 outputs (gitignored; uploaded as Actions artifacts)
        ├── Executive_Summary.csv
        ├── Visual_Dashboard.csv
        ├── PCA_Overview_and_Loadings.csv
        ├── PCA_Scores_TimeSeries.csv
        ├── Rolling_PCA_Dynamics.csv
        ├── Rolling_PCA_Loadings_Snapshots.csv
        ├── Factor_Analysis_Results.csv
        ├── Factor_Scores_TimeSeries.csv
        ├── Hierarchical_Clustering.csv
        └── Correlation_and_Distance_Matrix.csv
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

### Agent 3: Data Exploration (PCA, Rolling PCA, Factor Analysis & Hierarchical Clustering)
```powershell
# Full standalone run (fetches ML_Ready_Dataset from Agent 2 sheet and publishes 9 tabs)
py data_exploration.py

# Fast local re-run from Agent 2's local CSV without uploading
py data_exploration.py --skip-fetch --skip-upload

# Chained upload right after Agent 2
py data_exploration.py --skip-fetch
```

## Running on GitHub Actions

This repository includes [`.github/workflows/run-pipeline.yml`](.github/workflows/run-pipeline.yml), which runs Agents 1 → 2 → 3 automatically on every push to `main`, Tue–Sat at 09:00 SGT, and manually from the **Actions** tab.

- **Without secrets configured**: The workflow runs Agents 2 & 3 in offline verification mode (`--skip-fetch --skip-upload`) using `data/raw/financial_data_raw.csv` and uploads both `processed-ml-datasets` and `data-exploration-outputs` as downloadable workflow artifacts.
- **With Google Sheets sync enabled (Option A — Google Apps Script, no Google Cloud needed):**
  1. Open your destination Google Sheet and click **Extensions → Apps Script**.
  2. Paste [`apps_script/Code.gs`](apps_script/Code.gs), replace `API_TOKEN` with a strong secret token, and click **Deploy → New deployment → Web app** (*Execute as: Me*, *Who has access: Anyone*). *(Includes `apply_color_scales` and `rebuild_charts` so daily runs automatically refresh all 27 heatmaps and 11 charts.)*
  3. Add `APPS_SCRIPT_URL` (the `/exec` URL), `APPS_SCRIPT_TOKEN`, `SOURCE_SHEET_ID`, `DEST_SHEET_ID`, and `EXPLORATION_SHEET_ID` as GitHub repository secrets under **Settings → Secrets and variables → Actions**.
- **With Google Sheets sync enabled (Option B — Google Cloud Service Account):**
  1. In Google Cloud Console, enable the **Google Sheets API**, create a **Service Account**, and download its JSON key.
  2. Share your source Google Sheet (**Viewer**) and destination Google Sheets (**Editor**) with the service account's email address.
  3. Add `GOOGLE_SERVICE_ACCOUNT_JSON`, `SOURCE_SHEET_ID`, `DEST_SHEET_ID`, and `EXPLORATION_SHEET_ID` as GitHub repository secrets.

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

## Agent 3 Destination Google Sheet Tabs (`Data Exploration - PCA, FA, Clustering`)

Analyzes the clean 35 daily-traded multi-asset stationary panel (`is_core_ml_ready == 1`, `4,825` trading days from `2008-03-27` to `2026-11-20`; excluding stepped central bank policy rates to avoid zero-variance rolling windows, with full 5-universe sensitivities reported in `Executive_Summary`):

1. **`Executive_Summary`**: Run metadata, executive findings across all 4 methods, 5-universe sensitivity comparison (raw vs ±5σ winsorized vs 42-asset incl. policy rates vs post-2019 36-asset incl. BNDX vs full 138-feature ML matrix), and tab directory.
2. **`Visual_Dashboard`**: Executive visual gallery with **10 native embedded Google Sheets charts** (PCA Scree Plot with Horn PA & Marchenko-Pastur bounds, PCA Variance Explained & Cumulative Curve, Rolling 252d Absorption Ratios time series, Rolling Stock-Bond Regime Flip time series, Causal AR5 Z-Score vs Effective Rank, Cumulative Standardized PC1–PC3 trends, MLFA Factor Variance Explained, Ward Optimal-k Validation Silhouette & Separation Gap, **Ward Hierarchical Clustering Dendrogram** with optimal cut threshold + Leaf Index Key 1–35, and Hierarchical Risk Parity weights vs Equal Weight).
3. **`PCA_Overview_and_Loadings`**: Complete 35-component scree table (Eigenvalues, % variance, Kaiser > 1, Horn's Parallel Analysis 95th percentile, Marchenko-Pastur RMT upper bound, Effective Participation Ratio, economic labels), color-scale heatmap Correlation Loadings (`PC1`–`PC12` + reconstruction $R^2$), Orthonormal Eigenvector Weights (`PC1`–`PC10`), and Asset % Contributions.
4. **`PCA_Scores_TimeSeries`**: Daily standardized `PC1`–`PC10` factor score time series (`4,825` rows, sign-oriented with `PC1+` = Global Risk-On).
5. **`Rolling_PCA_Dynamics`**: Strictly causal trailing 252-day (1Y) & 63-day (1Q) Rolling PCA (`4,574` rows, zero lookahead) with color-scale heatmaps on `Absorption_Ratio_Top5`, `AR5_Causal_Z252`, and `Stock_Bond_PC1_Product`.
6. **`Rolling_PCA_Loadings_Snapshots`**: Annual regime averages (`2009`–`2026`), color-scale heatmap of year-end & latest PC1 loadings across all 35 assets, and Top 12 non-overlapping historical systemic fragility peaks.
7. **`Factor_Analysis_Results`**: Overall & per-asset KMO, Bartlett's Test of Sphericity, Jöreskog Maximum Likelihood Factor Analysis fit indices (RMSR, RMSEA, TLI, CFI), color-scale heatmaps on Varimax rotated loadings, Promax ($\kappa = 4$ oblique) pattern matrix, and Inter-Factor Correlation Matrix ($\Phi$).
8. **`Factor_Scores_TimeSeries`**: Daily standardized Thomson Regression Factor Scores from the Varimax MLFA model (`4,825` rows × `8` latent factors).
9. **`Hierarchical_Clustering`**: López de Prado metric correlation-distance ($d_{ij} = \sqrt{\frac{1}{2}(1-\rho_{ij})}$) benchmark across 4 linkage methods (`Ward`, `Average`, `Complete`, `Single`) and `k = 2..12`, color-scaled Optimal Leaf-Ordered cluster assignments, **Hierarchical Risk Parity (HRP)** diversification weights, cluster cohesion profiles, and complete 34-step dendrogram merge history.
10. **`Correlation_and_Distance_Matrix`**: Full `35 × 35` Pearson correlation matrix (Blue–White–Red heatmap) and correlation distance matrix (Green–Yellow–Red heatmap) ordered by Ward Optimal Leaf Ordering (quasi-diagonalized).

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
