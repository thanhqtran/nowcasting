#!/opt/homebrew/anaconda3/bin/python3
"""
Nowcast Vietnam GDP YOY Growth for 2026Q1
Using a MIDAS model with monthly PPI (Production Index) and PMI
(Purchasing Managers' Index) as high-frequency indicators.

Seasonal adjustment: STL (Seasonal-Trend decomposition via Loess) is applied
to the raw quarterly GDP level before computing YOY growth, removing recurring
intra-year patterns (e.g. the Q4 surge and Q1 dip common in Vietnamese GDP).

Model: MIDAS-ADL with Exponential Almon weights, rolling-window evaluation
Nowcast date: 2026Q1 (2026-03-31)
Available monthly data: through 2026M02; March 2026 imputed as average of
Jan–Feb 2026 (standard "ragged edge" fill for nowcasting).
"""

import sys
import os
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calendar import monthrange

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MIDAS_DIR    = os.path.join(PROJECT_ROOT, 'midas')
DATA_DIR     = os.path.join(SCRIPT_DIR, 'mydata')
OUT_DIR      = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# The midas package uses bare imports (e.g. `from weights import …`),
# so both the project root (for `midas.*`) and the midas dir (for internal
# cross-module references) must be on sys.path.
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MIDAS_DIR)

from midas.mix import mix_freq2                           # noqa: E402
from midas.adl import estimate2, forecast2, midas_adl2   # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_START = datetime.datetime(2014, 3, 31)   # start of rolling evaluation
TRAIN_END   = datetime.datetime(2020, 12, 31)  # end of initial training window
FINAL_END   = datetime.datetime(2025, 12, 31)  # last quarterly GDP observation
CI_Z        = 1.96                             # z-score for 95% CI
POLY        = 'expalmon'                       # Exponential Almon weights
XLAG        = '3m'                             # 3 monthly lags per quarter
YLAG        = 1                                # 1 autoregressive GDP lag
HORIZON     = 0                                # all 3 months available (incl. imputed)

# ── Seasonal-adjustment mode ───────────────────────────────────────────────────
# SA_MODE = 0 : compute seasonal adjustment via STL inside the script
# SA_MODE = 1 : import the pre-seasonally-adjusted GDP column ('GDP') from the
#               data file (data_vn_2025Q4.csv) and skip the STL step
SA_MODE = 1

# ── Date helpers ───────────────────────────────────────────────────────────────

def parse_quarterly(s: str) -> pd.Timestamp:
    """'2010Q1' → Timestamp at end of that quarter."""
    year, q = int(s[:4]), int(s[5])
    month = q * 3
    return pd.Timestamp(year, month, monthrange(year, month)[1])


def parse_monthly(s: str) -> pd.Timestamp:
    """'2012M01' → Timestamp at last day of that month."""
    year, month = int(s[:4]), int(s[5:])
    return pd.Timestamp(year, month, monthrange(year, month)[1])


# ── Load raw data ──────────────────────────────────────────────────────────────
print("Loading data …")

# GDP (quarterly) – first row is header, second is the comment row
gdp_raw = pd.read_csv(
    os.path.join(DATA_DIR, 'data_vn_2025Q4.csv'),
    index_col=0, encoding='utf-8-sig',
)
gdp_raw = gdp_raw[gdp_raw.index != 'comment']          # drop description row
gdp_raw.index = [parse_quarterly(s) for s in gdp_raw.index]
gdp_raw['GDP_U'] = pd.to_numeric(gdp_raw['GDP_U'], errors='coerce')
gdp_raw['GDP']   = pd.to_numeric(gdp_raw['GDP'],   errors='coerce')   # pre-SA column

# PPI (monthly)
ppi_raw = pd.read_csv(
    os.path.join(DATA_DIR, 'data_monthly_ppi.csv'),
    usecols=[0, 1], encoding='utf-8-sig',
)
ppi_raw.columns = ['DATE', 'PPI']
ppi_raw['DATE']  = ppi_raw['DATE'].apply(parse_monthly)
ppi_raw['PPI']   = pd.to_numeric(ppi_raw['PPI'], errors='coerce')
ppi_raw = ppi_raw.set_index('DATE').sort_index()

# PMI (monthly)
# PMI is defined as 50 × (Production_t / Production_{t-12}):
#   PMI = 50  →  production flat YOY
#   PMI > 50  →  production grew vs same month last year
#   PMI < 50  →  production fell vs same month last year
# Because it is already a YOY production ratio, it is used as-is in the
# MIDAS regression — no pct_change transformation is applied.
# This contrasts with PPI, which is a level index and requires pct_change(12).
pmi_raw = pd.read_csv(
    os.path.join(DATA_DIR, 'data_monthly_pmi.csv'),
    encoding='utf-8-sig',
)
pmi_raw.columns = ['DATE', 'PMI']
pmi_raw['DATE']  = pmi_raw['DATE'].apply(parse_monthly)
pmi_raw['PMI']   = pd.to_numeric(pmi_raw['PMI'], errors='coerce')
pmi_raw = pmi_raw.set_index('DATE').sort_index()

# ── Seasonal adjustment of GDP ─────────────────────────────────────────────────
gdp_level = gdp_raw['GDP_U'].dropna()

if SA_MODE == 0:
    # ── Mode 0: compute STL seasonal adjustment inside the script ────────────
    # STL decomposes the quarterly GDP level into trend, seasonal, and residual.
    # SA level = observed − seasonal component.
    # Parameters:
    #   period=4   – quarterly seasonality
    #   seasonal=7 – seasonal smoother window (must be odd and ≥ 7)
    #   robust=True – down-weights outliers (COVID quarters) during fitting
    from statsmodels.tsa.seasonal import STL   # only required for SA_MODE=0

    print("\nSA_MODE=0 — Applying STL seasonal adjustment to quarterly GDP …")

    stl        = STL(gdp_level, period=4, seasonal=7, robust=True)
    stl_result = stl.fit()
    gdp_sa     = (gdp_level - stl_result.seasonal).rename('GDP_SA')

    print(f"  Seasonal component range: "
          f"{stl_result.seasonal.min():.0f} – {stl_result.seasonal.max():.0f}")

    # Figure 0 – STL decomposition (four-panel)
    print("\nPlotting Figure 0: STL decomposition …")
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(gdp_level.index, gdp_level.values, color='steelblue', linewidth=1.5)
    axes[0].set_title('Original GDP Level (constant prices)')
    axes[0].set_ylabel('VND bn')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(stl_result.trend.index, stl_result.trend.values,
                 color='darkorange', linewidth=1.5)
    axes[1].set_title('Trend Component')
    axes[1].set_ylabel('VND bn')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(stl_result.seasonal.index, stl_result.seasonal.values,
                 color='seagreen', linewidth=1.2)
    axes[2].axhline(0, color='grey', linewidth=0.6, linestyle='--')
    axes[2].set_title('Seasonal Component')
    axes[2].set_ylabel('VND bn')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(gdp_sa.index, gdp_sa.values,
                 color='steelblue', linewidth=1.5, label='SA GDP (STL)')
    axes[3].plot(gdp_level.index, gdp_level.values,
                 color='lightsteelblue', linewidth=1.0, linestyle='--',
                 alpha=0.7, label='Original GDP')
    axes[3].set_title('Seasonally Adjusted GDP Level')
    axes[3].set_ylabel('VND bn')
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle('STL Seasonal Decomposition of Vietnam Quarterly GDP\n'
                 '(period=4, seasonal window=7, robust=True)', fontsize=11, y=1.01)
    plt.tight_layout()
    fig0_path = os.path.join(OUT_DIR, 'fig0_seasonal_decomposition.png')
    plt.savefig(fig0_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig0_path}")

else:
    # ── Mode 1: import pre-seasonally-adjusted 'GDP' column from the data file ─
    print("\nSA_MODE=1 — Importing pre-seasonally-adjusted GDP series from data file …")

    gdp_sa = gdp_raw['GDP'].dropna().rename('GDP_SA')

    print(f"  Imported SA GDP: {gdp_sa.index[0].date()} – {gdp_sa.index[-1].date()} "
          f"({len(gdp_sa)} obs)")

    # Figure 0 – compare raw vs imported SA level
    print("\nPlotting Figure 0: raw vs imported SA GDP …")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(gdp_level.index, gdp_level.values,
            color='lightsteelblue', linewidth=1.2, linestyle='--',
            alpha=0.8, label='Original GDP (GDP_U)')
    ax.plot(gdp_sa.index, gdp_sa.values,
            color='steelblue', linewidth=1.8, label='Imported SA GDP (GDP)')
    ax.set_title('Vietnam Quarterly GDP — Original vs Pre-Seasonally-Adjusted Level')
    ax.set_ylabel('VND bn')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig0_path = os.path.join(OUT_DIR, 'fig0_seasonal_decomposition.png')
    plt.savefig(fig0_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig0_path}")

# ── Compute YOY growth from the SA level ──────────────────────────────────────
gdp_yoy = (gdp_sa.pct_change(4) * 100).dropna().rename('GDP_YOY')
print(f"  SA GDP YOY: {gdp_yoy.index[0].date()} – {gdp_yoy.index[-1].date()} "
      f"({len(gdp_yoy)} obs)")

# ── Compute PPI YOY; keep PMI as-is ───────────────────────────────────────────
# PPI is a level index → transform to YOY % growth via pct_change(12).
# PMI = 50 × (Prod_t / Prod_{t-12}) → already a YOY ratio; used directly.
# Both series therefore express year-over-year production performance and are
# aligned with the GDP YOY target: for Q1 2026, the 3 monthly PMI / PPI-YOY
# values each compare 2026 months to the same months of 2025, which is the
# same temporal basis as GDP_YOY_2026Q1 = (GDP_2026Q1 / GDP_2025Q1 − 1) × 100.
ppi_yoy = (ppi_raw['PPI'].pct_change(12) * 100).dropna().rename('PPI_YOY')
pmi     = pmi_raw['PMI'].dropna()   # YOY production ratio, no further transform

print(f"  PPI YOY:  {ppi_yoy.index[0].date()} – {ppi_yoy.index[-1].date()} "
      f"({len(ppi_yoy)} obs)")
print(f"  PMI (YOY production ratio, base=50): "
      f"{pmi.index[0].date()} – {pmi.index[-1].date()} ({len(pmi)} obs)")

# ── Figure 1 – Data overview ───────────────────────────────────────────────────
print("\nPlotting Figure 1: data overview …")

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)

common_start = max(gdp_yoy.index[0], ppi_yoy.index[0], pmi.index[0])

gdp_yoy_raw = (gdp_level.pct_change(4) * 100).dropna().rename('GDP_YOY_RAW')
axes[0].plot(gdp_yoy.loc[common_start:].index, gdp_yoy.loc[common_start:],
             color='steelblue', linewidth=1.8, marker='o', markersize=3,
             label='SA GDP YOY (%)')
axes[0].plot(gdp_yoy_raw.loc[common_start:].index, gdp_yoy_raw.loc[common_start:],
             color='lightsteelblue', linewidth=1.1, linestyle='--', alpha=0.8,
             label='Unadjusted GDP YOY (%)')
axes[0].axhline(0, color='grey', linewidth=0.6, linestyle='--')
axes[0].set_title('Vietnam GDP — Year-on-Year Growth (seasonally adjusted vs raw, %)')
axes[0].set_ylabel('%')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(ppi_yoy.loc[common_start:].index, ppi_yoy.loc[common_start:],
             color='darkorange', linewidth=1.4, label='PPI YOY growth (%)')
axes[1].axhline(0, color='grey', linewidth=0.6, linestyle='--')
axes[1].set_title('Production Price Index — Year-on-Year Growth (%)')
axes[1].set_ylabel('%')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(pmi.loc[common_start:].index, pmi.loc[common_start:],
             color='seagreen', linewidth=1.4,
             label='PMI = 50 × (Prod_t / Prod_{t−12})')
axes[2].axhline(50, color='grey', linewidth=0.8, linestyle='--',
                label='50 = flat YOY (no growth)')
axes[2].set_title('PMI — Monthly Production YOY Ratio (base 50; >50 = growth vs same month last year)')
axes[2].set_ylabel('Index (base 50)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, 'fig1_data_overview.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig1_path}")

# ── Rolling forecast evaluation (MIDAS with two indicators) ────────────────────
print("\nRunning rolling-window MIDAS evaluation …")
print(f"  Training window: {TRAIN_START.date()} – {TRAIN_END.date()}, rolling forward")

rmse_two, fc_rolling = midas_adl2(
    gdp_yoy, ppi_yoy, pmi,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    x1lag=XLAG, x2lag=XLAG,
    ylag=YLAG,
    horizon=HORIZON,
    poly1=POLY, poly2=POLY,
    method='rolling',
)

# Benchmark RMSE: rolling in-sample mean of GDP YOY
n_window = gdp_yoy.index.get_loc(TRAIN_END) - gdp_yoy.index.get_loc(TRAIN_START)
benchmark_series = gdp_yoy.rolling(window=n_window).mean()
benchmark_aligned = benchmark_series.reindex(fc_rolling.index)
rmse_benchmark = float(np.sqrt(((fc_rolling['targets'] - benchmark_aligned) ** 2).mean()))
relative_mse = (rmse_two / rmse_benchmark) ** 2   # = MSE_MIDAS / MSE_benchmark

print(f"  Out-of-sample RMSE (MIDAS):     {rmse_two:.4f}")
print(f"  Out-of-sample RMSE (benchmark): {rmse_benchmark:.4f}")
print(f"  Relative MSE (MIDAS / BM):      {relative_mse:.4f}")

# ── Figure 2 – Model performance ──────────────────────────────────────────────
print("\nPlotting Figure 2: model performance …")

actual = fc_rolling['targets']
preds  = fc_rolling['preds']
errors = actual - preds

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

axes[0].plot(actual.index, actual.values, color='steelblue', linewidth=1.8,
             marker='o', markersize=4, label='Actual GDP YOY (%)')
axes[0].plot(preds.index,  preds.values,  color='tomato',    linewidth=1.4,
             linestyle='--', marker='s', markersize=3, label='MIDAS Forecast')
axes[0].plot(benchmark_aligned.index, benchmark_aligned.values,
             color='grey', linewidth=1.0, linestyle=':', label='Benchmark (rolling mean)')
axes[0].set_title(
    f'Rolling MIDAS Forecast vs Actual  '
    f'(RMSE={rmse_two:.3f}, Relative MSE={relative_mse:.3f})'
)
axes[0].set_ylabel('GDP YOY Growth (%)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].bar(errors.index, errors.values, color='steelblue', alpha=0.6, width=60)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Forecast Errors (Actual − Predicted)')
axes[1].set_ylabel('pp error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'fig2_model_performance.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}")

# ── Construct extended series for 2026Q1 nowcast ──────────────────────────────
print("\nConstructing 2026Q1 nowcast …")

mar_2026 = pd.Timestamp('2026-03-31')
feb_2026 = pd.Timestamp('2026-02-28')
jan_2026 = pd.Timestamp('2026-01-31')

# Impute March 2026 raw PPI as average of Jan–Feb 2026
ppi_raw_ext = ppi_raw.copy()
ppi_raw_ext.loc[mar_2026, 'PPI'] = ppi_raw_ext.loc[[jan_2026, feb_2026], 'PPI'].mean()

# Impute March 2026 PMI as average of Jan–Feb 2026
pmi_ext = pmi.copy()
pmi_ext.loc[mar_2026] = pmi.loc[[jan_2026, feb_2026]].mean()
pmi_ext = pmi_ext.sort_index()

# Recompute PPI YOY with extended raw data
ppi_yoy_ext = (ppi_raw_ext['PPI'].pct_change(12) * 100).dropna().rename('PPI_YOY')

# Extend GDP YOY with 2026Q1 placeholder (NaN — the target we are nowcasting)
gdp_yoy_ext = pd.concat([
    gdp_yoy,
    pd.Series([np.nan], index=[mar_2026], name='GDP_YOY'),
])

print(f"  Imputed PPI  for 2026-03:  {ppi_raw_ext.loc[mar_2026, 'PPI']:.2f}"
      f" (raw index);  PPI YOY = {ppi_yoy_ext.loc[mar_2026]:.2f}%")
print(f"  Imputed PMI  for 2026-03:  {pmi_ext.loc[mar_2026]:.2f}")

# ── Estimate final MIDAS model on all data through 2025Q4 ─────────────────────
y_train, yl_train, x1_train, x2_train, yf_out, ylf_out, x1f_out, x2f_out = mix_freq2(
    gdp_yoy_ext, ppi_yoy_ext, pmi_ext,
    XLAG, XLAG, YLAG, HORIZON,
    start_date=TRAIN_START,
    end_date=FINAL_END,
)

res_final = estimate2(y_train, yl_train, x1_train, x2_train,
                      poly1=POLY, poly2=POLY)

fc_out = forecast2(x1f_out, x2f_out, ylf_out, res_final,
                   poly1=POLY, poly2=POLY)

# 2026Q1 nowcast point estimate
nowcast_2026q1 = float(fc_out.loc[mar_2026, 'yfh'])

# Confidence interval using out-of-sample RMSE from rolling evaluation
ci_half   = CI_Z * rmse_two
ci_lower  = nowcast_2026q1 - ci_half
ci_upper  = nowcast_2026q1 + ci_half

print(f"\n  ══════════════════════════════════════════")
print(f"  Nowcast GDP YOY Growth 2026Q1: {nowcast_2026q1:.2f}%")
print(f"  95% Confidence Interval:       [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"  ══════════════════════════════════════════")

# ── Build forecast table ───────────────────────────────────────────────────────
# Combine historical rolling forecasts + the 2026Q1 nowcast into one table
hist_table = fc_rolling[['preds', 'targets']].copy()
hist_table.columns = ['Forecast_GDP_YOY', 'Actual_GDP_YOY']
hist_table['CI_Lower_95'] = hist_table['Forecast_GDP_YOY'] - ci_half
hist_table['CI_Upper_95'] = hist_table['Forecast_GDP_YOY'] + ci_half
hist_table.index.name = 'Date'

nowcast_row = pd.DataFrame({
    'Forecast_GDP_YOY': [round(nowcast_2026q1, 4)],
    'Actual_GDP_YOY':   [np.nan],
    'CI_Lower_95':      [round(ci_lower, 4)],
    'CI_Upper_95':      [round(ci_upper, 4)],
}, index=pd.DatetimeIndex([mar_2026], name='Date'))

forecast_table = pd.concat([hist_table, nowcast_row]).sort_index()
forecast_table = forecast_table.round(4)

csv_path = os.path.join(OUT_DIR, 'forecast_table.csv')
forecast_table.to_csv(csv_path)
print(f"\nForecast table saved: {csv_path}")
print(forecast_table.tail(8).to_string())

# ── Figure 3 – Nowcast with confidence interval ────────────────────────────────
print("\nPlotting Figure 3: 2026Q1 nowcast …")

# Show recent history (last 3 years of actuals) + nowcast
plot_start = pd.Timestamp('2022-12-31')
hist_actual = gdp_yoy.loc[plot_start:]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(hist_actual.index, hist_actual.values,
        color='steelblue', linewidth=2, marker='o', markersize=5,
        label='Actual GDP YOY (%)')

# Historical forecasts for recent period
hist_fc_plot = hist_table.loc[plot_start:]
if not hist_fc_plot.empty:
    ax.plot(hist_fc_plot.index, hist_fc_plot['Forecast_GDP_YOY'],
            color='tomato', linewidth=1.5, linestyle='--',
            marker='s', markersize=3, label='MIDAS Forecast (historical)')
    ax.fill_between(hist_fc_plot.index,
                    hist_fc_plot['CI_Lower_95'],
                    hist_fc_plot['CI_Upper_95'],
                    color='tomato', alpha=0.10)

# 2026Q1 nowcast point + CI
ax.scatter([mar_2026], [nowcast_2026q1], color='crimson', s=120, zorder=5,
           label=f'2026Q1 Nowcast: {nowcast_2026q1:.2f}%')
ax.errorbar([mar_2026], [nowcast_2026q1],
            yerr=[[nowcast_2026q1 - ci_lower], [ci_upper - nowcast_2026q1]],
            fmt='none', color='crimson', capsize=8, linewidth=2,
            label=f'95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]')

ax.axhline(0, color='grey', linewidth=0.7, linestyle=':')
ax.axvline(pd.Timestamp('2025-12-31'), color='grey', linewidth=0.8,
           linestyle='--', alpha=0.6, label='Last known GDP')
ax.set_title('MIDAS Nowcast — Vietnam GDP YOY Growth 2026Q1\n'
             '(Monthly PPI + PMI as high-frequency indicators, Exp. Almon weights)',
             fontsize=11)
ax.set_ylabel('GDP YOY Growth (%)')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'fig3_nowcast_2026Q1.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig3_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MIDAS NOWCAST SUMMARY — Vietnam GDP YOY Growth 2026Q1")
print("=" * 55)
sa_desc = "STL (period=4, seasonal=7, robust)" if SA_MODE == 0 else "Imported pre-SA series (GDP column)"
print(f"  Seasonal adjustment: {sa_desc}")
print(f"  Model:              MIDAS-ADL, Exponential Almon weights")
print(f"  Indicators:         PPI (YOY%) and PMI (YOY production ratio, base=50; used as-is)")
print(f"  High-freq lags:     {XLAG} per indicator")
print(f"  AR lags (GDP):      {YLAG}")
print(f"  Train window:       {TRAIN_START.date()} – {FINAL_END.date()}")
print(f"  Monthly data:       through 2026-02-28 (March imputed)")
print(f"  Out-of-sample RMSE: {rmse_two:.4f} pp")
print(f"  Relative MSE:       {relative_mse:.4f}")
print(f"\n  Nowcast 2026Q1 GDP YOY (SA):  {nowcast_2026q1:.2f}%")
print(f"  95% CI:                        [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print("=" * 55)
print(f"\nOutputs written to: {OUT_DIR}/")
print("  fig0_seasonal_decomposition.png")
print("  fig1_data_overview.png")
print("  fig2_model_performance.png")
print("  fig3_nowcast_2026Q1.png")
print("  forecast_table.csv")
