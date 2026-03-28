#!/opt/homebrew/anaconda3/bin/python
"""
Nowcast Vietnam GDP YOY Growth for 2026Q1
Single-indicator MIDAS model — PMI only (no PPI).

PMI = 50 × (Production_t / Production_{t-12}): a monthly YOY production ratio
anchored at 50.  It is used as-is (no further transformation) because it
already expresses the same year-over-year temporal basis as the GDP YOY target.

Seasonal adjustment: controlled by SA_MODE (see below).
Model: MIDAS-ADL with Exponential Almon weights, rolling-window evaluation.
Nowcast date: 2026Q1 (2026-03-31).
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

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MIDAS_DIR)

# Single-indicator functions (no *2 suffix)
from midas.mix import mix_freq                     # noqa: E402
from midas.adl import estimate, forecast, rmse     # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_START = datetime.datetime(2012, 3, 31)   # start of rolling evaluation
TRAIN_END   = datetime.datetime(2018, 12, 31)  # end of initial training window
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

# GDP (quarterly)
gdp_raw = pd.read_csv(
    os.path.join(DATA_DIR, 'data_vn_2025Q4.csv'),
    index_col=0, encoding='utf-8-sig',
)
gdp_raw = gdp_raw[gdp_raw.index != 'comment']
gdp_raw.index = [parse_quarterly(s) for s in gdp_raw.index]
gdp_raw['GDP_U'] = pd.to_numeric(gdp_raw['GDP_U'], errors='coerce')
gdp_raw['GDP']   = pd.to_numeric(gdp_raw['GDP'],   errors='coerce')   # pre-SA column

# PMI (monthly)
# PMI = 50 × (Production_t / Production_{t-12}) — already a YOY ratio.
# Values > 50 indicate YOY production growth; < 50 indicate contraction.
# No pct_change transformation is applied.
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
    from statsmodels.tsa.seasonal import STL   # only required for SA_MODE=0

    print("\nSA_MODE=0 — Applying STL seasonal adjustment to quarterly GDP …")

    stl        = STL(gdp_level, period=4, seasonal=7, robust=True)
    stl_result = stl.fit()
    gdp_sa     = (gdp_level - stl_result.seasonal).rename('GDP_SA')

    print(f"  Seasonal component range: "
          f"{stl_result.seasonal.min():.0f} – {stl_result.seasonal.max():.0f}")

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
    fig0_path = os.path.join(OUT_DIR, 'fig0_seasonal_decomposition_1i.png')
    plt.savefig(fig0_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig0_path}")

else:
    print("\nSA_MODE=1 — Importing pre-seasonally-adjusted GDP series from data file …")

    gdp_sa = gdp_raw['GDP'].dropna().rename('GDP_SA')

    print(f"  Imported SA GDP: {gdp_sa.index[0].date()} – {gdp_sa.index[-1].date()} "
          f"({len(gdp_sa)} obs)")

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
    fig0_path = os.path.join(OUT_DIR, 'fig0_seasonal_decomposition_1i.png')
    plt.savefig(fig0_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig0_path}")

# ── Compute YOY growth from the SA level ──────────────────────────────────────
gdp_yoy = (gdp_sa.pct_change(4) * 100).dropna().rename('GDP_YOY')
print(f"  SA GDP YOY: {gdp_yoy.index[0].date()} – {gdp_yoy.index[-1].date()} "
      f"({len(gdp_yoy)} obs)")

# PMI used as-is — already a YOY production ratio (base 50)
pmi = pmi_raw['PMI'].dropna()
print(f"  PMI (YOY production ratio, base=50): "
      f"{pmi.index[0].date()} – {pmi.index[-1].date()} ({len(pmi)} obs)")

# ── Figure 1 – Data overview ───────────────────────────────────────────────────
print("\nPlotting Figure 1: data overview …")

common_start = max(gdp_yoy.index[0], pmi.index[0])
gdp_yoy_raw  = (gdp_level.pct_change(4) * 100).dropna().rename('GDP_YOY_RAW')

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

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

axes[1].plot(pmi.loc[common_start:].index, pmi.loc[common_start:],
             color='seagreen', linewidth=1.4,
             label='PMI = 50 × (Prod_t / Prod_{t−12})')
axes[1].axhline(50, color='grey', linewidth=0.8, linestyle='--',
                label='50 = flat YOY (no growth)')
axes[1].set_title('PMI — Monthly Production YOY Ratio (base 50; >50 = growth vs same month last year)')
axes[1].set_ylabel('Index (base 50)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, 'fig1_data_overview_1i.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig1_path}")

# ── Rolling-window MIDAS evaluation (single indicator: PMI only) ───────────────
# The library's rolling() helper ignores the poly argument internally, so the
# rolling evaluation is implemented explicitly here to ensure poly='expalmon'
# is applied consistently throughout.
print("\nRunning rolling-window MIDAS evaluation (PMI only) …")

preds_r, targets_r, dates_r = [], [], []
# Snap TRAIN_START / TRAIN_END to the nearest quarterly date that exists in
# gdp_yoy.index (handles the case where the user supplies a monthly date).
_ts = gdp_yoy.index[gdp_yoy.index.searchsorted(pd.Timestamp(TRAIN_START), side='left')]
_te = gdp_yoy.index[min(gdp_yoy.index.searchsorted(pd.Timestamp(TRAIN_END), side='right') - 1,
                        len(gdp_yoy.index) - 1)]
start_loc   = gdp_yoy.index.get_loc(_ts)
window_size = gdp_yoy.index.get_loc(_te) - start_loc
print(f"  Training window: {_ts.date()} – {_te.date()}, rolling forward")
fc_horizon  = 1   # one-step-ahead rolling forecast

while start_loc + window_size < len(gdp_yoy.index) - fc_horizon:
    y, yl, x, yf, ylf, xf = mix_freq(
        gdp_yoy, pmi,
        XLAG, YLAG, HORIZON,
        start_date=gdp_yoy.index[start_loc],
        end_date=gdp_yoy.index[start_loc + window_size],
    )
    if len(xf) - fc_horizon <= 0:
        break

    res = estimate(y, yl, x, poly=POLY)
    fc  = forecast(xf, ylf, res, poly=POLY)

    preds_r.append(fc.iloc[fc_horizon - 1].values[0])
    targets_r.append(yf.iloc[fc_horizon - 1])
    dates_r.append(yf.index[fc_horizon - 1])

    start_loc += 1

preds_r   = np.array(preds_r)
targets_r = np.array(targets_r)
fc_rolling = pd.DataFrame(
    {'preds': preds_r, 'targets': targets_r},
    index=pd.DatetimeIndex(dates_r),
)
rmse_one = rmse(preds_r, targets_r)

# Benchmark: rolling in-sample mean of GDP YOY
n_window         = gdp_yoy.index.get_loc(_te) - gdp_yoy.index.get_loc(_ts)
benchmark_series = gdp_yoy.rolling(window=n_window).mean()
benchmark_aligned = benchmark_series.reindex(fc_rolling.index)
rmse_benchmark   = float(np.sqrt(((fc_rolling['targets'] - benchmark_aligned) ** 2).mean()))
relative_mse     = (rmse_one / rmse_benchmark) ** 2

print(f"  Out-of-sample RMSE (MIDAS-PMI):  {rmse_one:.4f}")
print(f"  Out-of-sample RMSE (benchmark):  {rmse_benchmark:.4f}")
print(f"  Relative MSE (MIDAS / BM):        {relative_mse:.4f}")

# ── Figure 2 – Model performance ──────────────────────────────────────────────
print("\nPlotting Figure 2: model performance …")

actual = fc_rolling['targets']
preds  = fc_rolling['preds']
errors = actual - preds

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

axes[0].plot(actual.index, actual.values, color='steelblue', linewidth=1.8,
             marker='o', markersize=4, label='Actual GDP YOY (%)')
axes[0].plot(preds.index, preds.values, color='tomato', linewidth=1.4,
             linestyle='--', marker='s', markersize=3, label='MIDAS-PMI Forecast')
axes[0].plot(benchmark_aligned.index, benchmark_aligned.values,
             color='grey', linewidth=1.0, linestyle=':', label='Benchmark (rolling mean)')
axes[0].set_title(
    f'Rolling MIDAS-PMI Forecast vs Actual  '
    f'(RMSE={rmse_one:.3f}, Relative MSE={relative_mse:.3f})'
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
fig2_path = os.path.join(OUT_DIR, 'fig2_model_performance_1i.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}")

# ── Construct extended series for 2026Q1 nowcast ──────────────────────────────
print("\nConstructing 2026Q1 nowcast …")

mar_2026 = pd.Timestamp('2026-03-31')
feb_2026 = pd.Timestamp('2026-02-28')
jan_2026 = pd.Timestamp('2026-01-31')

# Impute March 2026 PMI as average of Jan–Feb 2026
pmi_ext = pmi.copy()
pmi_ext.loc[mar_2026] = pmi.loc[[jan_2026, feb_2026]].mean()
pmi_ext = pmi_ext.sort_index()
print(f"  Imputed PMI for 2026-03: {pmi_ext.loc[mar_2026]:.2f}")

# Extend GDP YOY with 2026Q1 placeholder (NaN — the target we are nowcasting)
gdp_yoy_ext = pd.concat([
    gdp_yoy,
    pd.Series([np.nan], index=[mar_2026], name='GDP_YOY'),
])

# ── Estimate final MIDAS model on all data through 2025Q4 ─────────────────────
y_train, yl_train, x_train, yf_out, ylf_out, xf_out = mix_freq(
    gdp_yoy_ext, pmi_ext,
    XLAG, YLAG, HORIZON,
    start_date=TRAIN_START,
    end_date=FINAL_END,
)

res_final = estimate(y_train, yl_train, x_train, poly=POLY)
fc_out    = forecast(xf_out, ylf_out, res_final, poly=POLY)

# 2026Q1 nowcast point estimate
nowcast_2026q1 = float(fc_out.loc[mar_2026, 'yfh'])

# Confidence interval using out-of-sample RMSE from rolling evaluation
ci_half  = CI_Z * rmse_one
ci_lower = nowcast_2026q1 - ci_half
ci_upper = nowcast_2026q1 + ci_half

print(f"\n  ══════════════════════════════════════════")
print(f"  Nowcast GDP YOY Growth 2026Q1: {nowcast_2026q1:.2f}%")
print(f"  95% Confidence Interval:       [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"  ══════════════════════════════════════════")

# ── Build forecast table ───────────────────────────────────────────────────────
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

forecast_table = pd.concat([hist_table, nowcast_row]).sort_index().round(4)

csv_path = os.path.join(OUT_DIR, 'forecast_table_1i.csv')
forecast_table.to_csv(csv_path)
print(f"\nForecast table saved: {csv_path}")
print(forecast_table.tail(8).to_string())

# ── Figure 3 – Nowcast with confidence interval ────────────────────────────────
print("\nPlotting Figure 3: 2026Q1 nowcast …")

plot_start  = pd.Timestamp('2022-12-31')
hist_actual = gdp_yoy.loc[plot_start:]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(hist_actual.index, hist_actual.values,
        color='steelblue', linewidth=2, marker='o', markersize=5,
        label='Actual GDP YOY (%)')

hist_fc_plot = hist_table.loc[plot_start:]
if not hist_fc_plot.empty:
    ax.plot(hist_fc_plot.index, hist_fc_plot['Forecast_GDP_YOY'],
            color='tomato', linewidth=1.5, linestyle='--',
            marker='s', markersize=3, label='MIDAS-PMI Forecast (historical)')
    ax.fill_between(hist_fc_plot.index,
                    hist_fc_plot['CI_Lower_95'],
                    hist_fc_plot['CI_Upper_95'],
                    color='tomato', alpha=0.10)

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
             '(PMI-only model, Exp. Almon weights)',
             fontsize=11)
ax.set_ylabel('GDP YOY Growth (%)')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'fig3_nowcast_2026Q1_1i.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig3_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MIDAS NOWCAST SUMMARY — Vietnam GDP YOY Growth 2026Q1")
print("=" * 55)
sa_desc = "STL (period=4, seasonal=7, robust)" if SA_MODE == 0 else "Imported pre-SA series (GDP column)"
print(f"  Seasonal adjustment: {sa_desc}")
print(f"  Model:              MIDAS-ADL (single indicator), Exponential Almon weights")
print(f"  Indicator:          PMI only (YOY production ratio, base=50; used as-is)")
print(f"  High-freq lags:     {XLAG}")
print(f"  AR lags (GDP):      {YLAG}")
print(f"  Train window:       {TRAIN_START.date()} – {FINAL_END.date()}")
print(f"  Monthly data:       through 2026-02-28 (March imputed)")
print(f"  Out-of-sample RMSE: {rmse_one:.4f} pp")
print(f"  Relative MSE:       {relative_mse:.4f}")
print(f"\n  Nowcast 2026Q1 GDP YOY (SA):  {nowcast_2026q1:.2f}%")
print(f"  95% CI:                        [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print("=" * 55)
print(f"\nOutputs written to: {OUT_DIR}/")
print("  fig0_seasonal_decomposition_1i.png")
print("  fig1_data_overview_1i.png")
print("  fig2_model_performance_1i.png")
print("  fig3_nowcast_2026Q1_1i.png")
print("  forecast_table_1i.csv")
