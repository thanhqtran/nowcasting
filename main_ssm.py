#!/opt/homebrew/anaconda3/bin/python
"""
Nowcast Vietnam GDP YOY Growth for 2026Q1
State-Space Mixed-Frequency Model — Mariano-Murasawa (2003), revised

Model (monthly grid):
  State x_t = [s_t, s_{t-1}, s_{t-2}]  — latent monthly GDP YOY growth

  Transition (PMI-driven, no AR persistence):
      s_t = c + gamma * (pmi_t − 50) + eta_t,  eta_t ~ N(0, sigma2_eta)

      phi is fixed at 0: the latent monthly state has NO AR dynamics.
      This is necessary because the MLE for this dataset always collapses
      to a near-deterministic AR(1) (phi→1, sigma_eta→0) which ignores PMI
      entirely.  Fixing phi=0 forces the model to explain GDP through the
      PMI signal, making c and gamma identifiable.

      c     : mean GDP growth rate when pmi = 50 (PMI neutral)
      gamma : sensitivity — each 1-unit PMI deviation from 50 → gamma pp GDP

  Observation (quarterly only — NaN at non-quarter-end months):
      GDP_q_t = (1/3)(s_t + s_{t-1} + s_{t-2}) + eps_q,  eps_q ~ N(0, SIGMA_Q^2)

  sigma_q fixed at SIGMA_Q (calibrated).  With only 56 quarterly obs, free
  sigma_q also collapses to zero and creates the same identification trap.

Parameters estimated by MLE (3):
  c, log_sigma_eta, gamma

Outputs tagged with _ssm suffix.
"""

import os
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calendar import monthrange

from statsmodels.tsa.statespace import mlemodel

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'mydata')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_START = datetime.datetime(2012, 3, 31)
TRAIN_END   = datetime.datetime(2018, 12, 31)
FINAL_END   = datetime.datetime(2025, 12, 31)
CI_Z        = 1.96
SA_MODE     = 1   # 0 = STL inside script; 1 = use pre-SA 'GDP' column from file
SIGMA_Q     = 1.0 # fixed GDP obs noise (pp): calibrated, not estimated
PHI         = 0.0 # fixed AR(1) persistence: 0 = no AR dynamics (see docstring)

# ── Date helpers ───────────────────────────────────────────────────────────────
def parse_quarterly(s):
    year, q = int(s[:4]), int(s[5])
    m = q * 3
    return pd.Timestamp(year, m, monthrange(year, m)[1])

def parse_monthly(s):
    year, m = int(s[:4]), int(s[5:])
    return pd.Timestamp(year, m, monthrange(year, m)[1])

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")

gdp_raw = pd.read_csv(os.path.join(DATA_DIR, 'data_vn_2025Q4.csv'),
                      index_col=0, encoding='utf-8-sig')
gdp_raw = gdp_raw[gdp_raw.index != 'comment']
gdp_raw.index = [parse_quarterly(s) for s in gdp_raw.index]
gdp_raw['GDP_U'] = pd.to_numeric(gdp_raw['GDP_U'], errors='coerce')
gdp_raw['GDP']   = pd.to_numeric(gdp_raw['GDP'],   errors='coerce')

pmi_raw = pd.read_csv(os.path.join(DATA_DIR, 'data_monthly_pmi.csv'),
                      encoding='utf-8-sig')
pmi_raw.columns = ['DATE', 'PMI']
pmi_raw['DATE'] = pmi_raw['DATE'].apply(parse_monthly)
pmi_raw['PMI']  = pd.to_numeric(pmi_raw['PMI'], errors='coerce')
pmi_raw = pmi_raw.set_index('DATE').sort_index()

# ── Seasonal adjustment ────────────────────────────────────────────────────────
gdp_level = gdp_raw['GDP_U'].dropna()

if SA_MODE == 0:
    from statsmodels.tsa.seasonal import STL
    print("SA_MODE=0 — Applying STL …")
    stl_result = STL(gdp_level, period=4, seasonal=7, robust=True).fit()
    gdp_sa = (gdp_level - stl_result.seasonal).rename('GDP_SA')
else:
    print("SA_MODE=1 — Using pre-SA GDP column …")
    gdp_sa = gdp_raw['GDP'].dropna().rename('GDP_SA')

gdp_yoy = (gdp_sa.pct_change(4) * 100).dropna().rename('GDP_YOY')
pmi     = pmi_raw['PMI'].dropna()

print(f"  GDP YOY: {gdp_yoy.index[0].date()} – {gdp_yoy.index[-1].date()} ({len(gdp_yoy)} obs)")
print(f"  PMI:     {pmi.index[0].date()} – {pmi.index[-1].date()} ({len(pmi)} obs)")

# ── Figure 0: SA overview ──────────────────────────────────────────────────────
print("\nPlotting Figure 0 …")
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(gdp_level.index, gdp_level.values,
        color='lightsteelblue', linewidth=1.2, linestyle='--', alpha=0.8,
        label='Original GDP (GDP_U)')
ax.plot(gdp_sa.index, gdp_sa.values,
        color='steelblue', linewidth=1.8, label='SA GDP')
ax.set_title('Vietnam Quarterly GDP — Original vs Seasonally Adjusted Level')
ax.set_ylabel('VND bn'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig0_seasonal_decomposition_ssm.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 1: data overview ────────────────────────────────────────────────────
print("Plotting Figure 1 …")
common_start = max(gdp_yoy.index[0], pmi.index[0])
gdp_yoy_raw  = (gdp_level.pct_change(4) * 100).dropna()

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
axes[0].plot(gdp_yoy.loc[common_start:].index, gdp_yoy.loc[common_start:].values,
             color='steelblue', linewidth=1.8, marker='o', markersize=3,
             label='SA GDP YOY (%)')
axes[0].plot(gdp_yoy_raw.loc[common_start:].index, gdp_yoy_raw.loc[common_start:].values,
             color='lightsteelblue', linewidth=1.1, linestyle='--', alpha=0.8,
             label='Unadjusted GDP YOY (%)')
axes[0].axhline(0, color='grey', linewidth=0.6, linestyle='--')
axes[0].set_title('Vietnam GDP — Year-on-Year Growth (%)')
axes[0].set_ylabel('%'); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[1].plot(pmi.loc[common_start:].index, pmi.loc[common_start:].values,
             color='seagreen', linewidth=1.4,
             label='PMI = 50 × (Prod_t / Prod_{t−12})')
axes[1].axhline(50, color='grey', linewidth=0.8, linestyle='--', label='50 = flat YOY')
axes[1].set_title('PMI — Monthly Production YOY Ratio (base 50)')
axes[1].set_ylabel('Index'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_data_overview_ssm.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ── Build monthly observation grid ─────────────────────────────────────────────
def build_obs(gdp_yoy_series, pmi_series, pmi_end=None, mask_gdp_after=None):
    """
    Return a DataFrame with columns [GDP_OBS, PMI] on a monthly (ME) grid.

    GDP_OBS is NaN at non-quarter-end months and optionally after mask_gdp_after.
    PMI is the observed monthly PMI value (used as state equation input).
    """
    start = max(gdp_yoy_series.index[0], pmi_series.index[0])
    end   = pmi_series.index[-1]
    if pmi_end is not None:
        end = min(end, pd.Timestamp(pmi_end))

    monthly_idx = pd.date_range(start=start, end=end, freq='ME')
    df = pd.DataFrame({'GDP_OBS': np.nan, 'PMI': np.nan}, index=monthly_idx)

    for ts, val in gdp_yoy_series.items():
        if ts in df.index:
            if mask_gdp_after is None or ts <= pd.Timestamp(mask_gdp_after):
                df.loc[ts, 'GDP_OBS'] = val

    df['PMI'] = pmi_series.reindex(monthly_idx)
    return df.dropna(subset=['PMI'])


# ── State-Space Model ──────────────────────────────────────────────────────────
class MixedFreqSSM(mlemodel.MLEModel):
    """
    Mariano-Murasawa (2003) — PMI-driven state equation, centered PMI.

    State x_t = [s_t, s_{t-1}, s_{t-2}]  (latent monthly GDP YOY)

    Transition:
      s_t = c + phi * s_{t-1} + gamma * (pmi_t − 50) + eta_t
      ↑ constant c = long-run GDP mean when pmi=50 (identified separately from gamma)
      ↑ gamma = sensitivity to PMI deviations from neutral

    Observation (GDP only, scalar; NaN at non-quarter-end months):
      GDP_q_t = (1/3)(s_t + s_{t-1} + s_{t-2}) + eps_q

    Parameters (3): c, log_sigma_eta, gamma
    phi is fixed at PHI=0; sigma_q is fixed at SIGMA_Q.
    """

    def __init__(self, gdp_obs, pmi_centered, **kwargs):
        """
        gdp_obs     : (T, 1) array — quarterly GDP YOY; NaN at non-quarter-end
        pmi_centered: (T,)   array — (pmi_t − 50), centered monthly PMI
        """
        super().__init__(gdp_obs, k_states=3, k_posdef=1,
                         initialization='diffuse', **kwargs)
        self._pmi_c = np.asarray(pmi_centered, dtype=float)

        self['design'] = np.array([[1/3, 1/3, 1/3]], dtype=float)
        self['transition'] = np.array([[0., 0., 0.],
                                        [1., 0., 0.],
                                        [0., 1., 0.]], dtype=float)
        self['selection'] = np.array([[1.], [0.], [0.]], dtype=float)
        # Declare state_intercept as time-varying (k_states × nobs).
        # We keep a reference to the underlying array and modify it in-place
        # in update() — this ensures gradient computation sees the changes.
        self._si = np.zeros((3, self.nobs), dtype=float)
        self['state_intercept'] = self._si

    @property
    def param_names(self):
        return ['c', 'log_sigma_eta', 'gamma']

    @property
    def start_params(self):
        # c≈6 (mean Vietnam GDP YOY when pmi=50),
        # sigma_eta≈2 (within-quarter state variance), gamma≈0.3
        return np.array([6.0, np.log(2.0), 0.3])

    def update(self, params, **kwargs):
        super().update(params, **kwargs)
        c, log_sig_eta, gamma = params

        # phi fixed at PHI (=0): no AR persistence in the latent state
        self['transition', 0, 0] = PHI
        self['state_cov', 0, 0]  = np.exp(log_sig_eta) ** 2
        self['obs_cov', 0, 0]    = SIGMA_Q ** 2   # fixed

        # PMI-driven time-varying state intercept (reassigned each call)
        si = np.zeros((3, self.nobs), dtype=float)
        si[0, :] = c + gamma * self._pmi_c
        self['state_intercept'] = si


def fit_ssm(obs_df, start_params=None):
    """Fit SSM via MLE (L-BFGS-B). Returns (model, result)."""
    gdp_arr = obs_df[['GDP_OBS']].values   # (T, 1)
    pmi_c   = obs_df['PMI'].values - 50.0  # center PMI at neutral=50
    model   = MixedFreqSSM(gdp_arr, pmi_c)
    kw = dict(disp=False, maxiter=400, method='lbfgs', optim_score='approx')
    if start_params is not None:
        kw['start_params'] = start_params
    result = model.fit(**kw)
    return model, result


def get_filtered_nowcast(result, obs_df, target_date):
    """
    Filtered nowcast at target_date.
    filtered_state[:, t] = E[x_t | all PMI through t, GDP through last quarter].
    Nowcast = (s_t + s_{t-1} + s_{t-2}) / 3.
    """
    ts    = pd.Timestamp(target_date)
    t_idx = obs_df.index.get_loc(ts)
    fs    = result.filter_results.filtered_state   # (k_states, nobs)
    return (fs[0, t_idx] + fs[1, t_idx] + fs[2, t_idx]) / 3.0


# ── Rolling out-of-sample evaluation (expanding window) ───────────────────────
print("\nRunning rolling-window SSM evaluation …")
print("  PMI drives state equation directly (time-varying intercept).")

_ts = gdp_yoy.index[gdp_yoy.index.searchsorted(pd.Timestamp(TRAIN_START), side='left')]
_te = gdp_yoy.index[min(gdp_yoy.index.searchsorted(pd.Timestamp(TRAIN_END), side='right') - 1,
                        len(gdp_yoy.index) - 1)]
print(f"  Evaluation from: {_te.date()} + 1Q  through  {FINAL_END.date()}")

eval_quarters = [d for d in gdp_yoy.index
                 if _te < d <= pd.Timestamp(FINAL_END)]

preds_r, targets_r, dates_r = [], [], []
prev_params = None

for i, eval_date in enumerate(eval_quarters):
    prev_gdp_date = gdp_yoy.index[gdp_yoy.index.get_loc(eval_date) - 1]
    obs = build_obs(gdp_yoy, pmi,
                    pmi_end=eval_date,
                    mask_gdp_after=prev_gdp_date)

    if len(obs) < 24:
        continue

    try:
        _, res = fit_ssm(obs, start_params=prev_params)
        prev_params = res.params.copy()

        nowcast = get_filtered_nowcast(res, obs, eval_date)
        actual  = float(gdp_yoy.loc[eval_date])

        preds_r.append(nowcast)
        targets_r.append(actual)
        dates_r.append(eval_date)

        print(f"  [{i+1:2d}/{len(eval_quarters)}] {eval_date.date()}: "
              f"nowcast={nowcast:+.2f}%  actual={actual:+.2f}%  "
              f"err={actual - nowcast:+.2f}pp")

    except Exception as e:
        print(f"  [{i+1:2d}/{len(eval_quarters)}] {eval_date.date()}: skipped ({e})")

preds_r   = np.array(preds_r)
targets_r = np.array(targets_r)
fc_rolling = pd.DataFrame({'preds': preds_r, 'targets': targets_r},
                           index=pd.DatetimeIndex(dates_r))

rmse_ssm = float(np.sqrt(np.mean((preds_r - targets_r) ** 2)))

n_window  = gdp_yoy.index.get_loc(_te) - gdp_yoy.index.get_loc(_ts)
bm_series = gdp_yoy.rolling(window=n_window).mean().reindex(fc_rolling.index)
rmse_bm   = float(np.sqrt(((fc_rolling['targets'] - bm_series) ** 2).mean()))
rel_mse   = (rmse_ssm / rmse_bm) ** 2

print(f"\n  Out-of-sample RMSE (SSM):        {rmse_ssm:.4f}")
print(f"  Out-of-sample RMSE (benchmark):  {rmse_bm:.4f}")
print(f"  Relative MSE (SSM / BM):          {rel_mse:.4f}")

# ── Figure 2: rolling performance ─────────────────────────────────────────────
print("\nPlotting Figure 2 …")
actual_r  = fc_rolling['targets']
preds_r_s = fc_rolling['preds']
errors    = actual_r - preds_r_s

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
axes[0].plot(actual_r.index, actual_r.values, color='steelblue', linewidth=1.8,
             marker='o', markersize=4, label='Actual GDP YOY (%)')
axes[0].plot(preds_r_s.index, preds_r_s.values, color='tomato', linewidth=1.4,
             linestyle='--', marker='s', markersize=3, label='SSM Nowcast')
axes[0].plot(bm_series.index, bm_series.values, color='grey', linewidth=1.0,
             linestyle=':', label='Benchmark (rolling mean)')
axes[0].set_title(f'Rolling SSM Nowcast vs Actual  '
                  f'(RMSE={rmse_ssm:.3f}, Relative MSE={rel_mse:.3f})')
axes[0].set_ylabel('GDP YOY Growth (%)')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].bar(errors.index, errors.values, color='steelblue', alpha=0.6, width=60)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Forecast Errors (Actual − SSM Nowcast)')
axes[1].set_ylabel('pp error'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_model_performance_ssm.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ── 2026Q1 nowcast ─────────────────────────────────────────────────────────────
print("\nConstructing 2026Q1 nowcast …")

mar_2026 = pd.Timestamp('2026-03-31')
feb_2026 = pd.Timestamp('2026-02-28')
jan_2026 = pd.Timestamp('2026-01-31')

pmi_ext = pmi.copy()
pmi_ext.loc[mar_2026] = pmi.loc[[jan_2026, feb_2026]].mean()
pmi_ext = pmi_ext.sort_index()
print(f"  Imputed PMI for 2026-03: {pmi_ext.loc[mar_2026]:.2f}")

obs_final = build_obs(gdp_yoy, pmi_ext,
                      pmi_end=mar_2026,
                      mask_gdp_after=pd.Timestamp(FINAL_END))

print(f"  Monthly grid: {obs_final.index[0].date()} – {obs_final.index[-1].date()} "
      f"({len(obs_final)} months, {obs_final['GDP_OBS'].notna().sum()} GDP obs)")

_, res_final = fit_ssm(obs_final, start_params=prev_params)

nowcast_2026q1 = get_filtered_nowcast(res_final, obs_final, mar_2026)
ci_half  = CI_Z * rmse_ssm
ci_lower = nowcast_2026q1 - ci_half
ci_upper = nowcast_2026q1 + ci_half

print(f"\n  ══════════════════════════════════════════")
print(f"  Nowcast GDP YOY Growth 2026Q1: {nowcast_2026q1:.2f}%")
print(f"  95% Confidence Interval:       [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"  ══════════════════════════════════════════")

print("\n  Fitted model parameters (final estimation):")
for name, val in zip(res_final.model.param_names, res_final.params):
    print(f"    {name:20s} = {val:8.4f}")
sigma_eta = np.exp(res_final.params[1])
print(f"    {'sigma_eta':20s} = {sigma_eta:8.4f}   [monthly state noise, pp]")
print(f"    {'phi (fixed)':20s} = {PHI:8.4f}   [AR persistence, calibrated=0]")
print(f"    {'sigma_q (fixed)':20s} = {SIGMA_Q:8.4f}   [GDP obs noise, calibrated]")

# ── Forecast table ─────────────────────────────────────────────────────────────
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
csv_path = os.path.join(OUT_DIR, 'forecast_table_ssm.csv')
forecast_table.to_csv(csv_path)
print(f"\nForecast table saved: {csv_path}")
print(forecast_table.tail(8).to_string())

# ── Figure 3: nowcast with CI ──────────────────────────────────────────────────
print("\nPlotting Figure 3 …")
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
            marker='s', markersize=3, label='SSM Nowcast (historical)')
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
ax.set_title('State-Space Nowcast — Vietnam GDP YOY Growth 2026Q1\n'
             '(Mariano-Murasawa: PMI-driven AR(1) state, Kalman filter)',
             fontsize=11)
ax.set_ylabel('GDP YOY Growth (%)')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_nowcast_2026Q1_ssm.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 4: filtered latent state ───────────────────────────────────────────
print("Plotting Figure 4: filtered latent state …")

fs_final       = res_final.filter_results.filtered_state   # (3, T)
monthly_dates  = obs_final.index
latent_monthly = fs_final[0, :]
implied_q      = (fs_final[0, :] + fs_final[1, :] + fs_final[2, :]) / 3.0

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
axes[0].plot(monthly_dates, latent_monthly,
             color='seagreen', linewidth=1.2, alpha=0.8,
             label='Filtered latent monthly GDP YOY (s_t)')
axes[0].axhline(0, color='grey', linewidth=0.6, linestyle='--')
axes[0].set_title('Kalman Filtered Latent Monthly GDP YOY Growth')
axes[0].set_ylabel('Monthly GDP YOY (%)')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot(monthly_dates, implied_q,
             color='darkorange', linewidth=1.6,
             label='Implied quarterly aggregate: (s_t + s_{t-1} + s_{t-2})/3')
axes[1].scatter(gdp_yoy.index, gdp_yoy.values,
                color='steelblue', s=30, zorder=5,
                label='Observed quarterly GDP YOY')
axes[1].scatter([mar_2026], [nowcast_2026q1],
                color='crimson', s=80, zorder=6,
                label=f'2026Q1 SSM nowcast: {nowcast_2026q1:.2f}%')
axes[1].axhline(0, color='grey', linewidth=0.6, linestyle='--')
axes[1].set_title('Implied Quarterly Aggregate vs Observed GDP YOY')
axes[1].set_ylabel('GDP YOY (%)'); axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('State-Space Mixed-Frequency Model — Filtered States\n'
             '(PMI as state equation input, GDP as observation)',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_filtered_state_ssm.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("SSM NOWCAST SUMMARY — Vietnam GDP YOY Growth 2026Q1")
print("=" * 62)
sa_desc = "STL (period=4, seasonal=7, robust)" if SA_MODE == 0 else "Imported pre-SA series (GDP column)"
print(f"  Seasonal adjustment: {sa_desc}")
print(f"  Model:              Mariano-Murasawa State-Space (no AR dynamics)")
print(f"                      s_t = c + gamma*(pmi_t−50) + eta_t  [phi=0 fixed]")
print(f"                      GDP_q = (s_t+s_{{t-1}}+s_{{t-2}})/3 + eps_q")
print(f"  Estimation:         MLE via Kalman filter, 3 params (phi={PHI}, sigma_q={SIGMA_Q} fixed)")
print(f"  Evaluation:         Expanding-window OOS, GDP masked at forecast quarter")
print(f"  Final GDP obs:      {FINAL_END.date()}")
print(f"  Monthly PMI:        through 2026-02 (March imputed as Jan-Feb avg)")
print(f"  Out-of-sample RMSE: {rmse_ssm:.4f} pp")
print(f"  Relative MSE:       {rel_mse:.4f}")
print(f"\n  Nowcast 2026Q1 GDP YOY (SA):  {nowcast_2026q1:.2f}%")
print(f"  95% CI:                        [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print("=" * 62)
print(f"\nOutputs written to: {OUT_DIR}/")
for f in ['fig0_seasonal_decomposition_ssm.png',
          'fig1_data_overview_ssm.png',
          'fig2_model_performance_ssm.png',
          'fig3_nowcast_2026Q1_ssm.png',
          'fig4_filtered_state_ssm.png',
          'forecast_table_ssm.csv']:
    print(f"  {f}")
