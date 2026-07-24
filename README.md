This project fits the SVI model $$w(k) = a + b[ρ(k − m) + \sqrt{((k − m)² + σ²)}]$$ to the implied volatility smile across many daily snapshots of SPX options, targeting mid implied volatilities, and applies ridge penalisation on changes in SVI parameters so that parameter paths evolve smoothly over time. A second script turns the calibration into the figures, LaTeX tables and numeric macros used in the accompanying report.

---

## 1. Repository layout

```
svi_fit.py                      Core pipeline: clean data, estimate forwards, calibrate SVI (free + ridge)
make_figures.py                 Report pipeline: 7-way lambda sweep, event detection, all figures/tables

output/
├── options_data.csv            Input: raw SPX option chain (pricedate, expiry, strike, type, IV%)
├── svi_results_all_dates.csv   svi_fit.py output — per-date forward/params/RMSE, free vs. ridge
├── svi_report_results.csv      make_figures.py output — same, across the full lambda sweep
├── report_numbers.tex          \newcommand macros referenced by the write-up
├── tab_params.tex              Per-parameter summary table (mean/min/max/std(Δ), free vs. ridge)
└── tab_lambda.tex              Lambda sweep: RMSE vs. path roughness, one row per λ

figures/
├── fig_returns.png             Forward level, forward log-return, ATM IV — with flagged/event dates
├── fig_smile_{warmup,middle,good,event}.png   Representative smile fits at 4 points in the sample
├── fig_params_full.png         All 5 parameters + RMSE, full sample, free vs. ridge
├── fig_params_event.png        Same, zoomed to a window around the largest detected market event
├── fig_lambda_tradeoff.png     Mean RMSE vs. mean path roughness across the λ sweep
├── svi_demo_single_date.png    Single-date sanity-check plot (IV space + total-variance space)
└── svi_parameter_paths.png     Parameter-path comparison (3x2 panel, free vs. ridge)
```

`fig_*.png` files at the repo root are earlier renders of the same-named files under `figures/`; the `figures/` copies are the ones `make_figures.py` currently (re)writes.

## 2. Running it

Both scripts expect to be run from this directory (`spx-svi-calibration/`) so that the relative path `output/options_data.csv` resolves.

```bash
python svi_fit.py       # demo plot, all-dates fit, parameter-path plot, stability report -> stdout
                        # writes figures/svi_demo_single_date.png, figures/svi_parameter_paths.png, output/svi_results_all_dates.csv

python make_figures.py  # 7-track lambda sweep, event detection, full figure/table set -> stdout
                        # writes everything under figures/, output/{report_numbers,tab_params,tab_lambda}.tex,
                        # output/svi_report_results.csv
```

`make_figures.py` imports `load_and_clean`, `prepare_smile`, `fit_svi`, `svi_w`, `rmse_vol` from `svi_fit.py`, so the two must stay in the same directory; it does not require `svi_fit.py` to be run first.

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`.

---

## 3. `svi_fit.py`: calibration pipeline

```
svi_fit.py
├── load_and_clean()        Step 1 — data cleaning
├── estimate_forward()      Step 2 — forward from call/put IV crossing
├── svi_w()                 SVI total-variance function
├── auto_moneyness_clip()   per-day wing cutoff (replaces a manual constant)
├── smile_data()            Step 3 — OTM selection + (k, w, T) transforms
├── fit_svi()               Step 4 — multistart constrained fit, optional ridge
├── rmse_vol()              fit error in implied-vol units
├── prepare_smile()         shared forward-estimate + smile-data prep, used by both fits below
├── demo_one_date()         Step 5 — single-date sanity check + plots
├── fit_all_dates()         Step 6 — sequential fits, unpenalised vs ridge
├── plot_parameter_paths()  Step 7 — parameter-path and RMSE charts
└── stability_report()      Step 8 — quantitative stability comparison
```

Outputs produced: `svi_demo_single_date.png`, `svi_parameter_paths.png`, `output/svi_results_all_dates.csv`.

### Step 1 — `load_and_clean()`: data cleaning

The raw file is parsed with day-first dates, option types normalised to `'c'`/`'p'`, and IVs converted from percent to decimals. Cleaning filters, applied in line with the spec's instruction to remove missing quotes, stale prices and extreme outliers:

- Drop rows with missing IV
- Keep only IVs in the plausible band
- Require positive time-to-expiry `T = (expiry − date)/365 > 0`

The row counts before/after are printed, which feeds the data-provenance/filters documentation the spec requires.

### Step 2 — `estimate_forward()`: forward from the call/put IV crossing

SVI needs log-moneyness k = ln(K/F), but the dataset carries no spot or forward. The script recovers F per date from the data itself. The rationale, documented in the function: put–call parity gives C(K) − P(K) = DF·(F − K), and pricing both legs through Black–Scholes forces **IV_call(F) = IV_put(F)** — the call-IV and put-IV curves cross exactly at the forward when quotes are parity-consistent.

In this dataset d(K) = IV_call(K) − IV_put(K) turns out to be one-signed (a systematic offset, presumably discounting) rather than crossing zero, so the naive "strike of minimum |d|" estimator would be quantised to the strike grid and inject ~1 grid step of spurious daily noise. Implementation:

- build d(K) on the strikes common to both types (require ≥ 7), smoothing with a rolling mean;
- restrict the search to `F_prev·(1 ± window)` once a previous day's forward is known, for continuity;
- if a genuine sign change exists, take it directly, interpolating between the bracketing strikes for a sub-grid-accurate F (closest crossing to `F_prev`/median strike if there are several);
- otherwise fit a parabola to the points nearest the minimum of |d(K)| and take the vertex as a sub-grid estimate;
- last resort: the grid strike minimising |d(K)|.

The returned `flag` marks dates where F jumped more than `max_jump` (log terms) from the previous day — used later to flag discontinuous forward estimates and to gate event detection. This makes the pipeline self-contained (no external index/rates data) and doubles as a parity-consistency check on the quotes.

### Step 3 — `smile_data()`: OTM selection and coordinate transforms

Per the standard market convention, only out-of-the-money quotes are kept: puts for k ≤ 0, calls for k > 0. OTM contracts are the liquid side, and this leaves exactly one IV per strike. Deep wings are discarded using `auto_moneyness_clip()`, which computes the cutoff from each day's own quotes — keeping the central 85% by |k| and dropping the widest (least liquid) tail — instead of a manually chosen constant. On this dataset that comes out to ≈0.5–0.6, close to what a hand-picked threshold would give, but it now adapts per day/expiry automatically.

Transforms (spec steps 7–8): total variance **w = IV² · T** and log-moneyness **k = ln(K/F)**. All fitting happens in (k, w) space, where SVI's form is natural.

### Step 4 — `fit_svi()`: constrained multistart optimisation with optional ridge

The objective minimised for a single smile is:

> **loss(θ) = mean[ wts · (w_SVI(k; θ) − w_mkt)² ] + 10³·min(w_min(θ), 0)² + λ·mean[ ((θ − θ_prev)/s)² ]**

Term by term:

1. **Weighted fit-to-mid error.** Weights `wts = 1/(1 + 5k²)` down-weight the wings relative to the money — a vega-like weighting that mimics where market information is concentrated and prevents sparse noisy wing quotes from steering the fit.
2. **Soft no-negative-variance penalty.** The minimum of w_SVI over a grid k ∈ [−1, 1] is penalised if negative (weight 10³). This enforces the static-arbitrage-motivated condition a + bσ√(1−ρ²) ≥ 0 in soft form, keeping the optimiser in the admissible region without a hard nonlinear constraint.
3. **Ridge on parameter changes** (active only when `lam > 0` and a previous-day θ exists). Differences are divided by the per-parameter scale vector **s = (0.05, 0.5, 0.5, 0.2, 0.2)** before squaring. This scaling is essential: the parameters live on very different magnitudes (a ~ 0.01, m ~ 0.1), and without it λ would effectively regularise only the numerically largest parameter.

Hard box bounds are imposed via L-BFGS-B: a ∈ (−0.5, 1), b ∈ (10⁻⁴, 5) (positive wings), ρ ∈ (−0.999, 0.999), m ∈ (−1, 1), σ ∈ (10⁻³, 2) (positive curvature).

**Multistart strategy** (8 starts, configurable via `n_starts`): the previous day's solution (warm start), a skew-informed default (a = 0.8·w_ATM, b = 0.1, ρ = −0.6, m = 0, σ = 0.1 — encoding the prior that SPX has a strongly negative skew), and randomised starts drawn around plausible ranges. The best of the starts is kept. This guards against the local-minimum traps that a single-start nonlinear fit of SVI is prone to.

`rmse_vol()` converts fitted total variance back to implied vol and reports RMSE in vol units (multiplied by 10⁴ in printouts → vol basis points), which is the economically interpretable error measure.

### Step 5 — `demo_one_date()`: single-date sanity check

Before the full time series, one date (the earliest by default) is fitted and plotted in both IV space and total-variance space, with the estimated forward, T, sample size, fitted parameters, and RMSE printed. This corresponds to spec step 9 ("fit baseline SVI") and provides the smile-fit and total-variance plots the spec lists among expected outputs.

### Step 6 — `fit_all_dates()`: sequential unpenalised vs ridge-penalised fits

Dates are processed **chronologically**. For each date the forward is estimated, the smile constructed via `prepare_smile()` (skipping dates with a failed forward or fewer than 15 usable quotes), and **two** fits run in parallel tracks:

- `theta_free`: λ = 0, warm-started from yesterday's free solution;
- `theta_ridge`: λ = `lamb` (1e-5 by default in `__main__`), penalised toward and warm-started from yesterday's *ridge* solution.

Keeping the two tracks separate ensures a clean comparison: each track's dynamics depend only on its own history. Results per date — forward, T, sample size, both parameter vectors, both RMSEs, the forward-continuity flag — accumulate into a DataFrame exported as `output/svi_results_all_dates.csv` (the spec's fitted-parameter and calibration-error tables).

### Step 7 - `plot_parameter_paths()`: visual comparison

A 3×2 panel: one subplot per parameter showing the unpenalised and ridge paths overlaid, plus a final panel of daily fit RMSE (vol bps) for both tracks. This is the spec's "parameter time-series charts showing smoother parameter evolution after ridge penalisation" and the per-snapshot calibration-error chart.


---

## 4. `make_figures.py`: report figures and tables

Reuses `load_and_clean`, `prepare_smile`, `fit_svi`, `svi_w`, `rmse_vol` from `svi_fit.py` and drives them across a wider set of scenarios than the base script, to produce every figure, table and numeric macro referenced by the write-up.

### `run_pipeline()` / `as_frame()`: the lambda sweep

Instead of a single free-vs-ridge comparison, this script runs **seven independent tracks in parallel**, one per λ in `sweep = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`, each with its own warm start and ridge anchor from its own history (so tracks don't interfere) and an equal multistart budget (`n_starts = 8`) so RMSEs are directly comparable across λ. Alongside the SVI fit, each date's record also carries the ATM implied vol (`np.interp` of the smile at k=0), used for event corroboration below. Results assemble into `res` (indexed by date, all λ columns) and are exported to `output/svi_report_results.csv`.

### `returns_and_events()`: market-event detection

A date counts as an event only when a large forward log-return (|return| > 2.5σ) is **corroborated** by a large ATM-IV change (|Δiv| > 1.0σ) — a forward move with no vol repricing is more likely a forward-estimation artefact than a genuine index move. Dates with a flagged (discontinuous) forward are excluded from candidacy unless doing so would exclude almost everything. The detector degrades gracefully through three fallbacks (`basis` records which was used): corroborated events → large returns alone → largest ATM-vol moves. Returns `(ret, d_atm, events, thresh, n_flagged, basis)`.

### Figure generation

- `plot_returns()` → `fig_returns.png`: forward level (flagged dates marked), daily forward log-return with the ±threshold band, ATM IV — with vertical lines at detected events.
- `plot_smile()` → `fig_smile_{warmup,middle,good,event}.png`: one smile fit each at the cold-start date (no history, free fit only), a mid-sample date, the best-fitting non-event date, and the largest corroborated event date — free vs. ridge overlaid where applicable.
- `mask_outliers()` / `robust_ylim()`: rolling-median/MAD-based masking used only for plotting, so that isolated bad local-optimum fits (visible as spikes) don't blow out the y-axis of the parameter-path charts; masked points are marked with an "x" rather than dropped silently.
- `plot_params_full()` → `fig_params_full.png`: all 5 parameters + RMSE across the whole sample, free vs. the selected ridge λ (1e-5), with event dates marked and outliers masked per above.
- `plot_params_event()` → `fig_params_event.png`: the same panel zoomed to a ±12-day window around the largest event, to show how ridge parameters transition to a new level rather than lagging or over-smoothing through a genuine regime change.
- `plot_lambda_tradeoff()` → `fig_lambda_tradeoff.png`: mean scaled path roughness (mean of the scaled per-step parameter-change norm) vs. mean RMSE, one point per λ — the stability/accuracy trade-off curve referenced in the write-up's discussion of the "rigidity threshold."

### Tables and macros

- `write_tables()` → `output/tab_params.tex`: per-parameter mean/min/max/std(Δ) for both the free and selected-ridge tracks. `output/tab_lambda.tex`: one row per swept λ with mean RMSE, mean roughness, and a short qualitative note (`notes` dict, e.g. "selected: optimum on both axes").
- `write_numbers()` → `output/report_numbers.tex`: `\newcommand` macros for every headline figure quoted in the write-up — date counts, event dates/basis, residuals split by moneyness region (core |k|≤0.15 vs wing |k|>0.35), calm-vs-event-window std(Δρ), per-track mean RMSE, etc. These are consumed directly by a LaTeX report (not included in this repo); `SVI_Volatility_Calibration_Report.md` is the rendered write-up built from the same numbers.

Running the script's `__main__` block also prints forward-estimation diagnostics (flag rate, log-return percentiles) and the first date's call/put IV difference, to eyeball the forward estimator's `estimate_forward()` behaviour described above.

---

## 5. Further reading

`SVI_Volatility_Calibration_Report.md` is the write-up built on top of these two scripts' outputs — data-cleaning rationale, the calibration-stability numbers from §2/§8 above, the "inter-temporal guidance" finding that ridge *improves* mean RMSE rather than trading it off, and a discussion of the skew ($ρ$) and shift ($m$) parameters' economic interpretation.
