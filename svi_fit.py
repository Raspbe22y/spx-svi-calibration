"""
SVI calibration pipeline for single-expiry SPX option IV data.

Steps:
  1. Load + clean options_data.csv
  2. Estimate the forward per date from the call/put IV crossing (put-call parity:
     call IV == put IV at K = F when quotes are parity-consistent)
  3. Baseline SVI fit for one date (with plots)
  4. Sequential SVI fit across all dates, unpenalised vs ridge-penalised,
     with parameter-path and RMSE comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# 1. Data loading

def load_and_clean(path="output/options_data.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"pricedate": "date", "expiry": "expiry",
                            "option_type": "type", "impliedvol": "iv"})

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["expiry"] = pd.to_datetime(df["expiry"], dayfirst=True)
    df["type"] = df["type"].str.strip().str.lower().str[0]   # 'c' / 'p'

    # IVs are quoted in percent -> decimals
    df["iv"] = df["iv"] / 100.0

    n0 = len(df)
    df = df.dropna(subset=["iv"])
    df["T"] = (df["expiry"] - df["date"]).dt.days / 365.0
    df = df[df["T"] > 0]

    print(f"Cleaning: {n0} rows -> {len(df)} rows "
          f"({n0 - len(df)} dropped: NaN/outlier IVs)")
    return df.sort_values(["date", "type", "strike"]).reset_index(drop=True)


# 2. Forward estimation from call/put IV crossing

def estimate_forward(day_df, F_prev=None, max_jump=0.03, window=0.15,
                     smooth=5, n_fit=7):
    """Forward from the call/put IV difference d(K) = IV_c(K) - IV_p(K).

    Put-call parity implies d(F) = 0. In this dataset d is one-signed (a
    systematic offset, presumably discounting), so no zero crossing exists and
    the naive 'strike of minimum |d|' estimator is quantised to the strike grid,
    injecting ~1 grid step of spurious daily noise. Instead we smooth |d| and
    fit a parabola to the points nearest its minimum, taking the vertex as a
    sub-grid estimate of F. If a genuine sign change exists it is used directly.
    Continuity: when F_prev is known the search is restricted to
    F_prev*(1 +/- window).

    Returns (F, flag); flag is True if F moved more than max_jump in log terms.
    """
    c = day_df[day_df["type"] == "c"].set_index("strike")["iv"]
    p = day_df[day_df["type"] == "p"].set_index("strike")["iv"]
    common = c.index.intersection(p.index)
    if len(common) < 7:
        return np.nan, True
    diff = (c.loc[common] - p.loc[common]).sort_index()
    k = diff.index.values.astype(float)
    d = pd.Series(diff.values).rolling(smooth, center=True,
                                       min_periods=1).mean().values

    if F_prev is not None and np.isfinite(F_prev):
        sel = (k > F_prev * (1 - window)) & (k < F_prev * (1 + window))
        if sel.sum() >= n_fit:
            k, d = k[sel], d[sel]

    F = np.nan
    flips = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(flips):                                  # genuine crossing
        roots = np.array([k[i] - d[i] * (k[i + 1] - k[i]) / (d[i + 1] - d[i])
                          for i in flips])
        ref = F_prev if (F_prev is not None and np.isfinite(F_prev)) \
            else np.median(k)
        F = float(roots[np.argmin(np.abs(roots - ref))])
    else:                                           # sub-grid vertex of |d|
        y = np.abs(d)
        i = int(np.argmin(y))
        lo, hi = max(0, i - n_fit // 2), min(len(k), i + n_fit // 2 + 1)
        if hi - lo >= 3:
            a2, a1, _ = np.polyfit(k[lo:hi], y[lo:hi], 2)
            if a2 > 0:
                cand = -a1 / (2 * a2)
                if k[lo] <= cand <= k[hi - 1]:      # vertex inside the fit range
                    F = float(cand)
        if not np.isfinite(F):
            F = float(k[i])                         # last resort: grid point

    flag = (F_prev is not None and np.isfinite(F_prev)
            and abs(np.log(F / F_prev)) > max_jump)
    return F, flag

# 3. SVI model + single-smile objective

def svi_w(k, theta):
    a, b, rho, m, sig = theta
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def auto_moneyness_clip(k, pct=85.0, floor=0.15):
    """Wing cutoff computed per smile instead of a manual constant: keep
    the central % of quotes by |k|, dropping the widest (least
    liquid) tail automatically."""
    absk = np.abs(k)
    return max(np.percentile(absk, pct), floor)


def smile_data(day_df, F):
    """Return (k, w, T) using OTM options only: puts for k<0, calls for k>0.
    Deep wings are dropped using a per-day auto-computed moneyness clip
    (see auto_moneyness_clip) rather than a fixed threshold."""
    d = day_df.copy()
    d["k"] = np.log(d["strike"] / F)
    otm = ((d["type"] == "p") & (d["k"] <= 0)) | ((d["type"] == "c") & (d["k"] > 0))
    d = d[otm]
    clip = auto_moneyness_clip(d["k"].values)
    d = d[d["k"].abs() <= clip]
    T = d["T"].iloc[0]
    w = d["iv"].values ** 2 * T
    return d["k"].values, w, T


BOUNDS = [(-0.5, 1.0),      # a
          (1e-4, 5.0),      # b
          (-0.999, 0.999),  # rho
          (-1.0, 1.0),      # m
          (1e-3, 2.0)]      # sigma


def fit_svi(k, w, theta0=None, lam=0.0, theta_prev=None, n_starts=8, seed=0):
    """Weighted least squares on total variance, optional ridge on
    (theta - theta_prev). Vega-like weighting: heavier near the money."""
    wts = 1.0 / (1.0 + 5.0 * k ** 2)
    scale = np.array([0.05, 0.5, 0.5, 0.2, 0.2])  # Ridge scaling per parameter

    def obj(theta):
        r = svi_w(k, theta) - w
        loss = np.mean(wts * r ** 2)
        # Soft no-negative-variance penalty: min of svi on a grid must be >= 0
        wmin = svi_w(np.linspace(-1, 1, 41), theta).min()
        loss += 1e3 * min(wmin, 0.0) ** 2
        if lam > 0 and theta_prev is not None:
            loss += lam * np.mean(((theta - theta_prev) / scale) ** 2)
        return loss

    rng = np.random.default_rng(seed)
    starts = []
    if theta0 is not None:
        starts.append(np.asarray(theta0, float))
    w_atm = np.interp(0.0, np.sort(k), w[np.argsort(k)])
    starts.append(np.array([0.8 * w_atm, 0.1, -0.6, 0.0, 0.1]))
    while len(starts) < n_starts:
        starts.append(np.array([w_atm * rng.uniform(0.3, 1.2),
                                rng.uniform(0.02, 0.5),
                                rng.uniform(-0.95, 0.2),
                                rng.uniform(-0.3, 0.3),
                                rng.uniform(0.05, 0.5)]))

    best, best_val = None, np.inf
    for s in starts:
        res = minimize(obj, s, method="L-BFGS-B", bounds=BOUNDS)
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return best


def rmse_vol(k, w, T, theta):
    iv_fit = np.sqrt(np.maximum(svi_w(k, theta), 1e-12) / T)
    iv_mkt = np.sqrt(w / T)
    return float(np.sqrt(np.mean((iv_fit - iv_mkt) ** 2)))


def prepare_smile(day_df, F_prev=None, min_n=15):
    """Forward estimate + (k, w, T) smile data for one date.
    Returns (F, k, w, T, flag) or None if unusable."""
    F, flag = estimate_forward(day_df, F_prev=F_prev)
    if not np.isfinite(F):
        return None
    k, w, T = smile_data(day_df, F)
    if len(k) < min_n:
        return None
    return F, k, w, T, flag


# 4. Single-date demo

def demo_one_date(df, date=None):
    date = date or df["date"].min()
    day = df[df["date"] == date]
    F, k, w, T, _ = prepare_smile(day, min_n=1)
    theta = fit_svi(k, w)
    err = rmse_vol(k, w, T, theta)

    print(f"\n=== Demo date {date.date()} ===")
    print(f"Estimated forward F = {F:,.1f}   T = {T:.3f}y   n = {len(k)}")
    print("SVI params (a, b, rho, m, sigma):",
          np.round(theta, 5), f"  RMSE = {err*1e4:.1f} vol bps")

    kg = np.linspace(k.min(), k.max(), 200)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(k, np.sqrt(w / T) * 100, s=12, alpha=0.6, label="market IV")
    ax[0].plot(kg, np.sqrt(svi_w(kg, theta) / T) * 100, "r-", lw=2, label="SVI fit")
    ax[0].set(xlabel="log-moneyness k", ylabel="implied vol (%)",
              title=f"SVI smile fit — {date.date()}")
    ax[0].legend()
    ax[1].scatter(k, w, s=12, alpha=0.6, label="market total variance")
    ax[1].plot(kg, svi_w(kg, theta), "r-", lw=2, label="SVI w(k)")
    ax[1].set(xlabel="log-moneyness k", ylabel="total variance w",
              title="Total variance space")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("figures/svi_demo_single_date.png", dpi=300)
    plt.close()
    return theta


# 5. All dates: unpenalised vs ridge-penalised sequential fits

def fit_all_dates(df, lamb=1.0):
    dates = sorted(df["date"].unique())
    rows, F_prev = [], None
    theta_free_prev = theta_ridge_prev = None
    for t in dates:
        prepared = prepare_smile(df[df["date"] == t], F_prev=F_prev)
        if prepared is None:
            continue
        F, k, w, T, flag = prepared
        F_prev = F

        theta_free = fit_svi(k, w, theta0=theta_free_prev)
        theta_ridge = fit_svi(k, w, theta0=theta_ridge_prev,
                              lam=lamb, theta_prev=theta_ridge_prev)

        rows.append({"date": t, "F": F, "T": T, "n": len(k), "fwd_flag": flag,
                     **{f"{p}_free": v for p, v in
                        zip("a b rho m sigma".split(), theta_free)},
                     **{f"{p}_ridge": v for p, v in
                        zip("a b rho m sigma".split(), theta_ridge)},
                     "rmse_free": rmse_vol(k, w, T, theta_free),
                     "rmse_ridge": rmse_vol(k, w, T, theta_ridge)})
        theta_free_prev, theta_ridge_prev = theta_free, theta_ridge

    return pd.DataFrame(rows).set_index("date")


def plot_parameter_paths(res, lamb):
    params = "a b rho m sigma".split()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, p in zip(axes.ravel(), params):
        ax.plot(res.index, res[f"{p}_free"], "o-", ms=2.5, lw=0.8,
                alpha=0.7, label="unpenalised")
        ax.plot(res.index, res[f"{p}_ridge"], "o-", ms=2.5, lw=0.8,
                alpha=0.85, label=f"ridge (λ={lamb})")
        ax.set_title(p)
        ax.legend(fontsize=8)
    ax = axes.ravel()[-1]
    ax.plot(res.index, res["rmse_free"] * 1e4, label="unpenalised")
    ax.plot(res.index, res["rmse_ridge"] * 1e4, label="ridge")
    ax.set(title="fit RMSE (vol bps)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("figures/svi_parameter_paths.png", dpi=300)
    plt.close()


def stability_report(res):
    params = "a b rho m sigma".split()
    out = {}
    for p in params:
        out[p] = {"std_dtheta_free": res[f"{p}_free"].diff().std(),
                  "std_dtheta_ridge": res[f"{p}_ridge"].diff().std()}
    rep = pd.DataFrame(out).T
    rep["reduction_%"] = 100 * (1 - rep["std_dtheta_ridge"]
                                / rep["std_dtheta_free"])
    print("\n Parameter-change volatility (std of day-on-day diffs)")
    print(rep.round(4))
    print(f"\nMean RMSE  unpenalised: {res['rmse_free'].mean()*1e4:.1f} bps"
          f" | ridge: {res['rmse_ridge'].mean()*1e4:.1f} bps")



if __name__ == "__main__":
    lamb = 1e-5
    df = load_and_clean()
    demo_one_date(df)
    res = fit_all_dates(df, lamb)
    plot_parameter_paths(res, lamb)
    stability_report(res)
    res.to_csv("output/svi_results_all_dates.csv")