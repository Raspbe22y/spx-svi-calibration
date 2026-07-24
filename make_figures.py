"""
Report figures and LaTeX tables for the SVI calibration project.

Pipeline: sequential SVI fits across all snapshots in seven parallel tracks
(lambda = 0 to 1e-2), event detection from the forward-return proxy with
ATM-vol corroboration, and generation of every figure, table and numeric
macro referenced by report.tex.

Figures: fig_returns.png, fig_smile_{warmup,middle,good,event}.png,
         fig_params_full.png, fig_params_event.png, fig_lambda_tradeoff.png
Tables:  tab_params.tex, tab_lambda.tex
Macros:  report_numbers.tex
Data:    svi_report_results.csv

Requires svi_fit.py (with the continuity-constrained, sub-grid forward
estimator and an `if __name__ == "__main__":` guard) in the same directory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svi_fit import load_and_clean, prepare_smile, fit_svi, svi_w, rmse_vol

# Settings
lamb = 1e-5
sweep = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# Multistart budget for every track
n_starts = 8
# Ridge scaling (matches fit_svi)
scale = np.array([0.05, 0.5, 0.5, 0.2, 0.2])
params = "a b rho m sigma".split()
labels = {"a": "level", "b": "slope", "rho": "skew",
          "m": "shift", "sigma": "curvature"}
notes = {0.0: "baseline (unpenalised)", 1e-7: "penalty inert",
            1e-6: "mild smoothing", 1e-5: "selected: optimum on both axes",
            1e-4: "smoother, small fit cost", 1e-3: "over-smoothed",
            1e-2: "fit degraded"}

tag = lambda lam: f"{lam:g}"
fmt_date = lambda t: pd.Timestamp(t).strftime("%d %B %Y")


# Fitting
def run_pipeline(df, lams=sweep):
    """Sequential fits, one independent track per lambda, equal multistart
    budgets so sweep rows are directly comparable. Each track warm-starts and
    (if lam > 0) ridge-anchors on its own previous-day solution."""
    dates = sorted(df["date"].unique())
    prev = {lam: None for lam in lams}
    store, F_prev = {}, None
    for t in dates:
        prepared = prepare_smile(df[df["date"] == t], F_prev=F_prev)
        if prepared is None:
            continue
        F, k, w, T, flag = prepared
        F_prev = F
        order = np.argsort(k)
        rec = dict(F=F, k=k, w=w, T=T, flag=flag,
                   atm_iv=float(np.interp(0.0, k[order],
                                          np.sqrt(w[order] / T))))
        for lam in lams:
            th = fit_svi(k, w, theta0=prev[lam], lam=lam,
                         theta_prev=prev[lam], n_starts=n_starts)
            prev[lam] = th
            rec[f"th_{tag(lam)}"] = th
            rec[f"rm_{tag(lam)}"] = rmse_vol(k, w, T, th)
        store[t] = rec
    return store


def as_frame(store, lams=sweep):
    rows = []
    for t, s in store.items():
        r = {"date": t, "F": s["F"], "T": s["T"], "n": len(s["k"]),
             "atm_iv": s["atm_iv"], "fwd_flag": bool(s["flag"])}
        for lam in lams:
            for p, v in zip(params, s[f"th_{tag(lam)}"]):
                r[f"{p}_{tag(lam)}"] = v
            r[f"rmse_{tag(lam)}"] = s[f"rm_{tag(lam)}"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("date").sort_index()


# Events
def returns_and_events(res, z=2.5, atm_z=1.0, max_events=4):
    """Daily log returns of the estimated forward, plus event dates.

    A date counts as a market event only when a large forward return is
    corroborated by a large ATM implied-volatility change: a forward move with
    no vol repricing is more likely an estimation artefact than a real index
    move. Dates with flagged (discontinuous) forward estimates are excluded
    from candidacy unless that would exclude nearly everything. Degrades
    gracefully: if no corroborated event exists, falls back first to large
    returns alone, then to the largest ATM-vol moves; `basis` records which."""
    ret = np.log(res["F"] / res["F"].shift(1)).dropna()
    d_atm = res["atm_iv"].diff().dropna()

    flagged = res["fwd_flag"].reindex(ret.index).fillna(False).values
    ok = ~flagged
    if ok.sum() < 10:
        ok = np.ones(len(ret), dtype=bool)

    cand = ret[ok]
    thresh = z * cand.std()
    atm_thresh = atm_z * d_atm.std()

    big = cand[cand.abs() > thresh]
    corroborated = [t for t in big.index if abs(d_atm.get(t, 0.0)) > atm_thresh]

    if corroborated:
        events = list(big.loc[corroborated].abs()
                      .sort_values(ascending=False).index[:max_events])
        basis = "return and ATM-vol corroborated"
    elif len(big):
        events = list(big.abs().sort_values(ascending=False).index[:max_events])
        basis = "large return only (no ATM-vol corroboration)"
    else:
        events = list(d_atm.abs().sort_values(ascending=False)
                      .index[:max_events])
        basis = "largest ATM implied-volatility moves"

    print(f"Event basis: {basis}")
    return ret, d_atm, events, thresh, int(flagged.sum()), basis


# Figures
def plot_returns(res, ret, events, thresh, fname="figures/fig_returns.png"):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(res.index, res["F"], lw=1.2)
    flg = res["fwd_flag"]
    ax[0].scatter(res.index[flg], res.loc[flg, "F"], color="r", s=18,
                  zorder=3, label="flagged forward estimate")
    ax[0].set_title("Estimated SPX forward (continuity-constrained)")
    ax[0].legend(fontsize=8)

    ax[1].bar(ret.index, ret.values * 100, width=1.0)
    for s in (+1, -1):
        ax[1].axhline(s * thresh * 100, color="r", ls="--", lw=0.8)
    ax[1].set_title(f"Daily log return of forward (%), "
                    rf"threshold ±{thresh*100:.2f}% (2.5$\sigma$)")

    ax[2].plot(res.index, res["atm_iv"] * 100, lw=1.2, color="tab:green")
    ax[2].set_title("ATM implied volatility (%) — corroborating series")

    for t in events:
        for a in ax:
            a.axvline(t, color="r", alpha=0.35, lw=1)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def plot_smile(t, s, fname, extra="", show_ridge=True):
    k, w, T = s["k"], s["w"], s["T"]
    kg = np.linspace(k.min(), k.max(), 250)
    plt.figure(figsize=(7.2, 4.6))
    plt.scatter(k, np.sqrt(w / T) * 100, s=13, alpha=0.55, label="market IV")
    plt.plot(kg, np.sqrt(svi_w(kg, s["th_0"]) / T) * 100, "r-", lw=2,
             label=f"SVI free ({s['rm_0']*1e4:.0f} bps)")
    if show_ridge:
        plt.plot(kg, np.sqrt(svi_w(kg, s[f"th_{tag(lamb)}"]) / T) * 100,
                 "g--", lw=1.8,
                 label=f"SVI ridge ({s[f'rm_{tag(lamb)}']*1e4:.0f} bps)")
    plt.xlabel("log-moneyness k")
    plt.ylabel("implied vol (%)")
    plt.title(f"{pd.Timestamp(t).date()}  (T={T:.2f}y){extra}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def mask_outliers(res, tags, window=11, nmad=6.0):
    res = res.copy()
    for tg in tags:
        cols = [f"{p}_{tg}" for p in params]
        bad = pd.Series(False, index=res.index)
        for c in cols:
            med = res[c].rolling(window, center=True, min_periods=3).median()
            mad = (res[c] - med).abs().rolling(window, center=True,
                                               min_periods=3).median()
            floor = 0.05 * res[c].std()
            bad |= (res[c] - med).abs() > nmad * np.maximum(mad, floor)
        res.loc[bad, cols] = np.nan
        print(f"lambda={tg}: masked {int(bad.sum())} outlier fit(s) "
              "for plotting")
    return res


def robust_ylim(cols, pad=0.12):
    allv = np.concatenate([c.dropna().values for c in cols])
    lo, hi = allv.min(), allv.max()
    span = hi - lo if hi > lo else abs(hi) + 1e-6
    return lo - pad * span, hi + pad * span


def plot_params_full(res, events, fname="figures/fig_params_full.png"):
    m = mask_outliers(res, ["0", tag(lamb)])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, p in zip(axes.ravel(), params):
        f, r = m[f"{p}_0"], m[f"{p}_{tag(lamb)}"]
        raw_f = res[f"{p}_0"]
        for s_masked, lab, col in [(f, "free", "tab:blue"),
                                   (r, f"ridge λ={lamb:g}", "tab:orange")]:
            v = s_masked.dropna()                      # line bridges the gaps
            ax.plot(v.index, v.values, "o-", ms=2.2, lw=0.8,
                    alpha=0.8, color=col, label=lab)
        gaps = f.isna()                                # masked free-fit days
        if gaps.any():
            ax.plot(res.index[gaps], raw_f[gaps].clip(*robust_ylim([f, r])),
                    "x", ms=4, color="tab:red", alpha=0.6,
                    label="masked (off-trend fit)")
        ax.set_ylim(*robust_ylim([f, r]))
        for t in events:
            ax.axvline(t, color="r", alpha=0.25, lw=0.8)
        ax.set_title(f"{p} ({labels[p]})", fontsize=10)
        ax.legend(fontsize=7)
    ax = axes.ravel()[-1]
    ax.plot(res.index, res["rmse_0"] * 1e4, lw=1, label="free")
    ax.plot(res.index, res[f"rmse_{tag(lamb)}"] * 1e4, lw=1, label="ridge")
    ax.set_title("fit RMSE (vol bps)", fontsize=10)
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def plot_params_event(res, event, span=12, fname="figures/fig_params_event.png"):
    i = list(res.index).index(event)
    win = res.iloc[max(0, i - span): i + span + 1]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for ax, p in zip(axes.ravel(), params):
        ax.plot(win.index, win[f"{p}_0"], "o-", ms=3, lw=1, label="free")
        ax.plot(win.index, win[f"{p}_{tag(lamb)}"], "o-", ms=3, lw=1,
                label=f"ridge λ={lamb:g}")
        ax.axvline(event, color="r", alpha=0.4)
        ax.set_title(f"{p} ({labels[p]})", fontsize=10)
        ax.legend(fontsize=7)
    ax = axes.ravel()[-1]
    ax.plot(win.index, win["rmse_0"] * 1e4, label="free")
    ax.plot(win.index, win[f"rmse_{tag(lamb)}"] * 1e4, label="ridge")
    ax.axvline(event, color="r", alpha=0.4)
    ax.set_title("fit RMSE (vol bps)", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle(f"Parameter paths around {pd.Timestamp(event).date()}")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def plot_lambda_tradeoff(res, lams=sweep, fname="figures/fig_lambda_tradeoff.png"):
    rough, err = [], []
    for lam in lams:
        th = res[[f"{p}_{tag(lam)}" for p in params]].values
        rough.append(float(np.mean(np.sqrt(
            ((np.diff(th, axis=0) / scale) ** 2).sum(axis=1)))))
        err.append(float(res[f"rmse_{tag(lam)}"].mean() * 1e4))
    plt.figure(figsize=(7, 4.8))
    plt.plot(rough, err, "o-")
    for x, y, lam in zip(rough, err, lams):
        plt.annotate(f"λ={lam:g}", (x, y), fontsize=8,
                     xytext=(6, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel(r"mean scaled $\|\delta\theta\|$  (parameter path roughness)")
    plt.ylabel("mean fit RMSE (vol bps)")
    plt.title("Stability-accuracy trade-off across λ")
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()
    return pd.DataFrame({"lam": lams, "rough": rough, "rmse": err})


# Tables
def write_tables(res, sweep):
    rows = []
    for p in params:
        f, r = res[f"{p}_0"], res[f"{p}_{tag(lamb)}"]
        rows.append(f"${p}$ ({labels[p]}) & {f.mean():.4f} & {f.min():.4f} & "
                    f"{f.max():.4f} & {f.diff().std():.4f} & "
                    f"{r.mean():.4f} & {r.min():.4f} & {r.max():.4f} & "
                    f"{r.diff().std():.4f} \\\\")
    with open("output/tab_params.tex", "w") as fh:
        fh.write("\n".join(rows) + "\n")

    lines = [f"${row['lam']:g}$ & {row['rmse']:.1f} & {row['rough']:.4f} & "
             f"{notes.get(row['lam'], '')} \\\\"
             for _, row in sweep.iterrows()]
    with open("output/tab_lambda.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_numbers(res, store, ret, events, picks, thresh, n_flagged, basis):
    d_warm, d_mid, d_good, d_ev = picks
    g = lambda t, c: res.loc[t, c]

    # Residuals by moneyness region, pooled across dates
    core, wing = [], []
    for t, s in store.items():
        th = s[f"th_{tag(lamb)}"]
        k, w, T = s["k"], s["w"], s["T"]
        r = np.sqrt(np.maximum(svi_w(k, th), 1e-12) / T) - np.sqrt(w / T)
        core.append(r[np.abs(k) <= 0.15])
        wing.append(r[np.abs(k) > 0.35])
    core_resid = float(np.sqrt(np.mean(np.concatenate(core) ** 2)))
    wing_resid = float(np.sqrt(np.mean(np.concatenate(wing) ** 2)))

    # Parameter-change volatility: calm vs event windows (ridge track, rho)
    dr = res[f"rho_{tag(lamb)}"].diff().abs()
    near_event = pd.Series(False, index=res.index)
    for t in events:
        near_event |= ((res.index >= t - pd.Timedelta(days=5)) &
                       (res.index <= t + pd.Timedelta(days=5)))
    event_std = float(dr[near_event].std())
    calm_std = float(dr[~near_event].std())
    print(f"rho |Δ|: calm {calm_std:.4f} ({int((~near_event).sum())} days), "
          f"event {event_std:.4f} ({int(near_event.sum())} days)")

    lines = [
        r"\newcommand{\NDates}{%d}" % len(res),
        r"\newcommand{\NQuotes}{%d}" % int(res["n"].mean()),
        r"\newcommand{\NFlagged}{%d}" % n_flagged,
        r"\newcommand{\RetSigma}{%.2f}" % (ret.std() * 100),
        r"\newcommand{\EventThresh}{%.2f}" % (thresh * 100),
        r"\newcommand{\EventList}{%s}" % ", ".join(fmt_date(t)
                                                   for t in events),
        r"\newcommand{\EventBasis}{%s}" % basis,
        r"\newcommand{\DateWarm}{%s}" % fmt_date(d_warm),
        r"\newcommand{\DateMiddle}{%s}" % fmt_date(d_mid),
        r"\newcommand{\DateGood}{%s}" % fmt_date(d_good),
        r"\newcommand{\DateEvent}{%s}" % fmt_date(d_ev),
        r"\newcommand{\EventRet}{%.2f}" % (ret.get(d_ev, np.nan) * 100),
        r"\newcommand{\EventAtmChg}{%.2f}" % (
            res["atm_iv"].diff().get(d_ev, np.nan) * 100),
        r"\newcommand{\MeanRmseFree}{%.1f}" % (res["rmse_0"].mean() * 1e4),
        r"\newcommand{\MeanRmseRidge}{%.1f}" % (
            res[f"rmse_{tag(lamb)}"].mean() * 1e4),
        r"\newcommand{\RmseWarm}{%.1f}" % (g(d_warm, "rmse_0") * 1e4),
        r"\newcommand{\RmseGood}{%.1f}" % (g(d_good, "rmse_0") * 1e4),
        r"\newcommand{\RmseEventFree}{%.1f}" % (g(d_ev, "rmse_0") * 1e4),
        r"\newcommand{\RmseEventRidge}{%.1f}" % (
            g(d_ev, f"rmse_{tag(lamb)}") * 1e4),
        r"\newcommand{\RhoStdFree}{%.4f}" % res["rho_0"].diff().std(),
        r"\newcommand{\RhoStdRidge}{%.4f}" % (
            res[f"rho_{tag(lamb)}"].diff().std()),
        r"\newcommand{\SigStdFree}{%.4f}" % res["sigma_0"].diff().std(),
        r"\newcommand{\SigStdRidge}{%.4f}" % (
            res[f"sigma_{tag(lamb)}"].diff().std()),
        r"\newcommand{\LamBest}{10^{-5}}",
        r"\newcommand{\ResidCore}{%.1f}" % (core_resid * 1e4),
        r"\newcommand{\ResidWing}{%.1f}" % (wing_resid * 1e4),
        r"\newcommand{\TMin}{%.2f}" % res["T"].min(),
        r"\newcommand{\TMax}{%.2f}" % res["T"].max(),
        r"\newcommand{\RhoMean}{%.3f}" % res[f"rho_{tag(lamb)}"].mean(),
        r"\newcommand{\CalmStd}{%.4f}" % calm_std,
        r"\newcommand{\EventStd}{%.4f}" % event_std,
    ]
    with open("report_numbers.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


# Main
if __name__ == "__main__":
    print(f"lamb = {lamb:g}   sweep = {sweep}   n_starts = {n_starts}")
    df = load_and_clean()
    store = run_pipeline(df)
    res = as_frame(store)

    # Forward diagnostics
    fr = np.log(res["F"] / res["F"].shift(1)).dropna()
    print(f"\nForward diagnostics: "
          f"{int(res['fwd_flag'].sum())}/{len(res)} flagged")
    print(f"  |log return| — median {fr.abs().median()*100:.2f}%  "
          f"90th pct {fr.abs().quantile(0.9)*100:.2f}%  "
          f"max {fr.abs().max()*100:.2f}%")
    day0 = df[df["date"] == df["date"].min()]
    dd = (day0[day0["type"] == "c"].set_index("strike")["iv"]
          - day0[day0["type"] == "p"].set_index("strike")["iv"]).dropna()
    print(f"IV difference on first date: min {dd.min():.4f}  "
          f"max {dd.max():.4f}  sign changes: "
          f"{int((np.sign(dd.values[:-1]) * np.sign(dd.values[1:]) < 0).sum())}")

    # Events and representative dates
    ret, d_atm, events, thresh, n_flagged, basis = returns_and_events(res)
    dates = list(res.index)
    d_warm = dates[0]
    d_mid = dates[len(dates) // 2]

    excl = set(events) | set(res.index[res["fwd_flag"]])
    candidates = [t for t in dates if t not in excl]
    if not candidates:
        print("WARNING: exclusion set covers all dates; "
              "relaxing to non-event dates only.")
        candidates = [t for t in dates if t not in set(events)] or dates
    d_good = min(candidates, key=lambda t: res.loc[t, "rmse_0"])
    d_ev = events[0] if events else max(dates,
                                        key=lambda t: res.loc[t, "rmse_0"])

    # Figures
    plot_returns(res, ret, events, thresh)
    plot_smile(d_warm, store[d_warm], "figures/fig_smile_warmup.png",
               "  —  cold start, no history", show_ridge=False)
    plot_smile(d_mid, store[d_mid], "figures/fig_smile_middle.png", "  —  mid-sample")
    plot_smile(d_good, store[d_good], "figures/fig_smile_good.png",
               "  —  best fit (non-event dates)")
    plot_smile(d_ev, store[d_ev], "figures/fig_smile_event.png",
               "  —  largest corroborated market event")
    plot_params_full(res, events)
    plot_params_event(res, d_ev)
    sweep = plot_lambda_tradeoff(res)

    # Tables
    write_tables(res, sweep)
    write_numbers(res, store, ret, events, (d_warm, d_mid, d_good, d_ev),
                  thresh, n_flagged, basis)
    res.to_csv("output/svi_report_results.csv")

    # Summary
    print("\nForward estimates flagged:", n_flagged)
    print("Return sigma: %.2f%%   threshold: %.2f%%"
          % (ret.std() * 100, thresh * 100))
    print("Events:", [str(pd.Timestamp(t).date()) for t in events])
    print("Best-fit date:", pd.Timestamp(d_good).date())
    print(sweep.to_string(index=False))