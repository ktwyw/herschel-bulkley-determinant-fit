"""Reproduce Figures 1-5 of Mullineux (2008) and a synthetic-data demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    determinant,
    fit_herschel_bulkley,
    fit_with_yield_estimate,
    herschel_bulkley,
    refine_least_squares,
)
from .metrics import compare_nested_models
from .data import (
    PAPER_K,
    PAPER_N,
    PAPER_Y0,
    chlorine_as_hb,
    paper_curve_points,
    synthetic_data,
)


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_1(out: Path, dpi: int) -> None:
    """Typical Herschel-Bulkley curve, against x and against ln x."""

    x = np.linspace(0.5, 500.0, 600)
    y = herschel_bulkley(x, PAPER_Y0, PAPER_K, PAPER_N)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(x, y, lw=2)
    a.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)")
    a.set_ylabel("Shear stress y (Pa)")
    a.set_xlim(0, 500)
    a.set_ylim(0, 42)
    b.plot(np.log(x), y, lw=2)
    b.set_xlabel(r"Natural log of shear rate, $\ln\dot{\gamma}$")
    b.set_ylabel("Shear stress y (Pa)")
    b.set_ylim(0, 42)
    fig.suptitle(f"Figure 1: Herschel-Bulkley curve, y0={PAPER_Y0:g} Pa, K={PAPER_K:g} Pa s^n, n={PAPER_N:g}")
    _save(fig, out / "figure_01_hb_curve.png", dpi)


def figure_2(out: Path, dpi: int) -> dict:
    """Sensitivity of K and n to the yield-stress estimate (Section 2)."""

    x_all, y_all = paper_curve_points()
    starts = (5, 10, 20, 40, 80)
    deltas = np.linspace(-5.0, 5.0, 201)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    summary = {}
    for k, x_start in enumerate(starts):
        keep = x_all >= x_start
        x, y = x_all[keep], y_all[keep]
        Ks, ns, Es = [], [], []
        for d in deltas:
            f = fit_with_yield_estimate(x, y, PAPER_Y0 + d)
            Ks.append(f.K)
            ns.append(f.n)
            Es.append(f.rmse)
        color = cmap(k / (len(starts) - 1))
        ax.plot(deltas, Ks, color=color, lw=1.6)
        ax.plot(deltas, 10 * np.array(ns), color=color, lw=1.6, ls="--")
        ax.plot(deltas, 10 * np.array(Es), color=color, lw=1.6, ls=":")
        ax.text(5.05, Ks[-1], f"{x_start}", color=color, fontsize=8, va="center")
        ax.text(5.05, 10 * ns[-1], f"{x_start}", color=color, fontsize=8, va="center")
        ax.text(-5.6, 10 * Es[0], f"{x_start}", color=color, fontsize=8, va="center", ha="right")
        # error of a +1 Pa mis-estimate, as quoted in the paper (~12.5% in K, ~6% in n)
        f1 = fit_with_yield_estimate(x, y, PAPER_Y0 + 1.0)
        summary[f"x_start_{x_start}"] = {
            "points": int(x.size),
            "K_error_pct_for_dy0_plus1": 100.0 * (f1.K / PAPER_K - 1.0),
            "n_error_pct_for_dy0_plus1": 100.0 * (f1.n / PAPER_N - 1.0),
        }
    ax.plot([], [], color="k", lw=1.6, label="K")
    ax.plot([], [], color="k", lw=1.6, ls="--", label="10 n")
    ax.plot([], [], color="k", lw=1.6, ls=":", label="10 E,  E = sqrt(S/m)")
    ax.set_xlabel(r"$\Delta y_0$ (Pa): error in the estimated yield stress")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 8.5)
    ax.set_title("Figure 2: sensitivity of the yield-estimate method\n(labels: lowest x-value kept in the data set)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    _save(fig, out / "figure_02_yield_estimate_sensitivity.png", dpi)
    return summary


def figures_3_4(out: Path, dpi: int) -> dict:
    """The scaled determinant Fbar(n) for the paper's 80-point data set."""

    x, y = paper_curve_points()
    fit = fit_herschel_bulkley(x, y)
    for name, n_max, ylim in (("figure_03_determinant_wide.png", 20.0, (-230, 30)),
                              ("figure_04_determinant_zoom.png", 0.5, (-40, 12))):
        n = np.linspace(0.0, n_max, 2000)
        F = determinant(n, x, y)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(n, F, lw=1.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.plot([fit.n], [0], "o", color="tab:red", label=f"root n = {fit.n:.4f}")
        ax.set_xlabel("n")
        ax.set_ylabel(r"Determinant $\bar F(n)$")
        ax.set_xlim(0, n_max)
        ax.set_ylim(*ylim)
        ax.set_title(("Figure 3" if n_max > 1 else "Figure 4") + r": $\bar F(n)$, X = 400, Y = 37.57")
        ax.legend()
        ax.grid(alpha=0.2)
        _save(fig, out / name, dpi)
    return {"root_n": fit.n, "y0": fit.y0, "K": fit.K, "Fbar_min": float(np.min(determinant(np.linspace(0.01, 20, 4000), x, y)))}


def figure_5(out: Path, dpi: int) -> dict:
    """The 'Chlorine' example of Section 7: data, fitted curve and F(n)."""

    x, y = chlorine_as_hb()
    fit = fit_herschel_bulkley(x, y, n_max=1.4)
    a, b = fit.y0, -fit.K / fit.y0
    n = np.linspace(0.0, 1.4, 1500)
    F = determinant(n, x, y, scale=False)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 8))
    top.plot(x, y, "o", ms=4, label="data (Bates & Watts 'Chlorine')")
    xx = np.linspace(x.min(), x.max(), 300)
    top.plot(xx, fit(xx), lw=2, color="tab:red", label=f"a = {a:.2f}, b = {b:.3f}, n = {fit.n:.3f}")
    top.set_xlabel(r"$x = e^{-t}$")
    top.set_ylabel("Concentration y (%)")
    top.set_ylim(0, 42)
    top.legend()
    top.grid(alpha=0.2)
    bottom.plot(n, F, lw=1.8)
    bottom.axhline(0, color="k", lw=0.8)
    bottom.plot([fit.n], [0], "o", color="tab:red")
    bottom.set_xlabel("n")
    bottom.set_ylabel("F(n)")
    bottom.set_xlim(0, 1.4)
    bottom.set_ylim(-15, 55)
    bottom.grid(alpha=0.2)
    fig.suptitle("Figure 5: 'Chlorine' example, y = a(1 - b e^{-nt})")
    _save(fig, out / "figure_05_chlorine_example.png", dpi)
    return {"a": a, "b": b, "n": fit.n, "paper_values": {"a": 39.09, "b": 0.828, "n": 0.159}}


def synthetic_demo(out: Path, dpi: int, *, seed: int = 0, noise: float = 0.03) -> dict:
    """Noisy synthetic flow curve: Mullineux fit vs. yield-estimate vs. LM."""

    x, y = synthetic_data(seed=seed, noise=noise)
    fit = fit_herschel_bulkley(x, y)
    lm = refine_least_squares(x, y, fit.y0, fit.K, fit.n)
    # a plausible naive estimate: extrapolate the two lowest points to x=0
    y0_guess = float(y[0] - (y[1] - y[0]) * x[0] / (x[1] - x[0]))
    y0_guess = max(y0_guess, 0.0)
    naive = fit_with_yield_estimate(x, y, y0_guess) if np.all(y > y0_guess) else None

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    xx = np.geomspace(x.min(), x.max(), 300)
    a.plot(x, y, "o", ms=4, label=f"synthetic data, {noise:.0%} noise")
    a.plot(xx, herschel_bulkley(xx, 5, 4, 0.35), "k--", lw=1, label="true: y0=5, K=4, n=0.35")
    a.plot(xx, fit(xx), color="tab:red", lw=2,
           label=f"Mullineux: y0={fit.y0:.2f}, K={fit.K:.2f}, n={fit.n:.3f}")
    if naive is not None:
        a.plot(xx, herschel_bulkley(xx, naive.y0, naive.K, naive.n), color="tab:green", lw=1.5, ls=":",
               label=f"yield-estimate (y0={naive.y0:.2f}): K={naive.K:.2f}, n={naive.n:.3f}")
    a.set_xscale("log")
    a.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)")
    a.set_ylabel(r"Shear stress $\sigma$ (Pa)")
    a.legend(fontsize=8)
    a.grid(alpha=0.2, which="both")
    nn = np.linspace(0.0, 2.0, 1000)
    b.plot(nn, determinant(nn, x, y), lw=1.8)
    b.axhline(0, color="k", lw=0.8)
    b.plot([fit.n], [0], "o", color="tab:red", label=f"root n = {fit.n:.4f}")
    b.set_xlabel("n")
    b.set_ylabel(r"$\bar F(n)$")
    b.legend()
    b.grid(alpha=0.2)
    fig.suptitle("Synthetic demonstration: the root of the determinant gives n directly")
    _save(fig, out / "demo_synthetic_fit.png", dpi)

    # goodness-of-fit panel: residuals and the nested-model comparison
    g = fit.gof
    comparison = compare_nested_models(x, y, fit, {"y0": fit.y0, "K": fit.K, "n": fit.n})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    a.axhline(0, color="k", lw=0.8)
    a.plot(x, g.residuals, "o", ms=4)
    a.axhspan(-2 * g.residual_std_error, 2 * g.residual_std_error, color="tab:blue", alpha=0.08, label=r"$\pm 2s$")
    a.set_xscale("log")
    a.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)")
    a.set_ylabel(r"Residual $\sigma_{fit}-\sigma$ (Pa)")
    a.set_title(f"Residuals: s = {g.residual_std_error:.3g} Pa, MAPE = {g.mape:.2f}%, Durbin-Watson = {g.durbin_watson:.2f}", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.2, which="both")
    for mf in comparison.models:
        if mf.name.startswith("Newtonian"):
            continue
        b.plot(xx, mf.predict(xx), lw=1.8, label=f"{mf.name.split('  ')[0]}: RMSE {mf.rmse:.3g}, AICc {mf.aicc:.1f}")
    b.plot(x, y, "o", ms=3, color="k", alpha=0.5, label="data")
    b.set_xscale("log")
    b.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)")
    b.set_ylabel(r"Shear stress $\sigma$ (Pa)")
    tests = "; ".join(f"{t.reduced}: p = {t.p_value:.2g}" for t in comparison.tests[:2])
    b.set_title("Nested models (F-test against Herschel-Bulkley)\n" + tests, fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.2, which="both")
    _save(fig, out / "demo_goodness_of_fit.png", dpi)
    return {
        "true": {"y0": 5.0, "K": 4.0, "n": 0.35},
        "mullineux": {"y0": fit.y0, "K": fit.K, "n": fit.n, "S": fit.S, "std_errors": fit.standard_errors},
        "goodness_of_fit": g.as_dict(),
        "model_comparison": {
            "models": [{"name": mf.name, "p": mf.p, "S": mf.S, "rmse": mf.rmse, "r_squared": mf.r_squared, "aicc": mf.aicc, "bic": mf.bic, "parameters": mf.parameters} for mf in comparison.models],
            "f_tests": [{"reduced": t.reduced, "F": t.F, "df": [t.df1, t.df2], "p_value": t.p_value} for t in comparison.tests],
        },
        "levenberg_marquardt_from_mullineux": {"y0": lm.y0, "K": lm.K, "n": lm.n, "S": lm.S},
        "yield_estimate": None if naive is None else {"y0": naive.y0, "K": naive.K, "n": naive.n, "S": naive.S},
    }


def reproduce_all(output_directory: str | Path, *, dpi: int = 160) -> dict:
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    figure_1(out, dpi)
    metrics = {
        "figure_2": figure_2(out, dpi),
        "figures_3_4": figures_3_4(out, dpi),
        "figure_5": figure_5(out, dpi),
        "synthetic_demo": synthetic_demo(out, dpi),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
