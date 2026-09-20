"""Goodness-of-fit statistics and nested-model comparison for flow-curve fits.

Everything here is model-agnostic given predictions, except the nested-model
comparison, which fits the Newtonian, power-law and Bingham special cases of
the Herschel-Bulkley model and tests whether the extra parameters are
justified.

Conventions
-----------
``m`` data points, ``p`` fitted parameters, residuals ``r_i = f(x_i) - y_i``,
``S = sum r_i^2``.  Information criteria use the Gaussian likelihood with the
error variance profiled out, ``m ln(S/m) + const``; constants cancel in
differences between models fitted to the same data, which is the only way
they should be used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GoodnessOfFit:
    m: int
    p: int
    S: float
    rmse: float
    """``E = sqrt(S/m)``, the paper's error measure."""
    residual_std_error: float
    """``s = sqrt(S/(m-p))``: unbiased estimate of the noise standard deviation."""
    mae: float
    max_abs_residual: float
    mape: float
    """Mean absolute percentage error, % of the measured stress."""
    r_squared: float
    adjusted_r_squared: float
    aic: float
    aicc: float
    bic: float
    durbin_watson: float
    """Near 2 for independent residuals; well below 2 indicates systematic
    misfit (runs of same-sign residuals), i.e. the model shape is wrong."""
    residuals: FloatArray = field(repr=False)
    relative_residuals_pct: FloatArray = field(repr=False)
    covariance: FloatArray | None = field(default=None, repr=False)
    correlation: FloatArray | None = field(default=None, repr=False)
    std_errors: FloatArray | None = None
    ci95_half_width: FloatArray | None = None
    """Half-widths of two-sided 95% confidence intervals, t-distribution with
    ``m-p`` degrees of freedom, from the linearised covariance."""

    def as_dict(self) -> dict:
        out = {
            "m": self.m, "p": self.p, "S": self.S, "rmse": self.rmse,
            "residual_std_error": self.residual_std_error, "mae": self.mae,
            "max_abs_residual": self.max_abs_residual, "mape_pct": self.mape,
            "r_squared": self.r_squared, "adjusted_r_squared": self.adjusted_r_squared,
            "aic": self.aic, "aicc": self.aicc, "bic": self.bic, "durbin_watson": self.durbin_watson,
        }
        if self.std_errors is not None:
            out["std_errors"] = self.std_errors.tolist()
            out["ci95_half_width"] = self.ci95_half_width.tolist()
            out["correlation"] = self.correlation.tolist()
        return out


def information_criteria(S: float, m: int, p: int) -> tuple[float, float, float]:
    """Gaussian AIC, AICc and BIC with the variance profiled out."""

    if S <= 0.0:
        return float("-inf"), float("-inf"), float("-inf")
    ll_term = m * np.log(S / m)
    aic = ll_term + 2.0 * p
    aicc = aic + (2.0 * p * (p + 1.0)) / (m - p - 1.0) if m - p - 1 > 0 else float("inf")
    bic = ll_term + p * np.log(m)
    return float(aic), float(aicc), float(bic)


def goodness_of_fit(
    x: ArrayLike,
    y: ArrayLike,
    predicted: ArrayLike,
    *,
    p: int,
    jacobian: FloatArray | None = None,
) -> GoodnessOfFit:
    """Compute the statistics for a fitted curve.

    ``jacobian`` (m x p, derivatives of the prediction with respect to the
    parameters) enables the covariance, standard errors and confidence
    intervals; pass ``None`` to skip them.
    """

    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    fs = np.asarray(predicted, dtype=float).reshape(-1)
    m = xs.size
    r = fs - ys
    S = float(np.dot(r, r))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - S / ss_tot if ss_tot > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (m - 1) / (m - p) if m > p and ss_tot > 0 else float("nan")
    aic, aicc, bic = information_criteria(S, m, p)
    dw = float(np.sum(np.diff(r) ** 2) / S) if S > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * r / ys
    mape = float(np.mean(np.abs(rel[np.isfinite(rel)]))) if np.any(np.isfinite(rel)) else float("nan")
    s2 = S / (m - p) if m > p else float("nan")

    cov = corr = se = ci = None
    if jacobian is not None and m > p:
        J = np.asarray(jacobian, dtype=float)
        try:
            cov = s2 * np.linalg.inv(J.T @ J)
            d = np.sqrt(np.diag(cov))
            if np.all(np.isfinite(d)) and np.all(d > 0):
                se = d
                corr = cov / np.outer(d, d)
                ci = float(stats.t.ppf(0.975, m - p)) * d
            else:
                cov = None
        except np.linalg.LinAlgError:
            cov = None
    return GoodnessOfFit(
        m=m, p=p, S=S, rmse=float(np.sqrt(S / m)), residual_std_error=float(np.sqrt(s2)),
        mae=float(np.mean(np.abs(r))), max_abs_residual=float(np.max(np.abs(r))), mape=mape,
        r_squared=r2, adjusted_r_squared=adj, aic=aic, aicc=aicc, bic=bic, durbin_watson=dw,
        residuals=r, relative_residuals_pct=rel, covariance=cov, correlation=corr,
        std_errors=se, ci95_half_width=ci,
    )


# --------------------------------------------------------------------------- #
# Nested models: Newtonian (K x), power law (K x^n), Bingham (y0 + K x)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelFit:
    name: str
    parameters: dict[str, float]
    p: int
    S: float
    rmse: float
    r_squared: float
    aicc: float
    bic: float
    predict: Callable[[FloatArray], FloatArray] = field(repr=False, compare=False)


def _linear_fit(columns: list[FloatArray], y: FloatArray) -> FloatArray:
    A = np.column_stack(columns)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_newtonian(x: FloatArray, y: FloatArray) -> ModelFit:
    (eta,) = _linear_fit([x], y)
    f = lambda xx: eta * np.asarray(xx, dtype=float)
    return _make("Newtonian  y = eta x", {"eta": float(eta)}, 1, x, y, f)


def fit_bingham(x: FloatArray, y: FloatArray) -> ModelFit:
    y0, K = _linear_fit([np.ones_like(x), x], y)
    f = lambda xx: y0 + K * np.asarray(xx, dtype=float)
    return _make("Bingham  y = y0 + K x", {"y0": float(y0), "K": float(K)}, 2, x, y, f)


def fit_power_law(x: FloatArray, y: FloatArray, n_max: float = 5.0) -> ModelFit:
    """Least squares on the original scale (not the log-log regression)."""

    def S_of_n(n: float) -> float:
        xn = x**n
        K = float(np.dot(xn, y) / np.dot(xn, xn))
        r = K * xn - y
        return float(np.dot(r, r))

    grid = np.linspace(0.01, n_max, 500)
    k = int(np.argmin([S_of_n(v) for v in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = minimize_scalar(S_of_n, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    n = float(res.x)
    xn = x**n
    K = float(np.dot(xn, y) / np.dot(xn, xn))
    f = lambda xx: K * np.asarray(xx, dtype=float) ** n
    return _make("Power law  y = K x^n", {"K": K, "n": n}, 2, x, y, f)


def _make(name: str, params: dict[str, float], p: int, x: FloatArray, y: FloatArray, f) -> ModelFit:
    r = f(x) - y
    S = float(np.dot(r, r))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    _, aicc, bic = information_criteria(S, x.size, p)
    return ModelFit(name, params, p, S, float(np.sqrt(S / x.size)), 1.0 - S / ss_tot if ss_tot > 0 else float("nan"), aicc, bic, f)


@dataclass(frozen=True)
class NestedTest:
    reduced: str
    full: str
    F: float
    df1: int
    df2: int
    p_value: float

    @property
    def extra_parameters_justified(self) -> bool:
        return self.p_value < 0.05


def extra_sum_of_squares_test(S_reduced: float, p_reduced: int, S_full: float, p_full: int, m: int,
                              *, reduced: str = "reduced", full: str = "full") -> NestedTest:
    """F-test of a nested (reduced) model against the full model."""

    df1, df2 = p_full - p_reduced, m - p_full
    if df1 <= 0 or df2 <= 0 or S_full <= 0:
        return NestedTest(reduced, full, float("nan"), df1, df2, float("nan"))
    F = ((S_reduced - S_full) / df1) / (S_full / df2)
    return NestedTest(reduced, full, float(F), df1, df2, float(stats.f.sf(F, df1, df2)))


@dataclass(frozen=True)
class ModelComparison:
    models: list[ModelFit]
    tests: list[NestedTest]

    def table(self) -> str:
        best = min(self.models, key=lambda mf: mf.aicc)
        lines = [f"{'model':<26}{'p':>3}{'S':>13}{'RMSE':>10}{'R^2':>10}{'AICc':>10}{'BIC':>10}  parameters"]
        for mf in self.models:
            mark = " <- lowest AICc" if mf is best else ""
            pars = ", ".join(f"{k}={v:.4g}" for k, v in mf.parameters.items())
            lines.append(f"{mf.name:<26}{mf.p:>3}{mf.S:>13.5g}{mf.rmse:>10.4g}{mf.r_squared:>10.5f}{mf.aicc:>10.2f}{mf.bic:>10.2f}  {pars}{mark}")
        for t in self.tests:
            verdict = "justified" if t.extra_parameters_justified else "NOT justified at the 5% level"
            lines.append(f"  F-test {t.reduced} -> {t.full}: F({t.df1},{t.df2}) = {t.F:.3g}, p = {t.p_value:.3g}: extra parameter(s) {verdict}")
        return "\n".join(lines)


def compare_nested_models(x: ArrayLike, y: ArrayLike, hb_predict: Callable[[FloatArray], FloatArray],
                          hb_params: dict[str, float]) -> ModelComparison:
    """Fit the three special cases and test them against the Herschel-Bulkley fit."""

    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    hb = _make("Herschel-Bulkley  y = y0 + K x^n", hb_params, 3, xs, ys, hb_predict)
    newt, power, bing = fit_newtonian(xs, ys), fit_power_law(xs, ys), fit_bingham(xs, ys)
    m = xs.size
    tests = [
        extra_sum_of_squares_test(bing.S, 2, hb.S, 3, m, reduced="Bingham (n=1)", full="Herschel-Bulkley"),
        extra_sum_of_squares_test(power.S, 2, hb.S, 3, m, reduced="power law (y0=0)", full="Herschel-Bulkley"),
        extra_sum_of_squares_test(newt.S, 1, hb.S, 3, m, reduced="Newtonian", full="Herschel-Bulkley"),
    ]
    return ModelComparison([newt, power, bing, hb], tests)
