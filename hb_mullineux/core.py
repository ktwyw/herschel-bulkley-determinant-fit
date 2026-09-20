"""Herschel-Bulkley fitting by the consistency-condition method of Mullineux.

Reference
---------
G. Mullineux, "Non-linear least squares fitting of coefficients in the
Herschel-Bulkley model", Applied Mathematical Modelling 32 (2008) 2538-2551,
doi:10.1016/j.apm.2007.09.010.

The model is ``y = y0 + K x**n`` (shear stress ``y`` against shear rate ``x``).
The normal equations of the least-squares problem are linear in ``y0`` and
``K`` with coefficients that depend on ``n``; the condition that they have a
non-trivial solution is that a 3x3 determinant ``F(n)`` vanishes (paper
Eq. 5).  ``n`` is therefore found first as a root of ``F``, after which ``y0``
and ``K`` follow from a 2x2 linear solve.  Following Eq. 6 the data are scaled
by ``X = max x`` and ``Y = max y`` so that the determinant is well
conditioned; the scaling does not move the roots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from .metrics import GoodnessOfFit, goodness_of_fit

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #
def _as_data(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length.")
    if x.size < 4:
        raise ValueError("At least four data points are needed to fit three parameters.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x and y must be finite.")
    if np.any(x <= 0.0):
        raise ValueError("Shear rates must be strictly positive (the model uses x**n and ln x).")
    order = np.argsort(x)
    return x[order], y[order]


def herschel_bulkley(x: ArrayLike, y0: float, K: float, n: float) -> FloatArray:
    """Evaluate ``y = y0 + K x**n``."""

    return y0 + K * np.power(np.asarray(x, dtype=float), n)


# --------------------------------------------------------------------------- #
# The determinant F(n)  (paper Eq. 5, scaled form Eq. 6)
# --------------------------------------------------------------------------- #
def determinant(
    n: ArrayLike, x: ArrayLike, y: ArrayLike, *, scale: bool = True
) -> FloatArray | float:
    """Evaluate the consistency determinant ``F(n)`` for one or many ``n``.

    With ``scale=True`` (default) the scaled determinant ``Fbar(n)`` of
    Eq. 6 is returned, computed from ``xbar = x/max(x)`` and
    ``ybar = y/max(y)``.  ``F`` and ``Fbar`` have the same roots.
    """

    xs, ys = _as_data(x, y)
    if scale:
        xs = xs / xs.max()
        ys = ys / np.abs(ys).max()
    lx = np.log(xs)
    nn = np.asarray(n, dtype=float)
    scalar = nn.ndim == 0
    nn = np.atleast_1d(nn)
    m = float(xs.size)

    # x**n as exp(n ln x): shape (len(n), m)
    xn = np.exp(np.outer(nn, lx))
    x2n = xn * xn
    s_xn = xn.sum(axis=1)
    s_xn_lx = (xn * lx).sum(axis=1)
    s_x2n = x2n.sum(axis=1)
    s_x2n_lx = (x2n * lx).sum(axis=1)
    s_y = float(ys.sum())
    s_xn_y = (xn * ys).sum(axis=1)
    s_xn_y_lx = (xn * ys * lx).sum(axis=1)

    # 3x3 determinant, rows: [m, Sx^n, Sx^n ln x], [Sx^n, Sx^2n, Sx^2n ln x],
    # [Sy, Sx^n y, Sx^n y ln x]
    a, b, c = m, s_xn, s_xn_lx
    d, e, f = s_xn, s_x2n, s_x2n_lx
    g, h, i = s_y, s_xn_y, s_xn_y_lx
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return float(det[0]) if scalar else det


# --------------------------------------------------------------------------- #
# Linear solve for y0 and K at fixed n
# --------------------------------------------------------------------------- #
def linear_parameters(n: float, x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Least-squares ``y0`` and ``K`` for a given ``n`` (2x2 normal equations).

    The solve is done on the scaled data and mapped back:
    ``y0 = Y*ybar0``, ``K = Y*Kbar / X**n``.
    """

    xs, ys = _as_data(x, y)
    X, Y = xs.max(), np.abs(ys).max()
    xb, yb = xs / X, ys / Y
    xn = xb**n
    A = np.array([[xb.size, xn.sum()], [xn.sum(), (xn * xn).sum()]])
    rhs = np.array([yb.sum(), (xn * yb).sum()])
    y0b, Kb = np.linalg.solve(A, rhs)
    return float(Y * y0b), float(Y * Kb / X**n)


def sum_of_squares(x: ArrayLike, y: ArrayLike, y0: float, K: float, n: float) -> float:
    xs, ys = _as_data(x, y)
    r = herschel_bulkley(xs, y0, K, n) - ys
    return float(np.dot(r, r))


# --------------------------------------------------------------------------- #
# Root search
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RootCandidate:
    n: float
    y0: float
    K: float
    S: float


def find_roots(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_min: float = 0.01,
    n_max: float = 5.0,
    grid_points: int = 1000,
) -> list[RootCandidate]:
    """All roots of ``Fbar(n)`` on ``[n_min, n_max]`` with their linear parameters.

    ``F(0)=0`` is a trivial root (the first two columns coincide), so the
    scan starts a little above zero.  Each sign change on the grid is
    refined by Brent's method; for each root the linear parameters and the
    residual sum of squares ``S`` are evaluated so the caller can pick the
    least-squares minimiser when more than one root exists.
    """

    if not (0.0 < n_min < n_max):
        raise ValueError("Need 0 < n_min < n_max.")
    xs, ys = _as_data(x, y)
    grid = np.linspace(n_min, n_max, grid_points)
    values = np.asarray(determinant(grid, xs, ys))
    roots: list[RootCandidate] = []
    for k in range(grid.size - 1):
        f0, f1 = values[k], values[k + 1]
        if f0 == 0.0:
            root = float(grid[k])
        elif f0 * f1 < 0.0:
            root = float(brentq(lambda nn: determinant(nn, xs, ys), grid[k], grid[k + 1],
                                xtol=1.0e-13, rtol=1.0e-13, maxiter=200))
        else:
            continue
        y0, K = linear_parameters(root, xs, ys)
        roots.append(RootCandidate(root, y0, K, sum_of_squares(xs, ys, y0, K, root)))
    return roots


# --------------------------------------------------------------------------- #
# Full fit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HBFit:
    y0: float
    K: float
    n: float
    S: float
    """Residual sum of squares, paper Eq. 2."""
    rmse: float
    """``E = sqrt(S/m)``, the error measure used in the paper's Fig. 2."""
    r_squared: float
    m: int
    candidates: list[RootCandidate] = field(default_factory=list)
    standard_errors: tuple[float, float, float] | None = None
    """Approximate standard errors of (y0, K, n) from the linearised
    covariance ``s^2 (J^T J)^-1`` at the solution."""
    gof: GoodnessOfFit | None = field(default=None, repr=False, compare=False)
    """Full goodness-of-fit statistics; see :mod:`hb_mullineux.metrics`."""

    def __call__(self, x: ArrayLike) -> FloatArray:
        return herschel_bulkley(x, self.y0, self.K, self.n)

    def summary(self) -> str:
        lines = [
            f"Herschel-Bulkley fit (Mullineux consistency-condition method), m = {self.m}",
            f"  yield stress      y0 = {self.y0:.6g}",
            f"  consistency       K  = {self.K:.6g}",
            f"  flow index        n  = {self.n:.6g}",
            f"  residual sum S = {self.S:.6g}, E = sqrt(S/m) = {self.rmse:.6g}, R^2 = {self.r_squared:.6f}",
        ]
        if self.gof is not None and self.gof.std_errors is not None:
            se, ci = self.gof.std_errors, self.gof.ci95_half_width
            lines.append(f"  std. errors (95% CI half-width): y0 {se[0]:.3g} ({ci[0]:.3g}), K {se[1]:.3g} ({ci[1]:.3g}), n {se[2]:.3g} ({ci[2]:.3g})")
            c = self.gof.correlation
            lines.append(f"  parameter correlations: r(y0,K) = {c[0,1]:+.3f}, r(y0,n) = {c[0,2]:+.3f}, r(K,n) = {c[1,2]:+.3f}")
        if self.gof is not None:
            g = self.gof
            lines.append(f"  goodness of fit: s = {g.residual_std_error:.4g}, MAE = {g.mae:.4g}, max|r| = {g.max_abs_residual:.4g}, "
                         f"MAPE = {g.mape:.3g}%, adj. R^2 = {g.adjusted_r_squared:.6f}")
            lines.append(f"                   AICc = {g.aicc:.3f}, BIC = {g.bic:.3f}, Durbin-Watson = {g.durbin_watson:.3f}")
        if len(self.candidates) > 1:
            lines.append(f"  ({len(self.candidates)} roots of F(n) found; the least-squares minimiser was kept)")
        return "\n".join(lines)


def hb_jacobian(xs: FloatArray, y0: float, K: float, n: float) -> FloatArray:
    """Derivatives of ``y0 + K x^n`` with respect to ``(y0, K, n)``."""

    xn = xs**n
    return np.column_stack((np.ones_like(xs), xn, K * xn * np.log(xs)))


def _gof(xs: FloatArray, ys: FloatArray, y0: float, K: float, n: float) -> GoodnessOfFit:
    return goodness_of_fit(xs, ys, herschel_bulkley(xs, y0, K, n), p=3, jacobian=hb_jacobian(xs, y0, K, n))


def _standard_errors(xs: FloatArray, ys: FloatArray, y0: float, K: float, n: float
                     ) -> tuple[float, float, float] | None:
    m = xs.size
    if m <= 3:
        return None
    xn = xs**n
    J = np.column_stack((np.ones(m), xn, K * xn * np.log(xs)))
    r = y0 + K * xn - ys
    s2 = float(np.dot(r, r)) / (m - 3)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    diag = np.diag(cov)
    if np.any(diag < 0.0):
        return None
    se = np.sqrt(diag)
    return float(se[0]), float(se[1]), float(se[2])


def fit_herschel_bulkley(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_min: float = 0.01,
    n_max: float = 5.0,
    grid_points: int = 1000,
) -> HBFit:
    """Fit ``y = y0 + K x**n`` by the Mullineux method.

    1. Form ``Fbar(n)`` from the scaled data (Eqs. 5-6).
    2. Locate its non-trivial roots on ``[n_min, n_max]`` (sign changes on a
       grid, refined by Brent's method).
    3. For each root solve the 2x2 linear problem for ``y0`` and ``K``.
    4. Return the root with the smallest residual sum of squares.
    """

    xs, ys = _as_data(x, y)
    candidates = find_roots(xs, ys, n_min=n_min, n_max=n_max, grid_points=grid_points)
    if not candidates:
        raise RuntimeError(
            f"F(n) has no sign change on [{n_min}, {n_max}]. Widen the search interval, "
            "check that the data show a monotone stress-rate relation, or check units."
        )
    best = min(candidates, key=lambda c: c.S)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - best.S / ss_tot if ss_tot > 0.0 else float("nan")
    return HBFit(
        y0=best.y0,
        K=best.K,
        n=best.n,
        S=best.S,
        rmse=float(np.sqrt(best.S / xs.size)),
        r_squared=r2,
        m=int(xs.size),
        candidates=candidates,
        standard_errors=_standard_errors(xs, ys, best.y0, best.K, best.n),
        gof=_gof(xs, ys, best.y0, best.K, best.n),
    )


# --------------------------------------------------------------------------- #
# The conventional "estimate the yield stress" method (paper Section 2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class YieldEstimateFit:
    y0: float
    K: float
    n: float
    S: float
    rmse: float


def fit_with_yield_estimate(x: ArrayLike, y: ArrayLike, y0: float) -> YieldEstimateFit:
    """Fix ``y0`` and regress ``ln(y - y0)`` on ``ln x`` (paper Section 2).

    Returns ``K`` and ``n`` from the 2x2 normal equations in ``(ln K, n)``
    together with the residual sum of squares of the *original* model, i.e.
    the quantity the paper plots as ``E = sqrt(S/m)`` in Fig. 2.
    """

    xs, ys = _as_data(x, y)
    if np.any(ys <= y0):
        raise ValueError("All stresses must exceed the yield-stress estimate y0.")
    lx = np.log(xs)
    ly = np.log(ys - y0)
    A = np.array([[xs.size, lx.sum()], [lx.sum(), (lx * lx).sum()]])
    rhs = np.array([ly.sum(), (lx * ly).sum()])
    lnK, n = np.linalg.solve(A, rhs)
    K = float(np.exp(lnK))
    S = sum_of_squares(xs, ys, y0, K, float(n))
    return YieldEstimateFit(float(y0), K, float(n), S, float(np.sqrt(S / xs.size)))


# --------------------------------------------------------------------------- #
# Cross-check with a general-purpose non-linear least-squares solver
# --------------------------------------------------------------------------- #
def refine_least_squares(x: ArrayLike, y: ArrayLike, y0: float, K: float, n: float) -> HBFit:
    """Polish (y0, K, n) with SciPy's Levenberg-Marquardt / trust-region solver.

    Used only as an independent check: starting from the Mullineux solution
    the solver should not move, because the consistency condition *is* the
    stationarity condition of S.
    """

    from scipy.optimize import least_squares

    xs, ys = _as_data(x, y)

    def residual(p: FloatArray) -> FloatArray:
        return herschel_bulkley(xs, p[0], p[1], p[2]) - ys

    sol = least_squares(residual, np.array([y0, K, n]), method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    p = sol.x
    S = float(np.dot(sol.fun, sol.fun))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return HBFit(
        y0=float(p[0]), K=float(p[1]), n=float(p[2]), S=S,
        rmse=float(np.sqrt(S / xs.size)),
        r_squared=1.0 - S / ss_tot if ss_tot > 0 else float("nan"),
        m=int(xs.size),
        standard_errors=_standard_errors(xs, ys, *map(float, p)),
        gof=_gof(xs, ys, *map(float, p)),
    )
