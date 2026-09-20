"""Synthetic Herschel-Bulkley data and the paper's 'Chlorine' example."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .core import herschel_bulkley

FloatArray = NDArray[np.float64]

# Parameters of the paper's illustrative curve (Fig. 1): y0 = 5 Pa,
# K = 4 Pa s^n, n = 0.35, sampled at x = 5, 10, ..., 400 1/s (80 points).
PAPER_Y0, PAPER_K, PAPER_N = 5.0, 4.0, 0.35


def paper_curve_points() -> tuple[FloatArray, FloatArray]:
    x = np.arange(5.0, 400.0 + 0.5, 5.0)
    return x, herschel_bulkley(x, PAPER_Y0, PAPER_K, PAPER_N)


def synthetic_data(
    y0: float = PAPER_Y0,
    K: float = PAPER_K,
    n: float = PAPER_N,
    *,
    x_min: float = 1.0,
    x_max: float = 400.0,
    m: int = 40,
    noise: float = 0.03,
    log_spaced: bool = True,
    seed: int | None = 0,
) -> tuple[FloatArray, FloatArray]:
    """Noisy Herschel-Bulkley data.

    ``noise`` is the relative standard deviation of Gaussian noise added to
    the stress (0.03 = 3%).  Shear rates are log-spaced by default, which is
    how rheometer flow curves are usually collected.
    """

    if x_min <= 0 or x_max <= x_min or m < 4:
        raise ValueError("Need 0 < x_min < x_max and m >= 4.")
    x = np.geomspace(x_min, x_max, m) if log_spaced else np.linspace(x_min, x_max, m)
    y_exact = herschel_bulkley(x, y0, K, n)
    rng = np.random.default_rng(seed)
    y = y_exact * (1.0 + noise * rng.standard_normal(m)) if noise > 0 else y_exact
    return x, y


# Bates & Watts "Chlorine" data, reproduced in the paper's Table 1:
# chlorine-ion concentration (%) against time (min).
CHLORINE_TIME = np.array([
    2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.05, 3.15, 3.25, 3.35, 3.45, 3.55, 3.65, 3.75, 3.85,
    3.95, 4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35,
    5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05, 6.15, 6.25, 6.35, 6.45, 6.55, 6.65, 6.75, 6.85,
    6.95, 7.05, 7.15, 7.25, 7.35, 7.45, 7.55, 7.65, 7.75,
])
CHLORINE_CONC = np.array([
    17.3, 17.6, 17.9, 18.3, 18.5, 18.9, 19.0, 19.3, 19.8, 19.9, 20.2, 20.5, 20.6, 21.1, 21.5,
    21.9, 22.0, 22.3, 22.6, 22.8, 23.0, 23.2, 23.4, 23.7, 24.0, 24.2, 24.5, 25.0, 25.4, 25.5,
    25.9, 25.9, 26.3, 26.2, 26.5, 26.5, 26.6, 27.0, 27.0, 27.0, 27.0, 27.3, 27.8, 28.1, 28.1,
    28.1, 28.4, 28.6, 29.0, 29.2, 29.3, 29.4, 29.4, 29.4,
])


def chlorine_as_hb() -> tuple[FloatArray, FloatArray]:
    """The chlorine data in Herschel-Bulkley form (paper Section 7).

    ``y = a(1 - b e^{-nt})`` becomes ``y = a - (ab) x**n`` with ``x = e^{-t}``,
    i.e. the model with ``y0 = a`` and a negative ``K = -ab``.
    """

    return np.exp(-CHLORINE_TIME), CHLORINE_CONC.copy()
