import numpy as np
import pytest

from hb_mullineux import (
    chlorine_as_hb,
    determinant,
    find_roots,
    fit_herschel_bulkley,
    fit_with_yield_estimate,
    linear_parameters,
    paper_curve_points,
    refine_least_squares,
    synthetic_data,
)


def test_exact_paper_curve_is_recovered() -> None:
    x, y = paper_curve_points()
    fit = fit_herschel_bulkley(x, y)
    assert abs(fit.n - 0.35) < 1e-9
    assert abs(fit.K - 4.0) < 1e-7
    assert abs(fit.y0 - 5.0) < 1e-7
    assert fit.S < 1e-18


def test_determinant_vanishes_at_zero_and_true_n() -> None:
    x, y = paper_curve_points()
    assert abs(determinant(0.0, x, y)) < 1e-9
    assert abs(determinant(0.35, x, y)) < 1e-8
    # positive for small n, negative for large n (paper Sections 5-6)
    assert determinant(0.2, x, y) > 0
    assert determinant(5.0, x, y) < 0


def test_scaling_does_not_move_the_root() -> None:
    x, y = paper_curve_points()
    unscaled = find_roots(x, y)
    scaled = find_roots(x, y)  # find_roots always scales; compare with raw determinant sign change
    n = scaled[0].n
    assert abs(determinant(n, x, y, scale=False)) < 1e-6 * abs(determinant(0.2, x, y, scale=False))
    assert unscaled[0].n == pytest.approx(0.35, abs=1e-9)


def test_chlorine_example_matches_paper() -> None:
    x, y = chlorine_as_hb()
    fit = fit_herschel_bulkley(x, y, n_max=1.4)
    a, b = fit.y0, -fit.K / fit.y0
    assert a == pytest.approx(39.09, abs=0.01)
    assert b == pytest.approx(0.828, abs=0.001)
    assert fit.n == pytest.approx(0.159, abs=0.001)


def test_linear_parameters_are_least_squares_optimal() -> None:
    x, y = synthetic_data(seed=3)
    n = 0.4
    y0, K = linear_parameters(n, x, y)
    A = np.column_stack((np.ones_like(x), x**n))
    ref = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose([y0, K], ref, rtol=1e-10)


def test_agrees_with_levenberg_marquardt_on_noisy_data() -> None:
    x, y = synthetic_data(seed=7, noise=0.05)
    fit = fit_herschel_bulkley(x, y)
    lm = refine_least_squares(x, y, fit.y0, fit.K, fit.n)
    assert np.allclose([fit.y0, fit.K, fit.n], [lm.y0, lm.K, lm.n], rtol=1e-6)
    assert fit.S <= lm.S * (1 + 1e-9)


def test_yield_estimate_sensitivity_matches_paper_statement() -> None:
    x, y = paper_curve_points()
    f = fit_with_yield_estimate(x, y, 5.0)
    assert f.K == pytest.approx(4.0, rel=1e-9) and f.n == pytest.approx(0.35, rel=1e-9)
    f1 = fit_with_yield_estimate(x, y, 6.0)
    assert 0.10 < abs(f1.K / 4.0 - 1.0) < 0.16
    assert 0.04 < abs(f1.n / 0.35 - 1.0) < 0.08


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        fit_herschel_bulkley([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        fit_herschel_bulkley([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
