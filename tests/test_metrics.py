import numpy as np
import pytest

from hb_mullineux import (
    compare_nested_models,
    extra_sum_of_squares_test,
    fit_herschel_bulkley,
    goodness_of_fit,
    information_criteria,
    synthetic_data,
)


def test_goodness_of_fit_basic_identities() -> None:
    x, y = synthetic_data(seed=1, noise=0.03)
    fit = fit_herschel_bulkley(x, y)
    g = fit.gof
    assert g.m == 40 and g.p == 3
    assert g.S == pytest.approx(fit.S)
    assert g.rmse == pytest.approx(np.sqrt(fit.S / 40))
    assert g.residual_std_error == pytest.approx(np.sqrt(fit.S / 37))
    assert g.r_squared == pytest.approx(fit.r_squared)
    assert g.adjusted_r_squared < g.r_squared
    assert g.mae <= g.rmse <= g.max_abs_residual
    assert np.allclose(g.std_errors, fit.standard_errors)
    assert np.all(g.ci95_half_width > g.std_errors)          # t(0.975, 37) > 1
    assert np.allclose(np.diag(g.correlation), 1.0)
    assert 1.0 < g.durbin_watson < 3.0


def test_exact_data_gives_perfect_scores() -> None:
    x = np.geomspace(1, 400, 30)
    y = 5 + 4 * x**0.35
    g = goodness_of_fit(x, y, y, p=3)
    assert g.r_squared == pytest.approx(1.0) and g.mape == pytest.approx(0.0)


def test_information_criteria_and_f_test() -> None:
    aic, aicc, bic = information_criteria(10.0, 40, 3)
    assert aicc > aic and bic > aic
    t = extra_sum_of_squares_test(20.0, 2, 10.0, 3, 40)
    assert t.df1 == 1 and t.df2 == 37 and t.F == pytest.approx(37.0)
    assert t.p_value < 1e-5 and t.extra_parameters_justified


def test_nested_comparison_detects_bingham_data() -> None:
    rng = np.random.default_rng(2)
    x = np.geomspace(1, 400, 40)
    y = (5 + 0.08 * x) * (1 + 0.03 * rng.standard_normal(40))
    fit = fit_herschel_bulkley(x, y)
    cmp = compare_nested_models(x, y, fit, {"y0": fit.y0, "K": fit.K, "n": fit.n})
    bingham_test = next(t for t in cmp.tests if t.reduced.startswith("Bingham"))
    assert not bingham_test.extra_parameters_justified
    bic_best = min(cmp.models, key=lambda mf: mf.bic)
    assert bic_best.name.startswith("Bingham")


def test_nested_comparison_supports_hb_on_hb_data() -> None:
    x, y = synthetic_data(seed=1, noise=0.03)
    fit = fit_herschel_bulkley(x, y)
    cmp = compare_nested_models(x, y, fit, {"y0": fit.y0, "K": fit.K, "n": fit.n})
    assert all(t.extra_parameters_justified for t in cmp.tests)
    assert min(cmp.models, key=lambda mf: mf.aicc).name.startswith("Herschel")
