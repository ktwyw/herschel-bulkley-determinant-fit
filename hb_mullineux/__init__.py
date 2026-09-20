"""Herschel-Bulkley curve fitting by the Mullineux (2008) determinant method."""

from .core import (
    HBFit,
    RootCandidate,
    YieldEstimateFit,
    determinant,
    find_roots,
    fit_herschel_bulkley,
    fit_with_yield_estimate,
    herschel_bulkley,
    linear_parameters,
    refine_least_squares,
    sum_of_squares,
)
from .data import chlorine_as_hb, paper_curve_points, synthetic_data
from .metrics import (
    GoodnessOfFit,
    ModelComparison,
    ModelFit,
    NestedTest,
    compare_nested_models,
    extra_sum_of_squares_test,
    goodness_of_fit,
    information_criteria,
)

__all__ = [
    "HBFit", "RootCandidate", "YieldEstimateFit", "determinant", "find_roots",
    "fit_herschel_bulkley", "fit_with_yield_estimate", "herschel_bulkley",
    "linear_parameters", "refine_least_squares", "sum_of_squares",
    "chlorine_as_hb", "paper_curve_points", "synthetic_data",
    "GoodnessOfFit", "ModelComparison", "ModelFit", "NestedTest", "compare_nested_models",
    "extra_sum_of_squares_test", "goodness_of_fit", "information_criteria",
]
__version__ = "1.0.0"
