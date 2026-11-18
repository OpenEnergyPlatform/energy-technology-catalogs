"""Tests of fitting function forms to data and calibrating cost functions.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from energy_technology_catalogs.costs import fitting
from energy_technology_catalogs.costs.core import CostFunction
from energy_technology_catalogs.costs.fitting import (
    FitResult,
    calibrate_cost_function,
    calibrate_to_point,
    fit_cost_form,
)
from energy_technology_catalogs.costs.forms import (
    Constant,
    Exponential,
    Linear,
    Logarithmic,
    PiecewiseLinear,
    Polynomial,
    PowerLaw,
)


X = np.array([1.0, 2.0, 4.0, 8.0, 16.0])


# ----------------------------------------------------------------------
# fit_cost_form
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, kwargs, truth",
    [
        ("linear", {}, Linear(3.0, 2.0)),
        ("polynomial", {"degree": 2}, Polynomial((1.0, 2.0, 3.0))),
        ("logarithmic", {}, Logarithmic(5.0, 2.0)),
        ("constant", {}, Constant(7.0)),
        ("power_law", {"fixed": {"offset": 0.0}}, PowerLaw(0.0, 500.0, -0.3)),
        ("power_law", {}, PowerLaw(20.0, 500.0, -0.3)),
        ("exponential", {}, Exponential(1.0, 2.0, 0.1)),
        (
            "piecewise_linear",
            {"nodes": [1.0, 4.0, 16.0]},
            PiecewiseLinear((1.0, 4.0, 16.0), (10.0, 30.0, 20.0)),
        ),
    ],
)
def test_fit_recovers_known_form(kind, kwargs, truth):
    result = fit_cost_form(X, truth.evaluate(X), kind, **kwargs)
    assert isinstance(result, FitResult)
    assert result.success and result.rmse < 1e-6
    assert result.n_points == len(X)
    assert result.form.kind == kind
    assert result.form.evaluate(X) == pytest.approx(truth.evaluate(X), rel=1e-6)
    assert result.residuals.shape == X.shape


def test_fit_rejects_underdetermined_input():  # Review Focus 4
    with pytest.raises(ValueError, match="points"):
        fit_cost_form([1.0, 2.0], [1.0, 2.0], "polynomial", degree=2)
    with pytest.raises(ValueError, match="distinct"):
        fit_cost_form([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], "linear")
    with pytest.raises(ValueError, match="degree"):
        fit_cost_form([1.0, 2.0], [1.0, 2.0], "polynomial")


def test_power_law_fit_with_non_positive_costs_converges():  # Review Focus 4
    truth = PowerLaw(-50.0, 40.0, 0.5)
    assert np.any(truth.evaluate(X) <= 0)
    assert fit_cost_form(X, truth.evaluate(X), "power_law").rmse < 1e-4


def test_power_law_fit_with_non_positive_costs_and_other_exponent():
    truth = PowerLaw(-80.0, 30.0, 0.8)
    result = fit_cost_form(X, truth.evaluate(X), "power_law")
    assert result.rmse < 1e-4


def test_exponential_fit_with_negative_costs():
    truth = Exponential(-10.0, 2.0, 0.2)
    result = fit_cost_form(X, truth.evaluate(X), "exponential")
    assert result.rmse < 1e-4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kind": "spline"}, "Unknown"),
        ({"kind": "piecewise_linear"}, "nodes"),
        ({"kind": "linear", "fixed": {"slope_x": 1.0}}, "slope_x"),
        ({"kind": "power_law", "initial": {"exp": 1.0}}, "exp"),
        ({"kind": "linear", "degree": 2}, "degree"),
        ({"kind": "linear", "nodes": [1.0, 2.0]}, "nodes"),
        ({"kind": "linear", "loss": "square"}, "loss"),
        ({"kind": "linear", "weights": [1.0, 1.0]}, "weights"),
        ({"kind": "linear", "weights": [1.0, -1.0, 1.0, 1.0, 1.0]}, "weights"),
    ],
)
def test_fit_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        fit_cost_form(X, 2.0 * X, **kwargs)


@pytest.mark.parametrize(
    "sizes", [[0.0, 1.0, 2.0], [-1.0, 1.0, 2.0], [np.nan, 1.0, 2.0]]
)
def test_fit_rejects_non_positive_sizes(sizes):
    with pytest.raises(ValueError, match="positive"):
        fit_cost_form(sizes, [1.0, 2.0, 3.0], "linear")


def test_fit_rejects_mismatched_lengths_and_non_finite_costs():
    with pytest.raises(ValueError, match="length"):
        fit_cost_form([1.0, 2.0, 3.0], [1.0, 2.0], "linear")
    with pytest.raises(ValueError, match="finite"):
        fit_cost_form([1.0, 2.0, 3.0], [1.0, np.inf, 2.0], "linear")


def test_fit_with_fixed_parameter_of_linear_kind():
    truth = Linear(3.0, 2.0)
    result = fit_cost_form(
        [5.0], [truth.evaluate(5.0)], "linear", fixed={"offset": 3.0}
    )
    assert result.form.offset == 3.0
    assert result.form.slope == pytest.approx(2.0, rel=1e-12)


def test_fit_with_fixed_shape_parameter_is_linear():
    truth = PowerLaw(20.0, 500.0, -0.3)
    result = fit_cost_form(X, truth.evaluate(X), "power_law", fixed={"exponent": -0.3})
    assert result.form.parameters() == pytest.approx(truth.parameters(), rel=1e-10)


def test_fit_with_piecewise_nodes_without_data_raises():
    with pytest.raises(ValueError, match="determined"):
        fit_cost_form(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "piecewise_linear", nodes=[1.0, 4.0, 16.0]
        )


def test_fit_weights_pull_towards_heavy_point():
    costs = np.array([10.0, 10.0, 10.0, 10.0, 20.0])
    light = fit_cost_form(X, costs, "constant")
    heavy = fit_cost_form(X, costs, "constant", weights=[1.0, 1.0, 1.0, 1.0, 100.0])
    assert light.form.value == pytest.approx(12.0)
    assert heavy.form.value == pytest.approx((40.0 + 2000.0) / 104.0)


def test_fit_statistics():
    costs = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    result = fit_cost_form(X, costs, "linear")
    expected = costs - result.form.evaluate(X)
    assert result.residuals == pytest.approx(expected)
    assert result.rmse == pytest.approx(math.sqrt(np.mean(expected**2)))
    ss_tot = np.sum((costs - costs.mean()) ** 2)
    assert result.r_squared == pytest.approx(1.0 - np.sum(expected**2) / ss_tot)


def test_robust_fit_resists_outlier():
    truth = Linear(3.0, 2.0)
    costs = truth.evaluate(X).copy()
    costs[2] += 100.0
    plain = fit_cost_form(X, costs, "linear")
    robust = fit_cost_form(X, costs, "linear", loss="soft_l1", f_scale=0.1)
    inliers = [0, 1, 3, 4]
    assert (
        np.abs(robust.residuals[inliers]).max() < np.abs(plain.residuals[inliers]).max()
    )


def test_fit_initial_values_are_used():
    truth = PowerLaw(20.0, 500.0, -0.3)
    result = fit_cost_form(
        X, truth.evaluate(X), "power_law", initial={"exponent": -0.5, "offset": 0.0}
    )
    assert result.rmse < 1e-6


# ----------------------------------------------------------------------
# calibrate_cost_function and calibrate_to_point: new behaviour
# ----------------------------------------------------------------------


def test_calibrate_proportional(function):
    points = [(s, 1.1 * function.form.evaluate(s)) for s in (20.0, 40.0, 80.0)]
    out = calibrate_cost_function(function, points, proportional=True)
    assert out.form.evaluate(50.0) == pytest.approx(
        1.1 * function.form.evaluate(50.0), rel=1e-8
    )
    assert out.key == function.key


def test_calibrate_to_point_methods(function):
    # PowerLaw(0, 1000, -0.5), tax 0, installation included
    assert calibrate_to_point(function, (25.0, 220.0)).form.evaluate(
        25.0
    ) == pytest.approx(220.0)
    assert calibrate_to_point(function, (25.0, 220.0), "vertical_shift").form == (
        function.form.shifted(20.0)
    )
    out = calibrate_to_point(function, (25.0, 220.0), "coefficient")
    assert out.form.parameters()["coefficient"] == pytest.approx(1100.0)
    poly = dataclasses.replace(function, form=Polynomial((1.0, 2.0, 3.0)))
    with pytest.raises(ValueError, match="coefficient"):
        calibrate_to_point(poly, (25.0, 220.0), "coefficient")


def test_calibrate_to_point_proportional_without_installation(function):
    # without installation the original cost at 25 kW is 200 * 0.6 = 120
    out = calibrate_to_point(function, (25.0, 132.0), include_installation=False)
    assert out.form.evaluate(25.0) == pytest.approx(220.0)


@pytest.mark.parametrize(
    "form, method",
    [
        (Linear(100.0, 2.0), "coefficient"),
        (Logarithmic(100.0, 20.0), "coefficient"),
        (Exponential(100.0, 20.0, 0.01), "coefficient"),
        (Polynomial((1.0, 2.0, 3.0)), "vertical_shift"),
        (Constant(50.0), "vertical_shift"),
        (PiecewiseLinear((10.0, 100.0), (300.0, 100.0)), "vertical_shift"),
        (PiecewiseLinear((10.0, 100.0), (300.0, 100.0)), "proportional"),
    ],
)
def test_calibrate_to_point_passes_through_point(function, form, method):
    original = dataclasses.replace(function, form=form)
    out = calibrate_to_point(original, (30.0, 400.0), method, include_installation=True)
    assert out.form.evaluate(30.0) == pytest.approx(400.0, rel=1e-12)
    assert out.form.kind == form.kind


@pytest.mark.parametrize(
    "form", [Constant(50.0), PiecewiseLinear((10.0, 100.0), (300.0, 100.0))]
)
def test_calibrate_to_point_coefficient_needs_one_size_term(function, form):
    with pytest.raises(ValueError, match="coefficient"):
        calibrate_to_point(
            dataclasses.replace(function, form=form), (30.0, 400.0), "coefficient"
        )


def test_calibrate_to_point_rejects_invalid_input(function):
    with pytest.raises(ValueError, match="method"):
        calibrate_to_point(function, (25.0, 220.0), "shift")
    with pytest.raises(ValueError, match="positive"):
        calibrate_to_point(function, (0.0, 220.0))
    with pytest.raises(ValueError, match="size-independent"):
        calibrate_to_point(function, (25.0, -1.0), "coefficient")
    zero = dataclasses.replace(function, form=PowerLaw(0.0, 0.0, -0.5))
    with pytest.raises(ValueError, match="zero"):
        calibrate_to_point(zero, (25.0, 220.0))


def test_calibrate_to_point_outside_range_warns(function):
    with pytest.warns(UserWarning, match="outside"):
        out = calibrate_to_point(function, (400.0, 60.0), "vertical_shift")
    assert out.form.evaluate(400.0) == pytest.approx(60.0)


def test_standard_calibration_of_linear_form(function):
    original = dataclasses.replace(function, form=Linear(100.0, 2.0))
    truth = Linear(150.0, 3.0)
    points = [(s, truth.evaluate(s)) for s in (20.0, 50.0, 90.0)]
    out = calibrate_cost_function(original, points)
    assert out.form.parameters() == pytest.approx(truth.parameters(), rel=1e-8)


def test_standard_calibration_of_polynomial_keeps_degree(function):
    original = dataclasses.replace(function, form=Polynomial((10.0, 1.0, 0.01)))
    truth = Polynomial((20.0, 2.0, 0.02))
    points = [(s, truth.evaluate(s)) for s in (10.0, 30.0, 60.0, 100.0)]
    out = calibrate_cost_function(original, points)
    assert out.form.parameters() == pytest.approx(truth.parameters(), rel=1e-6)


@dataclasses.dataclass(frozen=True)
class _TaggedCostFunction(CostFunction):
    tag: str = "x"


def test_calibration_keeps_subclass_and_fields(function):
    tagged = _TaggedCostFunction(
        **{f.name: getattr(function, f.name) for f in dataclasses.fields(function)},
        tag="kept",
    )
    for out in (
        calibrate_cost_function(tagged, [(20.0, 250.0), (60.0, 140.0)]),
        calibrate_cost_function(tagged, [(20.0, 250.0)], proportional=True),
        calibrate_to_point(tagged, (20.0, 250.0)),
    ):
        assert type(out) is _TaggedCostFunction
        assert out.tag == "kept"
        assert out.installation_share == function.installation_share
        assert out.form.exponent == function.form.exponent


def test_proportional_calibration_keeps_fixed_om(function):
    original = dataclasses.replace(function, fixed_om=Constant(12.0))
    points = [(s, 1.1 * function.form.evaluate(s)) for s in (20.0, 40.0)]
    for out in (
        calibrate_cost_function(original, points, proportional=True),
        calibrate_to_point(original, (25.0, 220.0)),
    ):
        assert out.form.coefficient == pytest.approx(1100.0)
        assert out.fixed_om == Constant(12.0)


def test_calibration_uses_with_form_hook(function, monkeypatch):
    calls = []
    original_with_form = CostFunction.with_form

    def spy(self, form):
        calls.append(form)
        return original_with_form(self, form)

    monkeypatch.setattr(CostFunction, "with_form", spy)
    calibrate_cost_function(function, [(20.0, 250.0), (60.0, 140.0)])
    calibrate_cost_function(function, [(20.0, 250.0)], proportional=True)
    for method in ("proportional", "vertical_shift", "coefficient"):
        calibrate_to_point(function, (25.0, 220.0), method)
    assert len(calls) == 5


def test_fit_wraps_solver_failure(monkeypatch):
    def failing(*args, **kwargs):
        raise ValueError("solver broke")

    monkeypatch.setattr(fitting, "least_squares", failing)
    with pytest.raises(RuntimeError, match="solver broke"):
        fit_cost_form(X, PowerLaw(1.0, 2.0, 0.5).evaluate(X), "power_law")


def test_fit_reports_solver_status_failure(monkeypatch):
    def failed_status(*args, **kwargs):
        return SimpleNamespace(status=-1, message="improper input", x=kwargs["x0"])

    monkeypatch.setattr(fitting, "least_squares", failed_status)
    with pytest.raises(RuntimeError, match="status=-1"):
        fit_cost_form(X, PowerLaw(1.0, 2.0, 0.5).evaluate(X), "power_law")


def test_calibration_rejects_negative_weight(function):
    with pytest.raises(ValueError, match="weight"):
        calibrate_cost_function(function, [(20.0, 250.0)], weight_functional=-1.0)


# ----------------------------------------------------------------------
# Parity with the Uni Kassel cost function module v3.0.0
# ----------------------------------------------------------------------

PARITY_REL = 1e-8
PARITY_ABS = 1e-9


def test_calibration_matches_reference(reference_data):
    data = reference_data("calibration")
    functions = [CostFunction.from_dict(f) for f in data["functions"]]
    mismatches = []
    for i, case in enumerate(data["cases"]):
        original = functions[case["function"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if case["kind"] == "adjust":
                options = dict(case["options"])
                out = calibrate_cost_function(
                    original, [tuple(p) for p in case["data_points"]], **options
                )
            else:
                out = calibrate_to_point(
                    original,
                    tuple(case["data_point"]),
                    case["method"],
                    include_installation=case["include_pi"],
                )
        got = (out.form.offset, out.form.coefficient)
        expected = (case["summand"], case["coefficient"])
        if got != pytest.approx(expected, rel=PARITY_REL, abs=PARITY_ABS):
            mismatches.append((i, got, expected))
        assert out.form.exponent == original.form.exponent
    assert len(data["cases"]) == 168
    assert not mismatches, mismatches


# ----------------------------------------------------------------------
# Reference tests of the adjustment, in the names used here
# ----------------------------------------------------------------------


def make_base_cf(
    summand: float = 100.0,
    coefficient: float = 50.0,
    exponent: float = 0.8,
    lo: float = 0.0,
    hi: float = 100_000.0,
    tax_rate: float = 0.0,
    pi_included: bool = False,
    pi_factor: float = 1.0,
    uncertainty_lower: float = 0.0,
    uncertainty_upper: float = 0.0,
) -> CostFunction:
    """Cost function built from the keys of the reference module."""
    return CostFunction.from_dict(
        {
            "component": "TestComponent",
            "source": "TestSource",
            "technology_type": "heat_pump",
            "reference_unit": "kW",
            "investment_year": 2023,
            "validity_range_lower": lo,
            "validity_range_upper": hi,
            "summand": summand,
            "coefficient": coefficient,
            "exponent": exponent,
            "fixed_om": 10.0,
            "tax_rate": tax_rate,
            "pi_included": pi_included,
            "pi_factor": pi_factor,
            "uncertainty_lower": uncertainty_lower,
            "uncertainty_upper": uncertainty_upper,
            "currency": "EUR",
        }
    )


CF0 = make_base_cf()


def value(function, size):
    """Raw form value, the ``base_cost`` of the reference tests."""
    return function.form.evaluate(size)


def params(function):
    return function.form.offset, function.form.coefficient


# Group 1: input validation


@pytest.mark.parametrize(
    "points, kwargs, match",
    [
        ([], {}, "point"),
        ([(-10.0, 200.0)], {}, "positive"),
        ([(0.0, 200.0)], {}, "positive"),
        ([(10.0, 200.0)], {"weight_data": 0.0}, "weight"),
        ([(10.0, 200.0)], {"loss": "invalid"}, "loss"),
    ],
)
def test_ref_input_validation(points, kwargs, match):
    with pytest.raises(ValueError, match=match):
        calibrate_cost_function(CF0, points, **kwargs)


# Group 2: return type and field preservation


def test_ref_returns_cost_function():
    result = calibrate_cost_function(CF0, [(10.0, 200.0), (20.0, 300.0)])
    assert isinstance(result, CostFunction)


def test_ref_exponent_preserved():
    result = calibrate_cost_function(CF0, [(10.0, 200.0), (20.0, 300.0)])
    assert result.form.exponent == CF0.form.exponent


def test_ref_technology_preserved():
    result = calibrate_cost_function(CF0, [(10.0, 200.0), (20.0, 300.0)])
    assert result.technology == CF0.technology


def test_ref_source_preserved():
    result = calibrate_cost_function(CF0, [(10.0, 200.0), (20.0, 300.0)])
    assert result.source == CF0.source


def _assert_other_fields_equal(result: CostFunction, original: CostFunction) -> None:
    for name in (f.name for f in dataclasses.fields(original) if f.name != "form"):
        assert getattr(result, name) == getattr(original, name), name


def test_ref_all_non_adjusted_fields_preserved():
    cf_full = make_base_cf(
        tax_rate=0.19,
        pi_included=True,
        pi_factor=0.15,
        uncertainty_lower=-0.20,
        uncertainty_upper=0.20,
    )
    result = calibrate_cost_function(cf_full, [(10.0, 200.0), (20.0, 300.0)])
    _assert_other_fields_equal(result, cf_full)


# Group 3: exact fitting with two points


def test_ref_exact_fit_two_points():
    data = [(100.0, 500.0), (1000.0, 2000.0)]
    result = calibrate_cost_function(CF0, data, weight_data=1.0)
    for x, y in data:
        assert abs(value(result, x) - y) < 0.5


def test_ref_exact_fit_linear_exponent():
    result = calibrate_cost_function(
        make_base_cf(exponent=1.0), [(10.0, 200.0), (20.0, 300.0)], weight_data=1.0
    )
    assert abs(result.form.coefficient - 10.0) < 1e-4
    assert abs(result.form.offset - 100.0) < 1e-4


def test_ref_single_data_point():
    result = calibrate_cost_function(CF0, [(50.0, 400.0)], weight_data=1.0)
    assert abs(value(result, 50.0) - 400.0) < 1.0


# Group 4: over-determined fitting


def test_ref_three_points_least_squares():
    truth = make_base_cf(summand=200.0, coefficient=80.0)
    data = [
        (100.0, value(truth, 100.0) + 10.0),
        (1000.0, value(truth, 1000.0) - 8.0),
        (5000.0, value(truth, 5000.0) + 5.0),
    ]
    result = calibrate_cost_function(CF0, data, weight_data=1.0)
    assert np.mean([(value(result, x) - y) ** 2 for x, y in data]) < 500.0


def test_ref_fit_recovers_true_params():
    truth = make_base_cf(summand=150.0, coefficient=75.0, exponent=0.9)
    data = [(s, value(truth, s)) for s in (100.0, 500.0, 1000.0, 5000.0, 10000.0)]
    result = calibrate_cost_function(
        make_base_cf(summand=100.0, coefficient=50.0, exponent=0.9),
        data,
        weight_data=1.0,
    )
    assert abs(result.form.offset - 150.0) < 1.0
    assert abs(result.form.coefficient - 75.0) < 1.0


# Group 5: regularisation


@pytest.mark.parametrize(
    "points, weights",
    [
        ([(100.0, 999.0), (1000.0, 9999.0)], {"weight_functional": 1.0}),
        ([(100.0, 999.0)], {"weight_absolute": 1.0}),
        ([(100.0, 999.0)], {"weight_relative": 1.0}),
    ],
)
def test_ref_regularisation_only_returns_original(points, weights):
    result = calibrate_cost_function(CF0, points, weight_data=0.0, **weights)
    assert abs(result.form.offset - CF0.form.offset) < 0.01
    assert abs(result.form.coefficient - CF0.form.coefficient) < 0.01


def test_ref_regularisation_pulls_toward_original():
    truth = make_base_cf(summand=500.0, coefficient=200.0)
    data_far = [(s, value(truth, s)) for s in (100.0, 500.0, 1000.0, 5000.0)]
    result_data = calibrate_cost_function(CF0, data_far, weight_data=1.0)
    result_reg = calibrate_cost_function(
        CF0, data_far, weight_data=0.01, weight_functional=0.99
    )

    def distance(result):
        return sum((a - b) ** 2 for a, b in zip(params(result), params(CF0)))

    assert distance(result_reg) < distance(result_data)


def test_ref_relative_with_zero_summand():
    original = make_base_cf(summand=0.0, coefficient=50.0, exponent=1.0)
    result = calibrate_cost_function(
        original, [(10.0, 500.0), (20.0, 1000.0)], weight_data=0.0, weight_relative=1.0
    )
    assert abs(result.form.offset) < 0.01
    assert abs(result.form.coefficient - 50.0) < 0.01


def test_ref_relative_with_zero_coefficient():
    original = make_base_cf(summand=100.0, coefficient=0.0, exponent=1.0)
    result = calibrate_cost_function(
        original, [(10.0, 100.0), (20.0, 100.0)], weight_data=0.0, weight_relative=1.0
    )
    assert abs(result.form.offset - 100.0) < 0.01
    assert abs(result.form.coefficient) < 0.01


# Group 6: edge cases


def test_ref_zero_exponent():
    original = make_base_cf(exponent=0.0, summand=100.0, coefficient=50.0)
    result = calibrate_cost_function(
        original, [(10.0, 150.0), (20.0, 150.0)], weight_data=1.0
    )
    assert abs(result.form.offset + result.form.coefficient - 150.0) < 1e-4


def test_ref_out_of_range_data_warns():
    with pytest.warns(UserWarning, match="outside"):
        result = calibrate_cost_function(
            make_base_cf(lo=10.0, hi=100.0), [(200.0, 500.0), (300.0, 600.0)]
        )
    assert isinstance(result, CostFunction)


def test_ref_weight_normalisation_invariant():
    data = [(100.0, 400.0), (1000.0, 1500.0)]
    result1 = calibrate_cost_function(CF0, data, weight_data=1.0, weight_functional=0.5)
    result2 = calibrate_cost_function(CF0, data, weight_data=2.0, weight_functional=1.0)
    assert abs(result1.form.offset - result2.form.offset) < 1e-6
    assert abs(result1.form.coefficient - result2.form.coefficient) < 1e-6


# Group 7: robust loss functions


@pytest.mark.parametrize("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan"])
def test_ref_all_valid_loss_functions_run(loss):
    data = [(100.0, 400.0), (500.0, 1200.0), (2000.0, 4000.0)]
    result = calibrate_cost_function(CF0, data, loss=loss, f_scale=100.0)
    assert isinstance(result, CostFunction)


def test_ref_robust_loss_less_sensitive_to_outlier():
    inliers = [(s, value(CF0, s)) for s in (100.0, 500.0, 2000.0)]
    data = inliers + [(1000.0, value(CF0, 1000.0) * 10.0)]
    result_linear = calibrate_cost_function(CF0, data, weight_data=1.0, loss="linear")
    result_huber = calibrate_cost_function(
        CF0, data, weight_data=1.0, loss="huber", f_scale=500.0
    )

    def inlier_mse(result):
        return np.mean([(value(result, x) - y) ** 2 for x, y in inliers])

    assert inlier_mse(result_huber) < inlier_mse(result_linear)


# Group 8: proportional scaling


def test_ref_proportional_returns_cost_function():
    result = calibrate_cost_function(CF0, [(500.0, 200.0)], proportional=True)
    assert isinstance(result, CostFunction)


def test_ref_proportional_preserves_exponent():
    result = calibrate_cost_function(CF0, [(500.0, 200.0)], proportional=True)
    assert result.form.exponent == CF0.form.exponent


def test_ref_proportional_scale_ratio_uniform():
    result = calibrate_cost_function(
        CF0, [(200.0, 150.0), (800.0, 600.0)], proportional=True, weight_data=1.0
    )
    ratios = [
        value(result, s) / value(CF0, s)
        for s in (100.0, 500.0, 1000.0, 5000.0, 10000.0)
    ]
    assert max(ratios) - min(ratios) < 1e-9


def test_ref_proportional_single_point_passes_through():
    cost_obs = value(CF0, 500.0) * 1.30
    result = calibrate_cost_function(
        CF0, [(500.0, cost_obs)], proportional=True, weight_data=1.0
    )
    assert abs(value(result, 500.0) - cost_obs) < 0.01


def test_ref_proportional_scale_equals_cost_ratio():
    cost_obs = value(CF0, 300.0) * 1.25
    result = calibrate_cost_function(
        CF0, [(300.0, cost_obs)], proportional=True, weight_data=1.0
    )
    assert abs(result.form.coefficient / CF0.form.coefficient - 1.25) < 1e-6


def test_ref_proportional_regularisation_pulls_to_unity():
    result = calibrate_cost_function(
        CF0,
        [(500.0, 9999.0)],
        proportional=True,
        weight_data=0.0,
        weight_functional=1.0,
    )
    assert abs(result.form.coefficient / CF0.form.coefficient - 1.0) < 1e-4


def test_ref_proportional_vs_standard_single_point():
    data = [(500.0, value(CF0, 500.0) * 1.20)]
    result_prop = calibrate_cost_function(CF0, data, proportional=True, weight_data=1.0)
    calibrate_cost_function(CF0, data, proportional=False, weight_data=1.0)
    ratios = [
        value(result_prop, s) / value(CF0, s) for s in (200.0, 500.0, 2000.0, 8000.0)
    ]
    assert max(ratios) - min(ratios) < 1e-9


def test_ref_proportional_all_non_adjusted_preserved():
    cf_full = make_base_cf(
        tax_rate=0.19,
        pi_included=True,
        pi_factor=0.15,
        uncertainty_lower=-0.20,
        uncertainty_upper=0.20,
    )
    result = calibrate_cost_function(cf_full, [(500.0, 200.0)], proportional=True)
    _assert_other_fields_equal(result, cf_full)


def test_ref_proportional_zero_original_cost_raises():
    with pytest.raises(ValueError, match="zero"):
        calibrate_cost_function(
            make_base_cf(summand=0.0, coefficient=0.0),
            [(500.0, 200.0)],
            proportional=True,
        )


def test_ref_proportional_multiple_points_recovers_scale():
    data = [(s, value(CF0, s) * 1.40) for s in (200.0, 500.0, 1000.0, 3000.0, 8000.0)]
    result = calibrate_cost_function(CF0, data, proportional=True, weight_data=1.0)
    assert abs(result.form.coefficient / CF0.form.coefficient - 1.40) < 1e-6
