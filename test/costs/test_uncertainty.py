"""Tests of distributions and uncertain cost functions.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import dataclasses
import json
import math
import warnings

import numpy as np
import pytest

from energy_technology_catalogs.costs.core import CostFunction
from energy_technology_catalogs.costs.forms import Constant, PowerLaw
from energy_technology_catalogs.costs.inflation import PriceIndex
from energy_technology_catalogs.costs.uncertainty import (
    Distribution,
    DistributionKind,
    OutputUncertaintyMode,
    UncertainCostFunction,
    distribution_from_bounds,
)


K = DistributionKind
M = OutputUncertaintyMode


def fields_of(function: CostFunction) -> dict:
    return {f.name: getattr(function, f.name) for f in dataclasses.fields(function)}


def uncertain(function: CostFunction, **changes) -> UncertainCostFunction:
    """Copy ``function`` into an uncertain cost function with ``changes``."""
    return UncertainCostFunction(**{**fields_of(function), **changes})


@pytest.fixture
def bounded(function) -> CostFunction:
    """The shared fixture with symmetric bounds of +-20 %."""
    return dataclasses.replace(function, uncertainty_lower=-0.2, uncertainty_upper=0.2)


# ----------------------------------------------------------------------
# Distribution
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind=K.DETERMINISTIC),
        dict(kind=K.NORMAL, mean=0.0),
        dict(kind=K.NORMAL, mean=0.0, std_dev=-1.0),
        dict(kind=K.LOGNORMAL, mean=0.0, std_dev=0.0),
        dict(kind=K.UNIFORM, lower=1.0),
        dict(kind=K.UNIFORM, lower=1.0, upper=1.0),
        dict(kind=K.TRIANGULAR, lower=0.0, upper=1.0),
        dict(kind=K.TRIANGULAR, lower=0.0, mode=2.0, upper=1.0),
        dict(kind=K.TRIANGULAR, lower=1.0, mode=1.0, upper=1.0),
        dict(kind=K.LAPLACE, mean=0.0, scale=0.0),
        dict(kind=K.NORMAL, mean=math.nan, std_dev=1.0),
    ],
)
def test_distribution_validation(kwargs):
    with pytest.raises(ValueError):
        Distribution(**kwargs)


def test_distribution_kind_must_be_enum():
    with pytest.raises(TypeError):
        Distribution("uniform", lower=0.0, upper=1.0)


@pytest.mark.parametrize(
    "distribution, mean",
    [
        (Distribution(K.DETERMINISTIC, mean=3.0), 3.0),
        (Distribution(K.NORMAL, mean=1.0, std_dev=2.0), 1.0),
        (Distribution(K.LOGNORMAL, mean=0.5, std_dev=0.4), math.exp(0.5 + 0.08)),
        (Distribution(K.UNIFORM, lower=-0.2, upper=0.4), 0.1),
        (Distribution(K.TRIANGULAR, lower=-0.1, mode=0.0, upper=0.4), 0.1),
        (Distribution(K.LAPLACE, mean=-1.0, scale=0.5), -1.0),
    ],
)
def test_mean_value_and_sampling(distribution, mean):
    assert distribution.mean_value == pytest.approx(mean)
    samples = distribution.sample(200_000, np.random.default_rng(7))
    assert samples.shape == (200_000,)
    assert samples.mean() == pytest.approx(mean, rel=2e-2, abs=1e-2)


def test_symmetry():
    assert Distribution(K.UNIFORM, lower=-0.2, upper=0.2).is_symmetric
    assert not Distribution(K.UNIFORM, lower=-0.2, upper=0.3).is_symmetric
    assert Distribution(K.TRIANGULAR, lower=-0.2, mode=0.05, upper=0.3).is_symmetric
    assert not Distribution(K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.3).is_symmetric
    assert Distribution(K.NORMAL, mean=1.0, std_dev=1.0).is_symmetric


def test_distribution_quantiles():
    uniform = Distribution(K.UNIFORM, lower=-0.2, upper=0.2)
    assert uniform.quantiles([0.05, 0.5, 0.95]) == pytest.approx(
        {0.05: -0.18, 0.5: 0.0, 0.95: 0.18}
    )
    normal = Distribution(K.NORMAL, mean=1.0, std_dev=2.0)
    assert normal.quantiles([0.975])[0.975] == pytest.approx(1.0 + 2.0 * 1.959963985)
    assert Distribution(K.DETERMINISTIC, mean=4.0).quantiles([0.1]) == {0.1: 4.0}
    with pytest.raises(ValueError):
        uniform.quantiles([0.0])
    with pytest.raises(ValueError):
        uniform.quantiles([1.0])


def test_sample_needs_positive_count():
    with pytest.raises(ValueError):
        Distribution(K.UNIFORM, lower=0.0, upper=1.0).sample(0, np.random.default_rng())


def test_distribution_dict_round_trip():
    triangular = Distribution(K.TRIANGULAR, lower=-0.1, mode=0.0, upper=0.3)
    data = triangular.to_dict()
    assert data == {"kind": "triangular", "lower": -0.1, "upper": 0.3, "mode": 0.0}
    assert Distribution.from_dict(json.loads(json.dumps(data))) == triangular
    # keys of the Uni Kassel cost function module
    reference = {
        "distribution": "triangular",
        "min_val": -0.1,
        "mode": 0.0,
        "max_val": 0.3,
    }
    assert Distribution.from_dict(reference) == triangular
    with pytest.raises(ValueError):
        Distribution.from_dict({"kind": "beta"})


# ----------------------------------------------------------------------
# Bounds
# ----------------------------------------------------------------------


def test_distribution_from_bounds():
    assert distribution_from_bounds(0.0, 0.0) is None
    assert distribution_from_bounds(-0.2, 0.2) == Distribution(
        K.UNIFORM, lower=-0.2, upper=0.2
    )
    assert distribution_from_bounds(-0.1, 0.3) == Distribution(
        K.TRIANGULAR, lower=-0.1, upper=0.3, mode=0.0
    )
    # one-sided bounds stay uniform
    assert distribution_from_bounds(0.0, 0.3) == Distribution(
        K.UNIFORM, lower=0.0, upper=0.3
    )
    normal = distribution_from_bounds(-0.2, 0.2, "normal", coverage=0.90)
    assert (normal.mean, normal.std_dev) == pytest.approx(
        (0.0, 0.2 / 1.6448536269514722)
    )
    laplace = distribution_from_bounds(-0.1, 0.3, "laplace", coverage=0.90)
    assert (laplace.mean, laplace.scale) == pytest.approx((0.1, 0.2 / -math.log(0.1)))
    assert distribution_from_bounds(-0.2, 0.2, "triangular") == Distribution(
        K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.2
    )
    assert distribution_from_bounds(-0.1, 0.3, "uniform").kind is K.UNIFORM
    with pytest.raises(ValueError):
        distribution_from_bounds(-0.2, 0.2, "beta")


def test_coverage_is_clipped():
    assert distribution_from_bounds(-0.2, 0.2, "normal", coverage=1.0) == (
        distribution_from_bounds(-0.2, 0.2, "normal", coverage=0.999)
    )
    assert distribution_from_bounds(-0.2, 0.2, "laplace", coverage=0.0) == (
        distribution_from_bounds(-0.2, 0.2, "laplace", coverage=0.01)
    )


# reference group "Uncertainty bounds" (the 7 tests without the serializer)


def test_bounds_symmetric_detection(bounded):
    assert UncertainCostFunction.from_bounds(bounded).output_uncertainty.is_symmetric


def test_bounds_asymmetric_detection(function):
    asymmetric = dataclasses.replace(
        function, uncertainty_lower=-0.2, uncertainty_upper=0.3
    )
    u = UncertainCostFunction.from_bounds(asymmetric)
    assert not u.output_uncertainty.is_symmetric
    assert u.output_uncertainty.kind is K.TRIANGULAR


def test_bounds_zero_gives_no_distribution(function):
    assert UncertainCostFunction.from_bounds(function) is None


def test_bounds_positive_lower_raises(function):
    with pytest.raises(ValueError):
        dataclasses.replace(function, uncertainty_lower=0.1)


def test_bounds_negative_upper_raises(function):
    with pytest.raises(ValueError):
        dataclasses.replace(function, uncertainty_upper=-0.1)


def test_bounds_symmetric_relative_equals_mean_preserving(bounded):
    # for symmetric bounds the plain relative mode is sufficient
    relative = UncertainCostFunction.from_bounds(bounded, mode=M.RELATIVE)
    preserving = UncertainCostFunction.from_bounds(bounded)
    p = [0.05, 0.5, 0.95]
    assert relative.cost_quantiles(25.0, p) == pytest.approx(
        preserving.cost_quantiles(25.0, p)
    )


def test_bounds_asymmetric_default_is_mean_preserving(function):
    asymmetric = dataclasses.replace(
        function, uncertainty_lower=-0.2, uncertainty_upper=0.3
    )
    u = UncertainCostFunction.from_bounds(asymmetric)
    assert u.output_mode is M.RELATIVE_MEAN_PRESERVING
    assert u.sample_cost(25.0, 20000, rng=3).mean() == pytest.approx(200.0, rel=1e-2)


# ----------------------------------------------------------------------
# Analytical quantiles and sampling
# ----------------------------------------------------------------------


def test_analytical_quantiles(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    assert u.supports_analytical_quantiles
    assert u.cost_quantiles(25.0, [0.05, 0.5, 0.95]) == pytest.approx(
        {0.05: 164.0, 0.5: 200.0, 0.95: 236.0}
    )


def test_quantiles_absolute_cost(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    assert u.cost_quantiles(25.0, [0.5], specific=False) == pytest.approx({0.5: 5000.0})


def test_order_installation_mode_tax(function):
    # form 200 EUR/kW gross with installation; without installation 120;
    # absolute deviation +-20 EUR/kW applied after the installation share,
    # tax removed after the deviation
    taxed = dataclasses.replace(function, tax_rate=0.19)
    u = uncertain(
        taxed,
        output_uncertainty=Distribution(K.UNIFORM, lower=-20.0, upper=20.0),
        output_mode=M.ABSOLUTE,
    )
    q = u.cost_quantiles(25.0, [0.5, 0.95], include_installation=False)
    assert q == pytest.approx({0.5: 120.0 / 1.19, 0.95: 138.0 / 1.19})
    direct = dataclasses.replace(
        u,
        output_uncertainty=Distribution(K.DETERMINISTIC, mean=500.0),
        output_mode=M.DIRECT,
    )
    assert direct.cost_quantiles(25.0, [0.5], include_installation=False) == (
        pytest.approx({0.5: 500.0 / 1.19})
    )
    assert direct.cost_quantiles(25.0, [0.5], include_tax=True) == {0.5: 500.0}


def test_cost_quantiles_needs_output_uncertainty(function):
    with pytest.raises(ValueError):
        uncertain(function).cost_quantiles(25.0, [0.5])


def test_cost_quantiles_needs_scalar_size(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    with pytest.raises(ValueError):
        u.cost_quantiles(np.array([20.0, 30.0]), [0.5])


def test_cost_quantiles_clamps_with_warning(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    with pytest.warns(UserWarning):
        q = u.cost_quantiles(400.0, [0.5])
    assert q == pytest.approx({0.5: 100.0})


def test_sampling_is_seeded_and_bounded(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    a, b = u.sample_cost(25.0, 20000, rng=12345), u.sample_cost(25.0, 20000, rng=12345)
    assert np.array_equal(a, b) and a.min() >= 160.0 and a.max() <= 240.0
    assert a.mean() == pytest.approx(200.0, rel=1e-2)


def test_sampling_accepts_generator(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    a = u.sample_cost(25.0, 100, rng=np.random.default_rng(5))
    assert np.array_equal(a, u.sample_cost(25.0, 100, rng=5))
    assert u.sample_cost(25.0, 10).shape == (10,)


def test_sampling_rejects_bad_arguments(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    with pytest.raises(ValueError):
        u.sample_cost(np.array([20.0, 30.0]), 10, rng=1)
    with pytest.raises(ValueError):
        u.sample_cost(25.0, 0, rng=1)


def test_mean_preserving_subtracts_analytical_mean(function):
    # with one sample a sample-mean correction would return the
    # deterministic cost exactly; the analytical mean does not
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.3),
    )
    (single,) = u.sample_cost(25.0, 1, rng=0)
    expected = Distribution(K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.3).sample(
        1, np.random.default_rng(0)
    )[0]
    assert single == pytest.approx(200.0 * (1.0 + expected - 1.0 / 30.0))
    q = u.cost_quantiles(25.0, [0.5])
    median = Distribution(K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.3).quantiles(
        [0.5]
    )[0.5]
    assert q[0.5] == pytest.approx(200.0 * (1.0 + median - 1.0 / 30.0))


def test_parameter_uncertainty(function):
    u = UncertainCostFunction(
        **fields_of(function),
        parameter_uncertainty={
            "coefficient": Distribution(K.NORMAL, mean=1000.0, std_dev=100.0)
        },
    )
    s = u.sample_cost(25.0, 50000, rng=1)
    assert s.mean() == pytest.approx(200.0, rel=2e-2)
    assert s.std() == pytest.approx(20.0, rel=5e-2)
    assert not u.supports_analytical_quantiles
    with pytest.raises(ValueError):
        u.cost_quantiles(25.0, [0.5])


def test_deterministic_parameter_sets_the_form_value(function):
    # one object, one parameter value: calculate_cost, quantiles, samples and
    # files all see the deterministic value (no tax, installation included)
    u = uncertain(
        function,
        parameter_uncertainty={
            "coefficient": Distribution(K.DETERMINISTIC, mean=2000.0)
        },
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.1, upper=0.1),
    )
    assert u.form == PowerLaw(0.0, 2000.0, -0.5)
    assert u.parameter_uncertainty == {}
    assert u.calculate_cost(25.0) == pytest.approx(400.0)
    assert u.supports_analytical_quantiles
    assert u.cost_quantiles(25.0, [0.5]) == pytest.approx({0.5: u.calculate_cost(25.0)})
    assert u.sample_cost(25.0, 2000, rng=2).mean() == pytest.approx(400.0, rel=1e-2)
    restored = UncertainCostFunction.from_dict(json.loads(json.dumps(u.to_dict())))
    assert restored == u
    assert restored.to_dict() == u.to_dict()


def test_deterministic_and_random_parameters(function):
    u = uncertain(
        function,
        parameter_uncertainty={
            "exponent": Distribution(K.DETERMINISTIC, mean=-1.0),
            "coefficient": Distribution(K.NORMAL, mean=1000.0, std_dev=100.0),
        },
    )
    assert u.form == PowerLaw(0.0, 1000.0, -1.0)
    assert list(u.parameter_uncertainty) == ["coefficient"]
    assert u.sample_cost(25.0, 20000, rng=3).mean() == pytest.approx(40.0, rel=1e-2)


def test_unknown_parameter_name_raises(function):
    with pytest.raises(ValueError, match="slope"):
        uncertain(
            function,
            parameter_uncertainty={
                "slope": Distribution(K.NORMAL, mean=1.0, std_dev=0.1)
            },
        )


def test_fixed_om_is_not_a_parameter(function):
    with pytest.raises(ValueError):
        uncertain(
            function,
            parameter_uncertainty={"fixed_om": Distribution(K.DETERMINISTIC, mean=1.0)},
        )


def test_field_types_are_checked(function):
    with pytest.raises(TypeError):
        uncertain(function, parameter_uncertainty={"coefficient": 1000.0})
    with pytest.raises(TypeError):
        uncertain(function, output_uncertainty=0.2)
    with pytest.raises(TypeError):
        uncertain(function, output_mode="relative")


def test_settings_survive_replace(bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    copy = dataclasses.replace(u, tax_rate=0.07)
    assert copy.output_uncertainty == u.output_uncertainty
    assert copy.output_mode is u.output_mode


def test_identity_hash_and_equality(function, bounded):
    u = UncertainCostFunction.from_bounds(bounded)
    assert hash(u) == hash(dataclasses.replace(u))
    assert u == dataclasses.replace(u)
    plain = uncertain(function)
    assert plain != function
    with_parameter = dataclasses.replace(
        plain,
        parameter_uncertainty={"exponent": Distribution(K.DETERMINISTIC, mean=-0.5)},
    )
    assert with_parameter == plain
    assert hash(with_parameter) == hash(plain)


def test_dict_round_trip(bounded):
    u = dataclasses.replace(
        UncertainCostFunction.from_bounds(bounded, mode=M.ABSOLUTE),
        parameter_uncertainty={
            "coefficient": Distribution(K.NORMAL, mean=1000.0, std_dev=100.0)
        },
    )
    data = json.loads(json.dumps(u.to_dict()))
    assert data["output_mode"] == "absolute"
    assert data["parameter_uncertainty"]["coefficient"]["kind"] == "normal"
    assert UncertainCostFunction.from_dict(data) == u


def test_from_dict_without_uncertainty(function):
    u = UncertainCostFunction.from_dict(function.to_dict())
    assert u.output_uncertainty is None
    assert u.parameter_uncertainty == {}
    assert u.output_mode is M.RELATIVE_MEAN_PRESERVING


# ----------------------------------------------------------------------
# reference group "Output uncertainty" (14) in the names used here
# ----------------------------------------------------------------------


def test_ou_default_disabled(function):
    u = uncertain(function)
    assert u.output_uncertainty is None
    assert not u.supports_analytical_quantiles


def test_ou_set_and_property(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.5, upper=0.5),
        output_mode=M.RELATIVE,
    )
    assert u.output_uncertainty is not None
    assert u.output_mode is M.RELATIVE


def test_ou_default_mode_is_mean_preserving(function):
    u = uncertain(
        function, output_uncertainty=Distribution(K.UNIFORM, lower=-0.5, upper=0.5)
    )
    assert u.output_mode is M.RELATIVE_MEAN_PRESERVING


def test_ou_clear(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.5, upper=0.5),
        output_mode=M.RELATIVE,
    )
    assert dataclasses.replace(u, output_uncertainty=None).output_uncertainty is None


def test_ou_relative_output_only(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.5, upper=0.5),
        output_mode=M.RELATIVE,
    )
    det = u.calculate_cost(25.0)
    s = u.sample_cost(25.0, 2000, rng=11)
    assert np.all(s >= det * 0.5 - 1e-6) and np.all(s <= det * 1.5 + 1e-6)


def test_ou_relative_mean_preserving(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.3),
        output_mode=M.RELATIVE_MEAN_PRESERVING,
    )
    det = u.calculate_cost(25.0)
    assert u.sample_cost(25.0, 5000, rng=12).mean() == pytest.approx(det, rel=2e-2)


def test_ou_asymmetric_relative_shifts_mean_without_warning(function):
    # the reference warned here; the caveat is documented instead, so that
    # copies (price adjustment, replace) stay silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        u = uncertain(
            function,
            output_uncertainty=Distribution(
                K.TRIANGULAR, lower=-0.2, mode=0.0, upper=0.3
            ),
            output_mode=M.RELATIVE,
        )
        dataclasses.replace(u, tax_rate=0.07)
    det = u.calculate_cost(25.0)
    s = u.sample_cost(25.0, 20000, rng=19)
    assert s.mean() == pytest.approx(det * (1.0 + 1.0 / 30.0), rel=5e-3)


def test_ou_lognormal_relative_never_below_deterministic(function):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        u = uncertain(
            function,
            output_uncertainty=Distribution(K.LOGNORMAL, mean=-3.0, std_dev=0.5),
            output_mode=M.RELATIVE,
        )
    assert np.all(u.sample_cost(25.0, 1000, rng=20) > u.calculate_cost(25.0))


def test_ou_absolute_output_only(function):
    spread = 50.0
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-spread, upper=spread),
        output_mode=M.ABSOLUTE,
    )
    det = u.calculate_cost(25.0)
    s = u.sample_cost(25.0, 2000, rng=13)
    assert np.all(s >= det - spread - 1e-6) and np.all(s <= det + spread + 1e-6)


def test_ou_direct_mode(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(
            K.TRIANGULAR, lower=5000.0, mode=10000.0, upper=15000.0
        ),
        output_mode=M.DIRECT,
    )
    s = u.sample_cost(25.0, 2000, rng=14)
    assert np.all(s >= 5000.0 - 1e-6) and np.all(s <= 15000.0 + 1e-6)


def test_ou_combined_parameter_and_output(function):
    u = uncertain(
        function,
        parameter_uncertainty={
            "coefficient": Distribution(K.NORMAL, mean=1000.0, std_dev=50.0)
        },
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.2, upper=0.2),
        output_mode=M.RELATIVE,
    )
    s = u.sample_cost(25.0, 2000, rng=15)
    assert s.std() > 0
    assert not u.supports_analytical_quantiles


def test_ou_without_uncertainty_samples_deterministic_cost(function):
    u = uncertain(function)
    s = u.sample_cost(25.0, 100, rng=16)
    assert np.allclose(s, u.calculate_cost(25.0))


def test_ou_quantiles_ordered(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.5, upper=0.5),
        output_mode=M.RELATIVE,
    )
    q = u.cost_quantiles(25.0, [0.05, 0.5, 0.95])
    assert q[0.05] < q[0.5] < q[0.95]
    sampled = np.quantile(u.sample_cost(25.0, 2000, rng=17), [0.05, 0.5, 0.95])
    assert sampled[0] < sampled[1] < sampled[2]


def test_ou_json_round_trip(function):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-0.5, upper=0.5),
        output_mode=M.RELATIVE,
    )
    raw = json.loads(json.dumps(u.to_dict()))
    assert raw["output_uncertainty"] == {"kind": "uniform", "lower": -0.5, "upper": 0.5}
    assert raw["output_mode"] == "relative"


def test_ou_reference_keys(function):
    data = function.to_dict()
    data["output_uncertainty"] = {
        "distribution": "uniform",
        "min_val": -0.5,
        "max_val": 0.5,
    }
    data["output_uncertainty_mode"] = "relative"
    u = UncertainCostFunction.from_dict(data)
    assert u.output_uncertainty == Distribution(K.UNIFORM, lower=-0.5, upper=0.5)
    assert u.output_mode is M.RELATIVE


# ----------------------------------------------------------------------
# Parity with the Uni Kassel cost function module v3.0.0
# ----------------------------------------------------------------------


def test_quantiles_match_reference(reference_data):
    data = reference_data("quantiles")
    functions = [CostFunction.from_dict(f) for f in data["functions"]]
    assert len(data["cases"]) == 96
    for number, case in enumerate(data["cases"]):
        spec = dict(case["distribution"])
        spec["kind"] = spec.pop("distribution")
        spec["lower"] = spec.pop("min_val", None)
        spec["upper"] = spec.pop("max_val", None)
        u = uncertain(
            functions[case["function"]],
            output_uncertainty=Distribution.from_dict(spec),
            output_mode=OutputUncertaintyMode(case["mode"]),
        )
        options = dict(
            include_installation=case["include_pi"], include_tax=case["return_gross"]
        )
        np.testing.assert_allclose(
            u.calculate_cost(case["size"], **options),
            case["deterministic_cost"],
            rtol=1e-9,
            err_msg=f"case {number}",
        )
        result = u.cost_quantiles(case["size"], case["probabilities"], **options)
        assert list(result) == case["probabilities"]
        np.testing.assert_allclose(
            list(result.values()),
            case["quantiles"],
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"case {number}",
        )


def test_constant_form_parameter_uncertainty(function):
    # a form other than the power law: the parameter is the constant value
    constant = dataclasses.replace(
        function,
        form=Constant(200.0),
        installation_included=False,
        installation_share=None,
    )
    u = uncertain(
        constant,
        parameter_uncertainty={
            "value": Distribution(K.UNIFORM, lower=100.0, upper=300.0)
        },
    )
    s = u.sample_cost(25.0, 1000, rng=18, specific=False)
    assert np.all(s >= 2500.0) and np.all(s <= 7500.0)
    assert isinstance(u.form, Constant) and not isinstance(u.form, PowerLaw)


# ----------------------------------------------------------------------
# Scaling (price adjustment, calibration) and form replacement
# ----------------------------------------------------------------------

PROBABILITIES = [0.05, 0.25, 0.5, 0.75, 0.95]


@pytest.mark.parametrize(
    "distribution",
    [
        Distribution(K.DETERMINISTIC, mean=3.0),
        Distribution(K.NORMAL, mean=1.0, std_dev=2.0),
        Distribution(K.NORMAL, mean=1.0, std_dev=0.0),
        Distribution(K.LOGNORMAL, mean=0.5, std_dev=0.4),
        Distribution(K.UNIFORM, lower=-0.2, upper=0.4),
        Distribution(K.TRIANGULAR, lower=-0.1, mode=0.0, upper=0.4),
        Distribution(K.LAPLACE, mean=-1.0, scale=0.5),
    ],
)
def test_distribution_scaled(distribution):
    k = 1.37
    scaled = distribution.scaled(k)
    assert scaled.kind is distribution.kind
    expected = {p: k * q for p, q in distribution.quantiles(PROBABILITIES).items()}
    assert scaled.quantiles(PROBABILITIES) == pytest.approx(expected, rel=1e-12)
    assert scaled.mean_value == pytest.approx(k * distribution.mean_value, rel=1e-12)


@pytest.mark.parametrize("factor", [0.0, -1.0, math.inf, math.nan])
def test_distribution_scaled_rejects_bad_factor(factor):
    with pytest.raises(ValueError):
        Distribution(K.UNIFORM, lower=0.0, upper=1.0).scaled(factor)


@pytest.mark.parametrize("mode", [M.ABSOLUTE, M.DIRECT])
def test_scaled_absolute_and_direct_output(function, mode):
    u = uncertain(
        function,
        output_uncertainty=Distribution(K.UNIFORM, lower=-20.0, upper=30.0),
        output_mode=mode,
    )
    k = 1.25
    scaled = u.scaled(k)
    assert isinstance(scaled, UncertainCostFunction)
    assert scaled.output_uncertainty == u.output_uncertainty.scaled(k)
    expected = {p: k * q for p, q in u.cost_quantiles(25.0, PROBABILITIES).items()}
    assert scaled.cost_quantiles(25.0, PROBABILITIES) == pytest.approx(expected)
    assert scaled.sample_cost(25.0, 100, rng=21) == pytest.approx(
        k * u.sample_cost(25.0, 100, rng=21)
    )


@pytest.mark.parametrize("mode", [M.RELATIVE, M.RELATIVE_MEAN_PRESERVING])
def test_scaled_relative_output(bounded, mode):
    u = UncertainCostFunction.from_bounds(bounded, mode=mode)
    k = 0.8
    scaled = u.scaled(k)
    assert scaled.output_uncertainty == u.output_uncertainty
    expected = {p: k * q for p, q in u.cost_quantiles(25.0, PROBABILITIES).items()}
    assert scaled.cost_quantiles(25.0, PROBABILITIES) == pytest.approx(expected)


def test_scaled_parameter_uncertainty(function):
    exponent = Distribution(K.UNIFORM, lower=-0.55, upper=-0.45)
    u = uncertain(
        function,
        parameter_uncertainty={
            "coefficient": Distribution(K.NORMAL, mean=1000.0, std_dev=100.0),
            "exponent": exponent,
        },
    )
    k = 1.1
    scaled = u.scaled(k)
    coefficient = scaled.parameter_uncertainty["coefficient"]
    assert coefficient.kind is K.NORMAL
    assert (coefficient.mean, coefficient.std_dev) == pytest.approx((1100.0, 110.0))
    assert scaled.parameter_uncertainty["exponent"] == exponent
    assert scaled.sample_cost(25.0, 200, rng=22) == pytest.approx(
        k * u.sample_cost(25.0, 200, rng=22), rel=1e-12
    )


def test_price_index_keeps_uncertainty(bounded):
    u = UncertainCostFunction.from_bounds(bounded, mode=M.ABSOLUTE)
    adjusted = PriceIndex({2026: 0.10}).adjust_function(u, 2026)
    assert type(adjusted) is UncertainCostFunction
    assert adjusted.price_year == 2026
    expected = {p: 1.1 * q for p, q in u.cost_quantiles(25.0, PROBABILITIES).items()}
    assert adjusted.cost_quantiles(25.0, PROBABILITIES) == pytest.approx(expected)


def test_with_form_refuses_random_parameters(function):
    u = uncertain(
        function,
        parameter_uncertainty={
            "coefficient": Distribution(K.NORMAL, mean=1000.0, std_dev=100.0)
        },
    )
    with pytest.raises(ValueError, match="parameter_uncertainty"):
        u.with_form(PowerLaw(0.0, 900.0, -0.5))


def test_with_form_without_random_parameters(function):
    # a deterministic entry is part of the form, so it does not block
    output = Distribution(K.UNIFORM, lower=-20.0, upper=20.0)
    u = uncertain(
        function,
        parameter_uncertainty={"coefficient": Distribution(K.DETERMINISTIC, mean=1.0)},
        output_uncertainty=output,
        output_mode=M.ABSOLUTE,
    )
    replaced = u.with_form(Constant(150.0))
    assert isinstance(replaced, UncertainCostFunction)
    assert replaced.form == Constant(150.0)
    assert replaced.parameter_uncertainty == {}
    assert replaced.output_uncertainty == output
    assert replaced.output_mode is M.ABSOLUTE
    assert replaced.cost_quantiles(25.0, [0.5]) == pytest.approx({0.5: 150.0})
