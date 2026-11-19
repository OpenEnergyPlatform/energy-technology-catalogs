"""Tests of the function forms.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import dataclasses
import math

import numpy as np
import pytest

from energy_technology_catalogs.costs.forms import (
    FORM_KINDS,
    Constant,
    Exponential,
    Linear,
    Logarithmic,
    PiecewiseLinear,
    Polynomial,
    PowerLaw,
    form_from_dict,
    form_from_parameters,
)


ALL_FORMS = [
    PowerLaw(100.0, 50.0, 0.5),
    Linear(10.0, 2.0),
    Polynomial((1.0, 2.0, 3.0)),
    Logarithmic(5.0, 2.0),
    Exponential(1.0, 2.0, 0.5),
    Constant(7.0),
    PiecewiseLinear((1.0, 3.0, 5.0), (10.0, 30.0, 20.0)),
]


def test_evaluate_known_values():
    assert PowerLaw(100.0, 50.0, 0.5).evaluate(4.0) == pytest.approx(200.0)
    assert Linear(10.0, 2.0).evaluate(5.0) == pytest.approx(20.0)
    assert Polynomial((1.0, 2.0, 3.0)).evaluate(2.0) == pytest.approx(17.0)
    assert Logarithmic(5.0, 2.0).evaluate(math.e) == pytest.approx(7.0)
    assert Exponential(1.0, 2.0, 0.5).evaluate(2.0) == pytest.approx(1.0 + 2.0 * math.e)
    assert Constant(7.0).evaluate(np.array([1.0, 2.0])).tolist() == [7.0, 7.0]


def test_piecewise_interpolates_and_extends_end_segments():
    form = PiecewiseLinear((1.0, 3.0, 5.0), (10.0, 30.0, 20.0))
    result = form.evaluate(np.array([2.0, 4.0, 7.0, 0.0]))
    assert result.tolist() == pytest.approx([20.0, 25.0, 10.0, 0.0])
    assert form.evaluate(3.0) == pytest.approx(30.0)
    assert form.evaluate(5.0) == pytest.approx(20.0)


def test_piecewise_from_points_sorts_and_rejects_duplicates():
    form = PiecewiseLinear.from_points([5.0, 1.0, 3.0], [20.0, 10.0, 30.0])
    assert form == PiecewiseLinear((1.0, 3.0, 5.0), (10.0, 30.0, 20.0))
    with pytest.raises(ValueError):
        PiecewiseLinear.from_points([1.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        PiecewiseLinear.from_points([1.0, 2.0], [1.0])


@pytest.mark.parametrize("form", ALL_FORMS, ids=lambda form: form.kind)
def test_scalar_in_float_out_array_in_array_out(form):
    assert isinstance(form.evaluate(2.0), float)
    result = form.evaluate(np.array([2.0, 4.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(form.evaluate(2.0))


@pytest.mark.parametrize("form", ALL_FORMS, ids=lambda form: form.kind)
def test_scaled_shifted_roundtrip(form):
    x = np.array([1.5, 2.0, 4.0])
    assert form.scaled(1.1).evaluate(x) == pytest.approx(1.1 * form.evaluate(x))
    assert form.shifted(20.0).evaluate(x) == pytest.approx(form.evaluate(x) + 20.0)
    assert form_from_dict(form.to_dict()) == form
    assert form.with_parameters(**form.parameters()) == form
    assert set(form.cost_parameter_names()) <= set(form.parameters())
    assert type(form.scaled(2.0)) is type(form)


@pytest.mark.parametrize("form", ALL_FORMS, ids=lambda form: form.kind)
def test_forms_are_frozen_and_registered(form):
    assert FORM_KINDS[form.kind] is type(form)
    field = dataclasses.fields(form)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(form, field, 1.0)


def test_parameter_names():
    assert list(PowerLaw(1.0, 2.0, 3.0).parameters()) == [
        "offset",
        "coefficient",
        "exponent",
    ]
    assert Polynomial((1.0, 2.0)).parameters() == {"c0": 1.0, "c1": 2.0}
    assert PiecewiseLinear((1.0, 2.0), (5.0, 6.0)).parameters() == {
        "v0": 5.0,
        "v1": 6.0,
    }
    assert PowerLaw(1.0, 2.0, 3.0).cost_parameter_names() == ("offset", "coefficient")
    assert Exponential(1.0, 2.0, 3.0).cost_parameter_names() == (
        "offset",
        "coefficient",
    )


def test_with_parameters_changes_one_value():
    assert Polynomial((1.0, 2.0, 3.0)).with_parameters(c1=5.0) == Polynomial(
        (1.0, 5.0, 3.0)
    )
    assert PiecewiseLinear((1.0, 2.0), (5.0, 6.0)).with_parameters(
        v1=9.0
    ) == PiecewiseLinear((1.0, 2.0), (5.0, 9.0))
    assert PowerLaw(1.0, 2.0, 3.0).with_parameters(exponent=0.5) == PowerLaw(
        1.0, 2.0, 0.5
    )


def test_evaluate_with_parameter_arrays():
    out = PowerLaw(0.0, 1000.0, -0.5).evaluate(
        25.0, {"coefficient": np.array([900.0, 1100.0])}
    )
    assert out.tolist() == pytest.approx([180.0, 220.0])


def test_piecewise_evaluate_with_value_arrays():
    form = PiecewiseLinear((1.0, 3.0), (10.0, 30.0))
    out = form.evaluate(2.0, {"v1": np.array([30.0, 50.0])})
    assert out.tolist() == pytest.approx([20.0, 30.0])


def test_constant_with_value_array():
    out = Constant(7.0).evaluate(3.0, {"value": np.array([1.0, 2.0])})
    assert out.tolist() == [1.0, 2.0]


def test_integer_input_is_stored_as_float():
    form = PowerLaw(0, 1000, 1)
    assert form == PowerLaw(0.0, 1000.0, 1.0)
    assert all(isinstance(v, float) for v in form.parameters().values())
    assert Polynomial([1, 2]).coefficients == (1.0, 2.0)


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        PiecewiseLinear((1.0, 1.0), (1.0, 2.0))
    with pytest.raises(ValueError):
        PiecewiseLinear((1.0,), (1.0,))
    with pytest.raises(ValueError):
        PiecewiseLinear((1.0, 2.0), (1.0,))
    with pytest.raises(ValueError):
        Polynomial(())
    with pytest.raises(ValueError):
        Linear(1.0, 2.0).with_parameters(exponent=1.0)
    with pytest.raises(ValueError):
        Linear(1.0, 2.0).evaluate(1.0, {"exponent": 1.0})
    with pytest.raises(ValueError):
        form_from_dict({"kind": "spline"})
    with pytest.raises(ValueError):
        form_from_dict({"offset": 1.0})
    with pytest.raises(ValueError):
        form_from_dict({"kind": "linear", "offset": 1.0, "exponent": 2.0})


def test_to_dict_is_json_ready():
    assert PiecewiseLinear((1.0, 2.0), (5.0, 6.0)).to_dict() == {
        "kind": "piecewise_linear",
        "sizes": [1.0, 2.0],
        "values": [5.0, 6.0],
    }
    assert Linear(1.0, 2.0).to_dict() == {"kind": "linear", "offset": 1.0, "slope": 2.0}


def test_size_array_with_parameter_array_raises():
    for form in ALL_FORMS:
        name = next(iter(form.parameters()))
        with pytest.raises(ValueError):
            form.evaluate(np.array([2.0, 3.0]), {name: np.array([1.0, 2.0])})


@pytest.mark.parametrize(
    "build",
    [
        lambda: PowerLaw(0.0, float("nan"), 1.0),
        lambda: Linear(float("inf"), 1.0),
        lambda: PiecewiseLinear((1.0, float("nan"), 3.0), (1.0, 2.0, 3.0)),
        lambda: Polynomial("12"),  # type: ignore[arg-type]
        lambda: PiecewiseLinear("12", (1.0, 2.0)),  # type: ignore[arg-type]
    ],
)
def test_non_finite_or_text_parameters_raise(build):
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize("form", ALL_FORMS, ids=lambda form: form.kind)
def test_form_from_parameters_inverts_parameters(form):
    sizes = getattr(form, "sizes", None)
    assert form_from_parameters(form.kind, form.parameters(), sizes=sizes) == form


def test_form_from_parameters_errors():
    with pytest.raises(ValueError):
        form_from_parameters("spline", {})
    with pytest.raises(ValueError):
        form_from_parameters("linear", {"offset": 1.0, "exponent": 2.0})
    with pytest.raises(ValueError):
        form_from_parameters("polynomial", {"c0": 1.0, "c2": 2.0})
    with pytest.raises(ValueError):
        form_from_parameters("piecewise_linear", {"v0": 1.0, "v1": 2.0})


def test_forms_are_hashable():
    assert len({form for form in ALL_FORMS + ALL_FORMS}) == len(ALL_FORMS)


def test_offset_parameter_names():
    assert PowerLaw(1.0, 2.0, 3.0).offset_parameter_names() == ("offset",)
    assert Polynomial((1.0, 2.0)).offset_parameter_names() == ("c0",)
    assert Constant(1.0).offset_parameter_names() == ("value",)
    assert PiecewiseLinear((1.0, 2.0), (5.0, 6.0)).offset_parameter_names() == (
        "v0",
        "v1",
    )
