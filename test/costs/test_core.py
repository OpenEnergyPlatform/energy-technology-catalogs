"""Tests of the cost function data model.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import csv
import dataclasses
import warnings
from importlib import resources

import numpy as np
import pytest

from energy_technology_catalogs.costs.core import (
    CostBasis,
    CostFunction,
    CostFunctionKey,
    TechnologyType,
)
from energy_technology_catalogs.costs.forms import (
    Constant,
    Linear,
    Logarithmic,
    Polynomial,
    PowerLaw,
)


def make(**changes) -> CostFunction:
    """Specific cost 1000 * size ** -0.5 EUR/kW, 19 % tax, installation share 0.4 included."""
    base = dict(
        technology="Boiler",
        source="Test",
        technology_type=TechnologyType.BIOMASS_BOILER,
        size_unit="kW",
        investment_year=2025,
        size_min=10.0,
        size_max=100.0,
        form=PowerLaw(0.0, 1000.0, -0.5),
        tax_rate=0.19,
        installation_included=True,
        installation_share=Constant(0.4),
    )
    return CostFunction(**{**base, **changes})


REFERENCE_ROW = {
    "name": "Synthetic boiler",
    "technology_type": "BIOMASS_BOILER",
    "reference_unit": "kW",
    "investment_year": 2030,
    "reference_year": 2025,
    "validity_range": "[10,100]",
    "summand": 0.0,
    "coefficient": 12000.0,
    "exponent": -0.6,
    "fixed_om": float("nan"),
    "P_&_I_included": True,
    "P_&_I_factor": 0.4,
    "uncertainty_lower": -0.2,
    "uncertainty_upper": 0.2,
    "source": "Synthetic",
    "tax_rate": 0.0,
    "lifetime": 20,
}


# --- calculation -----------------------------------------------------------


def test_tax_installation_and_basis():
    f = make()
    assert f.calculate_cost(25.0, include_tax=True) == pytest.approx(200.0)
    assert f.calculate_cost(25.0) == pytest.approx(200.0 / 1.19)
    assert f.calculate_cost(
        25.0, include_tax=True, include_installation=False
    ) == pytest.approx(120.0)
    assert f.calculate_cost(25.0, include_tax=True, specific=False) == pytest.approx(
        5000.0
    )


def test_installation_added_when_not_included():
    f = make(installation_included=False, installation_share=Constant(0.2))
    assert f.calculate_cost(25.0, include_tax=True) == pytest.approx(250.0)
    assert f.calculate_cost(
        25.0, include_tax=True, include_installation=False
    ) == pytest.approx(200.0)


def test_without_installation_share_the_flag_has_no_effect():
    for included in (True, False):
        f = make(installation_included=included, installation_share=None, tax_rate=0.0)
        assert f.calculate_cost(25.0) == pytest.approx(200.0)
        assert f.calculate_cost(25.0, include_installation=False) == pytest.approx(
            200.0
        )


def test_installation_share_as_size_dependent_form():
    f = make(tax_rate=0.0, installation_share=Linear(0.5, -0.004))
    # share at 25 kW: 0.5 - 0.1 = 0.4
    assert f.calculate_cost(25.0, include_installation=False) == pytest.approx(120.0)


def test_absolute_basis():
    f = make(
        form=Linear(1000.0, 50.0),
        cost_basis=CostBasis.ABSOLUTE,
        tax_rate=0.0,
        installation_share=None,
    )
    assert f.calculate_cost(20.0, specific=False) == pytest.approx(2000.0)
    assert f.calculate_cost(20.0) == pytest.approx(100.0)


def test_specific_flag_identity_on_arrays():
    f = make()
    sizes = np.array([10.0, 25.0, 100.0])
    assert f.calculate_cost(sizes, specific=False) == pytest.approx(
        f.calculate_cost(sizes) * sizes
    )


def test_scalar_in_float_out_array_in_array_out():
    f = make()
    assert isinstance(f.calculate_cost(25.0), float)
    assert isinstance(f.calculate_cost(25), float)
    result = f.calculate_cost(np.array([25.0, 50.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_tax_net_gross_relationship():
    f = make()
    net = f.calculate_cost(50.0)
    gross = f.calculate_cost(50.0, include_tax=True)
    assert gross == pytest.approx(net * 1.19)
    assert net < gross
    zero_tax = make(tax_rate=0.0)
    assert zero_tax.calculate_cost(50.0) == zero_tax.calculate_cost(
        50.0, include_tax=True
    )


def test_clamps_array_with_one_warning():
    f = make(tax_rate=0.0)
    with pytest.warns(UserWarning) as caught:
        out = f.calculate_cost(np.array([4.0, 25.0, 400.0]))
    assert len(caught) == 1
    assert out.tolist() == pytest.approx([1000.0 / 10**0.5, 200.0, 100.0])


def test_clamped_absolute_cost_uses_specific_cost_of_the_limit():
    f = make(tax_rate=0.0)
    with pytest.warns(UserWarning):
        assert f.calculate_cost(400.0, specific=False) == pytest.approx(100.0 * 400.0)


def test_extrapolate_evaluates_outside_the_range_without_warning():
    f = make(tax_rate=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f.calculate_cost(400.0, extrapolate=True) == pytest.approx(50.0)
        assert f.calculate_cost(25.0) == pytest.approx(200.0)


@pytest.mark.parametrize(
    "size", [0.0, -1.0, float("nan"), np.array([5.0, 0.0]), np.array([np.nan, 20.0])]
)
def test_non_positive_size_raises(size):
    with pytest.raises(ValueError):
        make().calculate_cost(size)
    with pytest.raises(ValueError):
        make(form=Logarithmic(5.0, 2.0)).calculate_cost(size, extrapolate=True)
    with pytest.raises(ValueError):
        make(cost_basis=CostBasis.ABSOLUTE).calculate_cost(size)


def test_installation_share_out_of_range_raises():
    f = make(installation_share=Linear(0.5, 0.01))  # reaches 1.0 at size 50
    with pytest.raises(ValueError, match="50"):
        f.calculate_cost(50.0, include_installation=False)
    with pytest.raises(ValueError, match="50"):
        f.calculate_cost(np.array([20.0, 50.0]), include_installation=False)
    assert f.calculate_cost(20.0, include_installation=False) > 0
    # the share is not needed when the cost is returned as stored
    assert f.calculate_cost(50.0) > 0


def test_fixed_om():
    f = make(fixed_om=Constant(12.0))
    assert f.calculate_fixed_om(25.0) == pytest.approx(12.0)
    assert f.calculate_fixed_om(25.0, specific=False) == pytest.approx(300.0)
    absolute = make(
        form=Linear(1000.0, 50.0),
        cost_basis=CostBasis.ABSOLUTE,
        fixed_om=Polynomial((100.0, 2.0)),
    )
    assert absolute.calculate_fixed_om(50.0, specific=False) == pytest.approx(200.0)
    assert absolute.calculate_fixed_om(50.0) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        make().calculate_fixed_om(25.0)


# --- identity and validation ----------------------------------------------


def test_key_equality_hash_and_str():
    key = CostFunctionKey("Boiler", "Test", 2025)
    assert key == CostFunctionKey("Boiler", "Test", 2025)
    assert hash(key) == hash(CostFunctionKey("Boiler", "Test", 2025))
    assert key != CostFunctionKey("Boiler", "Other", 2025)
    assert key != CostFunctionKey("Tank", "Test", 2025)
    assert key != CostFunctionKey("Boiler", "Test", 2030)
    assert {key: 1}[CostFunctionKey("Boiler", "Test", 2025)] == 1
    assert str(key) == "Boiler | Test | 2025"


def test_key_invalid_values_raise():
    with pytest.raises(ValueError):
        CostFunctionKey(" ", "Test", 2025)
    with pytest.raises(ValueError):
        CostFunctionKey("Boiler", "", 2025)
    with pytest.raises(TypeError):
        CostFunctionKey("Boiler", "Test", 2025.0)


def test_name_and_key():
    f = make()
    assert f.name == "Boiler | Test | 2025"
    assert f.key == CostFunctionKey("Boiler", "Test", 2025)


def test_price_year_defaults_to_investment_year():
    assert make().price_year == 2025
    f = make(investment_year=2030, price_year=2025)
    assert (f.investment_year, f.price_year) == (2030, 2025)
    assert f.key.investment_year == 2030


def test_function_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().tax_rate = 0.0


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"technology": " "}, ValueError),
        ({"source": ""}, ValueError),
        ({"technology_type": "biomass_boiler"}, TypeError),
        ({"cost_basis": "specific"}, TypeError),
        ({"form": 5.0}, TypeError),
        ({"fixed_om": 5.0}, TypeError),
        ({"installation_share": 0.4}, TypeError),
        ({"size_min": -1.0}, ValueError),
        ({"size_min": 100.0}, ValueError),
        ({"size_min": 200.0}, ValueError),
        ({"investment_year": 1800}, ValueError),
        ({"price_year": 2300}, ValueError),
        ({"investment_year": 2025.0}, TypeError),
        ({"investment_year": True}, TypeError),
        ({"tax_rate": -0.1}, ValueError),
        ({"uncertainty_lower": 0.1}, ValueError),
        ({"uncertainty_upper": -0.1}, ValueError),
        ({"lifetime_a": 0.0}, ValueError),
    ],
)
def test_invalid_fields_raise(changes, error):
    with pytest.raises(error):
        make(**changes)


def test_zero_lower_bound_and_numpy_years_are_accepted():
    f = make(size_min=0.0, investment_year=np.int64(2030))
    assert f.size_min == 0.0
    assert type(f.investment_year) is int


# --- dictionaries ----------------------------------------------------------


def test_dict_roundtrip():
    f = make(
        fixed_om=Constant(12.0),
        lifetime_a=20.0,
        price_year=2020,
        uncertainty_lower=-0.1,
        uncertainty_upper=0.3,
    )
    data = f.to_dict()
    assert CostFunction.from_dict(data) == f
    assert "name" not in data
    assert data["form"] == {
        "kind": "power_law",
        "offset": 0.0,
        "coefficient": 1000.0,
        "exponent": -0.5,
    }
    assert data["technology_type"] == "biomass_boiler"
    assert data["cost_basis"] == "specific"


def test_dict_roundtrip_absolute_basis_without_optional_forms():
    f = make(
        form=Polynomial((1.0, 2.0, 3.0)),
        cost_basis=CostBasis.ABSOLUTE,
        installation_share=None,
    )
    assert CostFunction.from_dict(f.to_dict()) == f


def test_from_dict_reference_layout():
    f = CostFunction.from_dict(REFERENCE_ROW)
    assert (f.technology, f.size_unit, f.size_min, f.size_max) == (
        "Synthetic boiler",
        "kW",
        10.0,
        100.0,
    )
    assert f.form == PowerLaw(0.0, 12000.0, -0.6)
    assert f.cost_basis is CostBasis.SPECIFIC
    assert f.installation_included is True
    assert f.installation_share == Constant(0.4)
    assert (f.investment_year, f.price_year) == (2030, 2025)
    assert (f.fixed_om, f.lifetime_a) == (None, 20.0)
    assert (f.uncertainty_lower, f.uncertainty_upper) == (-0.2, 0.2)


def test_from_dict_reference_installation_multiplier():
    row = {**REFERENCE_ROW, "P_&_I_included": False, "P_&_I_factor": 1.25}
    f = CostFunction.from_dict(row)
    assert f.installation_included is False
    assert f.installation_share.value == pytest.approx(0.2)
    # the reference multiplies by the factor when planning and installation are added
    base = 12000.0 * 50.0**-0.6
    assert f.calculate_cost(50.0) == pytest.approx(base * 1.25)
    assert f.calculate_cost(50.0, include_installation=False) == pytest.approx(base)


def test_from_dict_reference_installation_defaults():
    neutral = CostFunction.from_dict(
        {**REFERENCE_ROW, "P_&_I_included": False, "P_&_I_factor": 1.0}
    )
    assert neutral.installation_share is None
    missing = CostFunction.from_dict(
        {**REFERENCE_ROW, "P_&_I_included": True, "P_&_I_factor": float("nan")}
    )
    assert missing.installation_share is None
    with pytest.raises(ValueError):
        CostFunction.from_dict({**REFERENCE_ROW, "P_&_I_factor": 1.0})
    with pytest.raises(ValueError):
        CostFunction.from_dict(
            {**REFERENCE_ROW, "P_&_I_included": False, "P_&_I_factor": 0.5}
        )


def test_from_dict_reference_scalar_fixed_om_and_symmetric_uncertainty():
    row = {**REFERENCE_ROW, "fixed_om": 12.0, "uncertainty_pct": 0.15}
    del row["uncertainty_lower"], row["uncertainty_upper"]
    f = CostFunction.from_dict(row)
    assert f.fixed_om == Constant(12.0)
    assert (f.uncertainty_lower, f.uncertainty_upper) == (-0.15, 0.15)


def test_from_dict_size_range_variants():
    row = {k: v for k, v in REFERENCE_ROW.items() if k != "validity_range"}
    split = CostFunction.from_dict(
        {**row, "validity_range_lower": 5.0, "validity_range_upper": 50.0}
    )
    assert (split.size_min, split.size_max) == (5.0, 50.0)
    as_list = CostFunction.from_dict({**row, "validity_range": [5, 50]})
    assert (as_list.size_min, as_list.size_max) == (5.0, 50.0)
    nan_lower = CostFunction.from_dict(
        {**row, "validity_range_lower": float("nan"), "validity_range_upper": 50.0}
    )
    assert nan_lower.size_min == 0.0
    with pytest.raises(ValueError):
        CostFunction.from_dict({**row, "validity_range_lower": 5.0})
    with pytest.raises(ValueError):
        CostFunction.from_dict(row)
    with pytest.raises(ValueError):
        CostFunction.from_dict({**row, "validity_range": [1, 2, 3]})


def test_from_dict_years():
    f = CostFunction.from_dict(
        {**REFERENCE_ROW, "investment_year": 2030.0, "reference_year": float("nan")}
    )
    assert (f.investment_year, f.price_year) == (2030, 2030)
    row = {k: v for k, v in REFERENCE_ROW.items() if k != "investment_year"}
    with pytest.warns(DeprecationWarning):
        legacy = CostFunction.from_dict(row)
    assert legacy.investment_year == 2025
    del row["reference_year"]
    with pytest.raises(ValueError):
        CostFunction.from_dict(row)


def test_from_dict_errors():
    with pytest.raises(TypeError):
        CostFunction.from_dict({**REFERENCE_ROW, "technology_type": 5})
    with pytest.raises(ValueError):
        CostFunction.from_dict({**REFERENCE_ROW, "technology_type": "spaceship"})
    row = {k: v for k, v in REFERENCE_ROW.items() if k != "coefficient"}
    with pytest.raises(ValueError):
        CostFunction.from_dict(row)


# --- technology types ------------------------------------------------------


def test_technology_type_lookup():
    assert TechnologyType.from_string("heat_pump") is TechnologyType.HEAT_PUMP
    assert TechnologyType.from_string(" HEAT_PUMP ") is TechnologyType.HEAT_PUMP
    assert TechnologyType.from_string("ttes_small") is TechnologyType.TTES_SMALL
    with pytest.raises(ValueError):
        TechnologyType.from_string("spaceship")
    with pytest.raises(ValueError):
        TechnologyType.from_string(5)


def test_technology_type_metadata():
    path = resources.files("energy_technology_catalogs").joinpath(
        "data/technology_types.csv"
    )
    with path.open(encoding="utf-8") as file:
        rows = list(
            csv.DictReader(
                (line for line in file if not line.startswith("#")), delimiter=";"
            )
        )
    assert [row["technology_type"] for row in rows] == [t.name for t in TechnologyType]
    assert all(t.label("en") and t.label("de") for t in TechnologyType)
    assert TechnologyType.BIOMASS_BOILER.label() == "Biomass boiler"
    assert TechnologyType.BIOMASS_BOILER.label("de") == "Biomassekessel"
    assert isinstance(TechnologyType.BIOMASS_BOILER.oeo_iri, str)
    with pytest.raises(ValueError):
        TechnologyType.HEAT_PUMP.label("fr")


# --- parity with the reference module ---------------------------------------


def test_cost_grid_matches_reference(reference_data):
    data = reference_data("cost_grid")
    functions = [CostFunction.from_dict(row) for row in data["functions"]]
    assert len(data["cases"]) == 1040
    for case in data["cases"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cost = functions[case["function"]].calculate_cost(
                case["size"],
                include_installation=case["include_pi"],
                include_tax=case["return_gross"],
                extrapolate=case["extrapolate"],
            )
        assert cost == pytest.approx(case["cost"], rel=1e-9), case


# --- robustness of inputs ---------------------------------------------------


def test_technology_types_cover_the_reference_module():
    assert len(TechnologyType) == 34
    assert (
        TechnologyType.from_string("heat_pump_absorption")
        is TechnologyType.HEAT_PUMP_ABSORPTION
    )


def test_from_dict_names_of_this_package_win_over_reference_names():
    f = make(tax_rate=0.19, price_year=2020)
    for data in (
        {"name": "Label | x | 1", "taxes_included": 0.5, **f.to_dict()},
        {**f.to_dict(), "name": "Label | x | 1", "taxes_included": 0.5},
        {"reference_year": float("nan"), **f.to_dict()},
    ):
        assert CostFunction.from_dict(data) == f


def test_from_dict_component_wins_over_name_in_any_order():
    row = {k: v for k, v in REFERENCE_ROW.items() if k != "name"}
    first = CostFunction.from_dict({"name": "Label", "component": "Boiler", **row})
    last = CostFunction.from_dict({"component": "Boiler", **row, "name": "Label"})
    assert first.technology == last.technology == "Boiler"


def test_from_dict_empty_reference_value_does_not_hide_a_value():
    row = {**REFERENCE_ROW, "taxes_included": float("nan"), "tax_rate": 0.19}
    assert CostFunction.from_dict(row).tax_rate == 0.19


def test_from_dict_reference_zero_fixed_om_means_unknown():
    assert CostFunction.from_dict({**REFERENCE_ROW, "fixed_om": 0.0}).fixed_om is None
    # in the layout of this package a zero is a value
    data = make(fixed_om=Constant(0.0)).to_dict()
    assert CostFunction.from_dict(data).fixed_om == Constant(0.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"validity_range": "[10;100]"},
        {"validity_range": "10 100"},
        {"validity_range": [None, 100]},
        {"coefficient": None},
        {"coefficient": float("nan")},
        {"exponent": float("nan")},
        {"investment_year": 2030.7},
        {"P_&_I_included": "False"},
    ],
)
def test_from_dict_unreadable_values_raise_value_error(changes):
    with pytest.raises(ValueError):
        CostFunction.from_dict({**REFERENCE_ROW, **changes})


def test_from_dict_reads_years_given_as_text():
    f = CostFunction.from_dict({**REFERENCE_ROW, "investment_year": "2030.0"})
    assert f.investment_year == 2030


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"tax_rate": float("nan")}, ValueError),
        ({"uncertainty_lower": float("nan")}, ValueError),
        ({"uncertainty_upper": float("nan")}, ValueError),
        ({"lifetime_a": float("nan")}, ValueError),
        ({"installation_included": "False"}, TypeError),
        ({"installation_included": 2}, TypeError),
    ],
)
def test_nan_and_non_boolean_fields_raise(changes, error):
    with pytest.raises(error):
        make(**changes)


def test_numpy_bool_and_numpy_year_are_accepted():
    f = make(installation_included=np.bool_(True))
    assert f.installation_included is True
    key = CostFunctionKey("Boiler", "Test", np.int64(2025))
    assert key == CostFunctionKey("Boiler", "Test", 2025)
    assert type(key.investment_year) is int
    with pytest.raises(ValueError):
        CostFunctionKey(5, "Test", 2025)


def test_replace_keeps_price_year_and_function_is_hashable():
    f = make()
    later = dataclasses.replace(f, investment_year=2030)
    assert (later.investment_year, later.price_year) == (2030, 2025)
    assert hash(f) == hash(make())


def test_clamped_absolute_basis():
    f = make(
        form=Linear(1000.0, 50.0),
        cost_basis=CostBasis.ABSOLUTE,
        tax_rate=0.0,
        installation_share=None,
    )
    with pytest.warns(UserWarning):
        # specific cost of the 100 kW limit: (1000 + 5000) / 100 = 60 EUR/kW
        assert f.calculate_cost(400.0) == pytest.approx(60.0)
    with pytest.warns(UserWarning):
        assert f.calculate_cost(400.0, specific=False) == pytest.approx(24000.0)
    assert f.calculate_cost(400.0, specific=False, extrapolate=True) == pytest.approx(
        21000.0
    )


@pytest.mark.parametrize("size", [0.0, -1.0, float("nan")])
def test_fixed_om_non_positive_size_raises(size):
    with pytest.raises(ValueError):
        make(fixed_om=Constant(12.0)).calculate_fixed_om(size)


def test_fixed_om_is_returned_without_tax():
    assert make(tax_rate=0.19, fixed_om=Constant(12.0)).calculate_fixed_om(25.0) == 12.0


@pytest.mark.parametrize(
    "share, included",
    [
        (Constant(-0.1), True),
        (Linear(0.5, 0.01), False),
        (Constant(1.5), True),
    ],
)
def test_installation_share_outside_unit_interval_raises(share, included):
    f = make(installation_included=included, installation_share=share)
    with pytest.raises(ValueError):
        f.calculate_cost(50.0, include_installation=not included)


def test_installation_share_error_names_the_size_used():
    f = make(installation_share=Linear(0.5, 0.01))
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="100"):
        f.calculate_cost(400.0, include_installation=False)


def test_finalize_cost_accepts_sample_arrays_for_a_scalar_size():
    f = make()
    size = np.asarray(25.0)
    samples = np.array([180.0, 200.0, 220.0])
    out = f._finalize_cost(
        samples,
        size,
        size,
        include_installation=False,
        include_tax=False,
        specific=False,
    )
    assert out.tolist() == pytest.approx((samples * 0.6 / 1.19 * 25.0).tolist())
    gross = f._apply_installation(samples, size, False)
    assert gross.tolist() == pytest.approx((samples * 0.6).tolist())
    net = f._apply_tax_and_basis(gross, size, size, include_tax=False, specific=True)
    assert net.tolist() == pytest.approx((samples * 0.6 / 1.19).tolist())


def test_scaled_scales_money_only():
    f = make(fixed_om=Constant(12.0), uncertainty_lower=-0.1, uncertainty_upper=0.2)
    g = f.scaled(1.1)
    assert g.form == f.form.scaled(1.1)
    assert g.fixed_om.value == pytest.approx(13.2)
    assert dataclasses.replace(g, form=f.form, fixed_om=f.fixed_om) == f
    assert make().scaled(2.0).fixed_om is None
    for factor in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            f.scaled(factor)


def test_with_form_replaces_the_form_only():
    f = make()
    g = f.with_form(Linear(1.0, 2.0))
    assert g.form == Linear(1.0, 2.0)
    assert g.with_form(f.form) == f
