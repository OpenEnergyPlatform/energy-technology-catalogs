"""Tests of the unit sizing strategies.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import dataclasses
import warnings

import pytest

from energy_technology_catalogs.costs.core import CostBasis
from energy_technology_catalogs.costs.forms import Polynomial
from energy_technology_catalogs.costs.unit_sizing import (
    STRATEGIES,
    SizeSplit,
    split_equal,
    split_logarithmic,
    split_optimal,
    split_size,
)


@pytest.fixture
def progressive(function):
    """Absolute cost ``10 * size + size ** 2`` EUR for 10 to 100 kW."""
    return dataclasses.replace(
        function, form=Polynomial((0.0, 10.0, 1.0)), cost_basis=CostBasis.ABSOLUTE
    )


# ----------------------------------------------------------------------
# split_equal / split_logarithmic
# ----------------------------------------------------------------------


def test_split_equal():
    assert split_equal(250.0, 100.0) == pytest.approx([250 / 3] * 3)


def test_split_equal_fits_in_one_unit():
    assert split_equal(30.0, 100.0) == [30.0]


@pytest.mark.parametrize(
    ("total", "size_max"), [(0.0, 100.0), (-1.0, 100.0), (10.0, 0.0)]
)
def test_split_equal_rejects_non_positive(total, size_max):
    with pytest.raises(ValueError):
        split_equal(total, size_max)


def test_split_logarithmic_sums_to_total_and_increases():
    sizes = split_logarithmic(250.0, 10.0, 100.0)
    assert len(sizes) == 3
    assert sum(sizes) == pytest.approx(250.0)
    assert sizes == sorted(sizes)


def test_split_logarithmic_has_at_least_two_units():
    assert len(split_logarithmic(30.0, 10.0, 100.0)) == 2


@pytest.mark.parametrize(
    ("total", "size_min", "size_max"),
    [
        (250.0, 0.0, 100.0),
        (250.0, -1.0, 100.0),
        (250.0, 100.0, 100.0),
        (0.0, 10.0, 100.0),
    ],
)
def test_split_logarithmic_rejects_invalid_bounds(total, size_min, size_max):
    with pytest.raises(ValueError):
        split_logarithmic(total, size_min, size_max)


def test_unit_sizes_match_reference(reference_data):
    data = reference_data("unit_sizes")
    assert data["cases"]
    for case in data["cases"]:
        total, lo, hi = case["total_size"], case["size_min"], case["size_max"]
        assert split_equal(total, hi) == pytest.approx(case["equal"], rel=1e-9)
        assert split_logarithmic(total, lo, hi) == pytest.approx(
            case["logarithmic"], rel=1e-9
        )


# ----------------------------------------------------------------------
# split_optimal
# ----------------------------------------------------------------------


def test_optimal_prefers_few_units_for_degressive_cost(function):
    assert len(split_optimal(function, 250.0)) == 3


def test_optimal_prefers_many_units_for_progressive_cost(progressive):
    assert split_optimal(progressive, 250.0) == pytest.approx([10.0] * 25)


def test_optimal_respects_max_units(progressive):
    assert split_optimal(progressive, 250.0, max_units=5) == pytest.approx([50.0] * 5)


def test_optimal_with_zero_size_min_goes_up_to_max_units(progressive):
    f = dataclasses.replace(progressive, size_min=0.0)
    assert split_optimal(f, 250.0, max_units=50) == pytest.approx([5.0] * 50)


def test_optimal_ties_go_to_fewer_units(function):
    linear = dataclasses.replace(
        function, form=Polynomial((0.0, 10.0)), cost_basis=CostBasis.ABSOLUTE
    )
    assert split_optimal(linear, 250.0) == pytest.approx([250 / 3] * 3)


def test_optimal_below_size_min_gives_one_unit(progressive):
    assert split_optimal(progressive, 5.0) == pytest.approx([5.0])


def test_optimal_rejects_invalid_arguments(function):
    with pytest.raises(ValueError):
        split_optimal(function, 0.0)
    with pytest.raises(ValueError):
        split_optimal(function, 250.0, max_units=0)


# ----------------------------------------------------------------------
# split_size
# ----------------------------------------------------------------------


def test_strategies():
    assert STRATEGIES == ("single_unit", "equal", "logarithmic", "optimal")


def test_total_is_sum_of_absolute_unit_costs(function):
    split = split_size(function, 250.0, "equal")
    assert isinstance(split, SizeSplit)
    assert split.strategy == "equal"
    assert split.unit_sizes == pytest.approx((250 / 3,) * 3)
    assert split.unit_costs == pytest.approx([1000 * (250 / 3) ** 0.5] * 3)
    assert split.total_cost == pytest.approx(3 * 1000 * (250 / 3) ** 0.5)
    assert isinstance(split.unit_costs, tuple)
    assert all(isinstance(c, float) for c in split.unit_costs)


def test_single_unit_extrapolates_without_warning(function):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split = split_size(function, 250.0)
    assert split.strategy == "single_unit"
    assert split.unit_sizes == (250.0,)
    assert split.unit_costs == pytest.approx((1000 * 250**0.5,))
    assert split.total_cost == pytest.approx(1000 * 250**0.5)


def test_split_size_passes_installation_and_tax(function):
    taxed = dataclasses.replace(function, tax_rate=0.25)
    net = split_size(taxed, 250.0, "equal", include_installation=False)
    gross = split_size(taxed, 250.0, "equal", include_tax=True)
    # installation share 0.4 removed; form values are gross with 25 % tax
    assert net.total_cost == pytest.approx(3 * 1000 * (250 / 3) ** 0.5 * 0.6 / 1.25)
    assert gross.total_cost == pytest.approx(3 * 1000 * (250 / 3) ** 0.5)


def test_split_size_logarithmic(function):
    split = split_size(function, 250.0, "logarithmic")
    assert split.unit_sizes == pytest.approx(
        tuple(split_logarithmic(250.0, 10.0, 100.0))
    )
    assert split.total_cost == pytest.approx(
        sum(1000 * s**0.5 for s in split.unit_sizes)
    )


def test_split_size_optimal(progressive):
    split = split_size(progressive, 250.0, "optimal")
    assert len(split.unit_sizes) == 25
    assert split.total_cost == pytest.approx(25 * (10 * 10 + 10**2))


# reference "Size distribution" group, with a function of range 10-500 kW


@pytest.fixture
def wide(function):
    return dataclasses.replace(function, size_max=500.0)


def test_reference_single_unit(wide):
    split = split_size(wide, 200.0, "single_unit")
    assert len(split.unit_sizes) == 1
    assert split.unit_sizes[0] == pytest.approx(200.0)


def test_reference_equal_multiple_units(wide):
    split = split_size(wide, 2000.0, "equal")
    assert len(split.unit_sizes) >= 2
    assert sum(split.unit_sizes) == pytest.approx(2000.0)


def test_reference_unit_costs_returned(wide):
    split = split_size(wide, 2000.0, "equal")
    assert len(split.unit_costs) == len(split.unit_sizes)


def test_reference_total_cost_equals_sum(wide):
    split = split_size(wide, 2000.0, "equal")
    assert split.total_cost == pytest.approx(sum(split.unit_costs))


def test_reference_logarithmic_zero_lower_raises(wide):
    f = dataclasses.replace(wide, size_min=0.0)
    with pytest.raises(ValueError):
        split_size(f, 1500.0, "logarithmic")


def test_reference_unknown_strategy_raises(function):
    with pytest.raises(ValueError, match="unknown_strategy"):
        split_size(function, 250.0, "unknown_strategy")


@pytest.mark.parametrize("total", [0.0, -5.0, float("nan")])
def test_split_size_rejects_non_positive_total(function, total):
    with pytest.raises(ValueError):
        split_size(function, total, "single_unit")
