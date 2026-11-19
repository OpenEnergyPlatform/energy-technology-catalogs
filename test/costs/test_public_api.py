"""Tests of the public interface of the cost function sub-package.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import pytest

from energy_technology_catalogs import costs


PUBLIC_NAMES = {
    "CostForm",
    "PowerLaw",
    "Linear",
    "Polynomial",
    "Logarithmic",
    "Exponential",
    "Constant",
    "PiecewiseLinear",
    "form_from_dict",
    "form_from_parameters",
    "TechnologyType",
    "CostBasis",
    "CostFunctionKey",
    "CostFunction",
    "PriceIndex",
    "DistributionKind",
    "Distribution",
    "OutputUncertaintyMode",
    "distribution_from_bounds",
    "UncertainCostFunction",
    "SizeSplit",
    "split_size",
    "split_equal",
    "split_logarithmic",
    "split_optimal",
    "FitResult",
    "fit_cost_form",
    "calibrate_cost_function",
    "calibrate_to_point",
    "read_cost_functions",
    "write_cost_functions",
    "CostFunctionRegistry",
}


def test_public_names():
    assert set(costs.__all__) == PUBLIC_NAMES
    assert all(hasattr(costs, name) for name in costs.__all__)


@pytest.mark.parametrize("suffix", [".csv", ".json", ".xlsx"])
def test_fit_adjust_write_read(tmp_path, suffix):
    sizes = [10.0, 20.0, 50.0, 100.0, 200.0]  # m3
    unit_costs = [9000.0, 13500.0, 26000.0, 46500.0, 87000.0]  # EUR per unit
    result = costs.fit_cost_form(sizes, unit_costs, "linear")
    tank = costs.CostFunction(
        technology="Storage tank",
        source="Own offers",
        technology_type=costs.TechnologyType.TTES,
        size_unit="m3",
        investment_year=2025,
        size_min=10.0,
        size_max=200.0,
        form=result.form,
        cost_basis=costs.CostBasis.ABSOLUTE,
    )
    fitted = tank.calculate_cost(80.0, specific=False)
    assert fitted == pytest.approx(
        result.form.parameters()["offset"] + 80.0 * result.form.parameters()["slope"]
    )

    adjusted = costs.PriceIndex({2026: 0.10}).adjust_function(tank, 2026)
    path = tmp_path / f"functions{suffix}"
    costs.write_cost_functions([adjusted], path)

    registry = costs.CostFunctionRegistry()
    assert registry.load(path) == 1
    loaded = registry.get(tank.key)
    assert loaded.price_year == 2026
    assert loaded.calculate_cost(80.0, specific=False) == pytest.approx(1.10 * fitted)
    assert loaded.calculate_cost(80.0) == pytest.approx(1.10 * fitted / 80.0)
