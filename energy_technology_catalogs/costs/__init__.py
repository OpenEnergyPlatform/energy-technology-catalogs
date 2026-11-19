"""Cost functions of energy technologies.

A cost function gives the investment cost of a technology as a function of
its size. The sub-package provides

- function forms (power law, linear, polynomial, logarithmic, exponential,
  constant, piecewise linear) and the cost function data model,
- price adjustment with a price index,
- uncertainty of parameters and of the cost,
- splitting a total size into units,
- fitting forms to data and calibrating existing functions,
- reading and writing CSV, Excel and JSON files, and a registry.

Ported from the Uni Kassel cost function module v3.0.0 and extended by
further function forms.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from energy_technology_catalogs.costs.core import (
    CostBasis,
    CostFunction,
    CostFunctionKey,
    TechnologyType,
)
from energy_technology_catalogs.costs.fitting import (
    FitResult,
    calibrate_cost_function,
    calibrate_to_point,
    fit_cost_form,
)
from energy_technology_catalogs.costs.forms import (
    Constant,
    CostForm,
    Exponential,
    Linear,
    Logarithmic,
    PiecewiseLinear,
    Polynomial,
    PowerLaw,
    form_from_dict,
    form_from_parameters,
)
from energy_technology_catalogs.costs.inflation import PriceIndex
from energy_technology_catalogs.costs.registry import CostFunctionRegistry
from energy_technology_catalogs.costs.serialization import (
    read_cost_functions,
    write_cost_functions,
)
from energy_technology_catalogs.costs.uncertainty import (
    Distribution,
    DistributionKind,
    OutputUncertaintyMode,
    UncertainCostFunction,
    distribution_from_bounds,
)
from energy_technology_catalogs.costs.unit_sizing import (
    SizeSplit,
    split_equal,
    split_logarithmic,
    split_optimal,
    split_size,
)


__all__ = [
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
]
