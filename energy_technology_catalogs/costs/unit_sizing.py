"""Splitting a total size into units and costing them.

A required total size (e.g. 250 kW of boiler capacity) can be built as one
unit or as several smaller ones. The strategies here return the unit sizes:

- ``single_unit``: one unit of the total size,
- ``equal``: the fewest equal units that do not exceed ``size_max``,
- ``logarithmic``: at least two log-spaced units between ``size_min`` and
  ``size_max``, rescaled so that they sum to the total,
- ``optimal``: the number of equal units with the lowest total absolute cost,
  found by evaluating the cost function for every admissible unit count.

The total cost of a split is the sum of the absolute costs of its units,
evaluated with extrapolation outside the valid size range. Sizes are in the
``size_unit`` of the cost function, costs in its ``currency``.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from energy_technology_catalogs.costs.core import CostFunction


logger = logging.getLogger(__name__)

STRATEGIES: tuple[str, ...] = ("single_unit", "equal", "logarithmic", "optimal")
"""Names of the strategies accepted by :func:`split_size`."""

_TIE_TOLERANCE = 1e-9
"""Relative cost difference below which unit counts count as equally cheap."""


@dataclass(frozen=True)
class SizeSplit:
    """Result of :func:`split_size`.

    :param strategy: Name of the strategy used (see :data:`STRATEGIES`).
    :param unit_sizes: Size of each unit in the function's ``size_unit``.
    :param unit_costs: Absolute investment cost of each unit in the
        function's ``currency``.
    :param total_cost: Sum of ``unit_costs`` in the function's ``currency``.
    """

    strategy: str
    unit_sizes: tuple[float, ...]
    unit_costs: tuple[float, ...]
    total_cost: float


def _check_total(total_size: float) -> None:
    # negated comparison so that NaN is rejected as well
    if not total_size > 0:
        raise ValueError(f"total_size must be positive, got {total_size}.")


def _min_units(total_size: float, size_max: float) -> int:
    """Fewest units of at most ``size_max`` that make up ``total_size``."""
    return max(1, math.ceil(total_size / size_max))


def split_equal(total_size: float, size_max: float) -> list[float]:
    """Split a total size into the fewest equal units of at most ``size_max``.

    :param total_size: Total size (> 0), in any size unit.
    :param size_max: Largest allowed unit size (> 0), same unit.
    :return: ``n = ceil(total_size / size_max)`` equal unit sizes summing to
        ``total_size``.
    :raises ValueError: If ``total_size`` or ``size_max`` is not positive.
    """
    _check_total(total_size)
    if not size_max > 0:
        raise ValueError(f"size_max must be positive, got {size_max}.")
    n = _min_units(total_size, size_max)
    return [total_size / n] * n


def split_logarithmic(
    total_size: float, size_min: float, size_max: float
) -> list[float]:
    """Split a total size into log-spaced units.

    ``n = max(2, ceil(total_size / size_max))`` sizes are spaced evenly on a
    logarithmic scale from ``size_min`` to ``size_max`` and then scaled by a
    common factor so that they sum to ``total_size``; the scaled sizes may
    therefore leave ``[size_min, size_max]``.

    :param total_size: Total size (> 0), in any size unit.
    :param size_min: Lower end of the spacing (> 0), same unit.
    :param size_max: Upper end of the spacing (> ``size_min``), same unit.
    :return: Unit sizes in ascending order, summing to ``total_size``.
    :raises ValueError: If ``total_size`` or ``size_min`` is not positive or
        ``size_min >= size_max``.
    """
    _check_total(total_size)
    if not size_min > 0:
        raise ValueError(
            f"size_min must be > 0 for logarithmic spacing, got {size_min}. "
            "Use the 'equal' or 'optimal' strategy for a size range starting at 0."
        )
    if not size_min < size_max:
        raise ValueError(f"size_min ({size_min}) must be < size_max ({size_max}).")
    n = max(2, math.ceil(total_size / size_max))
    sizes = np.exp(np.linspace(np.log(size_min), np.log(size_max), n))
    return (sizes * (total_size / sizes.sum())).tolist()


def split_optimal(
    function: CostFunction,
    total_size: float,
    *,
    max_units: int = 50,
    include_installation: bool = True,
    include_tax: bool = False,
) -> list[float]:
    """Split a total size into the cheapest number of equal units.

    Candidate unit counts run from ``ceil(total_size / size_max)`` (no unit
    larger than ``size_max``) to ``min(max_units, floor(total_size /
    size_min))`` (no unit smaller than ``size_min``; with ``size_min == 0``
    up to ``max_units``), and at least the lower end. For every count the
    absolute cost of one unit is evaluated with extrapolation, in one
    vectorized call; the count with the lowest total cost wins. Counts whose
    totals differ by less than a relative 1e-9 count as equal, and ties go to
    fewer units.

    :param function: Cost function; its ``size_min``/``size_max`` bound the
        unit size.
    :param total_size: Total size (> 0) in the function's ``size_unit``.
    :param max_units: Largest number of units considered (>= 1).
    :param include_installation: Include planning and installation in the
        compared cost.
    :param include_tax: Include value-added tax in the compared cost.
    :return: Equal unit sizes in ``size_unit``, summing to ``total_size``.
    :raises ValueError: If ``total_size`` is not positive or ``max_units``
        is smaller than 1.
    """
    _check_total(total_size)
    if max_units < 1:
        raise ValueError(f"max_units must be >= 1, got {max_units}.")
    n_low = _min_units(total_size, function.size_max)
    if function.size_min > 0:
        n_high = min(max_units, math.floor(total_size / function.size_min))
    else:
        n_high = max_units
    counts = np.arange(n_low, max(n_low, n_high) + 1)
    unit_sizes = total_size / counts
    unit_costs = function.calculate_cost(
        unit_sizes,
        specific=False,
        extrapolate=True,
        include_installation=include_installation,
        include_tax=include_tax,
    )
    totals = counts * np.asarray(unit_costs)
    best = totals.min()
    n = int(counts[np.flatnonzero(totals <= best + _TIE_TOLERANCE * abs(best))[0]])
    logger.debug(
        "Optimal split of %g %s for '%s': %d units (candidates %d-%d).",
        total_size,
        function.size_unit,
        function.name,
        n,
        counts[0],
        counts[-1],
    )
    return [total_size / n] * n


def split_size(
    function: CostFunction,
    total_size: float,
    strategy: str = "single_unit",
    *,
    include_installation: bool = True,
    include_tax: bool = False,
) -> SizeSplit:
    """Split a total size into units and calculate their absolute costs.

    Units are costed with :meth:`CostFunction.calculate_cost` with
    ``specific=False`` and ``extrapolate=True``, so units outside the valid
    size range are evaluated there without a warning.

    :param function: Cost function of the technology.
    :param total_size: Total size (> 0) in the function's ``size_unit``.
    :param strategy: One of :data:`STRATEGIES`. ``"equal"`` uses the
        function's ``size_max``, ``"logarithmic"`` its ``size_min`` and
        ``size_max``, ``"optimal"`` see :func:`split_optimal`.
    :param include_installation: Include planning and installation in the
        unit costs.
    :param include_tax: Include value-added tax in the unit costs.
    :return: Unit sizes in ``size_unit``, absolute unit costs and their sum
        in ``currency``.
    :raises ValueError: If ``total_size`` is not positive, the strategy is
        unknown, or ``"logarithmic"`` is used with ``size_min == 0``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Valid options: {list(STRATEGIES)}"
        )
    _check_total(total_size)
    if strategy == "single_unit":
        sizes = [float(total_size)]
    elif strategy == "equal":
        sizes = split_equal(total_size, function.size_max)
    elif strategy == "logarithmic":
        sizes = split_logarithmic(total_size, function.size_min, function.size_max)
    else:
        sizes = split_optimal(
            function,
            total_size,
            include_installation=include_installation,
            include_tax=include_tax,
        )
    costs = np.asarray(
        function.calculate_cost(
            np.asarray(sizes),
            specific=False,
            extrapolate=True,
            include_installation=include_installation,
            include_tax=include_tax,
        ),
        dtype=float,
    )
    return SizeSplit(
        strategy=strategy,
        unit_sizes=tuple(float(s) for s in sizes),
        unit_costs=tuple(costs.tolist()),
        total_cost=float(costs.sum()),
    )
