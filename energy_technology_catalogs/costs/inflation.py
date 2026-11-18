"""Price index for moving cost functions between price years.

A :class:`PriceIndex` holds year-on-year rates of price change. The rate of a
year is the change of the price level against the previous year as a fraction
(0.02 = 2 %). The factor from ``from_year`` to ``to_year`` is the product of
``1 + rate`` over the years ``from_year + 1 ... to_year``; backwards it is the
reciprocal. A year without a rate counts as 0 % and logs a warning. Rates
derived from index values, ``rate[y] = index[y] / index[y - 1] - 1``, reproduce
the index ratios exactly.

The packaged default is the producer price index of industrial products
(Erzeugerpreisindex gewerblicher Produkte) of the German Federal Statistical
Office (Destatis), annual values 1949-2025.

Adjusting a cost function changes its price base (``price_year``) and scales
its investment cost form and its fixed O&M cost form (the latter deviates from
the Uni Kassel cost function module v3.0.0); ``investment_year``, and with it
the identity of the function, is never changed.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import logging
import math
import numbers
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from energy_technology_catalogs.costs.core import CostFunction


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=CostFunction)

YEAR_MIN = 1900
YEAR_MAX = 2200
_DEFAULT_FILE = "data/destatis_epi_1949-2025.csv"


def _check_year(year: Any, name: str) -> int:
    """Return ``year`` if it is an integer within ``[YEAR_MIN, YEAR_MAX]``.

    :param year: Calendar year.
    :param name: Argument name for the error message.
    :return: The year as ``int``.
    :raises TypeError: If ``year`` is not an integer (``bool`` included).
    :raises ValueError: If ``year`` is outside the plausible range.
    """
    if isinstance(year, bool) or not isinstance(year, numbers.Integral):
        raise TypeError(f"'{name}' must be int, got {type(year).__name__}.")
    if not YEAR_MIN <= year <= YEAR_MAX:
        raise ValueError(f"'{name}' ({year}) is outside {YEAR_MIN}-{YEAR_MAX}.")
    return int(year)


def _rates_from_index(index: Mapping[int, float]) -> dict[int, float]:
    """Year-on-year rates of an index series, for consecutive years only.

    :param index: Index value per year (any base, > 0).
    :return: ``{year: index[year] / index[year - 1] - 1}`` for every year whose
        previous year is present.
    :raises ValueError: If an index value is not finite or not positive.
    """
    values = {int(year): float(value) for year, value in index.items()}
    for year, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Index value of {year} must be finite and > 0, got {value}."
            )
    return {
        year: value / values[year - 1] - 1.0
        for year, value in sorted(values.items())
        if year - 1 in values
    }


def _rates_from_table(table: pd.DataFrame) -> dict[int, float]:
    """Rates from a table with ``year``/``rate`` columns or year and index columns.

    :param table: Either columns named ``year`` and ``rate`` (any case), or the
        year in the first and the index value in the second column. Rows
        without a numeric year or value are dropped.
    :return: Rates per year as fractions.
    """
    columns = {str(column).strip().lower(): column for column in table.columns}
    if "year" in columns and "rate" in columns:
        years = pd.to_numeric(table[columns["year"]], errors="coerce")
        values = pd.to_numeric(table[columns["rate"]], errors="coerce")
        is_rate = True
    else:
        if table.shape[1] < 2:
            raise ValueError("Price index table needs a year and a value column.")
        years = pd.to_numeric(table.iloc[:, 0], errors="coerce")
        values = pd.to_numeric(table.iloc[:, 1], errors="coerce")
        is_rate = False
    valid = years.notna() & values.notna()
    series = {int(y): float(v) for y, v in zip(years[valid], values[valid])}
    return series if is_rate else _rates_from_index(series)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a comma or semicolon separated file in UTF-8 or Latin-1."""
    try:
        table = pd.read_csv(path, encoding="utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        table = pd.read_csv(path, encoding="latin-1")
        encoding = "latin-1"
    if table.shape[1] == 1:
        table = pd.read_csv(path, sep=";", encoding=encoding)
    return table


@lru_cache(maxsize=1)
def _default_rates() -> tuple[tuple[int, float], ...]:
    """Rates of the packaged Destatis index, parsed once."""
    resource = resources.files("energy_technology_catalogs").joinpath(_DEFAULT_FILE)
    with resource.open("r", encoding="utf-8") as file:
        table = pd.read_csv(file)
    return tuple(_rates_from_table(table).items())


class PriceIndex:
    """Year-on-year price changes for moving costs between price years.

    :param rates: Change of the price level of each year against the
        previous year, as a fraction (0.02 = 2 %); keys are calendar years.
    :raises ValueError: If a rate is not finite or not greater than -1.
    """

    def __init__(self, rates: Mapping[int, float]) -> None:
        self.rates: dict[int, float] = {}
        for year, rate in rates.items():
            year_int = int(year)
            value = float(rate)
            if not math.isfinite(value) or value <= -1.0:
                raise ValueError(
                    f"Rate of {year_int} must be finite and > -1, got {value}."
                )
            self.rates[year_int] = value

    def __repr__(self) -> str:
        if not self.rates:
            return "PriceIndex({})"
        return (
            f"PriceIndex({len(self.rates)} years, {min(self.rates)}-{max(self.rates)})"
        )

    @classmethod
    def from_index_values(cls, index: Mapping[int, float]) -> PriceIndex:
        """Create a price index from index values (any base year).

        Rates are formed for consecutive years only:
        ``rate[y] = index[y] / index[y - 1] - 1``; a year whose previous year
        is missing gets no rate.

        :param index: Index value per year (> 0).
        :return: New price index.
        :raises ValueError: If an index value is not finite or not positive.
        """
        return cls(_rates_from_index(index))

    @classmethod
    def from_file(cls, path: str | Path) -> PriceIndex:
        """Read a price index from a CSV or Excel file.

        The file either has columns ``year`` and ``rate`` (rates as fractions,
        column names in any case) or holds the year in the first and the
        index value in the second column; further columns are ignored. CSV
        files may be comma or semicolon separated, UTF-8 or Latin-1 encoded.

        :param path: Path of a ``.csv`` or ``.xlsx`` file.
        :return: New price index.
        :raises ValueError: If the suffix is not supported or the table has
            fewer than two columns.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            table = _read_csv(path)
        elif suffix == ".xlsx":
            table = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file suffix '{suffix}'; use .csv or .xlsx.")
        return cls(_rates_from_table(table))

    @classmethod
    def default(cls) -> PriceIndex:
        """Packaged producer price index of industrial products (Destatis).

        Annual index 1949-2025, so rates exist for 1950-2025.

        :return: New price index; changing its rates does not affect later
            calls.
        """
        return cls(dict(_default_rates()))

    def factor(self, from_year: int, to_year: int) -> float:
        """Price factor from prices of ``from_year`` to prices of ``to_year``.

        :param from_year: Price year of the given cost.
        :param to_year: Price year of the result.
        :return: Dimensionless factor; a cost at ``from_year`` prices times
            this factor is the cost at ``to_year`` prices. Years without a
            rate count as 0 % and log a warning.
        :raises TypeError: If a year is not an integer.
        :raises ValueError: If a year is outside 1900-2200.
        """
        start = _check_year(from_year, "from_year")
        end = _check_year(to_year, "to_year")
        if start == end:
            return 1.0
        years = range(min(start, end) + 1, max(start, end) + 1)
        result = 1.0
        for year in years:
            rate = self.rates.get(year)
            if rate is None:
                logger.warning("No price change rate for %d; counted as 0 %%.", year)
                continue
            result *= 1.0 + rate
        return result if end > start else 1.0 / result

    def adjust_cost(self, cost: float, from_year: int, to_year: int) -> float:
        """Move a cost from prices of ``from_year`` to prices of ``to_year``.

        :param cost: Cost at ``from_year`` prices, in any currency unit (>= 0).
        :param from_year: Price year of ``cost``.
        :param to_year: Price year of the result.
        :return: Cost at ``to_year`` prices, in the unit of ``cost``.
        :raises ValueError: If ``cost`` is negative or a year is implausible.
        :raises TypeError: If a year is not an integer.
        """
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}.")
        return cost * self.factor(from_year, to_year)

    def adjust_function(self, function: F, to_year: int) -> F:
        """Move a cost function to prices of ``to_year``.

        The investment cost form ``form`` and the fixed O&M cost form
        ``fixed_om`` (if any) are scaled with :meth:`CostFunction.scaled` by
        ``k = factor(function.price_year, to_year)``, and ``price_year`` is set
        to ``to_year``. Both are money at prices of ``price_year``. Scaling
        ``fixed_om`` deviates from the Uni Kassel cost function module
        v3.0.0. ``investment_year`` is never changed, so the key of the
        function stays the same; the installation share, tax rate,
        uncertainty bounds, size bounds and all other fields are kept.
        Works for every subclass of
        :class:`CostFunction` via :func:`dataclasses.replace`.

        :param function: Cost function to adjust.
        :param to_year: Target price year.
        :return: New function of the same type, or ``function`` itself when
            its price year already equals ``to_year``.
        :raises TypeError: If ``to_year`` is not an integer.
        :raises ValueError: If ``to_year`` is outside 1900-2200.
        """
        end = _check_year(to_year, "to_year")
        start = function.price_year
        if start is None:  # pragma: no cover - set in CostFunction.__post_init__
            start = function.investment_year
        if start == end:
            return function
        k = self.factor(start, end)
        logger.debug(
            "Adjusted '%s' from %d to %d prices (factor %.6f).",
            function.name,
            start,
            end,
            k,
        )
        return dataclasses.replace(function.scaled(k), price_year=end)
