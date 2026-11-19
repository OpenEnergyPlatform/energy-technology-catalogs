"""In-memory registry of cost functions.

The registry stores cost functions under their identity
:class:`~energy_technology_catalogs.costs.core.CostFunctionKey`
``(technology, source, investment_year)``. One query method,
:meth:`CostFunctionRegistry.filter`, combines criteria on technology, source,
investment year, technology type and ontology class; results are sorted by
technology, source and investment year, so a filter on technology and source
returns the cost projection of that pair over the investment years.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from energy_technology_catalogs.costs.core import (
    CostFunction,
    CostFunctionKey,
    TechnologyType,
)
from energy_technology_catalogs.costs.serialization import (
    read_cost_functions,
    write_cost_functions,
)


logger = logging.getLogger(__name__)


class CostFunctionRegistry:
    """Collection of cost functions with unique keys.

    Keys are :class:`CostFunctionKey` objects; :meth:`get`, :meth:`remove`
    and ``in`` also accept a ``(technology, source, investment_year)`` tuple.
    Iteration yields the functions in insertion order.
    """

    def __init__(self) -> None:
        self._functions: dict[CostFunctionKey, CostFunction] = {}

    # ------------------------------------------------------------------
    # Changes
    # ------------------------------------------------------------------

    def add(self, function: CostFunction) -> None:
        """Register a cost function.

        :param function: Function with a key not yet registered.
        :raises TypeError: If ``function`` is not a ``CostFunction``.
        :raises ValueError: If its key is already registered; use
            :meth:`replace` to change a registered function.
        """
        self.extend([function])

    def extend(self, functions: Iterable[CostFunction]) -> None:
        """Register several cost functions; nothing is added if one fails.

        :param functions: Functions with keys not yet registered and unique
            among themselves.
        :raises TypeError: If an element is not a ``CostFunction``.
        :raises ValueError: If a key is already registered or repeats.
        """
        new = list(functions)
        keys: set[CostFunctionKey] = set()
        for function in new:
            if not isinstance(function, CostFunction):
                raise TypeError(
                    f"Expected CostFunction, got {type(function).__name__}."
                )
            key = function.key
            if key in self._functions or key in keys:
                raise ValueError(
                    f"A cost function '{key}' is already registered; "
                    "use replace() to change it."
                )
            keys.add(key)
        for function in new:
            self._functions[function.key] = function
        logger.debug("Registered %d cost function(s).", len(new))

    def replace(self, function: CostFunction) -> None:
        """Replace the registered function with the same key.

        The replacement keeps the position in the iteration order.

        :param function: New version of a registered function.
        :raises TypeError: If ``function`` is not a ``CostFunction``.
        :raises KeyError: If its key is not registered; use :meth:`add`.
        """
        if not isinstance(function, CostFunction):
            raise TypeError(f"Expected CostFunction, got {type(function).__name__}.")
        if function.key not in self._functions:
            raise KeyError(
                f"No cost function '{function.key}' is registered; use add()."
            )
        self._functions[function.key] = function

    def remove(self, key: CostFunctionKey | tuple[str, str, int]) -> CostFunction:
        """Remove a function and return it.

        :param key: Key or ``(technology, source, investment_year)`` tuple.
        :return: The removed function.
        :raises KeyError: If no function is registered under ``key``.
        :raises TypeError: If ``key`` is neither a key nor a three-element tuple.
        """
        resolved = _resolve_key(key)
        if resolved not in self._functions:
            raise KeyError(f"No cost function '{resolved}' is registered.")
        return self._functions.pop(resolved)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, key: CostFunctionKey | tuple[str, str, int]) -> CostFunction | None:
        """Return the function registered under ``key``.

        :param key: Key or ``(technology, source, investment_year)`` tuple.
        :return: The function, or None if none is registered under ``key``.
        :raises TypeError: If ``key`` is neither a key nor a three-element tuple.
        """
        return self._functions.get(_resolve_key(key))

    def filter(
        self,
        *,
        technology: str | None = None,
        source: str | None = None,
        investment_year: int | None = None,
        technology_type: TechnologyType | str | None = None,
        oeo_iri: str | None = None,
    ) -> list[CostFunction]:
        """Return the functions that meet all given criteria.

        Criteria left at None are not applied; without criteria all
        functions are returned.

        :param technology: Name of the technology; surrounding whitespace is
            ignored, as in stored names.
        :param source: Literature or data source; whitespace as for
            ``technology``.
        :param investment_year: Year of installation; an int, an integral
            float or a digit string.
        :param technology_type: Technology category, as member or as its
            value or name (see ``TechnologyType.from_string``).
        :param oeo_iri: IRI of the Open Energy Ontology class of the
            technology type; must not be empty.
        :return: Matching functions sorted by technology, source and
            investment year.
        :raises TypeError: If a criterion has the wrong type.
        :raises ValueError: If ``technology_type`` names no member or
            ``oeo_iri`` is empty.
        """
        technology = _text_criterion("technology", technology)
        source = _text_criterion("source", source)
        investment_year = _year_criterion(investment_year)
        if isinstance(technology_type, str):
            technology_type = TechnologyType.from_string(technology_type)
        elif technology_type is not None and not isinstance(
            technology_type, TechnologyType
        ):
            raise TypeError(
                "technology_type must be TechnologyType or str, "
                f"got {type(technology_type).__name__}."
            )
        oeo_iri = _text_criterion("oeo_iri", oeo_iri)
        if oeo_iri == "":
            raise ValueError("oeo_iri must not be empty.")
        found = [
            f
            for f in self._functions.values()
            if (technology is None or f.technology == technology)
            and (source is None or f.source == source)
            and (investment_year is None or f.investment_year == investment_year)
            and (technology_type is None or f.technology_type is technology_type)
            and (oeo_iri is None or f.technology_type.oeo_iri == oeo_iri)
        ]
        return sorted(found, key=lambda f: (f.technology, f.source, f.investment_year))

    def technologies(self) -> list[str]:
        """Return the registered technology names, sorted."""
        return sorted({f.technology for f in self._functions.values()})

    def sources(self) -> list[str]:
        """Return the registered sources, sorted."""
        return sorted({f.source for f in self._functions.values()})

    def technology_types(self) -> list[TechnologyType]:
        """Return the registered technology types in the order of the enum."""
        present = {f.technology_type for f in self._functions.values()}
        return [member for member in TechnologyType if member in present]

    def investment_years(self) -> list[int]:
        """Return the registered investment years, ascending."""
        return sorted({f.investment_year for f in self._functions.values()})

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def load(self, path: str | Path, **read_options: Any) -> int:
        """Register all functions of a CSV, Excel or JSON file.

        Nothing is added if a key of the file is already registered or
        repeats within the file.

        :param path: File to read (see :func:`read_cost_functions`).
        :param read_options: Keyword arguments of :func:`read_cost_functions`
            (``sep``, ``decimal``, ``encoding``, ``errors``).
        :return: Number of functions added.
        :raises ValueError: If the file cannot be read or a key is already
            registered (the message names the key).
        """
        functions = read_cost_functions(path, **read_options)
        self.extend(functions)
        logger.info("Loaded %d cost function(s) from '%s'.", len(functions), path)
        return len(functions)

    def save(self, path: str | Path, **write_options: Any) -> None:
        """Write all functions, sorted as in :meth:`filter`, to a file.

        :param path: Target file (see :func:`write_cost_functions`).
        :param write_options: Keyword arguments of
            :func:`write_cost_functions` (``sep``, ``decimal``, ``encoding``).
        :raises ValueError: If the suffix is not supported.
        """
        write_cost_functions(self.filter(), path, **write_options)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._functions)

    def __iter__(self) -> Iterator[CostFunction]:
        return iter(self._functions.values())

    def __contains__(self, key: object) -> bool:
        try:
            return _resolve_key(key) in self._functions
        except (TypeError, ValueError):
            return False


def _text_criterion(name: str, value: object) -> str | None:
    """Strip a text criterion; None stays None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}.")
    return value.strip()


def _year_criterion(value: object) -> int | None:
    """Accept an int, an integral float or a digit string as year."""
    if value is None:
        return None
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise TypeError(
        f"investment_year must be a whole number, got {type(value).__name__} {value!r}."
    )


def _resolve_key(key: object) -> CostFunctionKey:
    """Accept a key or a ``(technology, source, investment_year)`` tuple."""
    if isinstance(key, CostFunctionKey):
        return key
    if isinstance(key, tuple) and len(key) == 3:
        return CostFunctionKey(*key)
    raise TypeError(
        "Expected CostFunctionKey or a (technology, source, investment_year) "
        f"tuple, got {key!r}."
    )
