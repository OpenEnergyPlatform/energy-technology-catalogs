"""Reading and writing cost functions as CSV, Excel and JSON files.

The format is chosen by the file suffix (case-insensitive): ``.csv``,
``.xlsx`` or ``.json``. Every file holds one record per cost function.

JSON files hold a list of the dictionaries of
:meth:`~energy_technology_catalogs.costs.core.CostFunction.to_dict`, with the
forms as nested objects (UTF-8 without byte order mark, indented by two
spaces).

Tables (CSV, first sheet of an Excel workbook) flatten the same dictionary
into the columns of :data:`COLUMNS`, one row per function:

======================================  ==========================================
Column                                  Content
======================================  ==========================================
``technology``                          name of the technology as given by the source
``source``                              literature or data source
``technology_type``                     value of a ``TechnologyType`` member
``size_unit``                           unit of the size, e.g. ``kW``
``investment_year``                     year of installation (identity)
``price_year``                          price base of the parameters
``size_min``, ``size_max``              valid size range, in ``size_unit``
``cost_basis``                          ``specific`` or ``absolute``
``function_type``                       kind of the investment cost form
``function_parameters``                 JSON object of the fields of that form
``fixed_om_function_type``              kind of the fixed O&M form (empty: none)
``fixed_om_parameters``                 JSON object of its fields
``installation_included``               whether the form contains installation
``installation_share_function_type``    kind of the installation share form
``installation_share_parameters``       JSON object of its fields
``lifetime_a``                          technical lifetime in years (empty: unknown)
``tax_rate``                            value-added tax contained in the form
``uncertainty_lower``, ``_upper``       relative uncertainty bounds
``currency``                            ISO 4217 code
``parameter_uncertainty``               JSON object of parameter distributions
``output_uncertainty``                  JSON object of the output distribution
``output_uncertainty_mode``             value of an ``OutputUncertaintyMode``
======================================  ==========================================

The fields of a form are kept in one JSON cell, because the parameter set
differs between forms (JSON numbers always use ``.``; floats round-trip
exactly, as in CSV number cells; Excel keeps about 15 significant digits of
a number cell). The last three columns are written only if a function of the
file is an ``UncertainCostFunction``; rows with a value in any of them are
read as ``UncertainCostFunction``, all other rows as ``CostFunction``.

Reading also accepts the layout of the Uni Kassel cost function module
v3.0.0: ``component`` or ``name``, ``reference_unit``, ``reference_year``,
``validity_range_lower``/``_upper`` or ``validity_range``, ``summand``,
``coefficient``, ``exponent`` (a power law), a scalar ``fixed_om``,
``pi_included``/``P_&_I_included``/``P_&_I_incl``, ``pi_factor``/
``P_&_I_factor``, ``taxes_included``/``taxes_incl``/``tax_incl``,
``uncertainty``/``uncertainty_pct`` and ``lifetime``, resolved by
``CostFunction.from_dict`` through ``LEGACY_ALIASES``. For cell cleaning see
:func:`read_cost_functions`.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from energy_technology_catalogs.costs.core import LEGACY_ALIASES, CostFunction
from energy_technology_catalogs.costs.forms import PowerLaw
from energy_technology_catalogs.costs.uncertainty import UncertainCostFunction


logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "technology",
    "source",
    "technology_type",
    "size_unit",
    "investment_year",
    "price_year",
    "size_min",
    "size_max",
    "cost_basis",
    "function_type",
    "function_parameters",
    "fixed_om_function_type",
    "fixed_om_parameters",
    "installation_included",
    "installation_share_function_type",
    "installation_share_parameters",
    "lifetime_a",
    "tax_rate",
    "uncertainty_lower",
    "uncertainty_upper",
    "currency",
    "parameter_uncertainty",
    "output_uncertainty",
    "output_uncertainty_mode",
)
"""Columns of a written table, in this order."""

_FORM_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("form", "function_type", "function_parameters"),
    ("fixed_om", "fixed_om_function_type", "fixed_om_parameters"),
    (
        "installation_share",
        "installation_share_function_type",
        "installation_share_parameters",
    ),
)
"""Field of ``to_dict``, column of its kind, column of its other fields."""

_UNCERTAINTY_COLUMNS: tuple[str, ...] = COLUMNS[-3:]
_UNCERTAINTY_KEYS = frozenset({*_UNCERTAINTY_COLUMNS, "output_mode"})
_JSON_COLUMNS = frozenset(
    {params for _, _, params in _FORM_COLUMNS}
    | {"parameter_uncertainty", "output_uncertainty"}
)
# field names after LEGACY_ALIASES
_NUMERIC_FIELDS = frozenset(
    {
        "investment_year",
        "price_year",
        "size_min",
        "size_max",
        "lifetime_a",
        "tax_rate",
        "uncertainty_lower",
        "uncertainty_upper",
        "uncertainty_pct",
        "fixed_om",
        "installation_factor",
        "offset",
        "coefficient",
        "exponent",
    }
)
_BOOLEAN_FIELDS = frozenset({"installation_included"})
_TRUE_WORDS = frozenset({"true", "1", "yes", "wahr", "ja", "x"})
_FALSE_WORDS = frozenset({"false", "0", "no", "falsch", "nein"})
_SPACES = re.compile(r"[\s\u00a0\u2009\u202f]+")
_DASHES = re.compile(r"[-\u2012\u2013\u2014\u2212]+")
_THOUSANDS_DOT = re.compile(r"[+-]?\d{1,3}(\.\d{3})+")
_SUFFIXES = (".csv", ".xlsx", ".json")
_ERRORS = ("raise", "skip")


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def write_cost_functions(
    functions: Iterable[CostFunction],
    path: str | Path,
    *,
    sep: str = ";",
    decimal: str = ".",
    encoding: str = "utf-8-sig",
) -> None:
    """Write cost functions to a CSV, Excel or JSON file.

    :param functions: Functions to write, in this order.
    :param path: Target file; the suffix ``.csv``, ``.xlsx`` or ``.json``
        selects the format. An existing file is overwritten.
    :param sep: Field separator of CSV files.
    :param decimal: Decimal sign of number cells in CSV files; JSON cells
        always use ``.``.
    :param encoding: Text encoding of CSV files; the default (UTF-8 with byte
        order mark) shows umlauts correctly in Excel. JSON files are always
        UTF-8 without byte order mark.
    :raises ValueError: If the suffix is not supported or ``sep`` equals
        ``decimal``.
    :raises TypeError: If an element is not a ``CostFunction``.
    """
    _check_separators(sep, decimal)
    path = Path(path)
    suffix = _suffix(path)
    functions = list(functions)
    for function in functions:
        if not isinstance(function, CostFunction):
            raise TypeError(f"Expected CostFunction, got {type(function).__name__}.")
    if suffix == ".json":
        text = json.dumps(
            [f.to_dict() for f in functions], indent=2, ensure_ascii=False
        )
        with path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(text + "\n")
    else:
        frame = _to_table(functions)
        if suffix == ".csv":
            frame.to_csv(path, sep=sep, decimal=decimal, encoding=encoding, index=False)
        else:
            frame.to_excel(path, index=False)
    logger.info("Wrote %d cost function(s) to '%s'.", len(functions), path)


def _to_table(functions: list[CostFunction]) -> pd.DataFrame:
    """Flatten the dictionaries of ``functions`` into the table columns."""
    with_uncertainty = any(isinstance(f, UncertainCostFunction) for f in functions)
    columns = COLUMNS if with_uncertainty else COLUMNS[: -len(_UNCERTAINTY_COLUMNS)]
    rows = []
    for function in functions:
        data = function.to_dict()
        row = {name: data.get(name) for name in columns}
        for field, type_column, parameters_column in _FORM_COLUMNS:
            form = data.get(field)
            if form is not None:
                fields = {k: v for k, v in form.items() if k != "kind"}
                row[type_column] = form["kind"]
                row[parameters_column] = json.dumps(fields)
        if with_uncertainty and isinstance(function, UncertainCostFunction):
            parameters = data["parameter_uncertainty"]
            output = data["output_uncertainty"]
            row["parameter_uncertainty"] = (
                json.dumps(parameters) if parameters else None
            )
            row["output_uncertainty"] = None if output is None else json.dumps(output)
            row["output_uncertainty_mode"] = data["output_mode"]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(columns))


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def read_cost_functions(
    path: str | Path,
    *,
    sep: str = ";",
    decimal: str = ".",
    encoding: str = "utf-8-sig",
    errors: str = "raise",
) -> list[CostFunction]:
    """Read cost functions from a CSV, Excel or JSON file.

    See the module docstring for the columns and the reference layout that
    are read. Table cells are cleaned before conversion: empty and NaN cells
    and cells holding only dashes are missing values, in every column;
    numbers given as text may use the ``decimal`` sign and whitespace,
    no-break or thin spaces as thousands separators; in CSV files with a
    ``decimal`` sign other than ``.``, a dot is a thousands separator too
    (``1.000`` and ``1.000,5`` with ``decimal=","``); boolean words
    (true/1/yes/wahr/ja/x and false/0/no/falsch/nein, any case) are read in
    the ``installation_included`` column and its aliases. JSON cells are never
    changed by the ``decimal`` sign.

    :param path: Source file; the suffix ``.csv``, ``.xlsx`` or ``.json``
        selects the format. Excel files are read from the first sheet.
    :param sep: Field separator of CSV files.
    :param decimal: Decimal sign of numbers given as text in tables.
    :param encoding: Text encoding of CSV files; the default also reads UTF-8
        without byte order mark. JSON files are read as UTF-8.
    :param errors: ``"raise"`` (default) raises on the first unreadable
        record; ``"skip"`` logs it as a warning and continues.
    :return: Functions in file order; ``UncertainCostFunction`` for records
        with uncertainty values, otherwise ``CostFunction``. Blank table rows
        are ignored.
    :raises ValueError: If the suffix or ``errors`` is not supported, ``sep``
        equals ``decimal``, a JSON file holds no list, or (with
        ``errors="raise"``) a record cannot be read; the message then starts
        with ``"<technology> | <source>: "``.
    """
    if errors not in _ERRORS:
        raise ValueError(f"errors must be 'raise' or 'skip', got {errors!r}.")
    _check_separators(sep, decimal)
    path = Path(path)
    suffix = _suffix(path)
    if suffix == ".json":
        with path.open(encoding="utf-8-sig") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(
                f"'{path}' must hold a JSON list of cost functions, "
                f"got {type(records).__name__}."
            )
        functions = _build(records, _prepare_object, errors)
    else:
        if suffix == ".csv":
            frame = pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                index_col=False,
            )
        else:
            frame = pd.read_excel(path, sheet_name=0, dtype=str, keep_default_na=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        # native Excel numbers arrive as text with "." whatever ``decimal`` is
        thousands_dot = suffix == ".csv" and decimal != "."
        functions = _build(
            frame.to_dict(orient="records"),
            lambda row: _prepare_row(row, decimal, thousands_dot),
            errors,
        )
    logger.info("Read %d cost function(s) from '%s'.", len(functions), path)
    return functions


def _build(
    records: list[Any],
    prepare: Callable[[Any], dict[str, Any] | None],
    errors: str,
) -> list[CostFunction]:
    """Convert records to functions, raising or skipping unreadable ones."""
    functions: list[CostFunction] = []
    for index, record in enumerate(records):
        try:
            data = prepare(record)
            if data is not None:
                functions.append(_function_from_dict(data))
        except (ValueError, TypeError) as error:
            message = f"{_label(record, index)}: {error}"
            if errors == "raise":
                raise ValueError(message) from error
            logger.warning("%s", message)
    return functions


def _function_from_dict(data: Mapping[str, Any]) -> CostFunction:
    """Build an uncertain function if the record holds uncertainty values."""
    if any(_present(data.get(key)) for key in _UNCERTAINTY_KEYS):
        return UncertainCostFunction.from_dict(data)
    return CostFunction.from_dict(data)


def _label(record: Any, index: int) -> str:
    """``"<technology> | <source>"`` of a raw record for messages."""
    values = record if isinstance(record, Mapping) else {}
    technology = next(
        (
            values[key]
            for key in ("technology", "component", "name")
            if not _missing(values.get(key))
        ),
        f"#{index + 1}",
    )
    source = values.get("source")
    return f"{technology} | {'?' if _missing(source) else source}"


def _prepare_object(record: Any) -> dict[str, Any]:
    """Check a JSON record; its values are used as they are."""
    if not isinstance(record, Mapping):
        raise TypeError(f"Expected a JSON object, got {type(record).__name__}.")
    return dict(record)


def _prepare_row(
    row: Mapping[str, Any], decimal: str, thousands_dot: bool = False
) -> dict[str, Any] | None:
    """Clean the cells of a table row and rebuild the nested dictionary.

    :param row: Cells by column name.
    :param decimal: Decimal sign of numbers given as text.
    :param thousands_dot: See :func:`_number`.
    :return: Dictionary for ``from_dict``, or None for a blank row.
    :raises ValueError: If a JSON cell cannot be parsed or form cells do not
        fit together.
    """
    data: dict[str, Any] = {}
    for column, value in row.items():
        if isinstance(value, str) and _DASHES.fullmatch(_SPACES.sub("", value)):
            continue
        field = LEGACY_ALIASES.get(column, column)
        if field in _NUMERIC_FIELDS:
            value = _number(value, decimal, thousands_dot)
        elif field in _BOOLEAN_FIELDS:
            value = _boolean(value)
        elif column in _JSON_COLUMNS and not _missing(value):
            value = json.loads(value) if isinstance(value, str) else value
        if not _missing(value):
            data[column] = value
    if not data:
        return None

    for field, type_column, parameters_column in _FORM_COLUMNS:
        kind = data.pop(type_column, None)
        parameters = data.pop(parameters_column, None)
        if kind is None:
            if parameters is not None:
                raise ValueError(
                    f"'{parameters_column}' is given without '{type_column}'."
                )
            continue
        kind = str(kind).strip()
        if parameters is None:
            # reference layout: a power law given by summand, coefficient, exponent
            if field == "form" and kind == PowerLaw.kind and "coefficient" in data:
                continue
            parameters = {}
        if not isinstance(parameters, Mapping):
            raise ValueError(
                f"'{parameters_column}' must be a JSON object, got {parameters!r}."
            )
        data[field] = {**parameters, "kind": kind}

    mode = data.pop("output_uncertainty_mode", None)
    if mode is not None:
        data.setdefault("output_mode", mode)
    return data


def _number(value: Any, decimal: str, thousands_dot: bool = False) -> Any:
    """Parse a number cell; unreadable text is returned for ``from_dict`` to report.

    :param value: Cell value.
    :param decimal: Decimal sign of numbers given as text.
    :param thousands_dot: Read a dot as thousands separator where the text
        has the shape ``1.000`` or ``1.000,5`` (with ``decimal=","``).
    """
    if isinstance(value, str):
        text = _SPACES.sub("", value)
        if not text:
            return None
        if thousands_dot and (
            _THOUSANDS_DOT.fullmatch(text)
            or (
                "." in text
                and decimal in text
                and text.index(".") < text.index(decimal)
            )
        ):
            text = text.replace(".", "")
        if decimal != ".":
            text = text.replace(decimal, ".")
        try:
            value = float(text)
        except ValueError:
            return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _boolean(value: Any) -> Any:
    """Parse a boolean cell; unknown words are returned for ``from_dict`` to report."""
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _missing(value: Any) -> bool:
    """True for None, NaN and blank text."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and math.isnan(value)


def _present(value: Any) -> bool:
    """True for a value that is neither missing nor an empty mapping."""
    if isinstance(value, Mapping):
        return bool(value)
    return not _missing(value)


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------


def _suffix(path: Path) -> str:
    """Return the lower-case suffix of a supported file."""
    suffix = path.suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(
            f"Unsupported file type {path.suffix!r} of '{path}'; "
            f"use one of {list(_SUFFIXES)}."
        )
    return suffix


def _check_separators(sep: str, decimal: str) -> None:
    if sep == decimal:
        raise ValueError(f"sep and decimal must differ, both are {sep!r}.")
