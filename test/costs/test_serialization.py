"""Tests of the cost function file formats (CSV, Excel, JSON).

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import dataclasses
import json
import logging

import pandas as pd
import pytest

from energy_technology_catalogs.costs.core import CostBasis, CostFunction
from energy_technology_catalogs.costs.forms import (
    Constant,
    Exponential,
    Linear,
    Logarithmic,
    PiecewiseLinear,
    Polynomial,
    PowerLaw,
)
from energy_technology_catalogs.costs.serialization import (
    COLUMNS,
    read_cost_functions,
    write_cost_functions,
)
from energy_technology_catalogs.costs.uncertainty import (
    Distribution,
    DistributionKind,
    OutputUncertaintyMode,
    UncertainCostFunction,
)


SUFFIXES = [".csv", ".xlsx", ".json"]

ALL_FORMS = [
    PowerLaw(100.0, 50.0, 0.5),
    Linear(10.0, 2.0),
    Polynomial((1.0, 2.0, 3.0)),
    Logarithmic(5.0, 2.0),
    Exponential(1.0, 2.0, 0.5),
    Constant(7.0),
    PiecewiseLinear((1.0, 3.0, 5.0), (10.0, 30.0, 20.0)),
]

REFERENCE_HEADER = (
    "name;technology_type;reference_unit;investment_year;tax_rate;"
    "P_&_I_included;P_&_I_factor;validity_range_lower;"
    "validity_range_upper;summand;coefficient;exponent;fixed_om;"
    "source;currency;reference_year;"
)


def _uncertain(function: CostFunction, **fields) -> UncertainCostFunction:
    values = {f.name: getattr(function, f.name) for f in dataclasses.fields(function)}
    values.update(fields)
    return UncertainCostFunction(**values)


def _variants(function: CostFunction) -> list[CostFunction]:
    """Plain functions with every optional field set or left out."""
    return [
        dataclasses.replace(
            function,
            technology="Wärmepumpe groß",
            cost_basis=CostBasis.ABSOLUTE,
            fixed_om=Linear(50.0, 0.25),
            installation_share=Linear(0.1, 0.001),
            lifetime_a=20.0,
            tax_rate=0.19,
            uncertainty_lower=-0.1,
            uncertainty_upper=0.3,
            currency="USD",
            size_unit="m3",
            price_year=2020,
        ),
        dataclasses.replace(
            function,
            technology="No options",
            installation_included=False,
            installation_share=None,
            fixed_om=None,
            investment_year=2040,
            price_year=2025,
            form=PowerLaw(0.0, 850.3, 0.123456789012),
        ),
    ]


def _uncertain_functions(function: CostFunction) -> list[UncertainCostFunction]:
    """One function with output and one with parameter uncertainty."""
    output = UncertainCostFunction.from_bounds(
        dataclasses.replace(
            function, technology="Output", uncertainty_lower=-0.1, uncertainty_upper=0.3
        )
    )
    assert output is not None
    parameter = _uncertain(
        function,
        technology="Parameter",
        parameter_uncertainty={
            "coefficient": Distribution(
                DistributionKind.NORMAL, mean=1000.0, std_dev=50.0
            ),
            "exponent": Distribution(
                DistributionKind.TRIANGULAR, lower=-0.6, mode=-0.5, upper=-0.45
            ),
        },
        output_mode=OutputUncertaintyMode.RELATIVE,
    )
    return [output, parameter]


def _header(path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()[0].split(";")


# ----------------------------------------------------------------------
# Round trips
# ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_roundtrip_all_forms(tmp_path, suffix, function):
    functions = [
        dataclasses.replace(
            function, technology=f"T{i}", form=form, fixed_om=Constant(12.0)
        )
        for i, form in enumerate(ALL_FORMS)
    ]
    path = tmp_path / f"functions{suffix}"
    write_cost_functions(functions, path)
    assert read_cost_functions(path) == functions


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_roundtrip_forms_as_fixed_om_and_installation_share(tmp_path, suffix, function):
    functions = [
        dataclasses.replace(
            function,
            technology=f"T{i}",
            fixed_om=form,
            installation_share=form,
            installation_included=bool(i % 2),
        )
        for i, form in enumerate(ALL_FORMS)
    ]
    path = tmp_path / f"functions{suffix}"
    write_cost_functions(functions, path)
    assert read_cost_functions(path) == functions


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_roundtrip_optional_fields(tmp_path, suffix, function):
    functions = [function, *_variants(function)]
    path = tmp_path / f"functions{suffix}"
    write_cost_functions(functions, path)
    loaded = read_cost_functions(path)
    assert loaded == functions
    assert [type(f) for f in loaded] == [CostFunction] * 3


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_roundtrip_uncertain_functions(tmp_path, suffix, function):
    functions = _uncertain_functions(function)
    path = tmp_path / f"functions{suffix}"
    write_cost_functions(functions, path)
    loaded = read_cost_functions(path)
    assert loaded == functions
    assert all(type(f) is UncertainCostFunction for f in loaded)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_roundtrip_mixed_plain_and_uncertain(tmp_path, suffix, function):
    functions = [function, *_uncertain_functions(function), *_variants(function)]
    path = tmp_path / f"functions{suffix}"
    write_cost_functions(functions, path)
    loaded = read_cost_functions(path)
    assert loaded == functions
    assert [type(f) for f in loaded] == [type(f) for f in functions]


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_roundtrip_full_float_precision(tmp_path, suffix, function):
    functions = [
        dataclasses.replace(
            function,
            size_min=1.0 / 3.0,
            uncertainty_upper=0.30000000000000004,
            form=PowerLaw(0.1 + 0.2, 1.0 / 3.0, 0.123456789012),
        )
    ]
    path = tmp_path / f"precise{suffix}"
    write_cost_functions(functions, path)
    assert read_cost_functions(path) == functions


def test_excel_number_cells_keep_15_digits(tmp_path, function):
    """Excel stores numbers with about 15 significant digits; JSON cells are text."""
    precise = dataclasses.replace(
        function,
        uncertainty_upper=0.30000000000000004,
        form=PowerLaw(0.1 + 0.2, 1.0 / 3.0, 0.123456789012),
    )
    path = tmp_path / "precise.xlsx"
    write_cost_functions([precise], path)
    (loaded,) = read_cost_functions(path)
    assert loaded.form == precise.form
    assert loaded.uncertainty_upper == 0.3
    assert loaded == dataclasses.replace(precise, uncertainty_upper=0.3)


def test_uncertainty_roundtrip_from_plan(tmp_path, function):
    u = UncertainCostFunction.from_bounds(
        dataclasses.replace(function, uncertainty_lower=-0.1, uncertainty_upper=0.3)
    )
    path = tmp_path / "u.csv"
    write_cost_functions([u], path)
    assert read_cost_functions(path) == [u]


def test_uncertain_function_without_distributions_stays_uncertain(tmp_path, function):
    functions = [_uncertain(function)]
    path = tmp_path / "u.csv"
    write_cost_functions(functions, path)
    loaded = read_cost_functions(path)
    assert loaded == functions
    assert type(loaded[0]) is UncertainCostFunction


def test_decimal_comma_roundtrip(tmp_path, function):
    functions = [function, *_uncertain_functions(function), *_variants(function)]
    path = tmp_path / "comma.csv"
    write_cost_functions(functions, path, sep=";", decimal=",")
    text = path.read_text(encoding="utf-8-sig")
    assert ";0,19;" in text  # tax rate as a plain number cell
    assert '""exponent"": -0.5' in text  # JSON cells keep the decimal point
    assert read_cost_functions(path, sep=";", decimal=",") == functions


def test_sep_comma_roundtrip(tmp_path, function):
    functions = [function, *_uncertain_functions(function)]
    path = tmp_path / "comma_separated.csv"
    write_cost_functions(functions, path, sep=",")
    assert read_cost_functions(path, sep=",") == functions


@pytest.mark.parametrize("suffix", [".CSV", ".Xlsx", ".JSON"])
def test_suffix_is_case_insensitive(tmp_path, suffix, function):
    path = tmp_path / f"functions{suffix}"
    write_cost_functions([function], path)
    assert read_cost_functions(path) == [function]


def test_empty_list_roundtrip(tmp_path):
    path = tmp_path / "empty.json"
    write_cost_functions([], path)
    assert read_cost_functions(path) == []


# ----------------------------------------------------------------------
# Written format
# ----------------------------------------------------------------------


def test_columns_without_uncertainty(tmp_path, function):
    path = tmp_path / "plain.csv"
    write_cost_functions([function], path)
    assert _header(path) == list(COLUMNS[:-3])


def test_columns_with_uncertainty(tmp_path, function):
    path = tmp_path / "mixed.csv"
    write_cost_functions([function, *_uncertain_functions(function)], path)
    assert _header(path) == list(COLUMNS)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    plain, output, parameter = frame.to_dict(orient="records")
    assert pd.isna(plain["output_uncertainty_mode"])
    assert pd.isna(plain["output_uncertainty"])
    assert json.loads(output["output_uncertainty"])["kind"] == "triangular"
    assert output["output_uncertainty_mode"] == "relative_mean_preserving"
    assert pd.isna(output["parameter_uncertainty"])
    assert (
        json.loads(parameter["parameter_uncertainty"])["coefficient"]["kind"]
        == "normal"
    )
    assert parameter["output_uncertainty_mode"] == "relative"


def test_table_cells(tmp_path, function):
    path = tmp_path / "cells.xlsx"
    write_cost_functions(
        [dataclasses.replace(function, form=Polynomial((1.0, 2.0)))], path
    )
    row = pd.read_excel(path).to_dict(orient="records")[0]
    assert row["function_type"] == "polynomial"
    assert json.loads(row["function_parameters"]) == {"coefficients": [1.0, 2.0]}
    assert pd.isna(row["fixed_om_function_type"]) and pd.isna(
        row["fixed_om_parameters"]
    )
    assert row["installation_share_function_type"] == "constant"
    assert json.loads(row["installation_share_parameters"]) == {"value": 0.4}
    assert row["technology_type"] == "biomass_boiler"


def test_csv_written_format(tmp_path, function):
    """';' separator, UTF-8 with BOM, umlauts, full precision."""
    path = tmp_path / "format.csv"
    precise = dataclasses.replace(function, uncertainty_upper=0.30000000000000004)
    write_cost_functions([*_variants(function), precise], path)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "Wärmepumpe groß" in text
    assert "0.123456789012" in text
    assert ";0.30000000000000004;" in text


def test_json_file_format(tmp_path, function):
    functions = [function, *_variants(function), *_uncertain_functions(function)]
    path = tmp_path / "functions.json"
    write_cost_functions(functions, path)
    text = path.read_text(encoding="utf-8")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert text.endswith("]\n")
    assert "Wärmepumpe groß" in text
    assert (
        text
        == json.dumps([f.to_dict() for f in functions], indent=2, ensure_ascii=False)
        + "\n"
    )
    record = json.loads(text)[0]
    assert "name" not in record
    assert {"technology", "investment_year", "size_min", "size_max"} <= set(record)


# ----------------------------------------------------------------------
# Reference layout
# ----------------------------------------------------------------------


def test_reads_reference_layout(reference_layout_csv, reference_data):
    functions = read_cost_functions(reference_layout_csv)
    assert len(functions) == 10
    assert all(isinstance(f.form, PowerLaw) for f in functions)
    assert all(type(f) is CostFunction and f.fixed_om is None for f in functions)
    expected = [
        CostFunction.from_dict(d) for d in reference_data("cost_grid")["functions"]
    ]
    assert functions == expected


def test_reference_excel_export_format(tmp_path):
    """Trailing separator, space thousands separators, dashes, bool words."""
    rows = [
        "Boiler A;BIOMASS_BOILER;kW;2030;0.00;True;0.40; -   ; 100.00 ;0.00;"
        " 12 345.00 ;-0.5;;Test;EUR;2025;",
        "Heat pump B;HEAT_PUMP;kW;2025;0.19;False;1.10;"
        " 50.00 ;\u00a01\u202f000.00 ;0.00; 1\u20090 61.86 ;0.47;;Test;EUR;;",
    ]
    path = tmp_path / "excel_export.csv"
    path.write_bytes(
        ("\r\n".join([REFERENCE_HEADER, *rows]) + "\r\n").encode("utf-8-sig")
    )
    a, b = read_cost_functions(path)
    assert a.technology == "Boiler A"
    assert a.size_min == 0.0
    assert a.form == PowerLaw(0.0, 12345.0, -0.5)
    assert a.installation_included is True
    assert a.installation_share == Constant(0.4)
    assert (a.investment_year, a.price_year) == (2030, 2025)
    assert a.fixed_om is None
    assert b.technology == "Heat pump B"
    assert b.installation_included is False
    assert b.installation_share.value == pytest.approx(1.0 - 1.0 / 1.1)
    assert b.size_max == 1000.0
    assert b.form.coefficient == pytest.approx(1061.86)
    assert b.price_year == 2025  # empty: investment year


def _reference_comma_file(
    path, function_type: str = "", coefficient: str = "12345,5"
) -> None:
    header = (
        "component;source;technology_type;reference_unit;investment_year;"
        "validity_range_lower;validity_range_upper;function_type;summand;"
        "coefficient;exponent;pi_included;pi_factor"
    )
    values = f"{coefficient};-0,5;wahr;0,25"
    row = f"Boiler;Test;boiler;kW;2025;10;1\u00a0000;{function_type};0;{values}"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8-sig")


@pytest.mark.parametrize("function_type", ["", "power_law"])
def test_decimal_comma_reference_layout(tmp_path, function_type):
    """Decimal comma and no-break thousands separator in the reference layout."""
    path = tmp_path / "comma.csv"
    _reference_comma_file(path, function_type)
    (f,) = read_cost_functions(path, decimal=",")
    assert f.size_max == 1000.0
    assert f.form.parameters()["coefficient"] == 12345.5
    assert f.form.parameters()["exponent"] == -0.5
    assert f.installation_included is True
    assert f.installation_share == Constant(0.25)


def test_unknown_function_type_names_row(tmp_path, caplog):
    """An unknown form kind raises with technology and source of the row."""
    bad = tmp_path / "bad.csv"
    _reference_comma_file(bad, "spline")
    with pytest.raises(ValueError, match=r"Boiler \| Test: .*spline"):
        read_cost_functions(bad, decimal=",")
    with caplog.at_level(logging.WARNING):
        assert read_cost_functions(bad, decimal=",", errors="skip") == []
    assert "Boiler | Test" in caplog.text


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("1.000", 1000.0),
        ("1.000,5", 1000.5),
        ("12.345", 12345.0),
        ("-1.234.567", -1234567.0),
        ("12,5", 12.5),
        ("1.5", 1.5),
    ],
)
def test_csv_thousands_dot_with_decimal_comma(tmp_path, cell, expected):
    path = tmp_path / "dots.csv"
    _reference_comma_file(path, coefficient=cell)
    (f,) = read_cost_functions(path, decimal=",")
    assert f.form.parameters()["coefficient"] == expected


@pytest.mark.parametrize(
    ("cell", "expected"), [("1.000", 1.0), ("13.124", 13.124), ("1000.5", 1000.5)]
)
def test_csv_dot_is_decimal_point_with_decimal_dot(tmp_path, cell, expected):
    path = tmp_path / "dots.csv"
    _reference_comma_file(path, coefficient=cell)
    text = path.read_text(encoding="utf-8-sig").replace(",", ".")
    path.write_text(text.replace("1\u00a0000", "1000"), encoding="utf-8-sig")
    (f,) = read_cost_functions(path)
    assert f.form.parameters()["coefficient"] == expected


def test_excel_numbers_ignore_thousands_dot_rule(tmp_path, function):
    """Native Excel numbers arrive as text with a dot whatever ``decimal`` is."""
    functions = [
        dataclasses.replace(function, size_min=1.125, size_max=100.5, lifetime_a=12.5)
    ]
    path = tmp_path / "numbers.xlsx"
    write_cost_functions(functions, path)
    assert read_cost_functions(path, decimal=",") == functions


@pytest.mark.parametrize("dash", ["-", " \u2013 ", "--"])
def test_dash_only_cells_are_empty_in_all_columns(tmp_path, function, dash):
    functions = [function, *_uncertain_functions(function)]
    path = tmp_path / "dashes.csv"
    write_cost_functions(functions, path)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    columns = [
        "fixed_om_function_type",
        "fixed_om_parameters",
        "lifetime_a",
        "parameter_uncertainty",
        "output_uncertainty",
        "output_uncertainty_mode",
    ]
    frame.loc[0, columns] = dash
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    loaded = read_cost_functions(path)
    assert loaded == functions
    assert type(loaded[0]) is CostFunction


def test_reference_aliases_in_excel(tmp_path, function):
    record = {
        "name": "Boiler",
        "source": "Test",
        "technology_type": "boiler",
        "reference_unit": "kW",
        "investment_year": 2025,
        "validity_range": "[50.0, 500.0]",
        "summand": 0.0,
        "coefficient": 850.3,
        "exponent": 0.72,
        "fixed_om": 0.02,
        "taxes_incl": 0.19,
        "P_&_I_included": False,
        "P_&_I_factor": 1.25,
        "uncertainty": 0.2,
    }
    path = tmp_path / "aliases.xlsx"
    pd.DataFrame([record]).to_excel(path, index=False)
    (f,) = read_cost_functions(path)
    assert (f.technology, f.size_unit) == ("Boiler", "kW")
    assert (f.size_min, f.size_max) == (50.0, 500.0)
    assert f.tax_rate == pytest.approx(0.19)
    assert f.installation_included is False
    assert f.installation_share.value == pytest.approx(0.2)  # 1 - 1/1.25
    assert (f.uncertainty_lower, f.uncertainty_upper) == (-0.2, 0.2)
    assert f.fixed_om == Constant(0.02)


def test_missing_values_in_excel_use_defaults(tmp_path, function):
    record = function.to_dict()
    record.pop("form")
    parameters = {k: v for k, v in function.form.to_dict().items() if k != "kind"}
    record.update(function_type="power_law", function_parameters=json.dumps(parameters))
    record.update(currency=float("nan"), size_min=float("nan"), installation_share=None)
    path = tmp_path / "nan.xlsx"
    pd.DataFrame([record]).to_excel(path, index=False)
    (f,) = read_cost_functions(path)
    assert f.currency == "EUR"
    assert f.size_min == 0.0
    assert f.form == function.form


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("wahr", True),
        ("JA", True),
        ("x", True),
        ("1", True),
        ("Yes", True),
        ("falsch", False),
        ("Nein", False),
        ("0", False),
        ("no", False),
        ("FALSE", False),
    ],
)
def test_boolean_words(tmp_path, function, word, expected):
    path = tmp_path / "bool.csv"
    write_cost_functions([function], path)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    frame["installation_included"] = word
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    assert read_cost_functions(path)[0].installation_included is expected


def test_unreadable_boolean_raises(tmp_path, function):
    path = tmp_path / "bad_bool.csv"
    write_cost_functions([function], path)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    frame["installation_included"] = "vielleicht"
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=r"Boiler \| Test: .*vielleicht"):
        read_cost_functions(path)


def test_legacy_reference_year_only(tmp_path, function):
    record = {
        k: v
        for k, v in function.to_dict().items()
        if k not in ("form", "investment_year", "price_year")
    }
    record.update(reference_year=2017, summand=0.0, coefficient=1000.0, exponent=-0.5)
    record.pop("installation_share")
    record.pop("fixed_om")
    path = tmp_path / "legacy_year.csv"
    pd.DataFrame([record]).to_csv(path, sep=";", index=False)
    with pytest.warns(DeprecationWarning):
        (f,) = read_cost_functions(path)
    assert (f.investment_year, f.price_year) == (2017, 2017)


def test_blank_rows_are_ignored(tmp_path, function):
    path = tmp_path / "blank.xlsx"
    record = {
        "technology": "Boiler",
        "source": "Test",
        "technology_type": "boiler",
        "size_unit": "kW",
        "investment_year": 2025,
        "size_max": 100.0,
        "function_type": "constant",
        "function_parameters": '{"value": 5.0}',
    }
    pd.DataFrame([record, dict.fromkeys(record)]).to_excel(path, index=False)
    (f,) = read_cost_functions(path)
    assert f.form == Constant(5.0)


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


def test_bad_json_record_raises_or_is_skipped(tmp_path, function, caplog):
    path = tmp_path / "bad_record.json"
    path.write_text(
        json.dumps([{"technology": "broken", "source": "x"}, function.to_dict()]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"^broken \| x: "):
        read_cost_functions(path)
    with caplog.at_level(logging.WARNING):
        assert read_cost_functions(path, errors="skip") == [function]
    assert "broken | x: " in caplog.text


def test_json_item_that_is_no_object_raises(tmp_path, function):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([function.to_dict(), [1, 2]]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"^#2 \| \?: "):
        read_cost_functions(path)
    assert read_cost_functions(path, errors="skip") == [function]


def test_json_top_level_must_be_list(tmp_path, function):
    path = tmp_path / "object.json"
    path.write_text(json.dumps({"functions": [function.to_dict()]}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        read_cost_functions(path)


def test_invalid_json_cell_raises(tmp_path, function):
    path = tmp_path / "bad_cell.csv"
    write_cost_functions([function], path)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    frame["function_parameters"] = "{not json"
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=r"Boiler \| Test: "):
        read_cost_functions(path)


def test_parameters_without_function_type_raise(tmp_path, function):
    path = tmp_path / "no_type.csv"
    write_cost_functions([function], path)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    frame["installation_share_function_type"] = ""
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="installation_share_function_type"):
        read_cost_functions(path)


def test_unparseable_number_raises(tmp_path, function):
    path = tmp_path / "bad_number.csv"
    write_cost_functions([function], path)
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    frame["size_max"] = "1,5"  # decimal comma with decimal="."
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match=r"Boiler \| Test: "):
        read_cost_functions(path)


def test_missing_form_and_identity_raise(tmp_path, function):
    no_form = {
        k: v
        for k, v in function.to_dict().items()
        if k not in ("form", "installation_share")
    }
    no_identity = dict(
        no_form, function_type="constant", function_parameters='{"value": 1.0}'
    )
    del no_identity["technology"]
    for record in (no_form, no_identity):
        path = tmp_path / "missing.xlsx"
        pd.DataFrame([record]).to_excel(path, index=False)
        with pytest.raises(ValueError):
            read_cost_functions(path)


def test_invalid_errors_argument_raises(tmp_path, function):
    path = tmp_path / "f.json"
    write_cost_functions([function], path)
    with pytest.raises(ValueError, match="errors"):
        read_cost_functions(path, errors="ignore")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_sep_equal_to_decimal_raises(tmp_path, function, suffix):
    path = tmp_path / f"f{suffix}"
    with pytest.raises(ValueError, match="differ"):
        write_cost_functions([function], path, sep=",", decimal=",")
    write_cost_functions([function], path)
    with pytest.raises(ValueError, match="differ"):
        read_cost_functions(path, sep=",", decimal=",")


def test_unknown_suffix_raises(tmp_path, function):
    with pytest.raises(ValueError, match=".txt"):
        write_cost_functions([function], tmp_path / "f.txt")
    with pytest.raises(ValueError, match=".xls"):
        read_cost_functions(tmp_path / "f.xls")


def test_writing_something_else_raises(tmp_path, function):
    with pytest.raises(TypeError):
        write_cost_functions([function, function.to_dict()], tmp_path / "f.json")
