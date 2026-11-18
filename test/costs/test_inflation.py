"""Tests of the price index for cost functions.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import dataclasses
import logging

import pandas as pd
import pytest

from energy_technology_catalogs.costs.forms import Constant
from energy_technology_catalogs.costs.inflation import PriceIndex


def test_factor_forward_backward():
    index = PriceIndex({2021: 0.10, 2022: 0.20})
    assert index.factor(2020, 2022) == pytest.approx(1.32)
    assert index.factor(2022, 2020) == pytest.approx(1 / 1.32)
    assert index.factor(2021, 2021) == 1.0


def test_from_index_values():
    index = PriceIndex.from_index_values({2020: 100.0, 2021: 110.0, 2022: 132.0})
    assert index.rates == pytest.approx({2021: 0.10, 2022: 0.20})


def test_from_index_values_skips_gaps():
    index = PriceIndex.from_index_values({2018: 90.0, 2020: 100.0, 2021: 110.0})
    assert index.rates == pytest.approx({2021: 0.10})


def test_default_index():
    index = PriceIndex.default()
    assert min(index.rates) == 1950 and max(index.rates) == 2025
    assert index.factor(1949, 1950) == pytest.approx(22.9 / 23.5)
    assert index.factor(2024, 2025) == pytest.approx(126.2 / 127.7)


def test_default_index_is_independent_copy():
    index = PriceIndex.default()
    index.rates[2025] = 1.0
    assert PriceIndex.default().rates[2025] != 1.0


def test_adjust_function_scales_cost_forms_only(function):
    function = dataclasses.replace(
        function, fixed_om=Constant(12.0), tax_rate=0.19, uncertainty_upper=0.2
    )
    out = PriceIndex({2026: 0.10}).adjust_function(function, 2026)
    assert out.form == function.form.scaled(1.10) and out.price_year == 2026
    assert isinstance(out.fixed_om, Constant)
    assert out.fixed_om.value == pytest.approx(Constant(12.0).scaled(1.10).value)
    assert out.investment_year == 2025
    assert out.installation_share == function.installation_share
    assert (out.tax_rate, out.uncertainty_upper) == (0.19, 0.2)
    assert (out.size_min, out.size_max) == (function.size_min, function.size_max)
    assert out.key == function.key


def test_adjust_function_without_fixed_om(function):
    assert function.fixed_om is None
    out = PriceIndex({2026: 0.10}).adjust_function(function, 2026)
    assert out.fixed_om is None and out.form == function.form.scaled(1.10)


def test_adjust_function_backward(function):
    out = PriceIndex({2025: 0.25}).adjust_function(function, 2024)
    assert out.form == function.form.scaled(1 / 1.25) and out.price_year == 2024


def test_adjust_function_keeps_subclass(function):
    @dataclasses.dataclass(frozen=True)
    class Tagged(type(function)):
        tag: str = "x"

    tagged = Tagged(
        **{f.name: getattr(function, f.name) for f in dataclasses.fields(function)}
    )
    out = PriceIndex({2026: 0.10}).adjust_function(tagged, 2026)
    assert type(out) is Tagged and out.tag == "x"


def test_missing_year_logs_and_counts_as_zero(caplog):
    with caplog.at_level(logging.WARNING):
        assert PriceIndex({2022: 0.10}).factor(2020, 2022) == pytest.approx(1.10)
    assert "2021" in caplog.text


@pytest.mark.parametrize("year", [1899, 2201])
def test_implausible_year_raises(year):
    with pytest.raises(ValueError, match="year"):
        PriceIndex({2020: 0.02}).factor(year, 2020)


@pytest.mark.parametrize("year", [2020.0, "2020", True])
def test_non_int_year_raises(year):
    with pytest.raises(TypeError):
        PriceIndex({2020: 0.02}).factor(year, 2020)


@pytest.mark.parametrize("rate", [-1.0, -1.5, float("nan"), float("inf")])
def test_invalid_rate_raises(rate):
    with pytest.raises(ValueError, match="2020"):
        PriceIndex({2020: rate})


def test_from_file_rates_csv(tmp_path):
    path = tmp_path / "rates.csv"
    pd.DataFrame({"year": [2021, 2022], "rate": [0.10, 0.20]}).to_csv(path, index=False)
    assert PriceIndex.from_file(path).rates == pytest.approx({2021: 0.10, 2022: 0.20})


def test_from_file_index_xlsx(tmp_path):
    path = tmp_path / "index.xlsx"
    pd.DataFrame(
        {"Jahr": [2020.0, 2021.0, 2022.0], "Index": [100.0, 110.0, 132.0]}
    ).to_excel(path, index=False)
    index = PriceIndex.from_file(path)
    assert index.rates == pytest.approx({2021: 0.10, 2022: 0.20})
    assert all(type(year) is int for year in index.rates)


def test_from_file_semicolon_csv(tmp_path):
    path = tmp_path / "index.csv"
    path.write_text("Jahr;Index\n2020;100\n2021;110\n", encoding="utf-8")
    assert PriceIndex.from_file(path).rates == pytest.approx({2021: 0.10})


def test_from_file_german_index_csv(tmp_path):
    path = tmp_path / "index.csv"
    path.write_text("Jahr;Index\n2020;100,0\n2021;110,0\n2022;132,0", encoding="utf-8")
    index = PriceIndex.from_file(path)
    assert index.rates == pytest.approx({2021: 0.10, 2022: 0.20})
    assert index.factor(2020, 2022) == pytest.approx(1.32)


def test_from_file_german_rate_csv(tmp_path):
    path = tmp_path / "rates.csv"
    path.write_text("year;rate\n2021;0,02\n2022;1.5\n", encoding="utf-8")
    assert PriceIndex.from_file(path).rates == pytest.approx({2021: 0.02, 2022: 1.5})


@pytest.mark.parametrize(
    "text", ["Jahr;Index\nx;y\n-;n/a\n", "year;rate\nx;0,02\n2021;-\n"]
)
def test_from_file_without_numeric_row_raises(tmp_path, text):
    path = tmp_path / "empty.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="empty.csv"):
        PriceIndex.from_file(path)


def test_from_file_unknown_suffix_raises(tmp_path):
    with pytest.raises(ValueError, match="suffix"):
        PriceIndex.from_file(tmp_path / "rates.txt")


def test_price_factors_match_reference(reference_data):
    data = reference_data("price_factors")
    index = PriceIndex({int(year): rate for year, rate in data["rates"].items()})
    assert data["cases"]
    for case in data["cases"]:
        assert index.factor(case["base_year"], case["target_year"]) == pytest.approx(
            case["factor"], rel=1e-9
        )


# Reference group "Inflation adjuster", in the names of this package.


def test_reference_factor_forward():
    index = PriceIndex({2018: 0.02, 2019: 0.015, 2020: 0.01})
    assert index.factor(2017, 2020) == pytest.approx(1.02 * 1.015 * 1.01, abs=1e-9)


def test_reference_factor_same_year():
    assert PriceIndex({2020: 0.02}).factor(2020, 2020) == 1.0


def test_reference_adjust_cost():
    assert PriceIndex({2018: 0.05}).adjust_cost(1000.0, 2017, 2018) == pytest.approx(
        1050.0, abs=1e-9
    )


def test_reference_negative_cost_raises():
    with pytest.raises(ValueError):
        PriceIndex({2020: 0.02}).adjust_cost(-100.0, 2019, 2020)


def test_reference_adjust_function(function):
    source = dataclasses.replace(function, investment_year=2030, price_year=2017)
    out = PriceIndex({2018: 0.10}).adjust_function(source, 2018)
    assert out.price_year == 2018
    assert out.investment_year == 2030
    assert out.key == source.key
    assert out.form.coefficient == pytest.approx(
        source.form.coefficient * 1.10, abs=1e-9
    )
    assert out.form.offset == pytest.approx(source.form.offset * 1.10, abs=1e-9)


def test_reference_adjust_function_same_year(function):
    source = dataclasses.replace(function, investment_year=2030, price_year=2025)
    assert PriceIndex({2025: 0.02}).adjust_function(source, 2025) is source


def test_reference_adjusted_function_keeps_key(function):
    # The reference replaces the function in a registry; the registry comes in
    # a later task, so the test checks the property it relies on: equal keys.
    source = dataclasses.replace(function, investment_year=2030, price_year=2017)
    out = PriceIndex({2018: 0.10}).adjust_function(source, 2018)
    assert out.key == source.key and out.price_year == 2018 and out != source
