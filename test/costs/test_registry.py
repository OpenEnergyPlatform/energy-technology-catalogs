"""Tests of the cost function registry.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import dataclasses

import numpy as np
import pytest

from energy_technology_catalogs.costs.core import CostFunctionKey, TechnologyType
from energy_technology_catalogs.costs.registry import CostFunctionRegistry
from energy_technology_catalogs.costs.serialization import write_cost_functions
from energy_technology_catalogs.costs.uncertainty import UncertainCostFunction


@pytest.fixture
def heat_pump(function):
    return dataclasses.replace(
        function,
        technology="Heat pump",
        source="Source A",
        technology_type=TechnologyType.HEAT_PUMP,
    )


@pytest.fixture
def three(heat_pump):
    """Two sources of one technology and a chiller."""
    return [
        heat_pump,
        dataclasses.replace(heat_pump, source="Source B"),
        dataclasses.replace(
            heat_pump, technology="Chiller", technology_type=TechnologyType.CHILLER
        ),
    ]


@pytest.fixture
def registry(three):
    registry = CostFunctionRegistry()
    registry.extend(three)
    return registry


def _key_tuple(f):
    return (f.technology, f.source, f.investment_year)


# ----------------------------------------------------------------------
# CRUD
# ----------------------------------------------------------------------


def test_registry_filter_and_keys(function):
    registry = CostFunctionRegistry()
    registry.extend([function, dataclasses.replace(function, investment_year=2030)])
    assert len(registry) == 2 and function.key in registry
    assert registry.filter(investment_year=2030)[0].investment_year == 2030
    assert registry.investment_years() == [2025, 2030]
    with pytest.raises(ValueError):
        registry.add(function)


def test_add_and_get_by_key_and_tuple(function):
    registry = CostFunctionRegistry()
    registry.add(function)
    assert registry.get(function.key) is function
    assert registry.get(_key_tuple(function)) is function


def test_get_missing_returns_none():
    assert CostFunctionRegistry().get(("unknown", "unknown", 2025)) is None


def test_get_with_wrong_key_type_raises(registry):
    with pytest.raises(TypeError):
        registry.get("Heat pump | Source A | 2025")


def test_duplicate_add_names_the_key(function):
    registry = CostFunctionRegistry()
    registry.add(function)
    with pytest.raises(ValueError, match=r"Boiler \| Test \| 2025"):
        registry.add(dataclasses.replace(function, tax_rate=0.19))


def test_add_rejects_other_types(function):
    registry = CostFunctionRegistry()
    with pytest.raises(TypeError):
        registry.add(function.to_dict())
    with pytest.raises(TypeError):
        registry.extend([function, "not a function"])
    assert len(registry) == 0


def test_extend_with_duplicates_adds_nothing(function):
    registry = CostFunctionRegistry()
    with pytest.raises(ValueError):
        registry.extend([function, dataclasses.replace(function, tax_rate=0.19)])
    assert len(registry) == 0


def test_same_technology_different_years_allowed(function):
    registry = CostFunctionRegistry()
    registry.add(function)
    registry.add(dataclasses.replace(function, investment_year=2030))
    assert len(registry) == 2


def test_remove_by_tuple_and_key(registry, three):
    assert registry.remove(_key_tuple(three[0])) is three[0]
    assert registry.get(three[0].key) is None
    assert registry.remove(three[1].key) is three[1]
    assert len(registry) == 1


def test_remove_missing_raises():
    with pytest.raises(KeyError):
        CostFunctionRegistry().remove(("x", "y", 2025))


def test_replace(registry, heat_pump):
    updated = dataclasses.replace(heat_pump, tax_rate=0.19)
    registry.replace(updated)
    assert registry.get(heat_pump.key) is updated
    assert list(registry)[0] is updated  # position kept


def test_replace_missing_raises(function):
    with pytest.raises(KeyError):
        CostFunctionRegistry().replace(function)


def test_len(function):
    registry = CostFunctionRegistry()
    assert len(registry) == 0
    registry.add(function)
    assert len(registry) == 1


def test_contains(registry, heat_pump):
    assert heat_pump.key in registry
    assert _key_tuple(heat_pump) in registry
    assert CostFunctionKey("Heat pump", "Source C", 2025) not in registry
    assert ("Heat pump", "Source A") not in registry
    assert heat_pump.name not in registry  # display names are not keys
    assert heat_pump not in registry


def test_iter_in_insertion_order(registry, three):
    assert list(registry) == three


def test_uncertain_function_is_a_cost_function(heat_pump):
    values = {f.name: getattr(heat_pump, f.name) for f in dataclasses.fields(heat_pump)}
    uncertain = UncertainCostFunction(**values)
    registry = CostFunctionRegistry()
    registry.add(uncertain)
    assert registry.get(heat_pump.key) is uncertain


# ----------------------------------------------------------------------
# Queries
# ----------------------------------------------------------------------


def test_filter_by_technology_two_sources(registry):
    results = registry.filter(technology="Heat pump")
    assert [f.source for f in results] == ["Source A", "Source B"]


def test_filter_by_source(registry):
    assert len(registry.filter(source="Source A")) == 2


def test_filter_by_technology_type(registry, three):
    assert len(registry.filter(technology_type=TechnologyType.HEAT_PUMP)) == 2
    assert registry.filter(technology_type=TechnologyType.CHILLER) == [three[2]]
    assert registry.filter(technology_type=TechnologyType.PV) == []


def test_filter_after_remove(registry, heat_pump):
    registry.remove(heat_pump.key)
    results = registry.filter(technology="Heat pump")
    assert [f.source for f in results] == ["Source B"]


def test_filter_time_series_sorted_by_year(heat_pump):
    registry = CostFunctionRegistry()
    for year in (2040, 2025, 2030):
        registry.add(dataclasses.replace(heat_pump, investment_year=year))
    series = registry.filter(technology="Heat pump", source="Source A")
    assert [f.investment_year for f in series] == [2025, 2030, 2040]


def test_filter_by_year(heat_pump):
    registry = CostFunctionRegistry()
    registry.extend(
        [
            heat_pump,
            dataclasses.replace(heat_pump, source="Source B"),
            dataclasses.replace(heat_pump, investment_year=2030),
        ]
    )
    assert len(registry.filter(investment_year=2025)) == 2
    assert len(registry.filter(investment_year=2030)) == 1
    assert registry.filter(investment_year=2099) == []


def test_filter_by_type_and_year(heat_pump):
    registry = CostFunctionRegistry()
    registry.extend(
        [
            heat_pump,
            dataclasses.replace(heat_pump, investment_year=2030),
            dataclasses.replace(
                heat_pump, technology="Chiller", technology_type=TechnologyType.CHILLER
            ),
        ]
    )
    (result,) = registry.filter(
        technology_type=TechnologyType.HEAT_PUMP, investment_year=2025
    )
    assert result.investment_year == 2025
    assert result.technology_type is TechnologyType.HEAT_PUMP


def test_filter_by_oeo_iri(registry, three, monkeypatch):
    iri = "http://example.org/heat_pump"
    monkeypatch.setattr(
        TechnologyType,
        "oeo_iri",
        property(lambda self: iri if self is TechnologyType.HEAT_PUMP else ""),
    )
    assert registry.filter(oeo_iri=iri) == [three[0], three[1]]
    assert registry.filter(oeo_iri=f"  {iri} ") == [three[0], three[1]]
    assert registry.filter(oeo_iri="http://example.org/other") == []


@pytest.mark.parametrize("iri", ["", "   "])
def test_filter_by_empty_oeo_iri_raises(registry, iri):
    with pytest.raises(ValueError, match="oeo_iri"):
        registry.filter(oeo_iri=iri)


def test_filter_technology_type_as_text(registry, three):
    assert registry.filter(technology_type="chiller") == [three[2]]
    assert registry.filter(technology_type="HEAT_PUMP") == [three[0], three[1]]
    with pytest.raises(ValueError):
        registry.filter(technology_type="spaceship")
    with pytest.raises(TypeError):
        registry.filter(technology_type=1)


@pytest.mark.parametrize("year", [2025, np.int64(2025), 2025.0, "2025", " 2025 "])
def test_filter_investment_year_types(registry, year):
    assert len(registry.filter(investment_year=year)) == 3


@pytest.mark.parametrize("year", [2025.5, "twenty", True, [2025]])
def test_filter_invalid_investment_year_raises(registry, year):
    with pytest.raises(TypeError):
        registry.filter(investment_year=year)


def test_filter_strips_names(registry, three):
    assert registry.filter(technology=" Heat pump ", source="Source B\t") == [three[1]]
    with pytest.raises(TypeError):
        registry.filter(source=1)


def test_filter_without_criteria_is_sorted(registry, three):
    assert registry.filter() == [three[2], three[0], three[1]]


def test_investment_years_sorted(heat_pump):
    registry = CostFunctionRegistry()
    for year in (2030, 2025, 2040):
        registry.add(
            dataclasses.replace(heat_pump, source=f"src_{year}", investment_year=year)
        )
    assert registry.investment_years() == [2025, 2030, 2040]


def test_technologies_sources_and_types(registry):
    assert registry.technologies() == ["Chiller", "Heat pump"]
    assert registry.sources() == ["Source A", "Source B"]
    assert registry.technology_types() == [
        TechnologyType.HEAT_PUMP,
        TechnologyType.CHILLER,
    ]


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".xlsx", ".json"])
def test_save_and_load(tmp_path, registry, suffix):
    path = tmp_path / f"registry{suffix}"
    registry.save(path)
    loaded = CostFunctionRegistry()
    assert loaded.load(path) == 3
    assert list(loaded) == registry.filter()  # saved sorted
    assert len(loaded.filter(technology="Heat pump")) == 2
    assert len(loaded.filter(source="Source A")) == 2
    assert len(loaded.filter(technology_type=TechnologyType.HEAT_PUMP)) == 2
    assert loaded.investment_years() == [2025]


def test_save_and_load_pass_options(tmp_path, registry):
    path = tmp_path / "registry.csv"
    registry.save(path, sep=",", decimal=".")
    assert "Heat pump,Source A," in path.read_text(encoding="utf-8-sig")
    loaded = CostFunctionRegistry()
    loaded.load(path, sep=",", errors="raise")
    assert list(loaded) == registry.filter()


def test_load_duplicate_raises_and_adds_nothing(tmp_path, registry, three, function):
    path = tmp_path / "duplicate.json"
    write_cost_functions([function, three[0]], path)
    with pytest.raises(ValueError, match=r"Heat pump \| Source A \| 2025"):
        registry.load(path)
    assert len(registry) == 3
    assert function.key not in registry


def test_load_unknown_suffix_raises(tmp_path):
    with pytest.raises(ValueError):
        CostFunctionRegistry().load(tmp_path / "registry.txt")
