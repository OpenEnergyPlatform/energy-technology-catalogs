"""Shared fixtures of the cost function tests.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import json
from pathlib import Path

import pytest

from energy_technology_catalogs.costs.core import CostFunction, TechnologyType
from energy_technology_catalogs.costs.forms import Constant, PowerLaw


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "cost_functions"


@pytest.fixture
def function() -> CostFunction:
    """Specific cost ``1000 * size ** -0.5`` EUR/kW for 10 to 100 kW.

    No tax; planning and installation are included with a share of 0.4.
    At 25 kW the cost is 200 EUR/kW.
    """
    return CostFunction(
        technology="Boiler",
        source="Test",
        technology_type=TechnologyType.BIOMASS_BOILER,
        size_unit="kW",
        investment_year=2025,
        size_min=10.0,
        size_max=100.0,
        form=PowerLaw(0.0, 1000.0, -0.5),
        installation_included=True,
        installation_share=Constant(0.4),
    )


@pytest.fixture
def reference_data():
    """Load a parity fixture generated from the Uni Kassel cost function module v3.0.0."""

    def load(name: str):
        with open(FIXTURE_DIR / f"{name}.json", encoding="utf-8") as file:
            return json.load(file)

    return load


@pytest.fixture
def reference_layout_csv() -> Path:
    """Path of a cost function table in the column layout of the reference module."""
    return FIXTURE_DIR / "reference_layout.csv"
