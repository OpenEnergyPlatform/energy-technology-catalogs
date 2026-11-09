"""Smoke test: the package can be imported.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import energy_technology_catalogs


def test_package_imports():
    """The package is importable under its module name."""
    assert energy_technology_catalogs.__name__ == "energy_technology_catalogs"
