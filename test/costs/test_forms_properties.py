"""Property tests of the function forms (run where hypothesis is installed, e.g. tox).

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

import pytest


hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from energy_technology_catalogs.costs.forms import (
    Linear,
    Polynomial,
    PowerLaw,
    form_from_dict,
)


values = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
sizes = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False)
exponents = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)
factors = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)

forms = st.one_of(
    st.builds(PowerLaw, values, values, exponents),
    st.builds(Linear, values, values),
    st.builds(Polynomial, st.lists(values, min_size=1, max_size=4).map(tuple)),
)


@given(form=forms, size=sizes, factor=factors)
def test_scaled_multiplies_output(form, size, factor):
    expected = factor * form.evaluate(size)
    assert form.scaled(factor).evaluate(size) == pytest.approx(
        expected, rel=1e-9, abs=1e-6
    )


@given(form=forms, size=sizes, delta=values)
def test_shifted_adds_to_output(form, size, delta):
    expected = form.evaluate(size) + delta
    assert form.shifted(delta).evaluate(size) == pytest.approx(
        expected, rel=1e-9, abs=1e-6
    )


@given(form=forms)
def test_dict_roundtrip(form):
    assert form_from_dict(form.to_dict()) == form
