"""Cost function data model.

A :class:`CostFunction` describes the investment cost of one technology from
one source for one investment year. The size dependence is held by a function
form (see :mod:`energy_technology_catalogs.costs.forms`); the cost function
adds the size range, the cost basis, value-added tax, the share of planning
and installation, and relative uncertainty bounds.

Calculation order of :meth:`CostFunction.calculate_cost`:

1. sizes outside ``[size_min, size_max]`` are clamped unless extrapolation is
   requested,
2. the form is evaluated,
3. the planning and installation share is removed or added,
4. value-added tax is removed unless requested,
5. the result is converted between specific and absolute cost.

Years: ``investment_year`` is the year of installation and part of the
identity of a function; ``price_year`` is the price base of its parameters and
is only used for price adjustment.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import csv
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

from energy_technology_catalogs.costs.forms import (
    ArrayLike,
    Constant,
    CostForm,
    PowerLaw,
    form_from_dict,
)


class TechnologyType(Enum):
    """Technology category of a cost function.

    Labels and ontology references are kept in the packaged table
    ``data/technology_types.csv``.
    """

    HEAT_PUMP = "heat_pump"
    HEAT_PUMP_1STAGE = "heat_pump_1stage"
    HEAT_PUMP_1STAGE_AIR_SOURCE = "heat_pump_1stage_air_source"
    HEAT_PUMP_1STAGE_WATER_SOURCE = "heat_pump_1stage_water_source"
    HEAT_PUMP_1STAGE_GROUND_SOURCE = "heat_pump_1stage_ground_source"
    HEAT_PUMP_2STAGE_AIR_SOURCE = "heat_pump_2stage_air_source"
    HEAT_PUMP_2STAGE_WATER_SOURCE = "heat_pump_2stage_water_source"
    HEAT_PUMP_2STAGE_WASTE_HEAT_SOURCE = "heat_pump_2stage_waste_heat_source"
    HEAT_PUMP_2STAGE_WASTEWATER_SOURCE = "heat_pump_2stage_wastewater_source"
    HEAT_PUMP_ABSORPTION = "heat_pump_absorption"
    THERMAL_STORAGE = "thermal_storage"
    PTES = "PTES"
    TTES = "TTES"
    TTES_SMALL = "TTES_small"
    TTES_DHW = "TTES_dhw"
    ELECTRICAL_STORAGE = "electrical_storage"
    SOLAR_THERMAL = "solar_thermal"
    SOLAR_THERMAL_FLAT_PLATE = "solar_thermal_flat_plate"
    SOLAR_THERMAL_EVACUATED_TUBE = "solar_thermal_evacuated_tube"
    BOILER = "boiler"
    BIOMASS_BOILER = "biomass_boiler"
    OIL_BOILER = "oil_boiler"
    GAS_BOILER = "gas_boiler"
    H2_BOILER = "h2_boiler"
    ELECTRIC_BOILER = "electric_boiler"
    ELECTRIC_HEATER = "electric_heater"
    GEOTHERMAL_WELL = "geothermal_well"
    CHP = "CHP"
    WASTE_BOILER = "waste_boiler"
    PV = "pv"
    WIND = "wind"
    CHILLER = "chiller"
    HEATING_SUBSTATION = "heating_substation"
    DHW_SUBSTATION = "DHW_substation"

    @classmethod
    def from_string(cls, value: str) -> TechnologyType:
        """Look up a member by its value or name, ignoring case.

        :param value: Member value (``"heat_pump"``) or name (``"HEAT_PUMP"``).
        :return: The member.
        :raises ValueError: If ``value`` is not a string or matches no member.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}.")
        normalised = value.strip().lower()
        for member in cls:
            if normalised in (member.value.lower(), member.name.lower()):
                return member
        raise ValueError(
            f"'{value}' is not a valid {cls.__name__}. "
            f"Valid options: {[m.value for m in cls]}"
        )

    @property
    def oeo_iri(self) -> str:
        """IRI of the matching Open Energy Ontology class, empty if unknown."""
        return _technology_type_table()[self.name]["oeo_iri"]

    def label(self, lang: str = "en") -> str:
        """Return the display name.

        :param lang: ``"en"`` or ``"de"``.
        :return: Display name in that language.
        :raises ValueError: For another language code.
        """
        if lang not in ("en", "de"):
            raise ValueError(f"Unknown language {lang!r}; use 'en' or 'de'.")
        return _technology_type_table()[self.name][f"label_{lang}"]


@lru_cache(maxsize=1)
def _technology_type_table() -> dict[str, dict[str, str]]:
    """Read ``data/technology_types.csv`` once, keyed by member name."""
    path = resources.files("energy_technology_catalogs").joinpath(
        "data/technology_types.csv"
    )
    with path.open(encoding="utf-8") as file:
        rows = csv.DictReader(
            (line for line in file if not line.startswith("#")), delimiter=";"
        )
        return {row["technology_type"]: dict(row) for row in rows}


class CostBasis(Enum):
    """What the form of a cost function returns."""

    SPECIFIC = "specific"
    """Cost per size unit, e.g. EUR/kW."""
    ABSOLUTE = "absolute"
    """Cost of the whole unit, e.g. EUR."""


@dataclass(frozen=True)
class CostFunctionKey:
    """Identity of a cost function.

    :param technology: Name of the technology as given by the source.
    :param source: Literature or data source.
    :param investment_year: Year in which the technology is installed.
    """

    technology: str
    source: str
    investment_year: int

    def __post_init__(self) -> None:
        if not isinstance(self.technology, str) or not self.technology.strip():
            raise ValueError("'technology' must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("'source' must be a non-empty string.")
        year = self.investment_year
        if not isinstance(year, int | np.integer) or isinstance(year, bool):
            raise TypeError(
                f"'investment_year' must be int, got {type(year).__name__}."
            )
        object.__setattr__(self, "investment_year", int(year))

    def __str__(self) -> str:
        return f"{self.technology} | {self.source} | {self.investment_year}"


LEGACY_ALIASES: dict[str, str] = {
    "name": "technology",
    "component": "technology",
    "reference_unit": "size_unit",
    "reference_year": "price_year",
    "validity_range_lower": "size_min",
    "validity_range_upper": "size_max",
    "validity_range": "size_range",
    "taxes_included": "tax_rate",
    "taxes_incl": "tax_rate",
    "tax_incl": "tax_rate",
    "pi_included": "installation_included",
    "P_&_I_included": "installation_included",
    "P_&_I_incl": "installation_included",
    "pi_factor": "installation_factor",
    "P_&_I_factor": "installation_factor",
    "lifetime": "lifetime_a",
    "uncertainty": "uncertainty_pct",
    "summand": "offset",
}
"""Keys of the Uni Kassel cost function module v3.0.0 and their names here.

``installation_factor``, ``size_range``, ``uncertainty_pct`` and the power-law
keys ``offset``, ``coefficient``, ``exponent`` are only read, never written.
"""


def _is_missing(value: Any) -> bool:
    """True for None, NaN and empty strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(value))
    except TypeError:
        return False


def _float_or(value: Any, default: float) -> float:
    return default if _is_missing(value) else float(value)


def _year(value: Any) -> int:
    """Convert a year given as int, integral float or digit string."""
    year = float(value)
    if not year.is_integer():
        raise ValueError(f"A year must be a whole number, got {value!r}.")
    return int(year)


def _optional_form(value: Any) -> CostForm | None:
    """Accept a form, its dictionary, a scalar (constant form) or nothing."""
    if _is_missing(value):
        return None
    if isinstance(value, CostForm):
        return value
    if isinstance(value, Mapping):
        return form_from_dict(value)
    return Constant(float(value))


@dataclass(frozen=True)
class CostFunction:
    """Investment cost of a technology as a function of its size.

    :param technology: Name of the technology as given by the source.
    :param source: Literature or data source.
    :param technology_type: Technology category.
    :param size_unit: Unit of the size, e.g. ``"kW"`` or ``"m3"``.
    :param investment_year: Year of installation; part of the identity.
    :param size_min: Lower bound of the valid size range, in ``size_unit``
        (>= 0).
    :param size_max: Upper bound of the valid size range, in ``size_unit``
        (> ``size_min``).
    :param form: Function form of the investment cost. Its value is a gross
        cost (including value-added tax at ``tax_rate``) in ``currency`` at
        prices of ``price_year``, per size unit or absolute according to
        ``cost_basis``.
    :param cost_basis: Whether ``form`` and ``fixed_om`` yield specific or
        absolute values.
    :param fixed_om: Optional form of the fixed operation and maintenance
        cost per year, specific or absolute like ``form``. It is returned as
        stored: neither tax nor the installation share is applied to it.
    :param installation_included: Whether the value of ``form`` contains
        planning and installation.
    :param installation_share: Optional form of the share of planning and
        installation in the total cost including them (0 <= share < 1,
        dimensionless). Without it the cost is the same with and without
        planning and installation.
    :param lifetime_a: Optional technical lifetime in years.
    :param tax_rate: Value-added tax contained in the value of ``form``, as a
        fraction of the net cost (0.19 for 19 %).
    :param uncertainty_lower: Lower relative uncertainty bound as a fraction
        of the net cost (<= 0).
    :param uncertainty_upper: Upper relative uncertainty bound as a fraction
        of the net cost (>= 0).
    :param currency: ISO 4217 code.
    :param price_year: Price base of the cost parameters; defaults to
        ``investment_year``.
    """

    technology: str
    source: str
    technology_type: TechnologyType
    size_unit: str
    investment_year: int
    size_min: float
    size_max: float
    form: CostForm
    cost_basis: CostBasis = CostBasis.SPECIFIC
    fixed_om: CostForm | None = None
    installation_included: bool = False
    installation_share: CostForm | None = None
    lifetime_a: float | None = None
    tax_rate: float = 0.0
    uncertainty_lower: float = 0.0
    uncertainty_upper: float = 0.0
    currency: str = "EUR"
    price_year: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.technology, str) or not self.technology.strip():
            raise ValueError("'technology' must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("'source' must be a non-empty string.")
        if not isinstance(self.technology_type, TechnologyType):
            raise TypeError(
                f"'technology_type' must be TechnologyType, "
                f"got {type(self.technology_type).__name__}."
            )
        if not isinstance(self.cost_basis, CostBasis):
            raise TypeError(
                f"'cost_basis' must be CostBasis, got {type(self.cost_basis).__name__}."
            )
        if not isinstance(self.form, CostForm):
            raise TypeError(
                f"'form' must be a CostForm, got {type(self.form).__name__}."
            )
        for name in ("fixed_om", "installation_share"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, CostForm):
                raise TypeError(
                    f"'{name}' must be a CostForm or None, got {type(value).__name__}."
                )
        if self.size_min < 0:
            raise ValueError(f"size_min ({self.size_min}) must be >= 0.")
        if not self.size_min < self.size_max:
            raise ValueError(
                f"size_min ({self.size_min}) must be < size_max ({self.size_max})."
            )
        if self.price_year is None:
            object.__setattr__(self, "price_year", self.investment_year)
        for name in ("investment_year", "price_year"):
            year = getattr(self, name)
            if not isinstance(year, int | np.integer) or isinstance(year, bool):
                raise TypeError(f"'{name}' must be int, got {type(year).__name__}.")
            if not 1900 <= year <= 2200:
                raise ValueError(
                    f"{name} ({year}) is implausible (expected 1900-2200)."
                )
            object.__setattr__(self, name, int(year))
        if not isinstance(self.installation_included, bool | np.bool_):
            raise TypeError(
                "'installation_included' must be bool, "
                f"got {type(self.installation_included).__name__}."
            )
        object.__setattr__(
            self, "installation_included", bool(self.installation_included)
        )
        # negated comparisons so that NaN is rejected as well
        if not self.tax_rate >= 0:
            raise ValueError(f"tax_rate ({self.tax_rate}) must be >= 0.")
        if not self.uncertainty_lower <= 0:
            raise ValueError(
                f"uncertainty_lower ({self.uncertainty_lower}) must be <= 0."
            )
        if not self.uncertainty_upper >= 0:
            raise ValueError(
                f"uncertainty_upper ({self.uncertainty_upper}) must be >= 0."
            )
        if self.lifetime_a is not None and not self.lifetime_a > 0:
            raise ValueError(f"lifetime_a ({self.lifetime_a}) must be > 0.")

    @property
    def name(self) -> str:
        """``"<technology> | <source> | <investment_year>"``."""
        return str(self.key)

    @property
    def key(self) -> CostFunctionKey:
        """Identity of the function."""
        return CostFunctionKey(self.technology, self.source, self.investment_year)

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    def _effective_size(self, size: ArrayLike, extrapolate: bool) -> np.ndarray:
        """Validate ``size`` and clamp it to the valid range if required."""
        requested = np.asarray(size, dtype=float)
        if not np.all(requested > 0):
            raise ValueError(f"size must be positive, got {size}.")
        if extrapolate:
            return requested
        effective = np.clip(requested, self.size_min, self.size_max)
        if np.any(effective != requested):
            warnings.warn(
                f"Size outside the valid range [{self.size_min}, {self.size_max}] "
                f"{self.size_unit} of '{self.name}': the cost at the range limit "
                "is used. Pass extrapolate=True to evaluate the function there.",
                UserWarning,
                stacklevel=3,
            )
        return effective

    def _installation_share_at(self, effective_size: np.ndarray) -> Any:
        """Share of planning and installation at ``effective_size``."""
        if self.installation_share is None:
            return 0.0
        share = np.asarray(self.installation_share.evaluate(effective_size))
        invalid = ~((share >= 0) & (share < 1))
        if np.any(invalid):
            size = np.broadcast_to(effective_size, share.shape)[invalid].flat[0]
            raise ValueError(
                f"The installation share of '{self.name}' is "
                f"{share[invalid].flat[0]:g} at size {size:g} {self.size_unit}; "
                "it must be in [0, 1)."
            )
        return share

    def _to_requested_basis(
        self,
        value: Any,
        effective_size: np.ndarray,
        requested_size: np.ndarray,
        specific: bool,
    ) -> Any:
        """Convert a value of the form's basis to specific or absolute.

        Outside the valid range the specific cost of the range limit applies
        to the requested size.
        """
        if self.cost_basis is CostBasis.ABSOLUTE:
            value = value / effective_size
        return value if specific else value * requested_size

    def _finalize_cost(
        self,
        base: Any,
        effective_size: np.ndarray,
        requested_size: np.ndarray,
        *,
        include_installation: bool,
        include_tax: bool,
        specific: bool,
    ) -> Any:
        """Apply installation share, tax and basis conversion to a form value.

        :param base: Value(s) of ``form`` at ``effective_size``.
        :param effective_size: Size used in the form (clamped if required).
        :param requested_size: Size asked for by the caller.
        :param include_installation: Include planning and installation.
        :param include_tax: Include value-added tax.
        :param specific: Return specific instead of absolute cost.
        :return: Cost in ``currency`` per ``size_unit`` or in ``currency``.
        """
        cost = self._apply_installation(base, effective_size, include_installation)
        return self._apply_tax_and_basis(
            cost,
            effective_size,
            requested_size,
            include_tax=include_tax,
            specific=specific,
        )

    def _apply_installation(
        self, base: Any, effective_size: np.ndarray, include_installation: bool
    ) -> Any:
        """Remove or add planning and installation on a form value.

        :param base: Value(s) of ``form`` at ``effective_size``.
        :param effective_size: Size used in the form (clamped if required).
        :param include_installation: Include planning and installation.
        :return: Gross value on the basis of the form.
        """
        if self.installation_included == include_installation:
            return base
        share = self._installation_share_at(effective_size)
        return base / (1.0 - share) if include_installation else base * (1.0 - share)

    def _apply_tax_and_basis(
        self,
        cost: Any,
        effective_size: np.ndarray,
        requested_size: np.ndarray,
        *,
        include_tax: bool,
        specific: bool,
    ) -> Any:
        """Remove value-added tax if required and convert the basis.

        :param cost: Gross value(s) on the basis of the form.
        :param effective_size: Size used in the form (clamped if required).
        :param requested_size: Size asked for by the caller.
        :param include_tax: Keep value-added tax.
        :param specific: Return specific instead of absolute cost.
        :return: Cost in ``currency`` per ``size_unit`` or in ``currency``.
        """
        if not include_tax:
            cost = cost / (1.0 + self.tax_rate)
        return self._to_requested_basis(cost, effective_size, requested_size, specific)

    def calculate_cost(
        self,
        size: ArrayLike,
        *,
        include_installation: bool = True,
        include_tax: bool = False,
        specific: bool = True,
        extrapolate: bool = False,
    ) -> ArrayLike:
        """Calculate the investment cost.

        :param size: Size(s) in ``size_unit``; must be positive.
        :param include_installation: Include planning and installation
            (default). Has an effect only if ``installation_share`` is set.
        :param include_tax: Return the cost including value-added tax; the
            default is the net cost, which is comparable between sources.
        :param specific: Return the cost per size unit (default) or, if
            False, the absolute cost of a unit of that size.
        :param extrapolate: Evaluate the function outside the valid size
            range. By default the specific cost of the range limit is used
            and a ``UserWarning`` is issued.
        :return: Cost in ``currency`` per ``size_unit`` (specific) or in
            ``currency`` (absolute), at prices of ``price_year``; a float for
            a scalar ``size``, otherwise an array.
        :raises ValueError: If a size is not positive, or the installation
            share is outside [0, 1) at a size used in the calculation.
        """
        requested = np.asarray(size, dtype=float)
        effective = self._effective_size(size, extrapolate)
        cost = self._finalize_cost(
            self.form.evaluate(effective),
            effective,
            requested,
            include_installation=include_installation,
            include_tax=include_tax,
            specific=specific,
        )
        return float(cost) if requested.ndim == 0 else np.asarray(cost, dtype=float)

    def calculate_fixed_om(
        self,
        size: ArrayLike,
        *,
        specific: bool = True,
        extrapolate: bool = False,
    ) -> ArrayLike:
        """Calculate the fixed operation and maintenance cost per year.

        :param size: Size(s) in ``size_unit``; must be positive.
        :param specific: Return the cost per size unit and year (default) or
            the absolute cost per year.
        :param extrapolate: As in :meth:`calculate_cost`.
        :return: Cost in ``currency`` per ``size_unit`` and year, or in
            ``currency`` per year.
        :raises ValueError: If no ``fixed_om`` form is set or a size is not
            positive.
        """
        if self.fixed_om is None:
            raise ValueError(f"'{self.name}' has no fixed O&M cost.")
        requested = np.asarray(size, dtype=float)
        effective = self._effective_size(size, extrapolate)
        cost = self._to_requested_basis(
            self.fixed_om.evaluate(effective), effective, requested, specific
        )
        return float(cost) if requested.ndim == 0 else np.asarray(cost, dtype=float)

    # ------------------------------------------------------------------
    # Dictionary conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary with nested form dictionaries.

        :return: Dictionary accepted by :meth:`from_dict`.
        """

        def form_dict(form: CostForm | None) -> dict[str, Any] | None:
            return None if form is None else form.to_dict()

        return {
            "technology": self.technology,
            "source": self.source,
            "technology_type": self.technology_type.value,
            "size_unit": self.size_unit,
            "investment_year": self.investment_year,
            "price_year": self.price_year,
            "size_min": float(self.size_min),
            "size_max": float(self.size_max),
            "cost_basis": self.cost_basis.value,
            "form": self.form.to_dict(),
            "fixed_om": form_dict(self.fixed_om),
            "installation_included": bool(self.installation_included),
            "installation_share": form_dict(self.installation_share),
            "lifetime_a": None if self.lifetime_a is None else float(self.lifetime_a),
            "tax_rate": float(self.tax_rate),
            "uncertainty_lower": float(self.uncertainty_lower),
            "uncertainty_upper": float(self.uncertainty_upper),
            "currency": self.currency,
        }

    @classmethod
    def _fields_from_dict(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve a dictionary to constructor arguments of ``CostFunction``."""
        # names used here win over reference names; a reference name only
        # fills a value that is absent or empty
        d: dict[str, Any] = {k: v for k, v in data.items() if k not in LEGACY_ALIASES}
        for key, value in data.items():
            name = LEGACY_ALIASES.get(key)
            if name is None or _is_missing(value):
                continue
            # "component" is the technology name, "name" may be a composed label
            if _is_missing(d.get(name)) or key == "component":
                d[name] = value
        legacy_power_law = "form" not in d and "coefficient" in d

        fields: dict[str, Any] = {
            "technology": d.get("technology"),
            "source": d.get("source"),
            "size_unit": "" if _is_missing(d.get("size_unit")) else str(d["size_unit"]),
        }

        # years: a record with a price year only uses it as investment year
        if _is_missing(d.get("investment_year")):
            if _is_missing(d.get("price_year")):
                raise ValueError("Neither 'investment_year' nor a price year found.")
            warnings.warn(
                f"No 'investment_year' given; using the price year "
                f"({d['price_year']}) as investment year.",
                DeprecationWarning,
                stacklevel=3,
            )
            d["investment_year"] = d["price_year"]
        fields["investment_year"] = _year(d["investment_year"])
        fields["price_year"] = (
            None if _is_missing(d.get("price_year")) else _year(d["price_year"])
        )

        technology_type = d.get("technology_type")
        if isinstance(technology_type, str):
            technology_type = TechnologyType.from_string(technology_type)
        if not isinstance(technology_type, TechnologyType):
            raise TypeError(
                f"Cannot convert {type(technology_type).__name__!r} to TechnologyType."
            )
        fields["technology_type"] = technology_type

        # size range: separate bounds, or one "[lower, upper]" value
        if "size_min" in d or "size_max" in d:
            if _is_missing(d.get("size_max")):
                raise ValueError("The upper bound of the size range is missing.")
            fields["size_min"] = _float_or(d.get("size_min"), 0.0)
            fields["size_max"] = float(d["size_max"])
        elif not _is_missing(d.get("size_range")):
            size_range = d["size_range"]
            try:
                if isinstance(size_range, str):
                    size_range = ast.literal_eval(size_range)
                lower, upper = size_range
                fields["size_min"], fields["size_max"] = float(lower), float(upper)
            except (ValueError, SyntaxError, TypeError) as error:
                raise ValueError(
                    f"Cannot resolve the size range from {d['size_range']!r}."
                ) from error
        else:
            raise ValueError("No size range found ('size_min' and 'size_max').")

        # investment cost form: nested, or the three power-law values
        form = d.get("form")
        if isinstance(form, Mapping):
            form = form_from_dict(form)
        elif legacy_power_law:
            if _is_missing(d.get("coefficient")) or _is_missing(d.get("exponent")):
                raise ValueError("'coefficient' and 'exponent' must be numbers.")
            form = PowerLaw(
                _float_or(d.get("offset"), 0.0),
                float(d["coefficient"]),
                float(d["exponent"]),
            )
        if not isinstance(form, CostForm):
            raise ValueError("No function form found ('form').")
        fields["form"] = form

        cost_basis = d.get("cost_basis")
        if _is_missing(cost_basis):
            cost_basis = CostBasis.SPECIFIC
        fields["cost_basis"] = CostBasis(cost_basis)

        fixed_om = d.get("fixed_om")
        # the reference module stores 0.0 where no fixed O&M cost is known
        if legacy_power_law and not _is_missing(fixed_om) and fixed_om == 0:
            fixed_om = None
        fields["fixed_om"] = _optional_form(fixed_om)

        # installation: share as a form, or the factor of the reference module
        # (a share if included, a multiplier >= 1 if not included)
        included = d.get("installation_included")
        if _is_missing(included):
            included = False
        elif isinstance(included, str):
            raise ValueError(
                f"'installation_included' must be a boolean, got the text {included!r}."
            )
        fields["installation_included"] = bool(included)
        share = _optional_form(d.get("installation_share"))
        if share is None and not _is_missing(d.get("installation_factor")):
            factor = float(d["installation_factor"])
            if fields["installation_included"]:
                if not 0.0 <= factor < 1.0:
                    raise ValueError(
                        f"An included installation share must be in [0, 1), got {factor}."
                    )
                share = Constant(factor) if factor > 0.0 else None
            else:
                if factor < 1.0:
                    raise ValueError(
                        f"An installation multiplier must be >= 1, got {factor}."
                    )
                share = Constant(1.0 - 1.0 / factor) if factor > 1.0 else None
        fields["installation_share"] = share

        fields["lifetime_a"] = (
            None if _is_missing(d.get("lifetime_a")) else float(d["lifetime_a"])
        )
        fields["tax_rate"] = _float_or(d.get("tax_rate"), 0.0)

        if (
            not _is_missing(d.get("uncertainty_pct"))
            and "uncertainty_lower" not in d
            and "uncertainty_upper" not in d
        ):
            symmetric = abs(float(d["uncertainty_pct"]))
            fields["uncertainty_lower"] = -symmetric
            fields["uncertainty_upper"] = symmetric
        else:
            fields["uncertainty_lower"] = _float_or(d.get("uncertainty_lower"), 0.0)
            fields["uncertainty_upper"] = _float_or(d.get("uncertainty_upper"), 0.0)

        currency = d.get("currency")
        fields["currency"] = "EUR" if _is_missing(currency) else str(currency)
        return fields

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostFunction:
        """Build a cost function from a dictionary.

        Accepts the output of :meth:`to_dict` and the keys of the Uni Kassel
        cost function module v3.0.0 (see :data:`LEGACY_ALIASES`): there the
        three values ``summand``, ``coefficient``, ``exponent`` describe a
        power law of the specific cost, a scalar ``fixed_om`` a constant, and
        ``pi_factor`` either the included installation share or, if not
        included, a multiplier ``f >= 1`` (read as the share ``1 - 1/f``).
        Unknown keys are ignored.

        :param data: Dictionary of field values.
        :return: The cost function.
        :raises ValueError: If required values are missing or cannot be read.
        :raises TypeError: If the technology type cannot be resolved or a value
            has the wrong type.
        """
        return cls(**cls._fields_from_dict(data))
