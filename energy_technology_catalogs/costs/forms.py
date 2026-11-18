"""Function forms of cost functions.

A form maps a component size to a cost value and knows nothing about years,
taxes or units. Every form offers the same interface (evaluate, named
parameters, scaling and shifting of the output, dict conversion), so the other
modules of the sub-package never depend on a specific formula.

Whether the value is a specific cost (currency per size unit) or an absolute
cost (currency) is decided by the cost function that holds the form.

The power law is ported from the Uni Kassel cost function module v3.0.0; the
other forms are new.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias, TypeVar

import numpy as np


ArrayLike: TypeAlias = float | np.ndarray
FormT = TypeVar("FormT", bound="CostForm")


class CostForm(ABC):
    """Common interface of all function forms.

    Subclasses are frozen dataclasses. Their parameters are plain floats with
    stable names (see :meth:`parameters`).
    """

    kind: ClassVar[str]
    _tuple_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if field.name in self._tuple_fields:
                if isinstance(value, str):
                    raise ValueError(
                        f"'{field.name}' must be a sequence of numbers, got a string."
                    )
                value = tuple(float(v) for v in value)
                finite = all(math.isfinite(v) for v in value)
            else:
                value = float(value)
                finite = math.isfinite(value)
            if not finite:
                raise ValueError(
                    f"'{field.name}' of a {self.kind} form must be finite, got {value}."
                )
            object.__setattr__(self, field.name, value)
        self._validate()

    def _validate(self) -> None:
        """Check the field values; overridden where a form has rules."""

    @abstractmethod
    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        """Bare formula on a float array ``size`` and parameter values ``p``."""

    @abstractmethod
    def cost_parameter_names(self) -> tuple[str, ...]:
        """Names of the parameters that are linear in the output.

        :return: Parameter names; multiplying all of them by a factor
            multiplies the output by that factor.
        """

    @abstractmethod
    def _offset_parameter_names(self) -> tuple[str, ...]:
        """Names of the parameters that shift the output when increased."""

    def parameters(self) -> dict[str, float]:
        """Return all named parameters.

        :return: Mapping of parameter name to value, in field order.
        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)  # type: ignore[arg-type]
        }

    def with_parameters(self: FormT, **changes: float) -> FormT:
        """Return a copy with some parameters replaced.

        :param changes: New values by parameter name.
        :return: New form of the same kind.
        :raises ValueError: If a name is not a parameter of this form.
        """
        self._check_names(changes)
        return dataclasses.replace(self, **changes)  # type: ignore[type-var]

    def _check_names(self, names: Any) -> None:
        unknown = sorted(set(names) - set(self.parameters()))
        if unknown:
            raise ValueError(
                f"{unknown} are not parameters of a {self.kind} form; "
                f"valid names: {list(self.parameters())}."
            )

    def evaluate(
        self,
        size: ArrayLike,
        parameters: Mapping[str, ArrayLike] | None = None,
    ) -> ArrayLike:
        """Evaluate the form.

        :param size: Component size (scalar or array), in the size unit of the
            cost function that holds the form.
        :param parameters: Optional values that replace stored parameters by
            name for this call. Arrays are allowed, e.g. sampled parameter
            values together with a scalar ``size``.
        :return: Cost value(s); a float if ``size`` and all parameters are
            scalars, otherwise an array.
        :raises ValueError: If ``parameters`` contains an unknown name, or
            arrays are given for both ``size`` and a parameter.
        """
        values: dict[str, Any] = dict(self.parameters())
        x = np.asarray(size, dtype=float)
        if parameters:
            self._check_names(parameters)
            for name, value in parameters.items():
                values[name] = np.asarray(value, dtype=float)
                if x.ndim > 0 and values[name].ndim > 0:
                    raise ValueError(
                        "Parameter arrays need a scalar size; "
                        f"got arrays for the size and '{name}'."
                    )
        result = np.asarray(self._formula(x, values), dtype=float)
        return float(result) if result.ndim == 0 else result

    def scaled(self: FormT, factor: float) -> FormT:
        """Return a form whose output is ``factor`` times this output.

        :param factor: Dimensionless factor, e.g. a price index ratio.
        :return: New form of the same kind.
        """
        values = self.parameters()
        return self.with_parameters(
            **{name: values[name] * factor for name in self.cost_parameter_names()}
        )

    def shifted(self: FormT, delta: float) -> FormT:
        """Return a form whose output is this output plus ``delta``.

        :param delta: Shift in the unit of the output.
        :return: New form of the same kind.
        """
        values = self.parameters()
        return self.with_parameters(
            **{name: values[name] + delta for name in self._offset_parameter_names()}
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        :return: ``{"kind": ..., <field>: <value>, ...}``; tuples become lists.
        """
        data: dict[str, Any] = {"kind": self.kind}
        for field in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            data[field.name] = list(value) if isinstance(value, tuple) else value
        return data


@dataclass(frozen=True)
class PowerLaw(CostForm):
    """``offset + coefficient * size ** exponent``.

    :param offset: Additive term, in the unit of the output.
    :param coefficient: Factor of the size term.
    :param exponent: Dimensionless exponent of the size.
    """

    offset: float
    coefficient: float
    exponent: float

    kind: ClassVar[str] = "power_law"

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        return p["offset"] + p["coefficient"] * size ** p["exponent"]

    def cost_parameter_names(self) -> tuple[str, ...]:
        return ("offset", "coefficient")

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return ("offset",)


@dataclass(frozen=True)
class Linear(CostForm):
    """``offset + slope * size``.

    :param offset: Value at size zero, in the unit of the output.
    :param slope: Change of the output per size unit.
    """

    offset: float
    slope: float

    kind: ClassVar[str] = "linear"

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        return p["offset"] + p["slope"] * size

    def cost_parameter_names(self) -> tuple[str, ...]:
        return ("offset", "slope")

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return ("offset",)


@dataclass(frozen=True)
class Polynomial(CostForm):
    """``c0 + c1 * size + c2 * size ** 2 + ...``.

    :param coefficients: Coefficients in ascending order of the power of the
        size; at least one. Parameter names are ``c0``, ``c1``, ...
    """

    coefficients: tuple[float, ...]

    kind: ClassVar[str] = "polynomial"
    _tuple_fields: ClassVar[tuple[str, ...]] = ("coefficients",)

    def _validate(self) -> None:
        if not self.coefficients:
            raise ValueError("A polynomial needs at least one coefficient.")

    def parameters(self) -> dict[str, float]:
        return {f"c{i}": c for i, c in enumerate(self.coefficients)}

    def with_parameters(self, **changes: float) -> Polynomial:
        self._check_names(changes)
        values = {**self.parameters(), **changes}
        return Polynomial(tuple(values.values()))

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        # Horner scheme, highest power first
        result: Any = 0.0
        for i in range(len(self.coefficients) - 1, -1, -1):
            result = result * size + p[f"c{i}"]
        return result

    def cost_parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameters())

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return ("c0",)


@dataclass(frozen=True)
class Logarithmic(CostForm):
    """``offset + coefficient * ln(size)`` (natural logarithm, size > 0).

    :param offset: Value at size one, in the unit of the output.
    :param coefficient: Factor of the logarithm.
    """

    offset: float
    coefficient: float

    kind: ClassVar[str] = "logarithmic"

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        return p["offset"] + p["coefficient"] * np.log(size)

    def cost_parameter_names(self) -> tuple[str, ...]:
        return ("offset", "coefficient")

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return ("offset",)


@dataclass(frozen=True)
class Exponential(CostForm):
    """``offset + coefficient * exp(rate * size)``.

    :param offset: Additive term, in the unit of the output.
    :param coefficient: Factor of the exponential term.
    :param rate: Rate per size unit.
    """

    offset: float
    coefficient: float
    rate: float

    kind: ClassVar[str] = "exponential"

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        return p["offset"] + p["coefficient"] * np.exp(p["rate"] * size)

    def cost_parameter_names(self) -> tuple[str, ...]:
        return ("offset", "coefficient")

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return ("offset",)


@dataclass(frozen=True)
class Constant(CostForm):
    """The same value for every size.

    :param value: Output value.
    """

    value: float

    kind: ClassVar[str] = "constant"

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        return np.zeros(size.shape) + p["value"]

    def cost_parameter_names(self) -> tuple[str, ...]:
        return ("value",)

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return ("value",)


@dataclass(frozen=True)
class PiecewiseLinear(CostForm):
    """Linear interpolation between tabulated support points.

    Outside the first and last support point the end segments are extended
    linearly. The support sizes are structure, not parameters; the parameters
    are the values ``v0``, ``v1``, ... at the support sizes.

    :param sizes: Support sizes, strictly increasing, at least two.
    :param values: Output values at the support sizes.
    """

    sizes: tuple[float, ...]
    values: tuple[float, ...]

    kind: ClassVar[str] = "piecewise_linear"
    _tuple_fields: ClassVar[tuple[str, ...]] = ("sizes", "values")

    def _validate(self) -> None:
        if len(self.sizes) < 2:
            raise ValueError("A piecewise linear form needs at least two points.")
        if len(self.sizes) != len(self.values):
            raise ValueError(
                f"Got {len(self.sizes)} sizes but {len(self.values)} values."
            )
        if any(b <= a for a, b in zip(self.sizes, self.sizes[1:])):
            raise ValueError("Support sizes must be strictly increasing.")

    @classmethod
    def from_points(
        cls, sizes: Sequence[float], values: Sequence[float]
    ) -> PiecewiseLinear:
        """Build the form from support points in any order.

        :param sizes: Support sizes; duplicates are not allowed.
        :param values: Output values at the support sizes.
        :return: Form with the points sorted by size.
        :raises ValueError: If sizes repeat or the lengths differ.
        """
        if len(sizes) != len(values):
            raise ValueError(f"Got {len(sizes)} sizes but {len(values)} values.")
        pairs = sorted(zip(sizes, values))
        return cls(tuple(s for s, _ in pairs), tuple(v for _, v in pairs))

    def parameters(self) -> dict[str, float]:
        return {f"v{i}": v for i, v in enumerate(self.values)}

    def with_parameters(self, **changes: float) -> PiecewiseLinear:
        self._check_names(changes)
        values = {**self.parameters(), **changes}
        return PiecewiseLinear(self.sizes, tuple(values.values()))

    def _formula(self, size: np.ndarray, p: Mapping[str, Any]) -> Any:
        nodes = np.asarray(self.sizes)
        stacked = np.stack(
            np.broadcast_arrays(*[p[f"v{i}"] for i in range(len(nodes))])
        )
        segment = np.clip(
            np.searchsorted(nodes, size, side="right") - 1, 0, len(nodes) - 2
        )
        weight = (size - nodes[segment]) / (nodes[segment + 1] - nodes[segment])
        return stacked[segment] + weight * (stacked[segment + 1] - stacked[segment])

    def cost_parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameters())

    def _offset_parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameters())


FORM_KINDS: dict[str, type[CostForm]] = {
    form.kind: form
    for form in (
        PowerLaw,
        Linear,
        Polynomial,
        Logarithmic,
        Exponential,
        Constant,
        PiecewiseLinear,
    )
}


def form_from_parameters(
    kind: str,
    parameters: Mapping[str, float],
    *,
    sizes: Sequence[float] | None = None,
) -> CostForm:
    """Build a form from the names returned by :meth:`CostForm.parameters`.

    :param kind: Kind of the form, a key of :data:`FORM_KINDS`.
    :param parameters: Values by parameter name; for a polynomial ``c0``,
        ``c1``, ... without gaps, for a piecewise linear form ``v0``, ``v1``,
        ... (one per support size).
    :param sizes: Support sizes; required for ``"piecewise_linear"`` only.
    :return: The form.
    :raises ValueError: If the kind is unknown or the names do not fit it.
    """
    if kind not in FORM_KINDS:
        raise ValueError(
            f"Unknown function form {kind!r}; valid kinds: {sorted(FORM_KINDS)}."
        )
    try:
        if kind == Polynomial.kind:
            return Polynomial(
                tuple(parameters[f"c{i}"] for i in range(len(parameters)))
            )
        if kind == PiecewiseLinear.kind:
            if sizes is None:
                raise ValueError("A piecewise linear form needs 'sizes'.")
            return PiecewiseLinear(
                tuple(sizes), tuple(parameters[f"v{i}"] for i in range(len(parameters)))
            )
        return FORM_KINDS[kind](**parameters)
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Parameters {sorted(parameters)} do not fit a {kind} form."
        ) from error


def form_from_dict(data: Mapping[str, Any]) -> CostForm:
    """Build a form from the dictionary written by :meth:`CostForm.to_dict`.

    :param data: Mapping with the key ``kind`` and the fields of that form.
    :return: The form.
    :raises ValueError: If ``kind`` is missing or unknown, or fields do not
        match the form.
    """
    fields = dict(data)
    kind = fields.pop("kind", None)
    if kind not in FORM_KINDS:
        raise ValueError(
            f"Unknown function form {kind!r}; valid kinds: {sorted(FORM_KINDS)}."
        )
    try:
        return FORM_KINDS[kind](**fields)
    except TypeError as error:
        raise ValueError(f"Invalid fields for a {kind} form: {error}") from error
