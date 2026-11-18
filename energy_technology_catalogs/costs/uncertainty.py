"""Uncertainty of cost functions.

Two kinds of uncertainty can be attached to a cost function:

* **Parameter uncertainty**: a probability distribution of the value of a
  form parameter (e.g. the ``coefficient`` of a power law). The cost is
  obtained by Monte Carlo sampling: every uncertain parameter is drawn
  independently and the form is evaluated once with the sampled values.
* **Output uncertainty**: a distribution of a deviation ``eps`` applied to
  the deterministic gross cost ``raw`` (the form value after adding or
  removing planning and installation, on the cost basis of the form):

  ============================  ==========================
  mode                          cost
  ============================  ==========================
  absolute                      ``raw + eps``
  relative                      ``raw * (1 + eps)``
  relative, mean-preserving     ``raw * (1 + eps - E[eps])``
  direct                        ``eps``
  ============================  ==========================

  Value-added tax is removed and the cost basis converted after the mode is
  applied. With output uncertainty only, cost quantiles follow analytically
  from the quantiles of ``eps`` (the transforms are monotonically increasing
  for a non-negative ``raw``).

Relative uncertainty bounds of a cost function are turned into an output
distribution by :func:`distribution_from_bounds`: uniform for symmetric or
one-sided bounds, triangular with mode 0 for asymmetric two-sided bounds, or
a normal or Laplace distribution whose central interval of probability
``coverage`` equals the bounds.

Random numbers come from a :class:`numpy.random.Generator`, so results are
reproducible with a seed. The mean-preserving mode subtracts the analytical
mean of the deviation distribution in both the analytical and the sampled
calculation.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from energy_technology_catalogs.costs.core import CostFunction
from energy_technology_catalogs.costs.forms import CostForm


_SYMMETRY_TOLERANCE = 1e-9


class DistributionKind(Enum):
    """Probability distributions of :class:`Distribution`."""

    DETERMINISTIC = "deterministic"
    """A fixed value ``mean``."""
    NORMAL = "normal"
    """Normal distribution with ``mean`` and ``std_dev``."""
    LOGNORMAL = "lognormal"
    """Lognormal distribution; ``mean`` and ``std_dev`` of the logarithm."""
    UNIFORM = "uniform"
    """Uniform distribution on ``[lower, upper]``."""
    TRIANGULAR = "triangular"
    """Triangular distribution on ``[lower, upper]`` with peak ``mode``."""
    LAPLACE = "laplace"
    """Laplace distribution with location ``mean`` and ``scale``."""


class OutputUncertaintyMode(Enum):
    """How an output deviation ``eps`` acts on the deterministic gross cost.

    Caveats (not checked):

    * ``RELATIVE`` with an asymmetric distribution (``E[eps] != 0``, e.g.
      triangular bounds -0.2/+0.3 with mode 0) shifts the expected cost to
      ``raw * (1 + E[eps])``; use ``RELATIVE_MEAN_PRESERVING`` to keep ``raw``
      as expected value.
    * A lognormal deviation is always positive: with ``RELATIVE`` the cost
      never falls below ``raw``, and with ``RELATIVE_MEAN_PRESERVING`` the
      shift by ``E[eps] > 0`` cannot be read as a symmetric spread. Use
      ``DIRECT`` if the cost itself is lognormal.
    """

    ABSOLUTE = "absolute"
    """``raw + eps``; ``eps`` in the unit of the form value."""
    RELATIVE = "relative"
    """``raw * (1 + eps)``; the expected cost is ``raw * (1 + E[eps])``."""
    RELATIVE_MEAN_PRESERVING = "relative_mean_preserving"
    """``raw * (1 + eps - E[eps])``; the expected cost is ``raw``."""
    DIRECT = "direct"
    """``eps``; the distribution is the gross cost itself, ``raw`` is ignored."""


_REQUIRED: dict[DistributionKind, tuple[str, ...]] = {
    DistributionKind.DETERMINISTIC: ("mean",),
    DistributionKind.NORMAL: ("mean", "std_dev"),
    DistributionKind.LOGNORMAL: ("mean", "std_dev"),
    DistributionKind.UNIFORM: ("lower", "upper"),
    DistributionKind.TRIANGULAR: ("lower", "mode", "upper"),
    DistributionKind.LAPLACE: ("mean", "scale"),
}

_DISTRIBUTION_ALIASES = {"distribution": "kind", "min_val": "lower", "max_val": "upper"}
"""Keys of the Uni Kassel cost function module v3.0.0 and their names here."""


@dataclass(frozen=True)
class Distribution:
    """Univariate probability distribution.

    Required values per kind: deterministic ``mean``; normal and lognormal
    ``mean``, ``std_dev`` (of the logarithm for lognormal); uniform
    ``lower``, ``upper``; triangular ``lower``, ``mode``, ``upper``; Laplace
    ``mean``, ``scale``. Values not used by the kind are ignored. The unit of
    all values is the unit of the described quantity.

    :param kind: Distribution kind.
    :param mean: Mean (normal, Laplace, deterministic value) or mean of the
        logarithm (lognormal).
    :param std_dev: Standard deviation (>= 0; > 0 for lognormal, of the
        logarithm).
    :param lower: Lower limit (< ``upper``).
    :param upper: Upper limit.
    :param mode: Peak of a triangular distribution
        (``lower <= mode <= upper``).
    :param scale: Scale of a Laplace distribution (> 0).
    :raises ValueError: If a required value is missing, not finite or out of
        range.
    :raises TypeError: If ``kind`` is not a :class:`DistributionKind`.
    """

    kind: DistributionKind
    mean: float | None = None
    std_dev: float | None = None
    lower: float | None = None
    upper: float | None = None
    mode: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DistributionKind):
            raise TypeError(
                f"'kind' must be DistributionKind, got {type(self.kind).__name__}."
            )
        for name in ("mean", "std_dev", "lower", "upper", "mode", "scale"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"'{name}' must be finite, got {value}.")
            object.__setattr__(self, name, value)
        self._validate()

    def _values(self, *names: str) -> tuple[float, ...]:
        """Return required values; raise if one is missing."""
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"A {self.kind.value} distribution requires {list(names)}; "
                f"missing {missing}."
            )
        return tuple(getattr(self, name) for name in names)

    def _validate(self) -> None:
        values = self._values(*_REQUIRED[self.kind])
        kind = self.kind
        if kind is DistributionKind.NORMAL and not values[1] >= 0:
            raise ValueError(f"std_dev ({values[1]}) must be >= 0.")
        if kind is DistributionKind.LOGNORMAL and not values[1] > 0:
            raise ValueError(f"std_dev ({values[1]}) must be > 0 for lognormal.")
        if kind is DistributionKind.UNIFORM and not values[0] < values[1]:
            raise ValueError(f"lower ({values[0]}) must be < upper ({values[1]}).")
        if kind is DistributionKind.TRIANGULAR:
            lower, mode, upper = values
            if not lower < upper:
                raise ValueError(f"lower ({lower}) must be < upper ({upper}).")
            if not lower <= mode <= upper:
                raise ValueError(
                    f"mode ({mode}) must be within [lower, upper] = [{lower}, {upper}]."
                )
        if kind is DistributionKind.LAPLACE and not values[1] > 0:
            raise ValueError(f"scale ({values[1]}) must be > 0.")

    @property
    def is_symmetric(self) -> bool:
        """True if the distribution is symmetric about its mean.

        Uniform counts as symmetric only if centred at 0 (symmetric relative
        bounds); triangular if the mode is the midpoint of its limits.
        """
        if self.kind is DistributionKind.UNIFORM:
            lower, upper = self._values("lower", "upper")
            return abs(lower + upper) / 2.0 < _SYMMETRY_TOLERANCE
        if self.kind is DistributionKind.TRIANGULAR:
            lower, mode, upper = self._values("lower", "mode", "upper")
            return abs(mode - (lower + upper) / 2.0) < _SYMMETRY_TOLERANCE
        return True

    @property
    def mean_value(self) -> float:
        """Analytical mean in the unit of the described quantity."""
        if self.kind is DistributionKind.LOGNORMAL:
            mu, sigma = self._values("mean", "std_dev")
            return math.exp(mu + 0.5 * sigma**2)
        if self.kind is DistributionKind.UNIFORM:
            lower, upper = self._values("lower", "upper")
            return (lower + upper) / 2.0
        if self.kind is DistributionKind.TRIANGULAR:
            return sum(self._values("lower", "mode", "upper")) / 3.0
        (mean,) = self._values("mean")
        return mean

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw independent samples.

        :param n: Number of samples (>= 1).
        :param rng: Random number generator.
        :return: Array of shape ``(n,)``.
        :raises ValueError: If ``n < 1``.
        """
        if n < 1:
            raise ValueError(f"The number of samples must be >= 1, got {n}.")
        values = self._values(*_REQUIRED[self.kind])
        kind = self.kind
        if kind is DistributionKind.DETERMINISTIC:
            return np.full(n, values[0])
        if kind is DistributionKind.NORMAL:
            return rng.normal(values[0], values[1], n)
        if kind is DistributionKind.LOGNORMAL:
            return rng.lognormal(values[0], values[1], n)
        if kind is DistributionKind.UNIFORM:
            return rng.uniform(values[0], values[1], n)
        if kind is DistributionKind.TRIANGULAR:
            return rng.triangular(values[0], values[1], values[2], n)
        return rng.laplace(values[0], values[1], n)

    def quantiles(self, probabilities: Sequence[float]) -> dict[float, float]:
        """Return analytical quantiles.

        :param probabilities: Probability levels, each in (0, 1).
        :return: Quantile value by probability level, in the order given.
        :raises ValueError: If a level is outside (0, 1).
        """
        p = np.asarray(probabilities, dtype=float)
        if not np.all((p > 0) & (p < 1)):
            raise ValueError(
                f"Probabilities must be in (0, 1), got {list(probabilities)}."
            )
        values = self._values(*_REQUIRED[self.kind])
        kind = self.kind
        if kind is DistributionKind.DETERMINISTIC or (
            kind is DistributionKind.NORMAL and values[1] == 0
        ):
            q = np.full(p.shape, values[0])
        elif kind is DistributionKind.NORMAL:
            q = stats.norm.ppf(p, values[0], values[1])
        elif kind is DistributionKind.LOGNORMAL:
            q = stats.lognorm.ppf(p, values[1], scale=np.exp(values[0]))
        elif kind is DistributionKind.UNIFORM:
            q = stats.uniform.ppf(p, values[0], values[1] - values[0])
        elif kind is DistributionKind.TRIANGULAR:
            lower, mode, upper = values
            width = upper - lower
            q = stats.triang.ppf(p, (mode - lower) / width, loc=lower, scale=width)
        else:
            q = stats.laplace.ppf(p, values[0], values[1])
        return {float(k): float(v) for k, v in zip(probabilities, np.atleast_1d(q))}

    def scaled(self, factor: float) -> Distribution:
        """Return the distribution of ``factor * X``.

        Location and spread values are multiplied by ``factor``; for a
        lognormal distribution ``ln(factor)`` is added to the mean of the
        logarithm and its standard deviation is kept.

        :param factor: Dimensionless factor, > 0.
        :return: New distribution of the same kind.
        :raises ValueError: If ``factor`` is not a positive finite number.
        """
        if not (math.isfinite(factor) and factor > 0):
            raise ValueError(f"factor must be positive and finite, got {factor}.")
        if self.kind is DistributionKind.LOGNORMAL:
            (mu,) = self._values("mean")
            return dataclasses.replace(self, mean=mu + math.log(factor))
        changes = {
            name: getattr(self, name) * factor
            for name in ("mean", "std_dev", "lower", "upper", "mode", "scale")
            if getattr(self, name) is not None
        }
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        :return: ``{"kind": ..., <value>: ...}`` with the values that are set.
        """
        data: dict[str, Any] = {"kind": self.kind.value}
        for name in ("mean", "std_dev", "lower", "upper", "mode", "scale"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Distribution:
        """Build a distribution from a dictionary.

        Accepts the output of :meth:`to_dict` and the keys of the Uni Kassel
        cost function module v3.0.0 (``distribution``, ``min_val``,
        ``max_val``).

        :param data: Mapping with ``kind`` and the values of the kind.
        :return: The distribution.
        :raises ValueError: If the kind is unknown or values are invalid.
        """
        values = {_DISTRIBUTION_ALIASES.get(k, k): v for k, v in data.items()}
        kind = values.pop("kind", None)
        try:
            kind = DistributionKind(str(kind).strip().lower())
        except ValueError as error:
            raise ValueError(
                f"Unknown distribution {kind!r}; valid kinds: "
                f"{[k.value for k in DistributionKind]}."
            ) from error
        names = {f.name for f in dataclasses.fields(cls)} - {"kind"}
        return cls(kind, **{k: v for k, v in values.items() if k in names})


def distribution_from_bounds(
    lower: float,
    upper: float,
    distribution: str = "data",
    coverage: float = 0.90,
) -> Distribution | None:
    """Turn relative uncertainty bounds into a deviation distribution.

    :param lower: Lower relative bound (fraction, usually <= 0).
    :param upper: Upper relative bound (fraction, usually >= 0).
    :param distribution: ``"data"`` (uniform for symmetric or one-sided
        bounds, triangular with mode 0 for asymmetric two-sided bounds),
        ``"uniform"``, ``"triangular"`` (mode 0), ``"normal"`` or
        ``"laplace"``.
    :param coverage: For ``"normal"`` and ``"laplace"``: probability of the
        central interval ``[lower, upper]`` (0.90: the bounds are the 5 % and
        95 % quantiles); clipped to [0.01, 0.999].
    :return: Distribution of the deviation, or None if both bounds are zero.
    :raises ValueError: For an unknown ``distribution`` or invalid bounds.
    """
    lower, upper = float(lower), float(upper)
    if lower == 0.0 and upper == 0.0:
        return None
    if distribution == "data":
        asymmetric = abs(abs(lower) - upper) >= _SYMMETRY_TOLERANCE
        distribution = "triangular" if lower < 0.0 < upper and asymmetric else "uniform"
    if distribution == "uniform":
        return Distribution(DistributionKind.UNIFORM, lower=lower, upper=upper)
    if distribution == "triangular":
        return Distribution(
            DistributionKind.TRIANGULAR, lower=lower, mode=0.0, upper=upper
        )
    coverage = min(max(float(coverage), 0.01), 0.999)
    centre = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    if distribution == "normal":
        z = float(stats.norm.ppf(0.5 + 0.5 * coverage))
        return Distribution(
            DistributionKind.NORMAL, mean=centre, std_dev=half_width / z
        )
    if distribution == "laplace":
        # P(|X - mean| <= a) = 1 - exp(-a / scale)
        return Distribution(
            DistributionKind.LAPLACE,
            mean=centre,
            scale=half_width / -math.log(1.0 - coverage),
        )
    raise ValueError(
        f"Unknown distribution {distribution!r}; use 'data', 'uniform', "
        "'triangular', 'normal' or 'laplace'."
    )


@dataclass(frozen=True)
class UncertainCostFunction(CostFunction):
    """Cost function with parameter and/or output uncertainty.

    All fields of :class:`CostFunction` plus the fields below. The
    combinations are not checked for plausibility; see
    :class:`OutputUncertaintyMode` for the relative mode with an asymmetric
    distribution (the expected cost is shifted) and for lognormal deviations
    in the relative modes (always positive).

    :param parameter_uncertainty: Distribution of the value of form
        parameters, by name of :meth:`CostForm.parameters`. A deterministic
        distribution replaces the stored value.
    :param output_uncertainty: Distribution of the deviation applied to the
        deterministic gross cost, or None.
    :param output_mode: How the deviation is applied.
    :raises ValueError: If a key of ``parameter_uncertainty`` is not a
        parameter of ``form``.
    :raises TypeError: If a value has the wrong type.
    """

    parameter_uncertainty: Mapping[str, Distribution] = field(
        default_factory=dict, hash=False
    )
    output_uncertainty: Distribution | None = None
    output_mode: OutputUncertaintyMode = OutputUncertaintyMode.RELATIVE_MEAN_PRESERVING

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.parameter_uncertainty, Mapping):
            raise TypeError(
                "'parameter_uncertainty' must be a mapping, "
                f"got {type(self.parameter_uncertainty).__name__}."
            )
        unknown = sorted(set(self.parameter_uncertainty) - set(self.form.parameters()))
        if unknown:
            raise ValueError(
                f"{unknown} are not parameters of the {self.form.kind} form of "
                f"'{self.name}'; valid names: {list(self.form.parameters())}."
            )
        for name, value in self.parameter_uncertainty.items():
            if not isinstance(value, Distribution):
                raise TypeError(
                    f"The uncertainty of '{name}' must be a Distribution, "
                    f"got {type(value).__name__}."
                )
        object.__setattr__(
            self, "parameter_uncertainty", dict(self.parameter_uncertainty)
        )
        output = self.output_uncertainty
        if output is not None and not isinstance(output, Distribution):
            raise TypeError(
                "'output_uncertainty' must be a Distribution or None, "
                f"got {type(output).__name__}."
            )
        if not isinstance(self.output_mode, OutputUncertaintyMode):
            raise TypeError(
                "'output_mode' must be OutputUncertaintyMode, "
                f"got {type(self.output_mode).__name__}."
            )

    def scaled(self, factor: float) -> UncertainCostFunction:
        """Return a function whose money values are ``factor`` times as large.

        Besides ``form`` and ``fixed_om`` (see :meth:`CostFunction.scaled`)
        this scales the distributions of the cost parameters of the form
        (:meth:`CostForm.cost_parameter_names`; shape parameters such as an
        exponent stay) and the output distribution in the absolute and direct
        modes (relative deviations are dimensionless and stay).

        :param factor: Dimensionless factor, > 0.
        :return: New uncertain cost function; all its costs and cost
            quantiles are ``factor`` times those of this function.
        :raises ValueError: If ``factor`` is not a positive finite number.
        """
        scaled = super().scaled(factor)
        cost_parameters = set(self.form.cost_parameter_names())
        parameters = {
            name: d.scaled(factor) if name in cost_parameters else d
            for name, d in self.parameter_uncertainty.items()
        }
        output = self.output_uncertainty
        if output is not None and self.output_mode in (
            OutputUncertaintyMode.ABSOLUTE,
            OutputUncertaintyMode.DIRECT,
        ):
            output = output.scaled(factor)
        return dataclasses.replace(
            scaled, parameter_uncertainty=parameters, output_uncertainty=output
        )

    def with_form(self, form: CostForm) -> UncertainCostFunction:
        """Return a function with another investment cost form.

        Parameter distributions describe the values of the current form, so
        random ones block the replacement and deterministic ones are dropped.
        The output uncertainty is kept in every mode; absolute and direct
        deviations are not rescaled to the new form.

        :param form: New function form of the investment cost.
        :return: New uncertain cost function without parameter uncertainty.
        :raises ValueError: If a parameter has a non-deterministic
            distribution.
        """
        random = sorted(
            name
            for name, d in self.parameter_uncertainty.items()
            if d.kind is not DistributionKind.DETERMINISTIC
        )
        if random:
            raise ValueError(
                f"The distributions of {random} of '{self.name}' describe the "
                "values of the current form; remove or redefine them with "
                "dataclasses.replace(function, parameter_uncertainty={...}) "
                "before the form is replaced."
            )
        return dataclasses.replace(self, form=form, parameter_uncertainty={})

    @classmethod
    def from_bounds(
        cls,
        function: CostFunction,
        distribution: str = "data",
        mode: OutputUncertaintyMode = OutputUncertaintyMode.RELATIVE_MEAN_PRESERVING,
        coverage: float = 0.90,
    ) -> UncertainCostFunction | None:
        """Copy a function with output uncertainty from its relative bounds.

        :param function: Cost function with ``uncertainty_lower`` and
            ``uncertainty_upper``.
        :param distribution: See :func:`distribution_from_bounds`.
        :param mode: How the deviation is applied.
        :param coverage: See :func:`distribution_from_bounds`.
        :return: The uncertain cost function, or None if both bounds are zero.
        """
        output = distribution_from_bounds(
            function.uncertainty_lower,
            function.uncertainty_upper,
            distribution,
            coverage,
        )
        if output is None:
            return None
        values = {
            f.name: getattr(function, f.name) for f in dataclasses.fields(function)
        }
        values.update(output_uncertainty=output, output_mode=mode)
        return cls(**values)

    @property
    def supports_analytical_quantiles(self) -> bool:
        """True if output uncertainty is set and no parameter is random.

        Deterministic parameter distributions do not count as random.
        """
        return self.output_uncertainty is not None and all(
            d.kind is DistributionKind.DETERMINISTIC
            for d in self.parameter_uncertainty.values()
        )

    def _apply_output_mode(
        self, raw: Any, deviation: np.ndarray, output: Distribution
    ) -> Any:
        """Apply a deviation of ``output`` to the deterministic gross cost."""
        mode = self.output_mode
        if mode is OutputUncertaintyMode.ABSOLUTE:
            return raw + deviation
        if mode is OutputUncertaintyMode.RELATIVE:
            return raw * (1.0 + deviation)
        if mode is OutputUncertaintyMode.RELATIVE_MEAN_PRESERVING:
            return raw * (1.0 + deviation - output.mean_value)
        return deviation + np.zeros_like(raw)

    def _scalar_sizes(
        self, size: float, extrapolate: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Requested and effective size of a scalar ``size``."""
        if np.ndim(size) != 0:
            raise ValueError(
                f"Uncertain costs are calculated for one size at a time, got {size}."
            )
        return np.asarray(size, dtype=float), self._effective_size(size, extrapolate)

    def cost_quantiles(
        self,
        size: float,
        probabilities: Sequence[float],
        *,
        include_installation: bool = True,
        include_tax: bool = False,
        specific: bool = True,
        extrapolate: bool = False,
    ) -> dict[float, float]:
        """Calculate cost quantiles analytically from the output uncertainty.

        The quantiles of the deviation are transformed by the output mode;
        this assumes a non-negative deterministic gross cost for the relative
        modes.

        :param size: Size in ``size_unit`` (scalar, positive).
        :param probabilities: Probability levels, each in (0, 1).
        :param include_installation: As in :meth:`CostFunction.calculate_cost`.
        :param include_tax: As in :meth:`CostFunction.calculate_cost`.
        :param specific: As in :meth:`CostFunction.calculate_cost`.
        :param extrapolate: As in :meth:`CostFunction.calculate_cost`.
        :return: Cost in ``currency`` per ``size_unit`` (specific) or in
            ``currency`` by probability level, in the order given.
        :raises ValueError: If :attr:`supports_analytical_quantiles` is False,
            ``size`` is not a positive scalar or a level is outside (0, 1).
        """
        output = self.output_uncertainty
        if output is None or not self.supports_analytical_quantiles:
            raise ValueError(
                f"Analytical quantiles of '{self.name}' need output uncertainty "
                "and no random parameter; use sample_cost instead."
            )
        requested, effective = self._scalar_sizes(size, extrapolate)
        parameters = {
            name: d.mean_value for name, d in self.parameter_uncertainty.items()
        }
        raw = self._apply_installation(
            self.form.evaluate(effective, parameters), effective, include_installation
        )
        deviation = output.quantiles(probabilities)
        cost = self._apply_tax_and_basis(
            self._apply_output_mode(
                raw, np.fromiter(deviation.values(), dtype=float), output
            ),
            effective,
            requested,
            include_tax=include_tax,
            specific=specific,
        )
        return {p: float(c) for p, c in zip(deviation, cost)}

    def sample_cost(
        self,
        size: float,
        n_samples: int = 1000,
        *,
        rng: np.random.Generator | int | None = None,
        include_installation: bool = True,
        include_tax: bool = False,
        specific: bool = True,
        extrapolate: bool = False,
    ) -> np.ndarray:
        """Sample the cost by Monte Carlo simulation.

        Uncertain parameters are drawn first (in the order of
        :meth:`CostForm.parameters`), then the output deviation.

        :param size: Size in ``size_unit`` (scalar, positive).
        :param n_samples: Number of samples (>= 1).
        :param rng: Random number generator, seed, or None for a fresh
            generator.
        :param include_installation: As in :meth:`CostFunction.calculate_cost`.
        :param include_tax: As in :meth:`CostFunction.calculate_cost`.
        :param specific: As in :meth:`CostFunction.calculate_cost`.
        :param extrapolate: As in :meth:`CostFunction.calculate_cost`.
        :return: Cost samples in ``currency`` per ``size_unit`` (specific) or
            in ``currency``, shape ``(n_samples,)``.
        :raises ValueError: If ``size`` is not a positive scalar or
            ``n_samples < 1``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        requested, effective = self._scalar_sizes(size, extrapolate)
        sampled = {
            name: self.parameter_uncertainty[name].sample(n_samples, generator)
            for name in self.form.parameters()
            if name in self.parameter_uncertainty
        }
        base = np.broadcast_to(self.form.evaluate(effective, sampled), (n_samples,))
        raw = self._apply_installation(base, effective, include_installation)
        output = self.output_uncertainty
        if output is not None:
            raw = self._apply_output_mode(
                raw, output.sample(n_samples, generator), output
            )
        cost = self._apply_tax_and_basis(
            raw, effective, requested, include_tax=include_tax, specific=specific
        )
        return np.array(cost, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary with nested distribution dictionaries.

        :return: Dictionary accepted by :meth:`from_dict`.
        """
        data = super().to_dict()
        data["parameter_uncertainty"] = {
            name: d.to_dict() for name, d in self.parameter_uncertainty.items()
        }
        output = self.output_uncertainty
        data["output_uncertainty"] = None if output is None else output.to_dict()
        data["output_mode"] = self.output_mode.value
        return data

    @classmethod
    def _fields_from_dict(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve a dictionary to constructor arguments, incl. uncertainty.

        Reads ``output_uncertainty_mode`` of the reference module as
        ``output_mode``.
        """
        fields = super()._fields_from_dict(data)

        def distribution(value: Any) -> Distribution | None:
            if value is None or isinstance(value, Distribution):
                return value
            if isinstance(value, Mapping):
                return Distribution.from_dict(value)
            raise TypeError(
                f"Cannot read a distribution from {type(value).__name__!r}."
            )

        parameters = data.get("parameter_uncertainty") or {}
        if not isinstance(parameters, Mapping):
            raise TypeError(
                "'parameter_uncertainty' must be a mapping, "
                f"got {type(parameters).__name__}."
            )
        fields["parameter_uncertainty"] = {
            name: distribution(value) for name, value in parameters.items()
        }
        fields["output_uncertainty"] = distribution(data.get("output_uncertainty"))
        mode = data.get("output_mode", data.get("output_uncertainty_mode"))
        fields["output_mode"] = (
            OutputUncertaintyMode.RELATIVE_MEAN_PRESERVING
            if mode is None
            else OutputUncertaintyMode(mode)
        )
        return fields
