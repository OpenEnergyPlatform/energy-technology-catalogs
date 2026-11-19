"""Fitting of function forms to data and calibration of cost functions.

Three tasks share one idea: the output of every form is linear in its cost
parameters (:meth:`CostForm.cost_parameter_names`). For a fixed set of shape
parameters (e.g. the exponent of a power law) the form value at the sizes
``x`` is therefore ``A @ p`` with the design matrix ``A`` whose column ``j``
is the form evaluated with cost parameter ``j`` set to one and all other cost
parameters set to zero. This is exact, not a linearisation.

- :func:`fit_cost_form` creates a form from ``(size, cost)`` data. Forms
  without free shape parameters are solved by linear least squares on the
  design matrix; forms with free shape parameters (power law, exponential)
  by the trust-region reflective method of
  :func:`scipy.optimize.least_squares`.
- :func:`calibrate_cost_function` adjusts the cost parameters of an existing
  cost function to data while its shape parameters stay fixed. The residual
  vector consists of four weighted blocks: data residuals, the deviation from
  the original curve on an evaluation grid (functional distance), and the
  absolute and relative deviation of the parameters from their original
  values. The weights are normalised to a sum of one. In proportional mode a
  single factor scales all cost parameters.
- :func:`calibrate_to_point` makes a cost function pass through one point by
  scaling, shifting, or rescaling the factor of its size term.

Ported from the Uni Kassel cost function module v3.0.0.

SPDX-FileCopyrightText: 2026 energy-technology-catalogs contributors <https://github.com/OpenEnergyPlatform/energy-technology-catalogs>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from scipy.optimize import least_squares

from energy_technology_catalogs.costs.core import CostBasis, CostFunction
from energy_technology_catalogs.costs.forms import (
    FORM_KINDS,
    CostForm,
    PiecewiseLinear,
    Polynomial,
    form_from_parameters,
)


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=CostFunction)

LOSS_FUNCTIONS = frozenset({"linear", "soft_l1", "huber", "cauchy", "arctan"})
"""Loss functions accepted by :func:`scipy.optimize.least_squares`."""

POINT_METHODS = frozenset({"proportional", "vertical_shift", "coefficient"})
"""Methods of :func:`calibrate_to_point`."""

_ZERO_THRESHOLD = 1e-12
"""Absolute value below which a cost or size term counts as zero."""

_SHAPE_CANDIDATES: dict[str, Callable[[np.ndarray], tuple[float, ...]]] = {
    "exponent": lambda size: (-1.0, -0.5, 0.5, 1.0),
    "rate": lambda size: tuple(c / float(size.max()) for c in (-2.0, -1.0, 1.0, 2.0)),
}
"""Candidate start values of shape parameters, as a function of the sizes."""

_LOG_LINEAR_FEATURE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "exponent": np.log,
    "rate": lambda size: size,
}
"""``g(size)`` with ``ln(cost) = ln(coefficient) + shape * g(size)`` at offset 0."""


@dataclass(frozen=True, eq=False)
class FitResult:
    """Result of :func:`fit_cost_form`.

    :param form: Fitted form.
    :param residuals: Unweighted residuals ``cost - form(size)`` per data
        point, in the unit of the costs.
    :param rmse: Root mean square of ``residuals``, in the unit of the costs.
    :param r_squared: Coefficient of determination of the unweighted
        residuals; NaN if all costs are equal.
    :param n_points: Number of data points.
    :param success: Whether the solver reported convergence (always True for
        a direct linear solution).
    :param message: Solver message.
    """

    form: CostForm
    residuals: np.ndarray
    rmse: float
    r_squared: float
    n_points: int
    success: bool
    message: str


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _columns(
    form: CostForm, names: Sequence[str], size: np.ndarray
) -> list[np.ndarray]:
    """Design matrix columns of the cost parameters ``names`` at ``size``.

    Column ``j`` is the form with all cost parameters zero except
    ``names[j]``, which is one; the shape parameters keep their values.
    """
    zero = dict.fromkeys(form.cost_parameter_names(), 0.0)
    return [
        np.asarray(
            form.with_parameters(**{**zero, name: 1.0}).evaluate(size), dtype=float
        )
        * np.ones_like(size)
        for name in names
    ]


def _combine(columns: Sequence[np.ndarray], values: Sequence[float]) -> np.ndarray:
    """``sum_j values[j] * columns[j]``, summed in parameter order."""
    total = values[0] * columns[0]
    for value, column in zip(values[1:], columns[1:]):
        total = total + value * column
    return total


def _check_loss(loss: str) -> None:
    if loss not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss '{loss}'. Valid options: {sorted(LOSS_FUNCTIONS)}"
        )


def _run_optimizer(
    residuals: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    loss: str,
    f_scale: float,
    max_nfev: int | None,
    name: str,
) -> Any:
    """Run the trust-region reflective solver and handle its status.

    :raises RuntimeError: If the solver raises or reports a failure.
    """
    try:
        result = least_squares(
            residuals,
            x0=x0,
            loss=loss,
            f_scale=f_scale,
            bounds=([-np.inf] * len(x0), [np.inf] * len(x0)),
            method="trf",
            max_nfev=max_nfev,
            verbose=0,
        )
    except Exception as error:
        raise RuntimeError(
            f"Optimisation raised an exception for '{name}': {error}"
        ) from error
    if result.status < 0:
        raise RuntimeError(
            f"Optimisation failed for '{name}' (status={result.status}): {result.message}"
        )
    if result.status == 0:
        warnings.warn(
            f"Optimisation reached the maximum number of function evaluations "
            f"({result.nfev}) without convergence for '{name}'. "
            f"Final cost: {result.cost:.4g}",
            UserWarning,
            stacklevel=3,
        )
    return result


# ----------------------------------------------------------------------
# Fitting a form to data
# ----------------------------------------------------------------------


def _template(kind: str, degree: int | None, nodes: Sequence[float] | None) -> CostForm:
    """Form of ``kind`` with all parameters zero."""
    if kind not in FORM_KINDS:
        raise ValueError(
            f"Unknown function form {kind!r}; valid kinds: {sorted(FORM_KINDS)}."
        )
    if kind == Polynomial.kind:
        if degree is None:
            raise ValueError("A polynomial fit needs 'degree'.")
        if isinstance(degree, bool) or int(degree) != degree or degree < 0:
            raise ValueError(
                f"'degree' must be a non-negative integer, got {degree!r}."
            )
        names = [f"c{i}" for i in range(int(degree) + 1)]
    elif degree is not None:
        raise ValueError(f"'degree' applies to polynomial forms only, not to {kind}.")
    if kind == PiecewiseLinear.kind:
        if nodes is None:
            raise ValueError("A piecewise linear fit needs 'nodes' (support sizes).")
        names = [f"v{i}" for i in range(len(nodes))]
    elif nodes is not None:
        raise ValueError(
            f"'nodes' apply to piecewise linear forms only, not to {kind}."
        )
    elif kind != Polynomial.kind:
        names = [field.name for field in dataclasses.fields(FORM_KINDS[kind])]  # type: ignore[arg-type]
    return form_from_parameters(kind, dict.fromkeys(names, 0.0), sizes=nodes)


def _solve_cost_parameters(
    form: CostForm,
    names: Sequence[str],
    size: np.ndarray,
    cost: np.ndarray,
    sqrt_w: np.ndarray,
) -> tuple[dict[str, float], int]:
    """Weighted linear least squares for the cost parameters ``names``.

    The contribution of the other cost parameters of ``form`` is subtracted
    from ``cost`` first.

    :return: Parameter values by name and the rank of the design matrix.
    """
    if not names:
        return {}, 0
    rest = np.asarray(form.with_parameters(**dict.fromkeys(names, 0.0)).evaluate(size))
    design = np.vstack(_columns(form, names, size)).T
    solution, _, rank, _ = np.linalg.lstsq(
        design * sqrt_w[:, None], (cost - rest) * sqrt_w, rcond=None
    )
    return dict(zip(names, (float(v) for v in solution))), int(rank)


def _shape_candidates(
    name: str, size: np.ndarray, cost: np.ndarray, sqrt_w: np.ndarray
) -> list[float]:
    """Candidate start values of a shape parameter.

    The fixed candidates of :data:`_SHAPE_CANDIDATES`, preceded by the slope
    of a log-linear fit if all costs are positive.
    """
    candidates = list(_SHAPE_CANDIDATES.get(name, lambda s: (0.0,))(size))
    feature = _LOG_LINEAR_FEATURE.get(name)
    if feature is not None and np.all(cost > 0):
        design = np.column_stack([np.ones_like(size), feature(size)])
        solution = np.linalg.lstsq(
            design * sqrt_w[:, None], np.log(cost) * sqrt_w, rcond=None
        )[0]
        candidates.insert(0, float(solution[1]))
    return candidates


def _best_start(
    template: CostForm,
    free_cost: Sequence[str],
    shape_candidates: Mapping[str, Sequence[float]],
    size: np.ndarray,
    cost: np.ndarray,
    sqrt_w: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Shape start with the lowest weighted residual sum of squares.

    For every combination of shape candidates the free cost parameters are
    solved by linear least squares.

    :return: Shape values, cost parameter values, rank of the design matrix.
    :raises ValueError: If no candidate gives a finite residual.
    """
    best: tuple[float, dict[str, float], dict[str, float], int] | None = None
    names = list(shape_candidates)
    for combination in itertools.product(*shape_candidates.values()):
        shape = dict(zip(names, combination))
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                trial = template.with_parameters(**shape)
                linear, rank = _solve_cost_parameters(
                    trial, free_cost, size, cost, sqrt_w
                )
                predicted = trial.with_parameters(**linear).evaluate(size)
        except (ValueError, np.linalg.LinAlgError):
            continue
        sse = float(np.sum((sqrt_w * (cost - predicted)) ** 2))
        if not math.isfinite(sse):
            sse = math.inf
        if best is None or sse < best[0]:
            best = (sse, shape, linear, rank)
    if best is None:
        raise ValueError("No start value with a finite residual was found.")
    return best[1], best[2], best[3]


def fit_cost_form(
    sizes: Sequence[float] | np.ndarray,
    costs: Sequence[float] | np.ndarray,
    kind: str = "power_law",
    *,
    degree: int | None = None,
    nodes: Sequence[float] | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    fixed: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> FitResult:
    """Fit a function form to ``(size, cost)`` data.

    The residuals ``cost - form(size)`` are weighted by ``sqrt(weights)``.
    Forms without free shape parameters (linear, polynomial, logarithmic,
    constant, piecewise linear, or a power law with fixed exponent) are solved
    by linear least squares. Power law and exponential forms are solved by
    :func:`scipy.optimize.least_squares` (trust-region reflective). For the
    start value, the exponent candidates -1, -0.5, 0.5, 1 or the rate
    candidates ±1/max(size), ±2/max(size), plus the slope of a fit of
    ``ln(cost)`` if all costs are positive, are each combined with the linear
    least-squares solution of the cost parameters; the solver starts from
    the candidate with the lowest residual.
    With a loss other than ``"linear"`` every form is solved iteratively,
    starting from the linear solution.

    :param sizes: Component sizes, positive, in the size unit of the cost
        function the form is meant for.
    :param costs: Cost values in the basis the form is meant to return
        (specific, e.g. EUR/kW, or absolute, e.g. EUR), with tax and
        installation as the cost function will declare them.
    :param kind: Form kind, a key of
        :data:`~energy_technology_catalogs.costs.forms.FORM_KINDS`.
    :param degree: Degree of a polynomial; required for ``"polynomial"`` only.
    :param nodes: Support sizes of a piecewise linear form, strictly
        increasing; required for ``"piecewise_linear"`` only. Their values
        are fitted.
    :param weights: Non-negative weight per data point; default 1.
    :param fixed: Parameter values that are kept, by parameter name.
    :param initial: Start values by parameter name for the iterative solver;
        ignored by a direct linear solution.
    :param loss: Loss function of :func:`scipy.optimize.least_squares`
        (``"linear"``, ``"soft_l1"``, ``"huber"``, ``"cauchy"``,
        ``"arctan"``).
    :param f_scale: Residual scale of a robust loss, in the unit of the costs.
    :return: Fitted form with residual statistics.
    :raises ValueError: For an unknown kind or loss, missing or misplaced
        ``degree``/``nodes``, non-positive or non-finite sizes, non-finite
        costs, invalid weights, unknown names in ``fixed`` or ``initial``,
        fewer points or distinct sizes than free parameters, or data that do
        not determine all free parameters of a linear problem.
    :raises RuntimeError: If the iterative solver raises or reports a
        failure (status < 0).
    """
    size = np.asarray(sizes, dtype=float)
    cost = np.asarray(costs, dtype=float)
    if size.ndim != 1 or cost.ndim != 1:
        raise ValueError("'sizes' and 'costs' must be one-dimensional.")
    if size.shape != cost.shape:
        raise ValueError(
            f"Got {size.size} sizes but {cost.size} costs; the lengths must match."
        )
    if not np.all(np.isfinite(size) & (size > 0)):
        raise ValueError("All sizes must be positive and finite.")
    if not np.all(np.isfinite(cost)):
        raise ValueError("All costs must be finite.")
    _check_loss(loss)
    if weights is None:
        sqrt_w = np.ones_like(size)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != size.shape:
            raise ValueError(f"Got {w.size} weights for {size.size} data points.")
        if not np.all(np.isfinite(w) & (w >= 0)):
            raise ValueError("All weights must be non-negative and finite.")
        sqrt_w = np.sqrt(w)

    template = _template(kind, degree, nodes)
    names = list(template.parameters())
    fixed = dict(fixed or {})
    initial = dict(initial or {})
    for label, given in (("fixed", fixed), ("initial", initial)):
        unknown = sorted(set(given) - set(names))
        if unknown:
            raise ValueError(
                f"{unknown} in '{label}' are not parameters of a {kind} form; "
                f"valid names: {names}."
            )
    both = sorted(set(fixed) & set(initial))
    if both:
        raise ValueError(f"{both} are given as fixed and as initial values.")
    template = template.with_parameters(**fixed)

    free = [name for name in names if name not in fixed]
    if size.size < len(free):
        raise ValueError(
            f"{size.size} data points cannot determine {len(free)} free parameters {free}."
        )
    if np.unique(size).size < len(free):
        raise ValueError(
            f"{np.unique(size).size} distinct sizes cannot determine "
            f"{len(free)} free parameters {free}."
        )
    cost_names = set(template.cost_parameter_names())
    free_cost = [name for name in free if name in cost_names]
    free_shape = [name for name in free if name not in cost_names]

    candidates = {
        name: [initial[name]]
        if name in initial
        else _shape_candidates(name, size, cost, sqrt_w)
        for name in free_shape
    }
    shape, linear, rank = _best_start(
        template, free_cost, candidates, size, cost, sqrt_w
    )
    template = template.with_parameters(**shape)
    if not free_shape and rank < len(free_cost):
        raise ValueError(
            f"The free parameters {free_cost} of the {kind} form are not determined "
            f"by the data (rank {rank} of {len(free_cost)}); check the nodes or sizes."
        )
    start = {**shape, **linear, **{k: v for k, v in initial.items() if k in linear}}

    if free_shape or (loss != "linear" and free):

        def residuals(p: np.ndarray) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = template.evaluate(size, dict(zip(free, p)))
            return sqrt_w * (cost - predicted)

        result = _run_optimizer(
            residuals,
            np.array([start[name] for name in free], dtype=float),
            loss,
            f_scale,
            None,
            f"{kind} fit",
        )
        values = dict(zip(free, (float(v) for v in result.x)))
        success, message = bool(result.success), str(result.message)
    else:
        values = linear
        success, message = True, "Direct linear least-squares solution."

    form = form_from_parameters(kind, {**template.parameters(), **values}, sizes=nodes)
    fitted_residuals = cost - np.asarray(form.evaluate(size), dtype=float)
    ss_res = float(np.sum(fitted_residuals**2))
    ss_tot = float(np.sum((cost - cost.mean()) ** 2))
    logger.debug("fit_cost_form: %s with %s, %s", kind, form.parameters(), message)
    return FitResult(
        form=form,
        residuals=fitted_residuals,
        rmse=math.sqrt(ss_res / size.size),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        n_points=int(size.size),
        success=success,
        message=message,
    )


# ----------------------------------------------------------------------
# Calibration of a cost function to data
# ----------------------------------------------------------------------


def _clip_data(
    data_points: Sequence[tuple[float, float]],
    lower: float,
    upper: float,
    absolute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip the sizes to the valid range and warn if any changed.

    A clamped cost function applies the cost per size unit at the range limit
    to the requested size (see :meth:`CostFunction.calculate_cost`). For an
    absolute basis the cost of a clipped point is therefore converted to the
    limit size, ``cost * size_clipped / size``; a specific cost stays.
    """
    size_raw = np.array([s for s, _ in data_points], dtype=float)
    cost = np.array([c for _, c in data_points], dtype=float)
    size = np.clip(size_raw, lower if lower > 0 else 1e-9, upper)
    if absolute:
        cost = np.where(size != size_raw, cost * size / size_raw, cost)
    if not np.allclose(size_raw, size):
        indices = np.where(~np.isclose(size_raw, size))[0].tolist()
        warnings.warn(
            f"data_points at indices {indices} have sizes outside the valid range "
            f"[{lower}, {upper}] and were clipped.",
            UserWarning,
            stacklevel=3,
        )
    return size, cost


def calibrate_cost_function(
    original: F,
    data_points: Sequence[tuple[float, float]],
    *,
    weight_data: float = 1.0,
    weight_functional: float = 0.0,
    weight_absolute: float = 0.0,
    weight_relative: float = 0.0,
    n_eval_points: int = 100,
    epsilon: float = 1e-6,
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int | None = None,
    proportional: bool = False,
) -> F:
    """Adjust the cost parameters of a cost function to data.

    The data points are compared with the raw value of ``original.form``: in
    the basis of the form (specific or absolute according to ``cost_basis``),
    in ``currency`` at prices of ``price_year``, including tax at ``tax_rate``
    and with planning and installation as ``installation_included`` says.
    Sizes outside ``[size_min, size_max]`` are clipped to the range limit
    with a warning, as the cost function clamps them; for an absolute basis
    the cost of such a point is converted to the limit size
    (``cost * size_limit / size``), so the point is read as the clamped cost
    function reads it.

    Standard mode frees the cost parameters
    (:meth:`~energy_technology_catalogs.costs.forms.CostForm.cost_parameter_names`,
    e.g. offset and coefficient of a power law); shape parameters such as the
    exponent stay. The residual vector holds, each multiplied by the square
    root of its normalised weight:

    - data: ``cost_i - form(size_i)``;
    - functional: ``form_original(s) - form(s)`` on ``n_eval_points`` sizes
      evenly spaced over ``[max(size_min, 1e-9), size_max]``;
    - absolute: ``(p - p0) / max(1, |p0|)`` per cost parameter;
    - relative: ``(p - p0) / max(|p0|, epsilon)`` per cost parameter.

    The start value is the unweighted linear least-squares fit to the data.
    Proportional mode has one free factor ``k`` and returns
    ``form.scaled(k)``; there the absolute block penalises ``k - 1`` and the
    relative block ``(k - 1) / max(epsilon, mean |p0|)``.

    :param original: Cost function to calibrate.
    :param data_points: ``(size, cost)`` pairs; size in ``size_unit``,
        positive; cost as described above.
    :param weight_data: Weight of the data residuals.
    :param weight_functional: Weight of the deviation from the original curve.
    :param weight_absolute: Weight of the absolute parameter deviation.
    :param weight_relative: Weight of the relative parameter deviation.
    :param n_eval_points: Number of sizes of the functional distance.
    :param epsilon: Floor of the denominators of the relative deviation.
    :param loss: Loss function of :func:`scipy.optimize.least_squares`.
    :param f_scale: Residual scale of a robust loss.
    :param max_nfev: Maximum number of residual evaluations.
    :param proportional: Scale all cost parameters by one factor.
    :return: ``original.with_form(<calibrated form>)``: same class, all other
        fields kept. Only the investment cost is calibrated; ``fixed_om`` is
        not scaled, also not in proportional mode.
    :raises ValueError: On empty data, a non-positive or non-finite size, a
        non-finite cost, a negative or non-finite weight or a zero weight sum,
        an unknown loss, (proportional mode) an original cost of zero at a
        data point, or if ``original`` refuses a form replacement in
        :meth:`CostFunction.with_form` (e.g. an uncertain cost function with
        random parameter distributions); this is checked before the
        optimisation.
    :raises RuntimeError: If the solver fails.
    """
    weights = (weight_data, weight_functional, weight_absolute, weight_relative)
    if not data_points:
        raise ValueError("data_points must contain at least one point.")
    _check_loss(loss)
    for i, (size, cost_value) in enumerate(data_points):
        if not (math.isfinite(size) and size > 0):
            raise ValueError(
                f"data_points[{i}]: size must be positive and finite, got {size}."
            )
        if not math.isfinite(cost_value):
            raise ValueError(
                f"data_points[{i}]: cost must be finite, got {cost_value}."
            )
    if any(not (math.isfinite(w) and w >= 0) for w in weights):
        raise ValueError(f"All weights must be non-negative and finite, got {weights}.")
    original.with_form(original.form)
    total = sum(weights)
    if total <= 0:
        raise ValueError(
            "At least one weight must be positive (sum of all weights > 0)."
        )
    sqrt_w_data, sqrt_w_func, sqrt_w_abs, sqrt_w_rel = (
        np.sqrt(w / total) for w in weights
    )

    form = original.form
    names = form.cost_parameter_names()
    p0_values = form.parameters()
    p0 = np.array([p0_values[name] for name in names], dtype=float)
    lower, upper = original.size_min, original.size_max
    size, cost = _clip_data(
        data_points, lower, upper, original.cost_basis is CostBasis.ABSOLUTE
    )
    size_eval = np.linspace(max(lower, 1e-9), upper, n_eval_points)

    if proportional:
        cost_orig = np.asarray(form.evaluate(size), dtype=float)
        if np.any(np.abs(cost_orig) < _ZERO_THRESHOLD):
            raise ValueError(
                "The original cost is effectively zero at one or more data point sizes; "
                "proportional scaling is undefined. Use proportional=False instead."
            )
        cost_orig_eval = np.asarray(form.evaluate(size_eval), dtype=float)
        magnitudes = [abs(p0_values[name]) for name in names]
        denom_rel = max(epsilon, sum(magnitudes) / len(magnitudes))
        scale_init = float(np.dot(cost, cost_orig) / np.dot(cost_orig, cost_orig))

        def proportional_residuals(params: np.ndarray) -> np.ndarray:
            k = params[0]
            return np.concatenate(
                [
                    sqrt_w_data * (cost - k * cost_orig),
                    sqrt_w_func * (1.0 - k) * cost_orig_eval,
                    np.atleast_1d(sqrt_w_abs * (k - 1.0)),
                    np.atleast_1d(sqrt_w_rel * (k - 1.0) / denom_rel),
                ]
            )

        result = _run_optimizer(
            proportional_residuals,
            np.array([scale_init], dtype=float),
            loss,
            f_scale,
            max_nfev,
            original.name,
        )
        factor = float(result.x[0])
        logger.debug(
            "calibrate (proportional) '%s': factor %.6g, nfev=%d, cost=%.4g",
            original.name,
            factor,
            result.nfev,
            result.cost,
        )
        # the investment cost only: fixed O&M is not calibrated
        return original.with_form(form.scaled(factor))

    columns = _columns(form, names, size)
    columns_eval = _columns(form, names, size_eval)
    cost_orig_eval = np.asarray(form.evaluate(size_eval), dtype=float)
    scale_abs_vec = np.maximum(1.0, np.abs(p0))
    denom_rel_vec = np.maximum(np.abs(p0), epsilon)
    try:
        start = np.linalg.lstsq(np.vstack(columns).T, cost, rcond=None)[0]
    except np.linalg.LinAlgError:
        start = p0

    def standard_residuals(params: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                sqrt_w_data * (cost - _combine(columns, params)),
                sqrt_w_func * (cost_orig_eval - _combine(columns_eval, params)),
                sqrt_w_abs * (params - p0) / scale_abs_vec,
                sqrt_w_rel * (params - p0) / denom_rel_vec,
            ]
        )

    result = _run_optimizer(
        standard_residuals,
        np.array(start, dtype=float),
        loss,
        f_scale,
        max_nfev,
        original.name,
    )
    new_form = form.with_parameters(
        **{name: float(v) for name, v in zip(names, result.x)}
    )
    logger.debug(
        "calibrate '%s': %s -> %s, nfev=%d, cost=%.4g",
        original.name,
        dict(zip(names, p0)),
        new_form.parameters(),
        result.nfev,
        result.cost,
    )
    return original.with_form(new_form)


# ----------------------------------------------------------------------
# Calibration to one point
# ----------------------------------------------------------------------


def calibrate_to_point(
    original: F,
    data_point: tuple[float, float],
    method: str = "proportional",
    *,
    include_installation: bool = True,
) -> F:
    """Make a cost function pass through one ``(size, cost)`` point.

    - ``"proportional"``: all cost parameters are scaled by
      ``cost / calculate_cost(size)``. The observed cost is compared with
      :meth:`CostFunction.calculate_cost` at the size, extrapolated if
      needed: net of tax, in the basis of the form (specific or absolute
      according to ``cost_basis``), with planning and installation if
      ``include_installation``.
    - ``"vertical_shift"``: the form is shifted by
      ``cost - form(size)``; the raw form value is compared (gross as stored,
      installation as stored, basis of the form).
    - ``"coefficient"``: the factor of the size term (``coefficient`` or
      ``slope``) is set so that the raw form value equals ``cost``; the other
      parameters stay. Only for forms with exactly one such factor (power
      law, linear, logarithmic, exponential), and, as in the reference
      module, only where that factor comes out positive. The method is
      therefore unavailable for falling linear or logarithmic forms (negative
      slope or coefficient) and wherever the observed cost does not exceed
      the size-independent part of the form (for a logarithmic form at sizes
      below 1: does not fall below it).

    All methods change only the investment cost form (through
    :meth:`CostFunction.with_form`); ``fixed_om`` and all other fields are
    kept. A size outside ``[size_min, size_max]`` gives a warning.

    :param original: Cost function to calibrate.
    :param data_point: ``(size, cost)``; size in ``size_unit``, positive; cost
        in ``currency`` as described per method.
    :param method: ``"proportional"`` (default), ``"vertical_shift"`` or
        ``"coefficient"``.
    :param include_installation: Whether the observed cost includes planning
        and installation; used by ``"proportional"`` only.
    :return: ``original.with_form(<calibrated form>)``: same class, all other
        fields (including ``fixed_om``) kept.
    :raises ValueError: For an unknown method, a non-positive or non-finite
        size or a non-finite cost; if ``original`` refuses a form replacement
        in :meth:`CostFunction.with_form` (e.g. an uncertain cost function
        with random parameter distributions); for ``"proportional"`` if the
        original cost at the size is zero; for ``"coefficient"`` if the form
        has no single size-term factor, the size term is zero, or the new
        factor would not be positive.
    """
    if method not in POINT_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Valid options: {sorted(POINT_METHODS)}"
        )
    size, observed = data_point
    if not (math.isfinite(size) and size > 0):
        raise ValueError(f"data_point size must be positive and finite, got {size}.")
    if not math.isfinite(observed):
        raise ValueError(f"data_point cost must be finite, got {observed}.")
    original.with_form(original.form)
    if size < original.size_min or size > original.size_max:
        warnings.warn(
            f"data_point size {size} is outside the valid range "
            f"[{original.size_min}, {original.size_max}]; the result is an extrapolation.",
            UserWarning,
            stacklevel=2,
        )
    form = original.form

    if method == "proportional":
        current = original.calculate_cost(
            size,
            include_installation=include_installation,
            specific=original.cost_basis is CostBasis.SPECIFIC,
            extrapolate=True,
        )
        if abs(current) < _ZERO_THRESHOLD:
            raise ValueError(
                "The original cost at the size is effectively zero; proportional "
                "scaling is undefined. Use 'vertical_shift' or 'coefficient' instead."
            )
        new_form = form.scaled(observed / current)
    elif method == "vertical_shift":
        new_form = form.shifted(observed - form.evaluate(size))
    else:
        offsets = set(form.offset_parameter_names())
        factors = [name for name in form.cost_parameter_names() if name not in offsets]
        if len(factors) != 1:
            raise ValueError(
                f"The 'coefficient' method needs a form with exactly one factor of a "
                f"size term; a {form.kind} form has {len(factors)}."
            )
        name = factors[0]
        zero = dict.fromkeys(form.cost_parameter_names(), 0.0)
        size_independent = form.with_parameters(**{name: 0.0}).evaluate(size)
        size_term = form.with_parameters(**{**zero, name: 1.0}).evaluate(size)
        if abs(size_term) < _ZERO_THRESHOLD:
            raise ValueError(
                f"The size term of the {form.kind} form is zero at size {size}; "
                f"cannot solve for '{name}'."
            )
        value = (observed - size_independent) / size_term
        if not value > 0:
            raise ValueError(
                f"The 'coefficient' method needs a positive '{name}': the observed cost "
                f"({observed}) must exceed the size-independent part of the form "
                f"({size_independent:g}) at size {size}."
            )
        new_form = form.with_parameters(**{name: value})

    logger.debug(
        "calibrate_to_point (%s) '%s': %s -> %s",
        method,
        original.name,
        form.parameters(),
        new_form.parameters(),
    )
    return original.with_form(new_form)
