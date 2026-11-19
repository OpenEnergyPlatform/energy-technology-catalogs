# Cost functions

The sub-package `energy_technology_catalogs.costs` describes the investment
cost of energy technologies as functions of their size. It evaluates such
functions, moves them between price years, quantifies their uncertainty,
splits a total size into units, fits functions to data and reads and writes
them as CSV, Excel and JSON files.

The power-law model, the uncertainty model and the calibration are ported from
the Uni Kassel cost function module v3.0.0. The other function forms, the cost
basis and fitting from data are new.

## A cost function

A `CostFunction` belongs to one technology from one source for one investment
year. It holds a **function form** for the size dependence and the information
needed to interpret the value of that form.

```python
from energy_technology_catalogs.costs import (
    Constant, CostFunction, PowerLaw, TechnologyType,
)

boiler = CostFunction(
    technology="Pellet boiler",
    source="Example values",
    technology_type=TechnologyType.BIOMASS_BOILER,
    size_unit="kW",
    investment_year=2025,
    size_min=10.0,
    size_max=100.0,
    form=PowerLaw(offset=0.0, coefficient=1000.0, exponent=-0.5),
    installation_included=True,
    installation_share=Constant(0.4),
    tax_rate=0.19,
    uncertainty_lower=-0.2,
    uncertainty_upper=0.2,
)

boiler.calculate_cost(25.0)                              # 168.07 EUR/kW, net
boiler.calculate_cost(25.0, include_tax=True)            # 200.00 EUR/kW
boiler.calculate_cost(25.0, include_installation=False)  # 100.84 EUR/kW
boiler.calculate_cost(25.0, specific=False)              # 4201.68 EUR
```

`calculate_cost` accepts a single size or a NumPy array of sizes.

| Field | Meaning |
|---|---|
| `technology`, `source`, `investment_year` | Identity of the function (`CostFunctionKey`). |
| `technology_type` | Category from `TechnologyType`, with `label("en")`, `label("de")` and `oeo_iri`. |
| `size_unit`, `size_min`, `size_max` | Unit and valid range of the size. |
| `form` | Function form of the investment cost. |
| `cost_basis` | `CostBasis.SPECIFIC` (cost per size unit, default) or `CostBasis.ABSOLUTE` (cost of the whole unit). |
| `fixed_om` | Optional form of the fixed operation and maintenance cost per year. |
| `installation_included`, `installation_share` | Whether the form value contains planning and installation, and their share in the total cost (0 ≤ share < 1) as a form. |
| `tax_rate` | Value-added tax contained in the form value, as a fraction of the net cost. |
| `uncertainty_lower`, `uncertainty_upper` | Relative bounds as fractions of the net cost. |
| `lifetime_a` | Technical lifetime in years. |
| `currency`, `price_year` | Currency and price base of the cost parameters. |

### Function forms

| Class | Formula (x = size) | Parameters |
|---|---|---|
| `PowerLaw` | `offset + coefficient · x^exponent` | `offset`, `coefficient`, `exponent` |
| `Linear` | `offset + slope · x` | `offset`, `slope` |
| `Polynomial` | `c0 + c1 · x + c2 · x² + …` | `c0`, `c1`, … |
| `Logarithmic` | `offset + coefficient · ln(x)` | `offset`, `coefficient` |
| `Exponential` | `offset + coefficient · exp(rate · x)` | `offset`, `coefficient`, `rate` |
| `Constant` | `value` | `value` |
| `PiecewiseLinear` | linear interpolation between support points `(sizes, values)` | `v0`, `v1`, … |

Every form offers `evaluate(size)`, `parameters()`, `with_parameters(...)`,
`scaled(factor)`, `shifted(delta)` and `to_dict()`. The same forms describe the
fixed O&M cost and the installation share.

### Calculation order

1. A size outside `[size_min, size_max]` is replaced by the range limit and a
   `UserWarning` is issued. The specific cost of the limit then applies to the
   requested size. With `extrapolate=True` the function is evaluated at the
   requested size instead.
2. The form is evaluated. Its value is a gross cost on the cost basis of the
   function.
3. Planning and installation: if they are included and not wanted, the value
   is multiplied by `1 − share`; if they are not included and wanted, it is
   divided by `1 − share`.
4. Value-added tax is removed unless `include_tax=True`. Net costs are
   comparable between sources.
5. The result is converted to the requested basis
   (`specific = absolute / size`).

### Years

`investment_year` is the year in which the technology is installed. It is part
of the identity: projections for 2030 and 2040 are two functions.
`price_year` is the price base of the cost parameters. It defaults to the
investment year and only matters for price adjustment.

## Price adjustment

```python
from energy_technology_catalogs.costs import PriceIndex

index = PriceIndex({2026: 0.02, 2027: 0.03})   # change against the previous year
index.factor(2025, 2027)                        # 1.0506
later = index.adjust_function(boiler, 2027)     # price_year 2027, investment_year 2025
later.calculate_cost(25.0)                      # 176.57 EUR/kW
```

`adjust_function` scales the investment cost form and the fixed O&M form and
sets `price_year`. Years without a rate count as 0 % and are logged.
`PriceIndex.default()` provides the producer price index of industrial
products of the Federal Statistical Office of Germany with rates for 1950 to
2025; `PriceIndex.from_file` and `PriceIndex.from_index_values` read other
series.

## Uncertainty

`UncertainCostFunction` adds distributions to a cost function: for form
parameters (`parameter_uncertainty`) and for the cost itself
(`output_uncertainty` with one of the modes absolute, relative, relative
mean-preserving and direct). The cost uncertainty acts on the gross cost after
the installation step and before tax removal.

```python
from energy_technology_catalogs.costs import UncertainCostFunction

uncertain = UncertainCostFunction.from_bounds(boiler)   # uniform within ±20 %
uncertain.cost_quantiles(25.0, [0.05, 0.5, 0.95])
# {0.05: 137.82, 0.5: 168.07, 0.95: 198.32}
samples = uncertain.sample_cost(25.0, 10000, rng=1)     # NumPy array, seeded
```

`cost_quantiles` is analytical and needs a cost distribution without random
parameters; `sample_cost` covers every combination by Monte Carlo simulation.
`from_bounds` builds the cost distribution from `uncertainty_lower` and
`uncertainty_upper`: uniform for symmetric or one-sided bounds, triangular
with its mode at zero for two-sided asymmetric bounds, or a normal or Laplace
distribution that covers the bounds with a given probability. It returns
`None` if both bounds are zero.

## Unit sizing

`split_size` divides a total size into units and sums their absolute costs.

```python
from energy_technology_catalogs.costs import split_size

split = split_size(boiler, 250.0, "equal")
split.unit_sizes    # (83.33, 83.33, 83.33) kW
split.total_cost    # 23013.55 EUR
```

| Strategy | Units |
|---|---|
| `single_unit` | one unit of the total size, evaluated outside the valid range if necessary |
| `equal` | the fewest equal units not larger than `size_max` |
| `logarithmic` | log-spaced sizes between `size_min` and `size_max`, scaled to the total |
| `optimal` | the number of equal units with the lowest total cost |

## Fitting and calibration

`fit_cost_form` creates a form from data points. The costs are given on the
basis the function will have.

```python
from energy_technology_catalogs.costs import CostBasis, fit_cost_form

sizes = [10.0, 20.0, 50.0, 100.0, 200.0]              # m3
costs = [9000.0, 13500.0, 26000.0, 46500.0, 87000.0]  # EUR per unit
result = fit_cost_form(sizes, costs, "linear")
result.form        # Linear(offset=5275.29, slope=409.54)
result.r_squared   # 0.99992

tank = CostFunction(
    technology="Storage tank", source="Own offers",
    technology_type=TechnologyType.TTES, size_unit="m3", investment_year=2026,
    size_min=10.0, size_max=200.0,
    form=result.form, cost_basis=CostBasis.ABSOLUTE,
)
tank.calculate_cost(80.0, specific=False)   # 38038.1 EUR
tank.calculate_cost(80.0)                   # 475.48 EUR/m3
```

`kind` is one of `power_law`, `linear`, `polynomial` (with `degree`),
`logarithmic`, `exponential`, `constant` and `piecewise_linear` (with
`nodes`). Parameters can be fixed (`fixed={"offset": 0.0}`) and points
weighted. `PiecewiseLinear.from_points` builds a tabulated function without
fitting.

`calibrate_cost_function` adjusts an existing function to observed costs. It
keeps the shape parameters and either frees the parameters that are linear in
the cost or scales the form by one factor (`proportional=True`); weights
balance the fit to the data against the distance to the original function.
`calibrate_to_point` adjusts a function to a single observation.

Observed costs are compared with the value of the form, that is with the
gross cost on the cost basis of the function. Only the `proportional` method
of `calibrate_to_point` compares with `calculate_cost`, the net cost with or
without planning and installation. Calibration changes the investment cost
form only; the fixed O&M form is kept.

## Files and registry

`write_cost_functions` and `read_cost_functions` choose the format by the file
suffix: `.csv` (separator `;`), `.xlsx` and `.json`.

| Column | Content |
|---|---|
| `technology`, `source`, `technology_type`, `size_unit` | as above |
| `investment_year`, `price_year` | years |
| `size_min`, `size_max` | valid size range |
| `cost_basis` | `specific` or `absolute` |
| `function_type`, `function_parameters` | kind of the form and its parameters as JSON text |
| `fixed_om_function_type`, `fixed_om_parameters` | optional fixed O&M form |
| `installation_included`, `installation_share_function_type`, `installation_share_parameters` | planning and installation |
| `lifetime_a`, `tax_rate`, `uncertainty_lower`, `uncertainty_upper`, `currency` | as above |
| `parameter_uncertainty`, `output_uncertainty`, `output_uncertainty_mode` | optional distributions as JSON text |

JSON files hold the same content as a list of nested objects.

Tables in the layout of the Uni Kassel cost function module v3.0.0 are read as
well: `summand`, `coefficient` and `exponent` become a power law of the
specific cost, and its names `component` or `name`, `reference_unit`,
`reference_year`, `validity_range_lower`, `validity_range_upper`,
`pi_included` and `pi_factor` are mapped to the names above.

`CostFunctionRegistry` collects functions by key and offers `add`, `extend`,
`replace`, `remove`, `get`, `filter`, `load` and `save`.

## Data and licenses

| File | Source | License |
|---|---|---|
| `energy_technology_catalogs/data/destatis_epi_1949-2025.csv` | [Statistisches Bundesamt (Destatis)](https://www.destatis.de), Erzeugerpreisindex gewerblicher Produkte, 2025 | [Data licence Germany – attribution – version 2.0](https://www.govdata.de/dl-de/by-2-0) |
| `energy_technology_catalogs/data/technology_types.csv` | this project | MIT |
