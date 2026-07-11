# SIAGA Model Card

**Version:** v1
**Task:** predict a country's life expectancy at birth (years) from health-system
and contextual indicators.
**Model:** monotonicity-constrained histogram gradient-boosting regressor.
**Training rows:** 98 country-years (10 ASEAN states, 2004-2014).

## Intended use

Decision support for public-health resource allocation: ranking drivers of life
expectancy and simulating the effect of policy changes. Not a clinical or
individual-level tool. Operates on aggregate national indicators only.

## Performance (held-out)

| Test | R2 | RMSE (yrs) | MAE (yrs) |
|------|----|-----------|-----------|
| Unseen country (leave-one-country-out) | 0.686 | 3.09 | 2.58 |
| Future of known countries (2012-2014) | 0.923 | 1.46 | 1.12 |
| Out-of-source vs World Bank (2015-2019) | 0.693 | 2.73 | 2.29 |

## Features

- `physicians_per_1000` (actionable, monotonic +)
- `nurses_midwives_per_1000` (actionable, monotonic +)
- `pharma_workers_per_1000` (actionable, monotonic +)
- `immunization_dpt` (actionable, monotonic +)
- `immunization_measles` (actionable, monotonic +)
- `tb_prevalence` (actionable, monotonic -)
- `hiv_prevalence` (actionable, monotonic -)
- `undernourished_pct` (actionable, monotonic -)
- `crude_birth_rate` (context, monotonic 0)
- `log_capex_per_capita` (actionable, monotonic +)
- `log_gdp_per_capita` (context, monotonic +)
- `sanitation_pct` (actionable, monotonic +)
- `year` (context, monotonic 0)

Monotonic constraints guarantee the medically correct direction of each effect,
so the policy simulator cannot produce a perverse recommendation.

## Limitations

- Small panel (10 countries). Generalization to an unseen country is the weakest
  test; report ranges, not point certainty.
- Mortality indicators are excluded to prevent target leakage.
- The linear surrogate served by the API is an approximation for edge inference.

## Provenance

Trained by `python -m pipeline.train`. Source data: ASEAN Statistical Yearbook
files and the World Bank Open Data API. See `data_dictionary` for full schema.
