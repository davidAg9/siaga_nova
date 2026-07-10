"""Model registry: version, persist, and document each trained model.

Every training run writes the active model to models/ and appends a record to
models/registry.json, so model lineage and metric history are tracked over time.
"""
import json

import joblib

from . import config as C


def save(model, surrogate, metrics, feature_spec, n_rows, version):
    """Persist the active model and its metadata to models/, append to the registry."""
    C.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": C.FEATURES}, C.MODELS / "siaga_model.joblib")
    (C.MODELS / "linear_surrogate.json").write_text(json.dumps(surrogate, indent=2))
    (C.MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (C.MODELS / "feature_spec.json").write_text(json.dumps(
        [{"feature": f, "monotonic": m, "kind": k} for f, m, k in feature_spec], indent=2))
    _write_card(metrics, feature_spec, n_rows, version)
    _append_registry(metrics, version, n_rows)


def _append_registry(metrics, version, n_rows):
    path = C.MODELS / "registry.json"
    log = json.loads(path.read_text()) if path.exists() else []
    log.append({"version": version, "trained_rows": n_rows,
                "loco_mae": metrics["loco_cv"]["mae_years"],
                "temporal_mae": metrics["temporal_2012_2014"]["mae_years"],
                "oot_mae": metrics["worldbank_oot_2015_2019"]["mae_years"],
                "oot_r2": metrics["worldbank_oot_2015_2019"]["r2"]})
    path.write_text(json.dumps(log, indent=2))


def _write_card(metrics, feature_spec, n_rows, version):
    m = metrics
    levers = "\n".join(f"- `{f}` ({k}, monotonic {'+' if mo==1 else '-' if mo==-1 else '0'})"
                       for f, mo, k in feature_spec)
    card = f"""# SIAGA Model Card

**Version:** {version}
**Task:** predict a country's life expectancy at birth (years) from health-system
and contextual indicators.
**Model:** monotonicity-constrained histogram gradient-boosting regressor.
**Training rows:** {n_rows} country-years (10 ASEAN states, {C.TRAIN_YEARS[0]}-{C.TRAIN_YEARS[1]}).

## Intended use

Decision support for public-health resource allocation: ranking drivers of life
expectancy and simulating the effect of policy changes. Not a clinical or
individual-level tool. Operates on aggregate national indicators only.

## Performance (held-out)

| Test | R2 | RMSE (yrs) | MAE (yrs) |
|------|----|-----------|-----------|
| Unseen country (leave-one-country-out) | {m['loco_cv']['r2']} | {m['loco_cv']['rmse_years']} | {m['loco_cv']['mae_years']} |
| Future of known countries (2012-2014) | {m['temporal_2012_2014']['r2']} | {m['temporal_2012_2014']['rmse_years']} | {m['temporal_2012_2014']['mae_years']} |
| Out-of-source vs World Bank (2015-2019) | {m['worldbank_oot_2015_2019']['r2']} | {m['worldbank_oot_2015_2019']['rmse_years']} | {m['worldbank_oot_2015_2019']['mae_years']} |

## Features

{levers}

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
"""
    (C.MODELS / "model_card.md").write_text(card)
