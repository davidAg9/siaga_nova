"""SIAGA training pipeline entrypoint.

    python -m pipeline.train                 retrain on current data, version the model
    python -m pipeline.train --offline       use the cached World Bank snapshot
    python -m pipeline.train --version v3     label this run

Runs every stage end to end: ingest, clean, feature-build, train, validate,
explain, forecast, then versions the model and refreshes the serving files that
the Go API and Streamlit dashboard consume. This is the production path for
continuous improvement: drop new data into data/raw, rerun, and the deployed
model and dashboard update from one command.
"""
import argparse
import json

from . import config as C
from . import data as D
from . import model as M
from . import registry as R


def run(offline=False, version="v1"):
    print("[1/6] ingest: World Bank + raw ASEAN files")
    wb = D.pull_worldbank(use_cache=offline)
    panel, quirks = D.build_panel()
    C.CLEAN.mkdir(parents=True, exist_ok=True)
    panel.to_csv(C.CLEAN / "asean_panel.csv", index=False)
    (C.CLEAN / "quirks_log.md").write_text("# Data quirks log\n\n" + "\n".join(f"- {q}" for q in quirks))
    print(f"      panel {panel.shape}, {len(quirks)} quirks logged")

    print("[2/6] features")
    data = M.build_frame(panel, wb)
    train = M.training_frame(data)
    train.to_csv(C.CLEAN / "modeling_frame.csv", index=False)
    print(f"      training frame {train.shape}, {train[C.FEATURES].isna().sum().sum()} missing cells (GBM is NaN-native)")

    print("[3/6] validate (three held-out tests)")
    metrics = M.validate(train, panel, wb)
    for k, v in metrics.items():
        print(f"      {k:<26} {v}")

    print("[4/6] fit reference model + surrogate")
    reference = M.make_model().fit(train[C.FEATURES], train[C.TARGET])
    surrogate = M.linear_surrogate(train)
    print(f"      surrogate in-sample R2={surrogate['in_sample_r2']}")

    print("[5/6] explain (SHAP) + forecast (TB, malaria)")
    shap_imp = M.shap_importance(reference, train)
    forecasts = M.forecast_diseases(data)
    (C.CLEAN / "shap_importance.json").write_text(json.dumps(shap_imp))
    forecasts.to_csv(C.CLEAN / "disease_forecasts.csv", index=False)

    print("[6/6] version + export serving files")
    R.save(reference, surrogate, metrics, C.FEATURE_SPEC, len(train), version)
    _export_serving()
    print(f"      model versioned to models/ ({version}); serving files refreshed")
    return metrics


def _export_serving():
    """Copy what the Go API serves into api/data, and mirror model files to clean/."""
    C.API_DATA.mkdir(parents=True, exist_ok=True)
    (C.API_DATA / "linear_surrogate.json").write_bytes((C.MODELS / "linear_surrogate.json").read_bytes())
    (C.API_DATA / "metrics.json").write_bytes((C.MODELS / "metrics.json").read_bytes())
    for name in ["asean_panel.csv", "disease_forecasts.csv", "shap_importance.json"]:
        (C.API_DATA / name).write_bytes((C.CLEAN / name).read_bytes())
    # Keep the dashboard's expected copies in data/clean in sync with the registry.
    (C.CLEAN / "siaga_model.joblib").write_bytes((C.MODELS / "siaga_model.joblib").read_bytes())
    (C.CLEAN / "linear_surrogate.json").write_bytes((C.MODELS / "linear_surrogate.json").read_bytes())
    (C.CLEAN / "metrics.json").write_bytes((C.MODELS / "metrics.json").read_bytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and version the SIAGA model.")
    ap.add_argument("--offline", action="store_true", help="use cached World Bank snapshot")
    ap.add_argument("--version", default="v1", help="version label for this run")
    args = ap.parse_args()
    run(offline=args.offline, version=args.version)
