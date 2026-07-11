# ◎ SIAGA

**The eye that watches five years ahead. An early-warning intelligence layer for ASEAN public health.**

10Alytics Global Hackathon 2026, Track B (Data Science). Built for the ASEAN Digital
Health and Climate Resilience Initiative (ADHCRI).

*Siaga* is the Indonesian and Malay word for **alert, ready, on standby**, the term
Southeast Asian governments already use for emergency preparedness (for example
Indonesia's *Desa Siaga*, "Alert Village", programme). SIAGA carries that idea to the
regional scale: shift ASEAN public health from reacting after a crisis to acting
before one.

---

## What it does

1. **Predicts life expectancy** for each ASEAN member state from the drivers a health
   ministry actually controls (staffing, immunization, spend, disease burden,
   nutrition), validated on independent World Bank data it never saw.
2. **Isolates the drivers** with SHAP, producing a ranked, policy-ready priority list.
3. **Forecasts the disease burden** (TB and malaria) five years forward, with
   uncertainty bands and honest, backtested model selection.
4. **Simulates policy** interactively: move a lever, see the predicted change in life
   expectancy, with medically-correct guarantees baked into the model.

### Results at a glance

| Validation | R2 | MAE |
|---|---|---|
| Future of known countries (2012 to 2014) | 0.93 | 1.08 years |
| Unseen country (leave-one-country-out) | 0.70 | 2.68 years |
| Out-of-source vs World Bank (2015 to 2019) | 0.76 | 1.96 years |

### SDG alignment

- **SDG 3** Good Health and Well-being: the models and forecasts.
- **SDG 10** Reduced Inequalities: the cross-country equity lens.
- **SDG 17** Partnerships for the Goals: the interoperability API.

---

## Repository layout

```
siaga/
  pipeline/                  the production training pipeline (run: python -m pipeline.train)
    config.py                paths, schema, feature spec, hyperparameters
    data.py                  ingest + clean -> harmonized panel
    model.py                 features, training, validation, SHAP, forecasts
    registry.py              model versioning + model card
    train.py                 CLI: ingest -> clean -> train -> validate -> version -> serve
  models/                    THE MODEL lives here (produced by the pipeline)
    siaga_model.joblib         the deployed gradient-boosting model
    model_card.md            intended use, performance, limitations
    linear_surrogate.json    edge model for the API
    metrics.json, feature_spec.json, registry.json   metrics + version history
  notebooks/
    siaga_analysis.ipynb     the analysis narrative: cleaning, EDA, features, models, SHAP, forecasts
    siaga_analysis.py        the same, as a readable jupytext source
  api/
    main.go, middleware.go   Go interoperability API (SDG 17): auth + rate limiting, zero deps
    data/                    serving files (panel, forecasts, drivers, surrogate)
  dashboard/
    app.py                   Streamlit decision dashboard (map, drivers, simulator, forecasts)
  deploy/
    Dockerfile.api, Dockerfile.dashboard, docker-compose.yml   containerized deployment
  data/
    raw/                     the original ASEAN indicator files (unmodified)
    clean/                   harmonized panel, metrics, quirks log, serving copies
  report/                    technical_report, api_technical_report, data_dictionary, real_world_readiness, submission_form_answers, pitch_script (in ../docs/ as .md and .pdf/.docx)
  presentation/
    siaga_nova_deck.pptx     editable 12-slide pitch deck (add team photos here)
    siaga_nova_deck.pdf      the same deck as PDF
  assets/
    siaga_logo.svg/.png, siaga_icon.svg/.png   brand logo and mark
    charts/                  branded chart images used in the deck
  requirements.txt           Python dependencies (pip)
```

---

## How to run

### Setup

This folder is self-contained. With **pixi** (recommended, uses the local `pixi.toml`):

```bash
cd siaga
pixi install          # creates the env from siaga/pixi.toml
```

Convenience tasks are defined in `pixi.toml`:

```bash
pixi run train-offline   # retrain the model (cached World Bank data)
pixi run dashboard       # launch the Streamlit dashboard
pixi run notebook        # open the analysis notebook
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 1. Train the model (the production pipeline)

```bash
cd siaga
python -m pipeline.train              # ingest -> clean -> train -> validate -> version -> serve
python -m pipeline.train --offline    # use the cached World Bank snapshot (no network)
```

This is the continuous-improvement path: drop new or corrected data into `data/raw`,
rerun, and the versioned model in `models/` plus the API and dashboard serving files
all update from one command. Each run appends its metrics to `models/registry.json`.

### 2. The analysis notebook

```bash
cd siaga/notebooks
jupyter lab siaga_analysis.ipynb      # or: jupyter notebook
```

The notebook is the analysis narrative: it reads the raw files from `../data/raw`,
reproduces the cleaning and modeling, and regenerates every figure and metric. The
`pipeline/` package is the productionized version of the same steps.

### 3. The Go API (interoperability layer, with auth + rate limiting)

```bash
cd siaga/api
go run .                                          # dev mode: auth disabled, 60 req/min
SIAGA_API_KEYS="my-key" SIAGA_RATE_RPM=120 go run .   # production: auth on, 120 req/min
```

With auth enabled, pass the key as `-H "X-API-Key: my-key"` (or `Authorization: Bearer`).
`/health` is always open.

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and surrogate model info |
| `GET /countries` | member states and ISO3 codes |
| `GET /indicators?country=&indicator=&from=&to=` | harmonized panel data |
| `GET /forecast?disease=&country=` | 5-year TB and malaria forecasts |
| `GET /drivers` | SHAP driver ranking |
| `POST /predict` | life-expectancy edge inference (JSON body `{"features": {...}}`) |

Example:

```bash
curl "http://localhost:8080/forecast?disease=tb&country=Cambodia"
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"features":{"physicians_per_1000":1.5,"undernourished_pct":8,"tb_prevalence":300,"year":2019}}'
```

### 4. The dashboard

```bash
pixi run streamlit run siaga/dashboard/app.py
# or: streamlit run siaga/dashboard/app.py
```

The dashboard loads the model from `models/siaga_model.joblib`, reads from the Go API when it
is running, and falls back to the local clean files otherwise, so it always renders.
Set `SIAGA_API` to point at a remote API and `SIAGA_API_KEY` if the API has auth on.

### 5. Everything as containers

```bash
cd siaga
docker compose -f deploy/docker-compose.yml up --build   # API on :8080, dashboard on :8501
```

See `../docs/real_world_readiness.md` for an honest assessment of what is
production-grade today and what a funded pilot would harden next.

---

## Reproducing the data pipeline

The cleaning, modeling, and export steps live in the repository-root `workbench/`
folder (kept outside this submission folder to keep it clean). Running scripts 01
through 11 there regenerates everything in `siaga/data/clean` and `siaga/api/data`.
The notebook is the canonical, self-contained version of the same pipeline.

---

## Team

*Pod Nova.* Team details, bios, and headshots accompany the submission.

## License and data

Health indicators are from the ASEAN Statistical Yearbook and the World Bank Open Data
API. See `../docs/technical_report.md` for full references.
