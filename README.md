# metal-peer-risk-scoring

Unsupervised peer-group risk scoring demo for Metal/Vault home insurance submissions, with explainable statistical scoring, new submission simulation, notebooks, and Streamlit dashboard.

## Overview

This POC loads `Results.csv` (EDW / Metal-related submission export), cleans and deduplicates policies, builds peer groups (state / program / product / occupancy), and scores each submission using:

1. **Absolute feature risk** — deterministic, business-readable transforms.
2. **Peer-relative percentiles** — rank within comparable policies.
3. **Similarity context** — nearest historical neighbors.

The Streamlit app supports portfolio triage, submission drill-down, synthetic submission simulation, and model documentation.

> **Not a pricing engine or underwriting decision system.** This is risk prioritization and submission triage only.

## Install

```bash
pip install -r requirements.txt
```

## Run Streamlit

From the project root:

```bash
streamlit run app/main.py
```

Pages:

- **Portfolio Overview** — KPIs, filters, charts, ranked queue.
- **Submission Detail** — score breakdown, explanations, similar policies.
- **New Submission Simulator** — realistic synthetic submissions, rescoring, reset.
- **Model Explanation** — methodology and limitations.

## Notebooks

```bash
jupyter notebook notebooks/
```

Notebooks live in `notebooks/` and share `notebook_lib.py` (data load, peer scoring, ML benchmarks). Run from repo root or from `notebooks/`.

| Notebook | Purpose |
|----------|---------|
| `01_data_understanding.ipynb` | Profiling, nulls, dedupe |
| `02_peer_group_scoring_experiment.ipynb` | Peer groups & examples |
| `03_new_submission_simulation.ipynb` | Insert & rescore flow |
| `04_model_validation_summary.ipynb` | Sensitivity + **model comparison** (Ridge, RF, KNN, …) |

## Input data

- **File:** `Results.csv` (111 columns, header row).
- **Source:** SQL extract documented in `results.sql` (one row per policy + effective date).

## Tests

```bash
python -m pytest tests/ -q
```

## Project structure

```
app/           Streamlit UI
src/           Data, features, scoring, simulation
notebooks/     Validation notebooks
docs/          LaTeX technical brief
tests/         Unit tests
Results.csv    Portfolio extract
```

## Model configuration

Weights and feature groups: `src/config/feature_config.py`.

## Known limitations

- No approve/decline or loss labels.
- Percentiles shift as the reference book changes.
- Not connected to Metal APIs in this POC.

## Next steps

1. Collect underwriting decisions and claim outcomes.
2. Calibrate component weights with domain experts.
3. Deploy scoring API for Metal submission events.
4. Compare against supervised models when labels exist.

## Author

Federico Lievano
