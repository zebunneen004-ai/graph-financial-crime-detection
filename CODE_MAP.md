# Code Map

| Stage | Module | Role |
|---|---|---|
| Data preparation | `src/data_loading.py` | Validates counts, hashes raw files, preserves unknown nodes and builds memory-efficient processed tables |
| Graph audit | `src/graph_construction.py` | Verifies all nodes/edges, time consistency, self-loops, components and acyclicity |
| Graph features | `src/graph_features.py` | Degree, per-snapshot normalized PageRank, clustering, k-core and snapshot-context sensitivity |
| Statistics | `src/statistics.py` | Pooled descriptive tests, Holm correction, effect sizes and within-time sensitivity |
| Default ablation | `src/modeling.py` | All nine default model/feature combinations; thresholds selected on time steps 36–40 |
| Candidate selection | `src/tuning.py` | Identical temporal folds for default/tuned dataset and combined XGBoost, practical-margin selection and paired inference |
| Calibration | `src/calibration.py` | Sigmoid calibration fitted only on time steps 31–35 |
| Alert analysis | `src/alert_budget.py` | Budget selected only on time steps 36–40 and frozen for time steps 41–49 |
| Explainability | `src/explainability.py` | SHAP for the authoritative candidate and permutation importance for the best combined candidate |
| Reporting | `src/reporting.py` | Generates documentation from reproduced outputs only |
| Master runner | `src/corrected_master_pipeline.py` | Archives previous outputs and executes stages in isolated processes |

## Primary topology predictors

- `in_degree`
- `out_degree`
- `pagerank_relative`
- `clustering`
- `k_core`

## Excluded or separate variables

- `total_degree`: deterministic redundancy
- `pagerank_raw_snapshot`: descriptive only because its 1/N scale varies by snapshot size
- `weak_component_size`: snapshot-size/time-step sensitivity only
