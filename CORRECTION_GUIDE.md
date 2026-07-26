# Correction Guide

The authoritative implementation is the modular pipeline under `src/`. Old notebooks, old figures and old result tables are historical artifacts only.

## Non-negotiable rules

- Raw CSV files remain unchanged.
- Unknown labels remain in graph topology and are excluded from supervised metrics.
- `txId`, `time_step`, `total_degree`, `pagerank_raw_snapshot` and `weak_component_size` are excluded from primary model matrices.
- PageRank is calculated within snapshots and normalized against the uniform snapshot baseline.
- Model selection occurs only through temporal evidence within time steps 1–30.
- Calibration uses only 31–35.
- Threshold and alert-budget selection use only 36–40.
- Final evaluation uses only 41–49.
- Final-test results never choose a model, calibration method, threshold or alert budget.
- All numerical claims must originate from the latest successful run.
