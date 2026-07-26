# Research-Grade V2 Upgrade

This upgrade replaces the first corrected implementation with a stricter final design.

## Methodological changes

1. **Per-snapshot PageRank**
   - PageRank is computed independently within each completed time-step snapshot.
   - The primary predictor is `pagerank_relative = raw_snapshot_pagerank / (1 / snapshot_size)`.
   - A value of 1 equals the snapshot mean, which prevents raw PageRank's 1/N scale from acting as a snapshot-size proxy.
   - `pagerank_raw_snapshot` remains descriptive and is excluded from predictive matrices.

2. **Default-versus-tuned candidate selection**
   - Default and tuned dataset-only XGBoost use identical temporal folds.
   - Default and tuned combined XGBoost use identical temporal folds.
   - Tuning is accepted only if its mean temporal-CV PR-AUC improves by at least 0.002.
   - Combined is accepted only if its best candidate exceeds the best dataset-only candidate by at least 0.01 PR-AUC.
   - Final-test performance never selects the operational candidate.

3. **Dedicated chronological decision blocks**
   - Time steps 1–30: training, temporal tuning and model selection.
   - Time steps 31–35: probability calibration only.
   - Time steps 36–40: threshold and alert-budget selection only.
   - Time steps 41–49: final evaluation only.

4. **Additional reproducibility controls**
   - SHA-256 hashes for all raw files.
   - Repository-relative provenance paths.
   - Git commit capture when Git is initialized before execution.
   - `tabulate` permanently included in `requirements.txt`.
   - README records the actual Python version used by the run.

## Apply to an existing corrected project

Extract the V2 patch directly into the root of `GRAPH_PROJECT_CORRECTED` and allow Windows to replace matching files. Do not place the patch in a new nested folder.

Then run:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest tests -q
```

Before the definitive run, initialize Git and commit the code state:

```bat
git init
git add .gitignore src config tests tools requirements.txt run.bat Makefile pytest.ini RESEARCH_GRADE_V2_UPGRADE.md CODE_MAP.md CORRECTION_GUIDE.md SETUP_AND_RUN.md
git commit -m "Research-grade V2 methodology"
```

If `git` is unavailable, the pipeline still runs, but provenance will record the Git commit as unavailable.

Run the definitive pipeline once:

```bat
run.bat
```

Do not also run the master command afterward. `run.bat` already invokes it.

## Success checks

Confirm:

```text
results/final/run_completion.json
```

contains:

```json
"status": "success"
```

Also verify these new outputs exist:

```text
results/final/xgboost_candidate_cv_summary.csv
results/final/xgboost_candidate_metrics.csv
results/final/operational_model_selection.json
results/final/calibration_metrics.json
results/final/operating_alert_budget_analysis.csv
results/final/authoritative_test_predictions.parquet
results/final/data_manifest.json
figures/final/xgboost_candidate_comparison.png
```

The previous successful run is historical after this change because PageRank, split roles and model selection have changed. All final results must come from the V2 run.
