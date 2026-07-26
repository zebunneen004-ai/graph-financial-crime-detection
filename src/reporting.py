from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import load_config, project_path, read_json, record_provenance, setup_logging


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def format_model_table(metrics: pd.DataFrame) -> str:
    test = metrics.loc[metrics["split"] == "test"].copy()
    rows = [
        "| Model | Feature set | PR-AUC | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in test.sort_values(["model_family", "feature_set"]).iterrows():
        rows.append(
            f"| {row['model_family'].replace('_', ' ').title()} | "
            f"{row['feature_set'].replace('_', ' ').title()} | {row['pr_auc']:.4f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    return "\n".join(rows)


def selected_test_row(results: Path, selection: dict) -> pd.Series:
    metrics = pd.read_csv(results / "xgboost_candidate_metrics.csv")
    rows = metrics.loc[
        metrics["candidate_name"].eq(selection["selected_candidate_name"])
        & metrics["split"].eq("test")
    ]
    if len(rows) != 1:
        raise AssertionError("Could not uniquely identify the selected candidate's final-test metrics.")
    return rows.iloc[0]


def build_readme(config: dict) -> str:
    results = project_path(config, "results_final")
    graph_audit = read_json(results / "graph_audit.json")
    graph_metadata = read_json(results / "graph_feature_metadata.json")
    data_manifest = read_json(results / "data_manifest.json")
    selection = read_json(results / "operational_model_selection.json")
    inference = read_json(results / "incremental_value_inference.json")
    calibration = read_json(results / "calibration_metrics.json")
    sensitivity = read_json(results / "snapshot_size_sensitivity.json")
    alert_selection = read_json(results / "alert_budget_selection.json")
    runtime = read_json(results / "run_metadata.json")
    default_metrics = pd.read_csv(results / "default_model_metrics.csv")
    selected_test = selected_test_row(results, selection)
    alert = alert_selection["selected_budget_final_test_metrics"]
    ci_low, ci_high = inference["paired_bootstrap_confidence_interval"]
    python_version = str(runtime["python"]).split()[0]

    return f"""# Transaction Network Analysis for Financial Crime Detection and AML Alert Prioritization

## 30-Second Summary

This reproducible research implementation evaluates whether five additional interpretable node-level topology variables improve illicit-transaction classification beyond the Elliptic benchmark's supplied local and one-hop aggregated attributes. Unknown-labeled transactions are preserved in the graph and excluded from supervised metrics. Model selection, probability calibration, operating-point selection and final evaluation use chronologically separated evidence.

## Key Finding

The operational candidate selected using expanding-window temporal cross-validation within time steps 1–30 was **{selection['selected_candidate_name'].replace('_', ' ')}**. Its final labeled-holdout PR-AUC was **{selected_test['pr_auc']:.4f}**. The best combined-minus-best-dataset-only final-test difference was **{inference['point_delta_pr_auc']:+.4f} PR-AUC**, with a paired bootstrap {int(inference['confidence_level'] * 100)}% interval of **[{ci_low:+.4f}, {ci_high:+.4f}]**. {inference['interpretation']}

## Dataset

- {data_manifest['counts']['nodes']:,} transaction nodes
- {data_manifest['counts']['edges']:,} directed transaction-to-transaction edges
- {data_manifest['counts']['illicit']:,} illicit-labeled, {data_manifest['counts']['licit']:,} licit-labeled and {data_manifest['counts']['unknown']:,} unknown-labeled transactions
- {data_manifest['feature_schema']['supplied_predictor_count']} supplied model predictors: {data_manifest['feature_schema']['local_supplied_predictor_count_excluding_time_step']} local predictors after excluding `time_step`, plus {data_manifest['feature_schema']['one_hop_aggregated_predictor_count']} one-hop aggregated predictors
- 49 time steps
- SHA-256 hashes for all three raw files are recorded in `results/final/data_manifest.json`

Unknown means that no supervised ground-truth label is supplied. Unknown observations are retained for topology calculations and excluded from supervised evaluation.

## Research Question

**Primary:** Do additional interpretable node-level topological features improve illicit-transaction classification beyond the Elliptic dataset's supplied local and one-hop aggregated attributes under chronological evaluation?

**Secondary:** When topology variables do not materially improve predictive performance, do they provide useful descriptive network context for AML review?

## Methodology

Primary added graph variables:

- in-degree
- out-degree
- snapshot-normalized PageRank
- clustering coefficient on an undirected projection
- k-core number on an undirected projection

PageRank is computed independently within each completed time-step snapshot. The primary PageRank feature is divided by the snapshot's uniform baseline, so a value of 1 equals the snapshot mean and global 1/N scaling cannot act as an indirect snapshot-size feature. Raw within-snapshot PageRank is retained only as a descriptive output. `total_degree` is excluded because it is exactly `in_degree + out_degree`. `weak_component_size` is analyzed separately because it behaves as a snapshot-size/time-step proxy rather than transaction-level isolation. In the tuned sensitivity model, adding this snapshot variable changed final-test PR-AUC by **{sensitivity['final_test_delta_vs_tuned_primary_combined']:+.4f}** relative to the tuned primary combined model.

## Chronological Evaluation

- Training, temporal tuning and model selection: time steps 1–30
- Probability calibration only: time steps 31–35
- Threshold and alert-budget selection only: time steps 36–40
- Final untouched labeled holdout: time steps 41–49

The final holdout does not influence feature-set choice, default-versus-tuned selection, calibration fitting, threshold selection or alert-budget selection.

## Model Comparison

{format_model_table(default_metrics)}

Threshold-dependent metrics use thresholds selected on the operating-selection block. Accuracy is not used as the lead metric.

## Operational Candidate Selection

For dataset-only and combined XGBoost, the default and tuned variants are compared using the same expanding-window temporal folds. Tuning is accepted only when it improves mean temporal-CV PR-AUC by at least the configured minimum gain. The combined feature set is selected only when its best candidate exceeds the best dataset-only candidate by the configured practical PR-AUC margin. This prevents a tuned model from being selected merely because tuning was attempted.

## Incremental Graph-Feature Results

The direct incremental comparison uses aligned final-test predictions from the best temporally selected dataset-only and combined candidates. Paired observation-level bootstrap and time-step block-bootstrap intervals are reported. An interval containing zero is interpreted as failure to detect a statistically reliable improvement, not proof of model identity or universal uselessness of graph variables.

## Calibration

Raw XGBoost outputs are treated as model scores. A sigmoid calibrator was fitted only on labeled time steps 31–35. Final-test Brier score changed from **{calibration['raw_test_brier_score']:.5f}** before calibration to **{calibration['calibrated_test_brier_score']:.5f}** after calibration. Calibration and ranking discrimination are reported separately.

## Alert-Prioritization Analysis

Among the predefined 1%, 5%, 10% and 20% budgets, **{float(alert_selection['selected_budget']) * 100:.0f}%** was selected on time steps 36–40 using the documented F1 rule and applied unchanged to the final labeled holdout.

- Alerts: {int(alert['alerts']):,} of {int(alert['labeled_population']):,}
- Precision: {percent(float(alert['precision']))}
- Recall: {percent(float(alert['recall']))}
- F1: {float(alert['f1']):.3f}
- False positives per true positive: {float(alert['false_positives_per_true_positive']):.2f}
- Lift over labeled prevalence: {float(alert['lift_over_labeled_prevalence']):.2f}×

This is an experiment-specific operating point, not a universally optimal AML alert budget.

## Explainability

TreeSHAP outputs are generated from the authoritative operational candidate in raw model-output space. SHAP explains the fitted model's output decomposition; it does not establish criminal intent or independently justify an STR. Permutation importance is generated from the best combined candidate selected by temporal CV and reports uncertainty across repeated permutations.

## Network Context

The processed graph contains {graph_audit['weak_component_count']} weak components and every checked edge remains within its time step. The graph is a directed acyclic graph with no self-loops. These are properties of the processed Elliptic benchmark. Observed sources and sinks are defined only within the supplied subgraph and are not automatically mining transactions, cold storage or final endpoints.

## Limitations

- Public benchmark; not UAE-specific production data
- Anonymous supplied features
- Incomplete and potentially non-random label availability
- Unknown transactions cannot be evaluated with supervised metrics
- Completed-snapshot graph variables; batch rather than live evaluation
- Coarse time steps and only one blockchain dataset
- Dependence among connected observations
- Five simple topology variables rather than a complete network representation
- Hyperparameter search and candidate comparison reuse the same temporal folds within the training era, so CV estimates for the selected tuned candidate may be optimistic
- Benchmark prevalence does not represent real VASP prevalence
- Results may not generalize to production VASP populations

## UAE and AML Relevance

This public benchmark demonstrates analytical methods relevant to VASP financial-crime analytics, including chronological evaluation, network context, explainability, calibration, alert prioritization, workload analysis and output provenance. It does not establish regulatory compliance and does not replace investigator or MLRO judgment.

## Future Work

Future work should examine causally available transaction-time features, nested temporal model selection, additional blockchain datasets, entity or wallet-level graphs, graph embeddings, time-aware network models, label-selection bias and institution-specific alert-utility functions.

## Reproducibility

This final run used Python **{python_version}**. Create an isolated environment using an available compatible Python installation:

```bat
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
run.bat
```

Cross-platform execution:

```bash
python src/corrected_master_pipeline.py --config config/config.yaml --overwrite
```

All authoritative outputs are written under `results/final`, `figures/final` and `models/final`. `results/final/provenance_manifest.csv` records output lineage and the Git commit when the repository was committed before execution.

## Repository Structure

```mermaid
flowchart TD
    A[Raw Elliptic files + SHA-256 hashes] --> B[Memory-efficient data preparation]
    B --> C[All-node graph audit]
    C --> D[Per-snapshot graph features]
    D --> E[Statistics]
    B --> F[Default nine-model ablation]
    D --> F
    F --> G[Temporal default-vs-tuned XGBoost selection within 1-30]
    G --> H[Calibration on 31-35]
    H --> I[Threshold and alert selection on 36-40]
    I --> J[Authoritative final-test predictions on 41-49]
    G --> K[SHAP and permutation importance]
    J --> L[Final documentation and provenance]
    K --> L
```

## Disclaimer

This repository is a reproducible research implementation. It is not a production AML platform, not real-time transaction monitoring, not legal advice and not evidence that any transaction or person committed a crime.
"""


def build_employer_brief(config: dict) -> str:
    results = project_path(config, "results_final")
    selection = read_json(results / "operational_model_selection.json")
    inference = read_json(results / "incremental_value_inference.json")
    alert = read_json(results / "alert_budget_selection.json")
    selected_test = selected_test_row(results, selection)
    return f"""# Transaction Network Analysis for Financial Crime Detection

## Purpose
Evaluated whether five interpretable topology variables add predictive value beyond the Elliptic benchmark's 165 supplied transaction and one-hop aggregate predictors.

## Research design
- 203,769 transaction nodes and 234,355 directed payment-flow edges
- All unknown-labeled nodes retained for topology; 46,564 labeled observations used for supervised metrics
- Model selection by expanding-window temporal CV within time steps 1–30
- Dedicated calibration block 31–35, operating-selection block 36–40 and untouched test block 41–49
- Complete nine-model default ablation, default-versus-tuned XGBoost comparison, paired PR-AUC inference, calibration and alert-workload analysis

## Main result
The selected operational candidate was **{selection['selected_candidate_name'].replace('_', ' ')}**, with final-test PR-AUC **{selected_test['pr_auc']:.4f}**. The best combined-minus-best-dataset-only final-test ΔPR-AUC was **{inference['point_delta_pr_auc']:+.4f}**. Small or negative results are reported directly rather than rebranded as improvements.

## Operational analysis
A **{float(alert['selected_budget']) * 100:.0f}%** alert budget was selected on the dedicated operating block and applied unchanged to the final labeled holdout. It is an experiment-specific workload trade-off, not a universal AML policy.

## Relevance
The project demonstrates temporal validation, transaction-network context, probability calibration, explainability, reproducibility, provenance and analyst-workload measurement relevant to AML, VASP risk, fraud analytics and model governance.

## Boundaries
Public Bitcoin benchmark; not UAE production data; not a compliance determination; not real-time; does not replace investigators or MLRO judgment.
"""


def build_interview_assets(config: dict) -> str:
    results = project_path(config, "results_final")
    selection = read_json(results / "operational_model_selection.json")
    inference = read_json(results / "incremental_value_inference.json")
    return f"""# CV Bullets

- Built a reproducible Python pipeline over 203,769 Bitcoin transaction nodes and 234,355 directed edges, preserving unknown-labeled nodes for topology while isolating 46,564 labeled observations for supervised evaluation.
- Compared nine model/feature combinations and selected among default and tuned XGBoost candidates using expanding-window temporal CV, a practical improvement margin, paired PR-AUC bootstrap inference and an untouched future holdout.
- Added per-snapshot normalized PageRank, dedicated calibration and operating-selection periods, SHAP case explanations, permutation importance, automated tests, raw-data hashes and output-level provenance for AML alert-prioritization analysis.

# 30-Second Pitch

I evaluated whether five simple, interpretable graph-topology variables add predictive value beyond the Elliptic Bitcoin dataset's supplied transaction and one-hop aggregate features. I rebuilt the work as a reproducible Python pipeline with per-snapshot graph calculations, temporal model selection, dedicated calibration and operating blocks, paired PR-AUC inference, frozen alert-budget selection and an untouched future test. The selected operational candidate was {selection['selected_candidate_name'].replace('_', ' ')}, and the best combined-minus-best-dataset-only final-test difference was {inference['point_delta_pr_auc']:+.4f} PR-AUC. The project deliberately separates predictive lift, network context, calibration, explainability and analyst workload.

# Two-Minute Technical Explanation

The Elliptic benchmark contains transaction nodes, directed payment-flow edges, 165 supplied predictors and incomplete illicit, licit or unknown labels. I retained all nodes for graph calculations because dropping unknown nodes changes topology, but excluded unknown labels from every supervised metric. I calculated in-degree, out-degree, snapshot-normalized PageRank, clustering and k-core as primary topology variables. Total degree is excluded as deterministic redundancy, while weak-component size is handled separately after verifying that it is a snapshot-size proxy.

Time steps 1–30 are used for expanding-window tuning and model selection. Default and tuned dataset-only and combined XGBoost candidates are compared on identical folds, with minimum gains required before accepting additional complexity. Time steps 31–35 are used only for sigmoid calibration, 36–40 only for threshold and alert-budget selection, and 41–49 only for final evaluation. Incremental graph value is tested with paired observation-level and time-step block bootstrap intervals. SHAP explains the authoritative candidate, while permutation importance measures final-test score degradation for the best combined candidate. The conclusion is limited to this public benchmark and does not claim regulatory compliance or criminal intent.

# Interview Questions and Answers

## Why keep unknown nodes?
They lack supervised labels, but remain part of the supplied transaction graph. Removing them changes degree, centrality and connectivity calculations.

## Why normalize PageRank within each snapshot?
Standard PageRank sums to one. Across snapshots of different sizes, raw values therefore carry a 1/N scale. Dividing by the uniform snapshot baseline makes a value of one equal to the snapshot mean and reduces unintended snapshot-size information.

## Why exclude total degree?
It is exactly the sum of in-degree and out-degree, so it adds no independent information when both components are already included.

## Why separate weak-component size?
In this processed benchmark it is constant within each snapshot and equals snapshot size, so it behaves like time context rather than transaction-specific isolation.

## Why not use random cross-validation?
The research question concerns future-period generalization. Expanding-window folds preserve chronology and avoid training on later time steps than validation observations.

## What does a non-significant ΔPR-AUC mean?
The paired analysis did not detect a statistically reliable improvement under this design. It does not prove model equality or that richer graph methods could never help.

## Does SHAP prove a transaction is suspicious?
No. SHAP decomposes the fitted model output. It is a model explanation, not evidence of intent or an STR decision.
"""


def build_figure_captions() -> str:
    return """# Final Figure Captions

1. **Label distribution.** Counts of unknown, licit-labeled and illicit-labeled transaction nodes. Unknown nodes remain in graph calculations but are excluded from supervised metrics.
2. **Transactions by time step.** Processed snapshot size and illicit prevalence among labeled observations. The dashed line is the overall weighted labeled prevalence.
3. **Primary graph-feature distributions.** Pooled descriptive licit-versus-illicit comparisons for the five primary topology variables, with Holm-adjusted p-values and rank-biserial effects.
4. **Within-time effects.** Time-step-specific rank-biserial effects, included to show temporal heterogeneity and reduce reliance on pooled comparisons.
5. **Weak-component snapshot context.** Snapshot size versus labeled illicit prevalence; weak-component size is interpreted as snapshot context, not transaction-level isolation.
6. **Default model ablation.** Final-test PR-AUC for all nine default model and feature-set combinations; thresholds were frozen using the operating-selection block.
7. **XGBoost candidate comparison.** Mean expanding-window temporal-CV PR-AUC and later final-test PR-AUC for default and tuned dataset-only and combined candidates. Final-test values are reported after candidate selection and were not used to choose the candidate.
8. **Paired ΔPR-AUC bootstrap.** Distribution of aligned final-test best-combined-minus-best-dataset-only PR-AUC differences.
9. **Calibration curve.** Raw model scores and calibration-block-fitted sigmoid probabilities evaluated on the final labeled holdout.
10. **Alert-budget operating block.** Precision, recall and F1 across predefined budgets on time steps 36–40; the selected operating point is marked.
11. **Alert-budget final test.** The same predefined budgets on the final holdout, with the operating-block-selected budget marked and not reselected.
12. **SHAP summary.** Global model-output decomposition for a fixed final-test sample from the authoritative operational candidate.
13. **SHAP dependence.** Relationship between the top anonymized model feature and its SHAP contribution in raw model-output space.
14. **Case waterfalls.** One true positive, false positive, false negative and true negative model explanation, when each outcome exists.
15. **Permutation importance.** Final-test PR-AUC decrease after repeated feature permutations in the best combined XGBoost candidate selected by temporal CV.
"""


def build_provenance_appendix(config: dict) -> str:
    path = project_path(config, "results_final") / "provenance_manifest.csv"
    provenance = pd.read_csv(path)
    columns = [
        "output_path",
        "output_type",
        "run_id",
        "source_script_function",
        "model_family",
        "feature_set",
        "calibration_status",
        "data_split",
        "prediction_file",
        "seed",
        "git_commit",
        "notes",
    ]
    available = [column for column in columns if column in provenance.columns]
    return "# Result Provenance Appendix\n\n" + provenance[available].to_markdown(index=False) + "\n"


def run_reporting(config: dict) -> None:
    logger = setup_logging(config, "reporting")
    root = Path(__file__).resolve().parents[1]
    readme_path = root / "README.md"
    employer_path = project_path(config, "employer_brief") / "employer_brief.md"
    methodology_path = project_path(config, "report") / "final_methodology_results.md"
    interview_path = project_path(config, "results_final") / "interview_assets.md"
    captions_path = project_path(config, "report") / "figure_captions.md"
    provenance_appendix_path = project_path(config, "report") / "provenance_appendix.md"

    readme_text = build_readme(config)
    readme_path.write_text(readme_text, encoding="utf-8")
    employer_path.write_text(build_employer_brief(config), encoding="utf-8")
    methodology_path.write_text(readme_text, encoding="utf-8")
    interview_path.write_text(build_interview_assets(config), encoding="utf-8")
    captions_path.write_text(build_figure_captions(), encoding="utf-8")
    provenance_appendix_path.write_text(build_provenance_appendix(config), encoding="utf-8")

    record_provenance(
        config,
        [
            readme_path,
            employer_path,
            methodology_path,
            interview_path,
            captions_path,
            provenance_appendix_path,
        ],
        stage="reporting",
        source="src.reporting.run_reporting",
        output_type="final documentation",
        notes="All numerical claims are read from reproduced final result files.",
    )
    logger.info("Final documentation generated from reproduced results.")


def run(config_path: str | Path) -> None:
    run_reporting(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final documentation from reproduced results.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
