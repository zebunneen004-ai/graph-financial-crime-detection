# CV Bullets

- Built a reproducible Python pipeline over 203,769 Bitcoin transaction nodes and 234,355 directed edges, preserving unknown-labeled nodes for topology while isolating 46,564 labeled observations for supervised evaluation.
- Compared nine model/feature combinations and selected among default and tuned XGBoost candidates using expanding-window temporal CV, a practical improvement margin, paired PR-AUC bootstrap inference and an untouched future holdout.
- Added per-snapshot normalized PageRank, dedicated calibration and operating-selection periods, SHAP case explanations, permutation importance, automated tests, raw-data hashes and output-level provenance for AML alert-prioritization analysis.

# 30-Second Pitch

I evaluated whether five simple, interpretable graph-topology variables add predictive value beyond the Elliptic Bitcoin dataset's supplied transaction and one-hop aggregate features. I rebuilt the work as a reproducible Python pipeline with per-snapshot graph calculations, temporal model selection, dedicated calibration and operating blocks, paired PR-AUC inference, frozen alert-budget selection and an untouched future test. The selected operational candidate was tuned dataset only, and the best combined-minus-best-dataset-only final-test difference was +0.0004 PR-AUC. The project deliberately separates predictive lift, network context, calibration, explainability and analyst workload.

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
