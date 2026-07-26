# Transaction Network Analysis for Financial Crime Detection

## Purpose
Evaluated whether five interpretable topology variables add predictive value beyond the Elliptic benchmark's 165 supplied transaction and one-hop aggregate predictors.

## Research design
- 203,769 transaction nodes and 234,355 directed payment-flow edges
- All unknown-labeled nodes retained for topology; 46,564 labeled observations used for supervised metrics
- Model selection by expanding-window temporal CV within time steps 1–30
- Dedicated calibration block 31–35, operating-selection block 36–40 and untouched test block 41–49
- Complete nine-model default ablation, default-versus-tuned XGBoost comparison, paired PR-AUC inference, calibration and alert-workload analysis

## Main result
The selected operational candidate was **tuned dataset only**, with final-test PR-AUC **0.6422**. The best combined-minus-best-dataset-only final-test ΔPR-AUC was **+0.0004**. Small or negative results are reported directly rather than rebranded as improvements.

## Operational analysis
A **5%** alert budget was selected on the dedicated operating block and applied unchanged to the final labeled holdout. It is an experiment-specific workload trade-off, not a universal AML policy.

## Relevance
The project demonstrates temporal validation, transaction-network context, probability calibration, explainability, reproducibility, provenance and analyst-workload measurement relevant to AML, VASP risk, fraud analytics and model governance.

## Boundaries
Public Bitcoin benchmark; not UAE production data; not a compliance determination; not real-time; does not replace investigators or MLRO judgment.
