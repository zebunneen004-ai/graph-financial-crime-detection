# Final Figure Captions

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
