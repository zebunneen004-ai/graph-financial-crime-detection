================================================================================
RESEARCH PAPER: Graph-Theoretic Feature Engineering for Suspicious Transaction Detection
================================================================================

ABSTRACT

This project investigates whether graph-theoretic features improve suspicious transaction detection on the Elliptic Bitcoin Dataset. Using a strict inductive temporal split (train t=1-30, test t=41-49), three feature sets were compared: Dataset Only (A), Graph Only (B), and Combined (C). Results show that graph features alone perform poorly (PR-AUC = 0.062 vs 0.150 for dataset features), consistent with small effect sizes in Mann-Whitney U testing. When combined with dataset features, PR-AUC remained essentially unchanged (0.1490 vs 0.1498), indicating that dataset-provided features already capture substantial predictive signal. However, the combined model achieved modest improvements in precision (+5.7%) and F1-score (+4.8%), with 105 fewer false alerts at comparable recall. Permutation importance confirmed dataset features dominate the top predictors, with no graph features in the top 20. These findings suggest graph features may offer operational utility in reducing analyst workload and providing interpretable structural risk indicators for compliance workflows, even when they do not improve raw ranking power.

1. INTRODUCTION
- Rise of cryptocurrency and AML challenges
- Limitations of traditional rule-based systems
- Need for interpretable, graph-aware analytics

2. RESEARCH QUESTIONS
- Main: Do graph-theoretic features improve classification?
- Secondary: Which features provide strongest interpretable signals?

3. LITERATURE REVIEW
- Elliptic dataset (Weber et al., 2019)
- GNN benchmarks on Elliptic
- Interpretability gap in AML ML
- UAE/VARA regulatory context

4. DATASET DESCRIPTION
- 203,769 nodes, 234,355 edges
- 166 features, 46,564 labeled
- Temporal structure (49 time steps)

5. GRAPH CONSTRUCTION
- Directed graph (DiGraph)
- Unknown nodes preserved for topology
- Temporal validation (no violations)

6. GRAPH FEATURE ENGINEERING
- 7 features: degree, PageRank, clustering, k-core, component size
- Redundancy removal (perfect correlations)
- Correlation leakage check

7. STATISTICAL ANALYSIS
- Mann-Whitney U tests (7 features)
- Effect sizes (rank-biserial correlation)
- Key finding: 2 features with small effects, PageRank not significant (p=0.355)

8. CLASSIFICATION METHODOLOGY
- 3 feature sets (A/B/C)
- Temporal split (train 1-30, test 41-49)
- Logistic Regression with class_weight='balanced'
- PR-AUC as primary metric

9. RESULTS
- Model A (Dataset): PR-AUC 0.1498, Precision 0.1107
- Model B (Graph): PR-AUC 0.0622, Precision 0.0809
- Model C (Combined): PR-AUC 0.1490, Precision 0.1170
- Combined: +5.7% precision, 105 fewer false alerts

10. FEATURE IMPORTANCE AND INTERPRETATION
- Permutation importance on 172 features
- Top feature: feature_148 (importance 0.0676)
- Highest graph feature: in_degree (rank 166, importance -0.0002)
- Dataset features dominate top 20
- Graph features provide explainable risk indicators
- AML translations for compliance workflows

11. UAE/VARA RELEVANCE
- VARA mandates for VASP compliance
- Interpretable ML for regulatory reporting
- Distributed-ledger analytics capability

12. LIMITATIONS
- Public benchmark, not production data
- Coarse temporal granularity (time steps)
- Graph features may correlate with existing features
- Results may not generalize to other blockchains

13. CONCLUSION
- Graph features do not substantially improve PR-AUC
- But provide interpretable structural risk indicators
- Operational value in reducing false alerts
- Future work: subgraph analysis, real-time deployment

14. FUTURE WORK
- Elliptic2 subgraph approach
- Real-time streaming pipeline
- Multi-chain analytics
- GNN baseline for comparison

15. REFERENCES
- Weber et al. (2019) - Elliptic dataset
- VARA (2025) - AML/CFT rulebook
- FATF (2023) - Virtual assets guidance

================================================================================
