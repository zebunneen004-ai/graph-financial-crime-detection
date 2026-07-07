================================================================================
EMPLOYER BRIEF: Graph-Based Financial Crime Detection
================================================================================

PROBLEM
Financial crime is hidden in relationships between transactions, not just 
individual records. Current AML systems often miss structural patterns that 
reveal coordinated illicit activity.

WHAT I BUILT
A Python pipeline that models Bitcoin transactions as a directed graph and 
compares dataset-provided features with handcrafted graph-theoretic risk 
features.

METHODS
- Graph modeling: 203,769 nodes, 234,355 directed edges (NetworkX)
- Feature engineering: PageRank, degree, clustering, k-core, component size
- Statistical rigor: Mann-Whitney U + effect sizes before ML
- Modeling: 3 feature sets (Dataset / Graph / Combined), temporal split
- Interpretability: Permutation importance + AML translations

RESULTS
- Dataset features dominate predictive power (PR-AUC: 0.1498)
- Graph features alone are weak (PR-AUC: 0.0622)
- Combined model improves precision +5.7% with 105 fewer false alerts
- Graph features provide EXPLAINABLE risk indicators for compliance teams

BUSINESS TRANSLATION
At 80% detection rate:
- Combined model generates fewer false alerts than dataset-only
- Analysts can explain flags using structural behavior (e.g., "low degree + 
  small component size")
- Reduces investigation workload without sacrificing recall

UAE / VARA RELEVANCE
- VARA mandates real-time transaction monitoring and suspicious activity 
  reporting for licensed VASPs
- Interpretable features enable compliance officers to defend alerts to 
  regulators
- Demonstrates distributed-ledger analytics capability required by VARA 
  AML/CFT controls

KEY DIFFERENTIATOR
Unlike black-box GNNs, this approach prioritizes auditable, feature-level 
interpretability — exactly what regulators demand and courts accept.

================================================================================
