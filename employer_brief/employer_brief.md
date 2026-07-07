# Graph-Based Financial Crime Detection
## Using Transaction Network Analysis

---

**Built by:** [Your Name]  
**Contact:** zebunneen004@gmail.com | [LinkedIn](https://linkedin.com/in/yourprofile)  
**GitHub:** github.com/zebunneen004-ai/graph-financial-crime-detection

---

## The Problem

Financial crime is hidden in **relationships between transactions**, not just individual records. Standard AML systems analyze transactions in isolation, missing the network patterns that reveal layering, mixing, and structuring behavior.

## What I Built

A Python pipeline that models Bitcoin transactions as a **directed graph** and compares dataset-provided features with **graph-theoretic risk features** to detect suspicious activity.

| Component | Details |
|:---|:---|
| **Dataset** | Elliptic Bitcoin — 203,769 transactions, 234,355 edges |
| **Graph features** | PageRank, degree, clustering, k-core, component size |
| **Models** | XGBoost, Random Forest, Logistic Regression |
| **Evaluation** | Strict temporal split (train past → test future) |
| **Primary metric** | PR-AUC (not accuracy) |
| **Statistical rigor** | Mann-Whitney U, effect sizes, Bootstrap CIs, McNemar test |

## Key Results

| Metric | Value |
|:---|:---|
| Best PR-AUC | **0.6384** (XGBoost, Dataset Only) |
| Best F1-Score | **0.5303** (XGBoost, Dataset Only) |
| Precision improvement with graph | **+5.7%** |
| False alerts reduced at 80% recall | **105 fewer** |
| Statistical significance | Bootstrap 95% CIs + McNemar exact test |
| Tests passing | 6/6 pytest |

## Operational Translation

At a **5% alert budget** (flagging top 50 transactions per 1,000 for review):

- Catches **~40% of known illicit transactions**
- Generates **2.5 false alarms per true illicit caught**
- Provides **explainable structural indicators** (connectivity, isolation, hub status)
- Unlike black-box GNNs, flags can be **inspected and defended to regulators**

## What Makes This Different

| Standard Portfolio | This Project |
|:---|:---|
| "I trained a fraud model" | "I built an interpretable AML alert system" |
| Accuracy-focused | PR-AUC and recall-focused for imbalanced data |
| Random train/test split | Temporal split (simulates real AML workflow) |
| Black-box predictions | Explainable graph features for compliance teams |
| Positive results only | Honest negative results (PageRank not significant) |

## UAE / VARA Relevance

- **VARA-licensed VASPs** must maintain AML/CFT controls including distributed-ledger tracing and transaction monitoring
- **Interpretability requirement:** VASPs must explain alerts to regulators — not just predict them
- **Real-time monitoring:** Temporal split validates the model detects future illicit activity from past patterns
- **Skill intersection:** Graph ML + AML domain knowledge + regulatory compliance — scarce in UAE market

## Tools

Python, pandas, NetworkX, scikit-learn, XGBoost, statsmodels, matplotlib

## Limitations

- Public benchmark dataset (not production UAE data)
- Graph features add interpretability but not significant predictive power
- Results may not generalize to other blockchains

---

*Academic project. Not legal or compliance advice.*
