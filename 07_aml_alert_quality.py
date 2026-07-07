# NOTEBOOK 7: AML ALERT QUALITY & OPERATIONAL ANALYSIS
# Run from project root: python 07_aml_alert_quality.py

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display needed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import networkx as nx

from load_data import load_elliptic_data, clean_and_merge, split_labeled_unlabeled

print("="*70)
print("NOTEBOOK 7: AML ALERT QUALITY & OPERATIONAL ANALYSIS")
print("="*70)

# Load data
print("\nLoading data...")
features, classes, edges = load_elliptic_data('data')
df = clean_and_merge(features, classes)
df_all, df_labeled = split_labeled_unlabeled(df)

# Graph features
processed_path = 'data/processed/graph_features_final.csv'
if os.path.exists(processed_path):
    print("Loading existing graph features...")
    graph_features = pd.read_csv(processed_path)
else:
    print("Generating graph features (~30 seconds)...")
    G = nx.DiGraph()
    G.add_nodes_from(df_all['txId'].values)
    edge_list = [(int(row['txId1']), int(row['txId2'])) for _, row in edges.iterrows()]
    G.add_edges_from(edge_list)

    graph_features = pd.DataFrame()
    graph_features['txId'] = df_all['txId'].values
    graph_features['in_degree'] = graph_features['txId'].map(dict(G.in_degree())).fillna(0).astype(int)
    graph_features['out_degree'] = graph_features['txId'].map(dict(G.out_degree())).fillna(0).astype(int)
    graph_features['total_degree'] = graph_features['in_degree'] + graph_features['out_degree']
    graph_features['pagerank'] = graph_features['txId'].map(nx.pagerank(G, alpha=0.85)).fillna(0)

    G_undirected = G.to_undirected()
    graph_features['clustering'] = graph_features['txId'].map(nx.clustering(G_undirected)).fillna(0)
    graph_features['k_core'] = graph_features['txId'].map(nx.core_number(G_undirected)).fillna(0).astype(int)

    comp_size = {}
    for comp in nx.weakly_connected_components(G):
        for node in comp:
            comp_size[node] = len(comp)
    graph_features['weak_component_size'] = graph_features['txId'].map(comp_size).fillna(0).astype(int)

    os.makedirs('data/processed', exist_ok=True)
    graph_features.to_csv(processed_path, index=False)
    print(f"Saved: {processed_path}")

# Merge and split
df_model = df_labeled.merge(graph_features, on='txId')
dataset_features = [c for c in df_model.columns if 'feature_' in c]
train_df = df_model[df_model['time_step'] <= 30]
test_df = df_model[df_model['time_step'] > 40]

# Train XGBoost
X_train = train_df[dataset_features]
y_train = train_df['label']
X_test = test_df[dataset_features]
y_test = test_df['label']

scaler = StandardScaler()
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
model.fit(scaler.fit_transform(X_train), y_train)

y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
print(f"\nTest set: {len(y_test):,} | Illicit: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# Alert budget analysis
print("\n" + "="*70)
print("ALERT BUDGET ANALYSIS")
print("="*70)

test_results = pd.DataFrame({
    'txId': test_df['txId'].values,
    'time_step': test_df['time_step'].values,
    'risk_score': y_prob,
    'true_label': y_test.values
})
test_results = test_results.sort_values('risk_score', ascending=False).reset_index(drop=True)

budgets = [0.01, 0.05, 0.10, 0.20]
alert_results = []
for budget in budgets:
    n_alerts = int(len(test_results) * budget)
    alerts = test_results.head(n_alerts)
    tp = alerts['true_label'].sum()
    fp = n_alerts - tp
    fn = y_test.sum() - tp
    precision = tp / n_alerts if n_alerts > 0 else 0
    recall = tp / y_test.sum() if y_test.sum() > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    fp_per_tp = fp/tp if tp > 0 else float('inf')
    lift = precision / y_test.mean() if y_test.mean() > 0 else 0
    alert_results.append({
        'Budget': f"{budget*100:.0f}%", 'Alerts': n_alerts, 'True Positives': int(tp), 
        'False Positives': int(fp), 'False Negatives': int(fn),
        'Precision': f"{precision:.1%}", 'Recall': f"{recall:.1%}",
        'F1-Score': f"{f1:.3f}", 'FP per TP': f"{fp_per_tp:.1f}", 'Lift over Random': f"{lift:.1f}x"
    })

alert_df = pd.DataFrame(alert_results)
print("\n" + alert_df.to_string(index=False))

os.makedirs('results', exist_ok=True)
alert_df.to_csv('results/alert_budget_analysis.csv', index=False)
print("\nSaved: results/alert_budget_analysis.csv")

# Visualization
print("\nGenerating figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
budgets_pct = [1, 5, 10, 20]
precisions = [float(x.strip('%'))/100 for x in alert_df['Precision']]
recalls = [float(x.strip('%'))/100 for x in alert_df['Recall']]
f1s = [float(x) for x in alert_df['F1-Score']]
fp_per_tps = [float(x) for x in alert_df['FP per TP']]

ax1 = axes[0,0]
ax1.bar(budgets_pct, precisions, color='#3498db', edgecolor='black', width=3)
ax1.axhline(y=y_test.mean(), color='red', linestyle='--', linewidth=2, label=f'Random = {y_test.mean():.1%}')
ax1.set_xlabel('Alert Budget (%)'); ax1.set_ylabel('Precision')
ax1.set_title('Precision at Different Alert Budgets', fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')
for i,v in enumerate(precisions): ax1.text(budgets_pct[i], v+0.01, f'{v:.1%}', ha='center', fontweight='bold')

ax2 = axes[0,1]
ax2.bar(budgets_pct, recalls, color='#e74c3c', edgecolor='black', width=3)
ax2.set_xlabel('Alert Budget (%)'); ax2.set_ylabel('Recall')
ax2.set_title('Recall at Different Alert Budgets', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i,v in enumerate(recalls): ax2.text(budgets_pct[i], v+0.01, f'{v:.1%}', ha='center', fontweight='bold')

ax3 = axes[1,0]
ax3.bar(budgets_pct, f1s, color='#2ecc71', edgecolor='black', width=3)
ax3.set_xlabel('Alert Budget (%)'); ax3.set_ylabel('F1-Score')
ax3.set_title('F1-Score at Different Alert Budgets', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for i,v in enumerate(f1s): ax3.text(budgets_pct[i], v+0.01, f'{v:.3f}', ha='center', fontweight='bold')

ax4 = axes[1,1]
ax4.bar(budgets_pct, fp_per_tps, color='#f39c12', edgecolor='black', width=3)
ax4.set_xlabel('Alert Budget (%)'); ax4.set_ylabel('False Positives per True Positive')
ax4.set_title('Analyst Workload per Confirmed Illicit', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for i,v in enumerate(fp_per_tps): ax4.text(budgets_pct[i], v+0.1, f'{v:.1f}', ha='center', fontweight='bold')

plt.suptitle('AML Alert Budget Analysis: Operational Trade-offs', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/alert_budget_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/alert_budget_analysis.png")

# Case studies
print("\n" + "="*70)
print("CASE STUDIES: TOP 5 HIGHEST-RISK TRANSACTIONS")
print("="*70)

top5 = test_results.head(5).merge(graph_features, on='txId', how='left')
for idx, row in top5.iterrows():
    print(f"\nCASE #{idx+1}: Transaction {int(row['txId'])} | Score: {row['risk_score']:.4f} | {'ILICIT' if row['true_label']==1 else 'LICIT'}")
    print(f"  In: {row['in_degree']:.0f} | Out: {row['out_degree']:.0f} | Total: {row['total_degree']:.0f} | PR: {row['pagerank']:.6f}")
    flags = []
    if row['out_degree'] > 5: flags.append("High out-degree: potential layering")
    if row['in_degree'] > 5: flags.append("High in-degree: potential accumulation")
    if row['weak_component_size'] < 100: flags.append("Small component: isolated subgraph")
    if flags:
        for f in flags: print(f"  ⚠️ {f}")
    else:
        print("  Low structural risk profile")

# Business conclusion
print("\n" + "="*70)
print("BUSINESS CONCLUSION")
print("="*70)
bb = alert_df.iloc[1]
print(f"""
RECOMMENDED: 5% alert budget ({bb['Alerts']} transactions)
  Precision: {bb['Precision']} | Recall: {bb['Recall']} | F1: {bb['F1-Score']}
  Workload: {bb['FP per TP']} false positives per true illicit
  Lift: {bb['Lift over Random']} over random

VARA RELEVANCE: Explainable structural indicators for compliance teams.
Unlike black-box GNNs, these features can be defended to regulators.
""")

print("\n" + "="*70)
print("NOTEBOOK 7 COMPLETE")
print("="*70)
print("Files: results/alert_budget_analysis.csv | figures/alert_budget_analysis.png")
