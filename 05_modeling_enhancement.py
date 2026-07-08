# ============================================================
# NOTEBOOK 5 ENHANCEMENT: HYPERPARAMETER TUNING + SHAP + MODEL SAVE
# Standalone script — run from project root
# ============================================================

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score
from scipy.stats import uniform, randint
import xgboost as xgb
import joblib
import shap

from load_data import load_elliptic_data, clean_and_merge, split_labeled_unlabeled

print("="*70)
print("NOTEBOOK 5 ENHANCEMENT: HYPERPARAMETER TUNING + SHAP + MODEL SAVE")
print("="*70)

# Load data
print("\nLoading data...")
features, classes, edges = load_elliptic_data('data')
df = clean_and_merge(features, classes)
df_all, df_labeled = split_labeled_unlabeled(df)

# Load graph features
graph_features = pd.read_csv('data/processed/graph_features_final.csv')

# Merge
df_model = df_labeled.merge(graph_features, on='txId')

# Feature sets
dataset_features = [c for c in df_model.columns if 'feature_' in c]

# Temporal split
train_df = df_model[df_model['time_step'] <= 30]
test_df = df_model[df_model['time_step'] > 40]

# Prepare data
X_train = train_df[dataset_features]
y_train = train_df['label']
X_test = test_df[dataset_features]
y_test = test_df['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train illicit: {y_train.sum():,} | Test illicit: {y_test.sum():,}")

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

print("\n" + "="*70)
print("HYPERPARAMETER TUNING: XGBoost")
print("="*70)

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

print("\nRunning RandomizedSearchCV (20 iterations, 5-fold CV)...")
print("This may take 3-5 minutes...")

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=20,
    scoring='average_precision',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV PR-AUC: {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_
y_prob_tuned = best_model.predict_proba(X_test_scaled)[:, 1]

pr_auc_tuned = average_precision_score(y_test, y_prob_tuned)
print(f"Test set PR-AUC (tuned): {pr_auc_tuned:.4f}")

# Compare to default
pr_auc_default = 0.6384
improvement = (pr_auc_tuned - pr_auc_default) / pr_auc_default * 100
print(f"Improvement over default: {improvement:+.1f}%")

# Save tuned model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/xgboost_tuned.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("\nSaved: models/xgboost_tuned.pkl")
print("Saved: models/scaler.pkl")

# Save results
results_df = pd.DataFrame({
    'model': ['XGBoost (default)', 'XGBoost (tuned)'],
    'pr_auc': [pr_auc_default, pr_auc_tuned],
    'improvement_pct': [0, improvement]
})
results_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)
print("Saved: results/hyperparameter_tuning_results.csv")

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\n" + "="*70)
print("SHAP EXPLAINABILITY ANALYSIS")
print("="*70)

print("\nCreating SHAP explainer...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
print("Generating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=dataset_features, max_display=20, show=False)
plt.title('SHAP Feature Importance: Impact on Model Predictions', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/shap_summary.png")

# Dependence plot for top feature
print("Generating SHAP dependence plot...")
top_feature_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-1]
top_feature_name = dataset_features[top_feature_idx]
plt.figure(figsize=(10, 6))
shap.dependence_plot(top_feature_name, shap_values, X_test_scaled, feature_names=dataset_features, show=False)
plt.title(f'SHAP Dependence: {top_feature_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/shap_dependence_top.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/shap_dependence_top.png")

# Force plot for first illicit transaction
print("Generating SHAP force plot...")
illicit_indices = np.where(y_test.values == 1)[0]
if len(illicit_indices) > 0:
    illicit_idx = illicit_indices[0]
    plt.figure(figsize=(20, 4))
    shap.force_plot(explainer.expected_value, shap_values[illicit_idx], X_test_scaled[illicit_idx], 
                    feature_names=dataset_features, matplotlib=True, show=False)
    plt.title(f'SHAP Force Plot: Transaction {test_df.iloc[illicit_idx]["txId"]} (ILICIT)', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/shap_force_illicit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/shap_force_illicit.png")

# SHAP values DataFrame
shap_df = pd.DataFrame({
    'feature': dataset_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 10 SHAP features:")
print(shap_df.head(10).to_string(index=False))
shap_df.to_csv('results/shap_feature_importance.csv', index=False)
print("\nSaved: results/shap_feature_importance.csv")

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "="*70)
print("ENHANCEMENT COMPLETE")
print("="*70)
print("\nNew files created:")
print("  • models/xgboost_tuned.pkl (tuned model)")
print("  • models/scaler.pkl (feature scaler)")
print("  • figures/shap_summary.png (SHAP summary plot)")
print("  • figures/shap_dependence_top.png (SHAP dependence plot)")
print("  • figures/shap_force_illicit.png (SHAP force plot)")
print("  • results/shap_feature_importance.csv (SHAP values)")
print("  • results/hyperparameter_tuning_results.csv (tuning results)")
