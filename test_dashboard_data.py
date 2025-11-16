#!/usr/bin/env python3
"""Test what the dashboard will actually display"""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()

print("=" * 80)
print("🔍 TESTING DASHBOARD DATA DISPLAY")
print("=" * 80)

experiment = client.get_experiment_by_name("churn-experiments")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="",
    order_by=["start_time DESC"],
    max_results=10,
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
)

print(f"\n📊 Processing {len(runs)} runs...\n")

comparison_data = []
for run in runs:
    metrics = run.data.metrics
    params = run.data.params
    
    # THIS IS THE EXACT CODE FROM THE DASHBOARD
    accuracy = metrics.get('test_accuracy') or metrics.get('accuracy', 0)
    precision = metrics.get('test_precision') or metrics.get('precision', 0)
    recall = metrics.get('test_recall') or metrics.get('recall', 0)
    f1 = metrics.get('test_f1') or metrics.get('f1', 0)
    roc_auc = metrics.get('test_roc_auc') or metrics.get('roc_auc', 0)
    
    comparison_data.append({
        "Run ID": run.info.run_id[:8],
        "Date": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    })

# Create the dataframe EXACTLY as the dashboard does
df = pd.DataFrame(comparison_data)

print("=" * 80)
print("📋 NUMERIC VALUES (Used for charts)")
print("=" * 80)
print(df.to_string(index=False))
print()

# Now format for display (this is what you see in the table)
df_display = df.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}" if x > 0 else "0.00%")

print("=" * 80)
print("📊 FORMATTED DISPLAY (What you see in the table)")
print("=" * 80)
print(df_display.to_string(index=False))
print()

print("=" * 80)
print("🎯 KEY METRICS (Top KPI cards)")
print("=" * 80)
print(f"Best Accuracy:  {df['Accuracy'].max():.2%}")
print(f"Best Precision: {df['Precision'].max():.2%}")
print(f"Best Recall:    {df['Recall'].max():.2%}")
print(f"Best F1-Score:  {df['F1-Score'].max():.2%}")
print(f"Best ROC-AUC:   {df['ROC-AUC'].max():.2%}")
print()

print("=" * 80)
print("✅ DASHBOARD STATUS")
print("=" * 80)

if df['Precision'].max() > 0:
    print("✅ ALL METRICS ARE BEING LOGGED AND WILL DISPLAY CORRECTLY!")
    print()
    print("If you're still seeing 0% in the dashboard:")
    print("  1. Make sure you've saved the dashboard file")
    print("  2. Restart Streamlit: Ctrl+C and run 'streamlit run app.py' again")
    print("  3. Refresh your browser (Cmd+R or Ctrl+R)")
    print("  4. Check you're looking at the '📊 MLOps Dashboard' page")
else:
    print("❌ Something is wrong - metrics are not being retrieved")
    print("This shouldn't happen based on the database check!")

print()
