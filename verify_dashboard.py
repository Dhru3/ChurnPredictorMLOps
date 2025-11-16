#!/usr/bin/env python3
"""Quick verification of what the MLOps Dashboard should show."""

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()

print("=" * 70)
print("🔍 MLOPS DASHBOARD VERIFICATION")
print("=" * 70)

# Get experiment
experiment = client.get_experiment_by_name("churn-experiments")
if not experiment:
    print("❌ No experiment found!")
    exit(1)

# Get all ACTIVE runs (same as dashboard)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=50,
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY  # Only active runs
)

print(f"\n📊 Total runs found: {len(runs)}\n")

complete_runs = 0
incomplete_runs = 0

for i, run in enumerate(runs, 1):
    run_id = run.info.run_id[:8]
    metrics = run.data.metrics
    start_time = datetime.fromtimestamp(run.info.start_time / 1000)
    
    # Check for new metrics (test_*)
    has_new_metrics = all([
        'test_accuracy' in metrics,
        'test_precision' in metrics,
        'test_recall' in metrics,
        'test_f1' in metrics,
        'test_roc_auc' in metrics
    ])
    
    # Check for old metrics (no prefix)
    has_old_metrics = all([
        'accuracy' in metrics,
        'precision' in metrics,
        'recall' in metrics,
        'f1' in metrics,
        'roc_auc' in metrics
    ])
    
    has_all_metrics = has_new_metrics or has_old_metrics
    
    if has_all_metrics:
        complete_runs += 1
        status = "✅ COMPLETE"
        color = "\033[92m"
    else:
        incomplete_runs += 1
        status = "⚠️ PARTIAL"
        color = "\033[93m"
    
    reset = "\033[0m"
    
    print(f"{color}Run #{i}: {status}{reset}")
    print(f"  ID: {run_id}")
    print(f"  Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Metrics ({len(metrics)}):")
    
    for key in sorted(metrics.keys()):
        print(f"    - {key}: {metrics[key]:.4f}")
    
    print()

print("=" * 70)
print(f"✅ Complete runs: {complete_runs}")
print(f"⚠️ Partial runs: {incomplete_runs}")
print(f"📊 Total: {len(runs)}")
print("=" * 70)

if complete_runs > 0:
    print(f"\n🎉 SUCCESS! The dashboard should show {complete_runs} complete run(s)")
    print(f"   Latest complete run metrics:")
    
    # Find first complete run
    for run in runs:
        metrics = run.data.metrics
        has_new = all(k in metrics for k in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc'])
        has_old = all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        
        if has_new or has_old:
            if 'test_accuracy' in metrics:
                print(f"   - Accuracy:  {metrics['test_accuracy']:.2%}")
                print(f"   - Precision: {metrics['test_precision']:.2%}")
                print(f"   - Recall:    {metrics['test_recall']:.2%}")
                print(f"   - F1-Score:  {metrics['test_f1']:.2%}")
                print(f"   - ROC-AUC:   {metrics['test_roc_auc']:.2%}")
            else:
                print(f"   - Accuracy:  {metrics['accuracy']:.2%}")
                print(f"   - Precision: {metrics['precision']:.2%}")
                print(f"   - Recall:    {metrics['recall']:.2%}")
                print(f"   - F1-Score:  {metrics['f1']:.2%}")
                print(f"   - ROC-AUC:   {metrics['roc_auc']:.2%}")
            break
else:
    print("\n⚠️ No complete runs found!")
    print("   Solution: Run 'python train.py' to create a complete run")

print("\n" + "=" * 70)
print("🚀 To see the dashboard: streamlit run app.py")
print("=" * 70)
