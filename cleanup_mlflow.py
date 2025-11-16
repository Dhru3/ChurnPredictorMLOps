#!/usr/bin/env python3
"""
Clean up old MLflow runs that only have partial metrics.
This will help the MLOps dashboard show better data.
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()

print("=" * 70)
print("🧹 MLFLOW CLEANUP UTILITY")
print("=" * 70)

# Get experiment
experiment = client.get_experiment_by_name("churn-experiments")
if not experiment:
    print("❌ No 'churn-experiments' found!")
    exit(1)

# Get all runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=100
)

print(f"\n📊 Found {len(runs)} total runs\n")

# Analyze each run
incomplete_runs = []
complete_runs = []

REQUIRED_METRICS = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']

for run in runs:
    run_id = run.info.run_id
    metrics = run.data.metrics
    start_time = datetime.fromtimestamp(run.info.start_time / 1000)
    
    # Check if run has all required metrics
    has_all_metrics = all(metric in metrics for metric in REQUIRED_METRICS)
    
    if has_all_metrics:
        complete_runs.append(run)
        status = "✅ COMPLETE"
    else:
        incomplete_runs.append(run)
        status = "⚠️ INCOMPLETE"
    
    print(f"{status} - Run {run_id[:8]} - {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"         Metrics: {', '.join(metrics.keys())}")

print("\n" + "=" * 70)
print(f"✅ Complete runs (all 5 metrics): {len(complete_runs)}")
print(f"⚠️ Incomplete runs (missing metrics): {len(incomplete_runs)}")
print("=" * 70)

if incomplete_runs:
    print("\n🤔 What to do?")
    print("\nOption 1: Keep old runs (for history)")
    print("  - Old runs will show 0% for missing metrics")
    print("  - This is OK - shows your progress over time")
    
    print("\nOption 2: Delete old incomplete runs")
    print("  - Dashboard will only show complete runs")
    print("  - You lose historical data")
    print("  - Cleaner dashboard display")
    
    response = input("\n❓ Delete incomplete runs? (yes/no): ").lower().strip()
    
    if response == 'yes':
        print("\n🗑️ Deleting incomplete runs...")
        for run in incomplete_runs:
            run_id = run.info.run_id
            try:
                client.delete_run(run_id)
                print(f"   ✅ Deleted run {run_id[:8]}")
            except Exception as e:
                print(f"   ❌ Failed to delete {run_id[:8]}: {e}")
        
        print(f"\n✅ Cleanup complete! Deleted {len(incomplete_runs)} incomplete runs.")
        print(f"💾 Kept {len(complete_runs)} complete runs.")
    else:
        print("\n📊 No changes made. Old runs kept for historical tracking.")
        print("\nℹ️ The MLOps Dashboard will show:")
        print(f"   - {len(complete_runs)} complete runs with all metrics")
        print(f"   - {len(incomplete_runs)} old runs with only accuracy")

else:
    print("\n🎉 All runs are complete! No cleanup needed.")

print("\n" + "=" * 70)
print("💡 TIP: Run 'python train.py' to create more complete runs")
print("=" * 70)
