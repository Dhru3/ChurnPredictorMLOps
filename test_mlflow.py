#!/usr/bin/env python3
"""Test script to verify MLflow is working correctly."""
import sys
from pathlib import Path

print("=" * 60)
print("🔍 MLFLOW STATUS CHECK")
print("=" * 60)

# Check imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    print("✅ MLflow imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MLflow: {e}")
    sys.exit(1)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"

if not TRACKING_DB.exists():
    print(f"❌ MLflow database not found at: {TRACKING_DB}")
    sys.exit(1)

print(f"✅ MLflow database found: {TRACKING_DB}")
print(f"   Size: {TRACKING_DB.stat().st_size / 1024:.1f} KB")

# Connect to MLflow
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()

print("\n" + "=" * 60)
print("📊 EXPERIMENTS")
print("=" * 60)

experiments = client.search_experiments()
print(f"Total experiments: {len(experiments)}")
for exp in experiments:
    print(f"\n  📁 {exp.name}")
    print(f"     ID: {exp.experiment_id}")
    print(f"     Lifecycle: {exp.lifecycle_stage}")

print("\n" + "=" * 60)
print("🏃 TRAINING RUNS")
print("=" * 60)

# Get churn-experiments
experiment = client.get_experiment_by_name("churn-experiments")

if not experiment:
    print("❌ 'churn-experiments' experiment not found!")
    print("   Run 'python train.py' to create it.")
    sys.exit(1)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=50
)

print(f"Total runs: {len(runs)}")

if not runs:
    print("❌ No runs found! Run 'python train.py' first.")
    sys.exit(1)

print("\n" + "-" * 60)
for i, run in enumerate(runs, 1):
    print(f"\n🏃 Run #{i}")
    print(f"   ID: {run.info.run_id}")
    print(f"   Status: {run.info.status}")
    
    from datetime import datetime
    start_time = datetime.fromtimestamp(run.info.start_time / 1000)
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n   📊 Metrics ({len(run.data.metrics)}):")
    if run.data.metrics:
        for key, value in sorted(run.data.metrics.items()):
            print(f"      - {key}: {value:.4f}")
    else:
        print("      ⚠️ No metrics logged!")
    
    print(f"\n   ⚙️ Parameters ({len(run.data.params)}):")
    if run.data.params:
        for key, value in sorted(run.data.params.items()):
            print(f"      - {key}: {value}")
    else:
        print("      ⚠️ No parameters logged!")

print("\n" + "=" * 60)
print("📦 REGISTERED MODELS")
print("=" * 60)

registered_models = client.search_registered_models()
print(f"Total registered models: {len(registered_models)}")

for model in registered_models:
    print(f"\n  📦 {model.name}")
    print(f"     Description: {model.description or 'None'}")
    
    versions = client.search_model_versions(f"name='{model.name}'")
    print(f"     Versions: {len(versions)}")
    
    for version in versions:
        aliases = getattr(version, 'aliases', [])
        alias_str = f" [{', '.join(aliases)}]" if aliases else ""
        print(f"       - v{version.version}: {version.status}{alias_str}")

print("\n" + "=" * 60)
print("✅ SUMMARY")
print("=" * 60)

latest_run = runs[0]
has_all_metrics = all(
    key in latest_run.data.metrics 
    for key in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
)

if has_all_metrics:
    print("✅ Latest run has all 5 metrics logged correctly")
    print("\n📊 Latest Model Performance:")
    print(f"   - Accuracy:  {latest_run.data.metrics.get('test_accuracy', 0):.2%}")
    print(f"   - Precision: {latest_run.data.metrics.get('test_precision', 0):.2%}")
    print(f"   - Recall:    {latest_run.data.metrics.get('test_recall', 0):.2%}")
    print(f"   - F1-Score:  {latest_run.data.metrics.get('test_f1', 0):.2%}")
    print(f"   - ROC-AUC:   {latest_run.data.metrics.get('test_roc_auc', 0):.2%}")
else:
    print("⚠️ Latest run is missing some metrics")
    print("   Run 'python train.py' again to log all metrics")

print("\n" + "=" * 60)
print("🎯 MLOPS DASHBOARD STATUS")
print("=" * 60)

if len(runs) > 0 and has_all_metrics:
    print("✅ MLOps Dashboard should work correctly!")
    print("   Run: streamlit run app.py")
    print("   Then navigate to: 📊 MLOps Dashboard page")
else:
    print("⚠️ MLOps Dashboard may not display correctly")
    print("   Solution: Run 'python train.py' to create a complete training run")

print("\n" + "=" * 60)
