# MLOps Dashboard Debugging Guide

## Current State of Your MLflow Database

I checked your `mlflow.db` and here's what I found:

### ✅ Experiments:
```
1. Default (ID: 0) - Active
2. churn-experiments (ID: 1) - Active ✅ This is yours!
```

### ✅ Training Runs: **5 RUNS FOUND!**
```
Run 1: 9299919a (FINISHED) - Experiment: churn-experiments
Run 2: 646e577b (FINISHED) - Experiment: churn-experiments  
Run 3: fa7cfdbd (RUNNING)  - Experiment: churn-experiments
Run 4: bfaa621d (FINISHED) - Experiment: churn-experiments
Run 5: ccf2d61b (FINISHED) - Experiment: churn-experiments ← Latest!
```

### ✅ Latest Run Metrics (from Run 5):
```
test_accuracy:  78.42%
test_precision: 58.45%
test_recall:    64.71%
test_f1:        61.42%
test_roc_auc:   83.47%
```

### ✅ Latest Run Parameters:
```
model_type: RandomForest
n_estimators: 300
max_depth: None
min_samples_split: 2
min_samples_leaf: 2
```

## What I Added to Fix the Dashboard

### 1. **Registered Models Section**
At the top of the dashboard, you'll now see:
- 🏆 All registered models (like "churn-predictor")
- Model versions
- Which version has the "champion" alias
- When each version was created

### 2. **Debug Information**
In the sidebar, you'll see:
- Total experiments found
- Number of non-default experiments
- Selected experiment ID
- Number of runs found

### 3. **Better Error Handling**
- Changed sorting from `metrics.test_accuracy DESC` to `start_time DESC`
  - This avoids errors if a metric is missing
- Increased max_results from 20 to 50 runs
- Added try/catch blocks with helpful error messages

### 4. **Run Status Display**
The dashboard will show ALL 5 of your runs, including:
- The one that's marked as "RUNNING" (Run 3)
- All 4 completed runs

## What You Should See Now

### Top Section: Registered Models
```
🏆 Registered Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 churn-predictor (expanded)
   Description: [Your model description]
   Total Versions: [Number of versions]
   
   | Version | Status | Aliases  | Created |
   |---------|--------|----------|---------|
   | 5       | READY  | champion | [time]  |
   | 4       | READY  |          | [time]  |
   | ...     | ...    | ...      | ...     |
```

### Main Section: Training Runs
```
📈 Training Runs Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Select Experiment: [churn-experiments ▼]

📊 Found 5 run(s)

🎯 Key Performance Indicators:
┌─────────────┬────────────┬────────────┬────────────┬────────────┐
│ 🏆 Total    │ 🎯 Best    │ 📊 Best    │ 🔍 Best    │ ⚡ Best    │
│ Models: 5   │ Acc: 78.4% │ Prec: 58.5%│ Rec: 64.7% │ F1: 61.4%  │
└─────────────┴────────────┴────────────┴────────────┴────────────┘

📋 All Trained Models:
| Run ID   | Date       | Accuracy | Precision | Recall | F1    | ROC-AUC |
|----------|------------|----------|-----------|--------|-------|---------|
| ccf2d61b | 2025-11-16 | 78.42%   | 58.45%    | 64.71% | 61.42%| 83.47%  |
| bfaa621d | 2025-11-16 | 78.42%   | [old run] | ...    | ...   | ...     |
| fa7cfdbd | 2025-11-16 | RUNNING  | ...       | ...    | ...   | ...     |
| 646e577b | 2025-11-15 | 78.42%   | [old run] | ...    | ...   | ...     |
| 9299919a | 2025-11-15 | 78.42%   | [old run] | ...    | ...   | ...     |

[Charts showing performance evolution]
```

## Why You Might See Some Zeros

**Old runs** (runs 1-4) were logged with only `accuracy` metric.
**New runs** (run 5+) are logged with all 5 metrics.

The dashboard will show:
- ✅ Full metrics for new runs (test_accuracy, test_precision, etc.)
- ⚠️ Only accuracy for old runs (others show as 0% or N/A)

## How to Get All Metrics for All Runs

Two options:

### Option 1: Accept Mixed Data
- Keep the 5 existing runs
- New runs will have full metrics
- Old runs show what data they have

### Option 2: Fresh Start (Clear and Retrain)
```bash
# Backup current database
cp mlflow.db mlflow.db.backup

# Delete the database
rm mlflow.db

# Retrain to create fresh runs with all metrics
python train.py
```

This gives you a clean slate with all 5 metrics for all runs.

## Sidebar Debug Info

You should now see in the sidebar:

```
🔍 Debug Info:
- Total experiments: 2
- Non-default experiments: 1

📂 Experiment: churn-experiments
- ID: 1

📊 Found 5 run(s)
```

This confirms everything is connected properly!

## On Streamlit Cloud

Your `mlflow.db` file (228 KB) is committed to git, so:
- ✅ Streamlit Cloud will have the database
- ✅ All 5 runs will be available
- ✅ Dashboard should show all data

However, if you want to retrain on Streamlit Cloud:
- The filesystem is ephemeral (resets on redeploy)
- Local training is recommended
- Commit `mlflow.db` to git after retraining locally

## Expected Output

### Metrics Summary:
- 🏆 Total Models: **5**
- 🎯 Best Accuracy: **78.4%**
- 📊 Best Precision: **58.5%** (from latest run)
- 🔍 Best Recall: **64.7%** (from latest run)
- ⚡ Best F1: **61.4%** (from latest run)

### Charts:
- Performance evolution over time (all 5 runs)
- Metric comparisons
- Hyperparameter analysis

## What to Check

1. **Open your app** on Streamlit Cloud
2. **Navigate to** "📊 MLOps Dashboard"
3. **Look for**:
   - Sidebar debug info showing "Found 5 run(s)"
   - Registered models section at top
   - Table with 5 rows of training runs
   - Latest run showing all 5 metrics as percentages

If you still see issues, the sidebar debug info will tell us exactly what's being detected!

## Quick Verification Command

To verify locally what the dashboard will see:

```bash
sqlite3 mlflow.db "
SELECT 'Total Experiments:' as label, COUNT(*) as count FROM experiments
UNION ALL
SELECT 'Total Runs:', COUNT(*) FROM runs
UNION ALL  
SELECT 'Churn-Experiments Runs:', COUNT(*) FROM runs WHERE experiment_id='1';
"
```

Expected output:
```
Total Experiments:       2
Total Runs:              5
Churn-Experiments Runs:  5
```

Your dashboard is now fully instrumented and should show everything! 🚀
