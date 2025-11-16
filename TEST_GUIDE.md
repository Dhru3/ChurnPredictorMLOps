# 🚀 Quick Test Guide

## Test All MLOps Features in 5 Minutes

### Prerequisites
```bash
# Ensure you're in the project directory
cd ChurnPredictorMLOps

# Ensure dependencies are installed
pip install -r requirements.txt

# Ensure you have a trained model
# If not, run: python train.py
```

---

## ✅ Test Checklist

### 1. Main App (Prediction + Explainability)
```bash
# Start the app
streamlit run app.py
```

**Test Steps**:
- [ ] Fill in customer details (or use pre-filled test scenario)
- [ ] Click "🔮 Predict Churn"
- [ ] Verify prediction appears (Yes/No)
- [ ] Check probability gauge displays correctly
- [ ] Review SHAP explainability section
- [ ] Scroll down to see AI-generated retention email
- [ ] Click "📋 Copy to Clipboard" (verify it works)

**Expected Result**: Prediction works, SHAP plots display, email generates

---

### 2. Model Comparison Dashboard
```bash
# With app still running, navigate to:
# Sidebar → 📊 MLOps Dashboard
```

**Test Steps**:
- [ ] Page loads without errors
- [ ] See list of all training runs from MLflow
- [ ] Check "Performance Metrics Over Time" chart
- [ ] Check "Hyperparameter Impact" chart (n_estimators, max_depth)
- [ ] Check "Model Registry Status" section
- [ ] Check "Best Model Summary" displays
- [ ] Click "📥 Download Comparison Data" (verify CSV downloads)

**Expected Result**: All charts display, data exports successfully

---

### 3. Production Monitoring
```bash
# Navigate to: Sidebar → 📡 Production Monitor
```

**First Time (No Logs Yet)**:
- [ ] See message "No predictions logged yet!"
- [ ] Read instructions on how to enable monitoring
- [ ] See example log format

**After Making Predictions**:
- [ ] Go back to main page (🤖 Hybrid AI Churn-Bot)
- [ ] Make 3-5 predictions with different customer profiles
- [ ] Return to Production Monitor page
- [ ] Verify predictions appear in dashboard
- [ ] Check "Key Metrics" section updates
- [ ] Check "Prediction Volume Over Time" chart
- [ ] Check "Hourly Distribution" chart
- [ ] Check "Churn Probability Distribution" chart
- [ ] Check "Risk Breakdown" (🟢 Low, 🟡 Medium, 🔴 High)
- [ ] Check "Recent Predictions" table

**Expected Result**: All predictions logged and visualized

---

### 4. A/B Testing
```bash
# Navigate to: Sidebar → 🧪 A/B Testing
```

**Prerequisites**: Need at least 2 models registered in MLflow

**Test Steps**:
- [ ] Page loads, test dataset loads successfully
- [ ] Select Champion model from dropdown
- [ ] Select Challenger model from dropdown (different from Champion)
- [ ] Click "🧪 Run A/B Test" button
- [ ] Wait for test to complete (~5-10 seconds)
- [ ] Check "Performance Comparison" table
- [ ] Check radar chart comparing models
- [ ] Check bar chart comparing metrics
- [ ] Check "Statistical Significance Test" section
- [ ] Verify McNemar's test results display
- [ ] Check test interpretation (Winner/No significant difference)
- [ ] Check contingency table
- [ ] Check confusion matrices for both models
- [ ] Click "🔄 Clear Results and Run New Test"

**Expected Result**: Statistical comparison completes, winner declared

**If only 1 model exists**:
- [ ] See warning message
- [ ] Train another model: `python train.py` (with different hyperparameters)

---

### 5. Automated Model Validation
```bash
# In terminal (separate from Streamlit):
python model_validation.py "models:/ChurnPredictor/latest"
```

**Test Steps**:
- [ ] Script runs without errors
- [ ] See "PERFORMANCE VALIDATION" section
  - [ ] Accuracy check
  - [ ] Precision check
  - [ ] Recall check
  - [ ] F1-Score check
  - [ ] ROC-AUC check
- [ ] See "FAIRNESS VALIDATION" section
  - [ ] Gender fairness analysis
  - [ ] SeniorCitizen fairness analysis
  - [ ] Disparity calculations
- [ ] See "ROBUSTNESS VALIDATION" section
  - [ ] Baseline accuracy
  - [ ] Performance under noise (3 levels)
- [ ] See "MODEL VALIDATION REPORT" summary
- [ ] See "RECOMMENDATIONS" section
- [ ] Check exit code:
  - Exit code 0 = PASSED ✅
  - Exit code 1 = FAILED ❌

**Expected Result**: Validation completes, report generated

---

### 6. CI/CD Pipeline (Optional - Requires GitHub)
```bash
# Check the pipeline configuration:
cat .github/workflows/mlops.yml
```

**What it does**:
1. Code quality checks (flake8, black)
2. Run tests (pytest)
3. Train model
4. Validate model
5. Deploy to staging
6. Manual approval
7. Deploy to production
8. Monitor for drift

**To test** (requires GitHub):
- [ ] Push code to GitHub
- [ ] Check Actions tab in GitHub repository
- [ ] Verify pipeline runs automatically

---

## 🐛 Troubleshooting

### Issue: "No module named 'pages.production_monitor'"
**Solution**: The import has a fallback, should work anyway. If not, restart Streamlit.

### Issue: "No predictions logged yet!" in Production Monitor
**Solution**: Make predictions in the main app first, then return to monitor page.

### Issue: "Model not found" in A/B Testing
**Solution**: Register models using `train.py` or MLflow UI.

### Issue: "Test data not found" in A/B Testing
**Solution**: Ensure `Telco-Customer-Churn.csv` is in project root.

### Issue: Validation script fails
**Solution**: 
```bash
# Check if MLflow has models
python -c "from mlflow.tracking import MlflowClient; print(MlflowClient().search_registered_models())"

# If empty, train a model:
python train.py
```

---

## 📊 Expected Test Results Summary

| Feature | Status | Time |
|---------|--------|------|
| Main App Prediction | ✅ Working | <5s per prediction |
| Model Comparison | ✅ Working | Instant |
| Production Monitor | ✅ Working | Instant (after predictions) |
| A/B Testing | ✅ Working | ~10s per test |
| Model Validation | ✅ Working | ~30s |
| CI/CD Pipeline | ✅ Configured | Runs on GitHub push |

---

## 🎥 Demo Flow for Professor

**5-Minute Demo**:

1. **[30 sec]** Show main app → Make prediction → Show SHAP explanation
2. **[1 min]** Model Comparison Dashboard → Show multiple experiments
3. **[1 min]** Production Monitor → Show prediction logs and drift detection
4. **[1.5 min]** A/B Testing → Run test, show McNemar's results
5. **[1 min]** Terminal: Run validation script, show report

**Total: 5 minutes** ⏱️

---

## ✅ Final Checklist Before Showing Professor

- [ ] All 5 MLOps pages work correctly
- [ ] At least 2-3 models trained and registered
- [ ] Made 10+ predictions to populate monitoring dashboard
- [ ] Ran A/B test successfully
- [ ] Ran validation script successfully
- [ ] Read `MLOPS_FEATURES.md` for talking points
- [ ] (Optional) Deployed to Streamlit Cloud
- [ ] (Optional) Showed CI/CD pipeline in GitHub Actions

---

**You're ready to impress your professor! 🎓✨**

**Good luck! 🚀**
