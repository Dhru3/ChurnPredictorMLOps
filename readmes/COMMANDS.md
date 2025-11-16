# 🚀 Quick Commands Reference

## Your Complete Workflow (Copy-Paste Ready)

### 1️⃣ First Time Setup (One-time only)
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Set Up Groq API Key (One-time only)
```bash
# Create .env file from template
cp .env.example .env

# Edit the file and add your key
open -e .env
# Add: GROQ_API_KEY=gsk_your_key_here
```

Get your FREE key: https://console.groq.com/

---

## Daily Workflow

### Train/Retrain Model
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps
source .venv/bin/activate
python train.py
```

**Output you want to see:**
```
✅ Model registered as 'churn-predictor' version X with 'champion' alias
```

---

### Launch Streamlit Dashboard
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps
source .venv/bin/activate
streamlit run app.py
```

**Opens:** http://localhost:8501

**Stop:** Press `Ctrl+C` in terminal

---

### View MLflow UI (Optional)
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**Opens:** http://localhost:5001

**Stop:** Press `Ctrl+C` in terminal

---

## Troubleshooting Commands

### Port Already in Use
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9

# Kill process on port 8501 (Streamlit)
lsof -ti:8501 | xargs kill -9
```

### Check Python Environment
```bash
source .venv/bin/activate
which python
# Should show: /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps/.venv/bin/python
```

### List Installed Packages
```bash
source .venv/bin/activate
pip list
```

### Reinstall Dependencies
```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Check if Model Exists
```bash
source .venv/bin/activate
python -c "import mlflow; client = mlflow.MlflowClient('sqlite:///mlflow.db'); print(client.search_model_versions('name=\"churn-predictor\"'))"
```

---

## File Locations

| File | Purpose | Action |
|------|---------|--------|
| `train.py` | Train model | Run: `python train.py` |
| `app.py` | Streamlit dashboard | Run: `streamlit run app.py` |
| `telco_churn.csv` | Training data | Don't modify |
| `mlflow.db` | Model registry | Auto-generated |
| `.env` | API keys | Add your GROQ_API_KEY |
| `requirements.txt` | Dependencies | Run: `pip install -r requirements.txt` |

---

## Documentation Files

| File | What's Inside |
|------|---------------|
| `START_HERE.md` | 📖 Complete beginner walkthrough |
| `QUICK_START.md` | ⚡ 5-minute setup guide |
| `README.md` | 📚 Full project documentation |
| `WHAT_YOU_BUILT.md` | 🎯 Feature showcase & pitch guide |
| `MIGRATION_TO_ALIASES.md` | 🔄 MLflow modern approach explained |
| `COMMANDS.md` | 📋 This file - Quick command reference |

---

## URLs to Remember

| Service | URL | When to Use |
|---------|-----|-------------|
| Streamlit Dashboard | http://localhost:8501 | Main app interface |
| MLflow UI | http://localhost:5001 | View experiments & models |
| Groq Console | https://console.groq.com/ | Get API key |

---

## One-Liner Shortcuts

### Full Reset and Start
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps && source .venv/bin/activate && python train.py && streamlit run app.py
```

### Just Launch App (if model exists)
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps && source .venv/bin/activate && streamlit run app.py
```

### Train and View in MLflow
```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps && source .venv/bin/activate && python train.py && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## Git Commands (If Using Version Control)

### Initial Commit
```bash
git init
git add .
git commit -m "Initial commit: Hybrid AI Churn Predictor"
```

### Ignore Generated Files
Your `.gitignore` already includes:
- `.venv/`
- `mlflow.db`
- `mlruns/`
- `.env`

### Push to GitHub
```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

---

## Python Code Snippets

### Check Model in Python
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model with champion alias
model = mlflow.pyfunc.load_model("models:/churn-predictor@champion")
print("Model loaded successfully!")
```

### Make Prediction in Python
```python
import pandas as pd
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/churn-predictor@champion")

# Sample customer
customer = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": ["No"],
    "Partner": ["No"],
    "Dependents": ["No"],
    "tenure": [2],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [85.0],
    "TotalCharges": [170.0]
})

prediction = model.predict(customer)
print(f"Churn probability: {prediction[0]:.2%}")
```

---

## Environment Variables

### Required
```bash
GROQ_API_KEY=gsk_your_actual_key_here
```

### Optional (with defaults)
```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MODEL_NAME=churn-predictor
```

---

## Performance Tips

### Speed Up Model Loading
The app uses `@st.cache_resource` to cache the model. First load takes ~5 seconds, subsequent loads are instant!

### Reduce Training Time
Edit `train.py` line 78:
```python
# Current (slow but accurate)
n_estimators=300

# Faster for testing
n_estimators=100
```

### Reduce Data Size for Testing
Edit `train.py` after loading data:
```python
df = df.sample(n=1000, random_state=42)  # Use only 1000 rows
```

---

## System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 500MB for project + dependencies
- **Internet**: Required for Groq API calls

---

## Quick Debugging Checklist

**App won't start?**
- [ ] Virtual environment activated? (`source .venv/bin/activate`)
- [ ] Dependencies installed? (`pip install -r requirements.txt`)
- [ ] Port 8501 free? (`lsof -ti:8501 | xargs kill -9`)

**Model not found?**
- [ ] Trained model? (`python train.py`)
- [ ] Check MLflow: `ls mlflow.db mlruns/`

**AI emails not generating?**
- [ ] `.env` file exists with `GROQ_API_KEY`?
- [ ] Internet connection working?
- [ ] API key valid? Test at https://console.groq.com/

**Training fails?**
- [ ] Dataset present? (`ls telco_churn.csv`)
- [ ] Sklearn version compatible? (`pip install scikit-learn>=1.3.0`)

---

## 🎉 Most Common Command

**Just want to run the app?**

```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps && source .venv/bin/activate && streamlit run app.py
```

**That's it!** 🚀

---

## 📞 Where to Get Help

1. **Read documentation**
   - START_HERE.md for walkthrough
   - README.md for full docs
   - This file for commands

2. **Check MLflow docs**
   - https://mlflow.org/docs/latest/

3. **Check Streamlit docs**
   - https://docs.streamlit.io/

4. **Check Groq docs**
   - https://console.groq.com/docs

---

**Last Updated:** November 16, 2025
**Project:** Hybrid AI Churn Predictor
**Author:** Dhruti Purushotham
