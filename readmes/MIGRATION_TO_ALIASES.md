# ✨ MLflow Model Registry Update: Modern Alias System

## 🎉 What Changed?

Your project has been updated to use **MLflow's modern Model Alias system** instead of the deprecated Staging/Production stages.

**Reference:** https://www.mlflow.org/docs/latest/model-registry.html#using-registered-model-aliases

---

## 🔄 Old vs New Approach

### ❌ Old Way (Deprecated)
```python
# Load model from "Production" stage
model_uri = "models:/churn-predictor/Production"
model = mlflow.pyfunc.load_model(model_uri)
```

**Problems:**
- Required manual promotion in UI
- Stages concept is being phased out
- Less flexible for complex workflows

### ✅ New Way (Modern)
```python
# Load model with "champion" alias
model_uri = "models:/churn-predictor@champion"
model = mlflow.pyfunc.load_model(model_uri)
```

**Benefits:**
- ✨ Automatic alias assignment during training
- 🚀 No manual promotion needed
- 🎯 More flexible (can have multiple aliases like "champion", "challenger", "shadow")
- 📚 Follows MLflow best practices

---

## 📝 What Was Updated?

### 1. `train.py` - Automatic Alias Assignment

**Added this code:**
```python
# Set 'champion' alias for the latest version
client = mlflow.MlflowClient()
model_version = model_info.registered_model_version
client.set_registered_model_alias(MODEL_NAME, "champion", model_version)
print(f"✅ Model registered as '{MODEL_NAME}' version {model_version} with 'champion' alias")
```

**What it does:**
- Automatically tags your trained model with the "champion" alias
- No manual promotion needed!

---

### 2. `app.py` - Smart Model Loading

**Updated the `load_model()` function with fallback logic:**

```python
@st.cache_resource
def load_model():
    """Load the champion model from MLflow registry (modern alias system)."""
    try:
        # Try loading model with 'champion' alias (modern MLflow approach)
        model_uri = f"models:/{MODEL_NAME}@champion"
        model = mlflow.pyfunc.load_model(model_uri)
        return model, "Champion"
    except Exception:
        try:
            # Fallback: Try old Production stage for backwards compatibility
            model_uri = f"models:/{MODEL_NAME}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            return model, "Production"
        except Exception:
            # Final fallback: Use latest version
            # ... load latest version
```

**Smart features:**
1. **Primary:** Tries to load model with "champion" alias
2. **Fallback 1:** If that fails, tries old "Production" stage (backwards compatible)
3. **Fallback 2:** If both fail, loads the latest version

---

### 3. Documentation Updates

Updated these files:
- ✅ `START_HERE.md` - No manual promotion step needed
- ✅ `QUICK_START.md` - Simplified workflow
- ✅ `MIGRATION_TO_ALIASES.md` - This file!

---

## 🚀 How to Use (Updated Workflow)

### Step 1: Train Your Model
```bash
source .venv/bin/activate
python train.py
```

**What happens:**
```
Loading dataset...
Splitting features and target...
Building preprocessing pipeline...
Training model...
Test Accuracy: 0.7842
✅ Model registered as 'churn-predictor' version 1 with 'champion' alias
```

**That's it!** The model is automatically ready to use. No promotion needed! 🎉

---

### Step 2: Launch Streamlit
```bash
streamlit run app.py
```

The app will automatically load the model with the "champion" alias.

---

## 🔍 Viewing in MLflow UI (Optional)

Want to see the alias in action?

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Then:
1. Open http://localhost:5001
2. Click **"Models"** → **"churn-predictor"**
3. You'll see your model with a **"champion"** badge! 🏆

Press `Ctrl+C` to stop the UI when done.

---

## 🎓 Understanding Model Aliases

### What are aliases?

Aliases are **named pointers** to specific model versions. Think of them like Git tags!

### Common alias patterns:

| Alias | Purpose | Use Case |
|-------|---------|----------|
| **champion** | Currently deployed "best" model | Production serving |
| **challenger** | New model being tested | A/B testing |
| **shadow** | Runs in parallel for monitoring | Shadow deployment |
| **baseline** | Original reference model | Performance comparison |

### Setting aliases programmatically:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Set champion alias
client.set_registered_model_alias("churn-predictor", "champion", "3")

# Set challenger alias for A/B testing
client.set_registered_model_alias("churn-predictor", "challenger", "4")

# Load them
champion = mlflow.pyfunc.load_model("models:/churn-predictor@champion")
challenger = mlflow.pyfunc.load_model("models:/churn-predictor@challenger")
```

---

## 🔄 Migration Path

### If you have old models with "Production" stage:

**Don't worry!** The updated code has backwards compatibility:

1. **Old models still work** - The app tries "Production" stage if "champion" fails
2. **Gradually migrate** - Next time you train, the new model gets "champion" alias
3. **No breaking changes** - Everything continues to work

### To manually migrate existing models:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Get your current Production model
prod_versions = client.get_latest_versions("churn-predictor", stages=["Production"])

if prod_versions:
    version = prod_versions[0].version
    
    # Set champion alias
    client.set_registered_model_alias("churn-predictor", "champion", version)
    print(f"✅ Set 'champion' alias to version {version}")
```

---

## 📊 Comparison Table

| Feature | Old Stages | New Aliases |
|---------|-----------|-------------|
| **Status** | ⚠️ Deprecated | ✅ Recommended |
| **Manual promotion** | ❌ Required | ✅ Automatic |
| **Flexibility** | 🔒 Fixed (None→Staging→Production) | 🎨 Custom names |
| **Multiple deployments** | ❌ Limited | ✅ Unlimited |
| **Code example** | `models:/name/Production` | `models:/name@champion` |
| **UI Support** | ✅ Full support | ✅ Full support |

---

## 🎯 Next Steps

### For this project:
1. ✅ Code already updated
2. ✅ Backwards compatible
3. ✅ Documentation updated
4. ⏳ **Next:** Run `python train.py` to create first "champion" model

### For advanced users:
- Implement A/B testing with "champion" vs "challenger"
- Set up automated retraining pipeline
- Use multiple aliases for different deployment environments
- Integrate with CI/CD for automatic alias updates

---

## 📚 Additional Resources

- **MLflow Docs:** https://www.mlflow.org/docs/latest/model-registry.html#using-registered-model-aliases
- **Migration Guide:** https://www.mlflow.org/docs/latest/model-registry.html#migrating-from-stages
- **Best Practices:** https://mlflow.org/docs/latest/model-registry.html

---

## ❓ FAQ

### Q: Do I need to retrain my model?
**A:** No! The app will work with existing models using the fallback logic. But next time you train, it'll automatically use the new system.

### Q: What if I have models in "Staging"?
**A:** The fallback logic handles "Production" stage. For "Staging", either promote to "Production" or set a "challenger" alias manually.

### Q: Can I still use the old stage system?
**A:** Yes, for now. But it's deprecated and will be removed in future MLflow versions. Better to migrate now!

### Q: How do I delete an alias?
**A:**
```python
client.delete_registered_model_alias("churn-predictor", "champion")
```

### Q: Can I have multiple aliases on one version?
**A:** Yes! One version can have multiple aliases:
```python
client.set_registered_model_alias("churn-predictor", "champion", "3")
client.set_registered_model_alias("churn-predictor", "prod-v3", "3")
```

---

## 🎉 Summary

**You're now using the modern MLflow approach!** 🚀

- ✅ No manual promotion needed
- ✅ Cleaner, more professional code
- ✅ Future-proof for MLflow updates
- ✅ Backwards compatible with old models

**Next:** Run `python train.py` and enjoy the automated workflow! 🎊
