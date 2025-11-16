# 🤖 Churn Predictor MLOps

Predict customer churn with explainable AI. Built with MLflow, SHAP, and Streamlit.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 What It Does

- **Predict Churn**: 78% accuracy RandomForest model
- **Explain Why**: SHAP visualizations show top factors
- **Track Models**: MLflow dashboard compares training runs
- **Monitor Production**: Real-time prediction tracking

## 📁 Structure

```
├── app.py                          # Main prediction app
├── train.py                        # Train & log model
├── churn_pipeline.pkl              # Trained model
├── pages/
│   ├── 1_📊_MLOps_Dashboard.py     # Compare runs
│   └── 2_📡_Production_Monitor.py  # Track predictions
```

## 🎓 Training

```bash
python train.py
```

Logs accuracy, precision, recall, F1, and ROC-AUC to MLflow.

## 🔧 Tech Stack

MLflow • Streamlit • scikit-learn • SHAP • plotly

## 📈 Performance

- **Accuracy**: 78.42%
- **Precision**: 58.45%
- **Recall**: 64.71%
- **ROC-AUC**: 83.47%
