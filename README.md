# 🤖 Churn Predictor MLOps# The Hybrid AI Churn-Bot 🤖



**A complete MLOps system for predicting customer churn with explainable AI****An End-to-End MLOps + Generative AI System for Customer Retention**



Built with MLflow, SHAP, and Streamlit for production-ready churn prediction and monitoring.This project demonstrates a complete hybrid AI system combining **Predictive AI** (MLOps pipeline), **Explainable AI** (SHAP), and **Generative AI** (Google Gemini) to predict customer churn and generate personalized retention strategies.



---## 🎯 What Makes This "Hybrid AI"?



## 🎯 What This DoesThis isn't just another ML model—it's a **three-layer intelligent system**:



Predicts which customers are likely to churn (leave your service) and explains why, helping you take action to retain them.1. **🎯 Predictive AI (MLOps Foundation)**

   - Trains a RandomForest model to predict customer churn

### Key Features:   - Uses MLflow for experiment tracking and model registry

- **🔮 Churn Prediction**: RandomForest model with 78% accuracy   - Version-controlled models with production staging

- **🧠 Explainability**: SHAP analysis shows WHY customers might leave

- **📋 Retention Strategies**: Actionable recommendations based on risk factors2. **🧠 Explainable AI (SHAP Analysis)**

- **📊 MLOps Dashboard**: Track and compare multiple model training runs   - Explains *why* each customer is likely to churn

- **📡 Production Monitoring**: Real-time tracking of predictions in production   - Visual waterfall plots showing feature contributions

   - Identifies top risk factors for targeted intervention

---

3. **✨ Generative AI (Retention Strategies)**

## 📁 Project Structure   - Google Gemini generates personalized retention emails

   - Creates custom support scripts for each at-risk customer

```   - Provides immediate action plans based on churn factors

ChurnPredictorMLOps/

├── app.py                      # Main Streamlit app (churn predictions)## Project Architecture

├── train.py                    # Train and log model with MLflow

├── churn_pipeline.pkl          # Trained model (34MB)```

├── mlflow.db                   # MLflow experiment tracking databaseChurnPredictorMLOps/

├── prediction_logs.jsonl       # Production prediction logs├── 🎓 Training Pipeline (MLOps)

││   ├── train.py              # RandomForest training + MLflow registration

├── pages/                      # Multi-page Streamlit app│   ├── telco_churn.csv       # Telco Customer Churn dataset

│   ├── 1_📊_MLOps_Dashboard.py      # Compare training runs│   ├── mlflow.db             # Experiment tracking database

│   └── 2_📡_Production_Monitor.py   # Monitor predictions│   └── mlruns/               # Model artifacts

││

├── utils/├── 🤖 Hybrid AI Dashboard

│   └── prediction_logger.py    # Centralized prediction logging│   ├── app.py                # Streamlit "Mission Control" interface

││   └── .env                  # API keys (create from .env.example)

└── requirements.txt            # Python dependencies│

```└── 📦 Configuration

    ├── requirements.txt      # All dependencies

---    └── .env.example          # Template for API keys

```

## 🚀 Quick Start

## 🚀 Quick Start Guide

### 1. Install Dependencies

### Step 1: Set Up Your Environment

```bash

pip install -r requirements.txt```bash

```# Create virtual environment

python -m venv .venv

### 2. Run the Appsource .venv/bin/activate  # On Windows: .venv\Scripts\activate



```bash# Install dependencies

streamlit run app.pypip install --upgrade pip

```pip install -r requirements.txt

```

The app will open in your browser at `http://localhost:8501`

### Step 2: Configure Generative AI (Optional but Recommended)

### 3. Make Predictions

Get a **free** Google Gemini API key:

1. Fill in customer details in the sidebar1. Go to https://makersuite.google.com/app/apikey

2. Click "🔮 Predict Churn"2. Create a new API key

3. View prediction, explanation, and retention strategy3. Create `.env` file:



---```bash

cp .env.example .env

## 🎓 Training a New Model# Edit .env and add your key:

# GEMINI_API_KEY=your_actual_key_here

To retrain the model with your own data:```



```bash> 💡 **Note**: The app works without an API key, but you won't get AI-generated retention strategies.

python train.py

```### Step 3: Train the MLOps Model



This will:```bash

- ✅ Train a RandomForest model on the telco churn datasetpython train.py

- ✅ Log all metrics to MLflow (accuracy, precision, recall, F1, ROC-AUC)```

- ✅ Register the model with "champion" alias

- ✅ Save the model to `churn_pipeline.pkl`**What happens:**

- ✅ Create a new run in the MLOps Dashboard- ✅ Loads and preprocesses Telco dataset

- ✅ Trains RandomForest with 300 estimators

---- ✅ Logs to MLflow (params, metrics, artifacts)

- ✅ Registers model as `churn-predictor` Version 1

## 📊 App Features

**Expected output:**

### Main App (`app.py`)```

- **Churn Prediction**: Enter customer details, get instant churn probabilityTraining complete. Accuracy: 0.7842

- **SHAP Explanations**: Visual waterfall plots showing feature importanceSuccessfully registered model 'churn-predictor'.

- **Natural Language Summary**: Plain English explanation of top churn factorsCreated version '1' of model 'churn-predictor'.

- **Retention Strategies**: Personalized recommendations based on risk level```



### Page 1: MLOps Dashboard### Step 4: Promote Model to Production

- Compare multiple training runs

- View performance metrics across experiments**Terminal 1 - Start MLflow UI:**

- Track hyperparameter impact```bash

- Download comparison data as CSVmlflow ui --backend-store-uri sqlite:///mlflow.db

```

### Page 2: Production Monitor

- Real-time prediction tracking**Browser - Promote Model:**

- Churn rate over time1. Open http://localhost:5000

- Feature distribution monitoring2. Click **"Models"** tab

- Hourly prediction patterns3. Click **"churn-predictor"**

4. Click **Version 1**

---5. **"Stage: None"** → Select **"Transition to → Production"**

6. Confirm ✅

## 🔧 Technology Stack

### Step 5: Launch the Hybrid AI Dashboard

| Component | Technology |

|-----------|-----------|**Terminal 2 - Start Streamlit:**

| **ML Framework** | scikit-learn (RandomForest) |```bash

| **Experiment Tracking** | MLflow |streamlit run app.py

| **Explainability** | SHAP |```

| **Web App** | Streamlit |

| **Data Processing** | pandas, numpy |The dashboard opens automatically at **http://localhost:8501** 🎉

| **Visualization** | plotly, matplotlib |

## 📊 Using the Dashboard

---

### Main Features

## 📈 Model Performance

1. **📋 Customer Profile Input** (Sidebar)

Current production model:   - Fill in demographics, services, contract details

   - Click "🔮 Predict Churn"

```

Accuracy:  78.42%2. **🎯 Churn Prediction**

Precision: 58.45% (for churn class)   - Instant probability score

Recall:    64.71% (for churn class)   - Risk level indicator (Low/Moderate/High)

F1-Score:  61.42%   - Beautiful gauge visualization

ROC-AUC:   83.47%

```3. **🧠 Explainability (SHAP Analysis)**

   - See exactly *why* the model made its prediction

The model is trained on the **Telco Customer Churn** dataset with features like:   - Top factors increasing churn risk

- Customer tenure   - Top factors reducing churn risk

- Monthly charges   - Detailed waterfall plot showing feature contributions

- Contract type

- Internet service type4. **✨ AI Retention Strategy** (For high-risk customers)

- Payment method   - Personalized retention email (generated by Gemini)

- Additional services (tech support, online backup, etc.)   - Immediate action recommendations

   - Custom discount/offer suggestions

---   - Long-term loyalty strategy

   - Downloadable retention plan

## 🎯 How It Works

### Try These Scenarios

### 1. Training Pipeline (`train.py`)

```python**🔴 High Risk Customer:**

Load Data → Preprocess → Train RandomForest → Log to MLflow → Save Model- Tenure: 1-2 months

```- Contract: Month-to-month

- Monthly Charges: $80+

### 2. Prediction Flow (`app.py`)- No additional services (security, backup, etc.)

```python- Payment: Electronic check

User Input → Load Model → Predict → Generate SHAP Values → Log Prediction

```**🟢 Low Risk Customer:**

- Tenure: 60+ months

### 3. Monitoring (`Production Monitor`)- Contract: Two year

```python- Multiple bundled services

Read Logs → Aggregate Stats → Visualize Trends → Alert on Drift- Payment: Bank transfer (automatic)

```- Tech support: Yes



---## 🎓 Technical Highlights



## 📝 Key Files Explained### MLOps Pipeline

- **Framework**: scikit-learn Pipeline (preprocessing + model)

### `app.py` - Main Application- **Model**: RandomForestClassifier (300 estimators, balanced classes)

- Streamlit UI for churn predictions- **Tracking**: MLflow experiment logging

- Loads `churn_pipeline.pkl` model- **Registry**: Model versioning with lifecycle stages

- Generates SHAP explanations- **Backend**: SQLite for self-contained deployment

- Logs predictions to `prediction_logs.jsonl`

### Explainable AI

### `train.py` - Model Training- **Library**: SHAP (SHapley Additive exPlanations)

- Trains RandomForest classifier- **Method**: TreeExplainer for RandomForest

- Logs to MLflow with all metrics- **Visualization**: Waterfall plots, feature importance

- Registers model with "champion" alias- **Interpretation**: Per-prediction explanations

- Saves model as `churn_pipeline.pkl`

### Generative AI

### `mlflow.db` - Experiment Database- **Model**: Google Gemini Pro

- SQLite database storing all training runs- **Task**: Generate personalized retention strategies

- Tracks metrics, parameters, and artifacts- **Input Context**: Customer profile + SHAP factors

- Used by MLOps Dashboard- **Output**: Email templates, action plans, offers



### `churn_pipeline.pkl` - Production Model### Frontend

- Complete sklearn pipeline (preprocessing + model)- **Framework**: Streamlit (pure Python, no HTML/CSS needed)

- 34MB serialized model- **Features**: Interactive forms, real-time predictions, beautiful charts

- Used by main app for predictions- **Visualization**: Plotly gauges, matplotlib SHAP plots



---## 🌟 What Makes This Project Stand Out



## 🌐 Deployment| Feature | Traditional ML | This Hybrid AI System |

|---------|----------------|----------------------|

### Streamlit Cloud (Current)| Prediction | ✅ Yes | ✅ Yes |

| Model Versioning | ❌ Often manual | ✅ MLflow Registry |

Your app is live at:| Explainability | ❌ Black box | ✅ SHAP explanations |

```| Actionable Insights | ❌ Just numbers | ✅ AI-generated strategies |

https://churnpredictormlops-[your-id].streamlit.app| User Interface | ❌ Code/API only | ✅ Beautiful dashboard |

```| Retention Strategy | ❌ Manual work | ✅ Auto-generated by Gemini |



The app automatically deploys when you push to GitHub.## 🎯 Real-World Applications



### Local DevelopmentThis system demonstrates skills valuable for:

- **Customer Success Teams**: Proactive retention campaigns

```bash- **Support Centers**: Personalized customer outreach

# Run locally- **Product Managers**: Understanding churn drivers

streamlit run app.py- **Data Scientists**: Full MLOps workflow

- **ML Engineers**: Model deployment and monitoring

# Access at

http://localhost:8501## 📚 Learning Outcomes

```

By exploring this project, you'll understand:

---

1. **MLOps Best Practices**

## 📚 Documentation   - Experiment tracking with MLflow

   - Model registry and versioning

- **MLFLOW_STATUS.md** - Explains how MLflow is used in this project   - Production staging workflows

- **MLOPS_DEBUGGING_GUIDE.md** - Troubleshooting guide for the MLOps dashboard

- **MLOPS_DASHBOARD_FIXED.md** - Recent fixes to the dashboard2. **Explainable AI**

   - SHAP value computation

---   - Feature importance interpretation

   - Communicating model decisions

## 🎓 What You Can Learn

3. **Hybrid AI Systems**

This project demonstrates:   - Combining predictive + generative models

   - Building end-to-end intelligent applications

- ✅ **MLOps Best Practices**: Experiment tracking, model registry, versioning   - Integrating multiple AI technologies

- ✅ **Explainable AI**: SHAP for model interpretability

- ✅ **Production Monitoring**: Logging and tracking predictions in production4. **Production ML**

- ✅ **Multi-page Streamlit Apps**: Professional dashboard structure   - Preprocessing pipelines

- ✅ **Model Deployment**: From training to production   - Model serving strategies

- ✅ **Data Pipelines**: Preprocessing, feature engineering, model training   - User-friendly interfaces for ML



---## 🚀 Next Steps & Extensions



## 🔮 Future Enhancements### Beginner

- ✅ Run the complete pipeline

Potential improvements:- ✅ Test different customer profiles

- [ ] Add model retraining automation- ✅ Compare model versions in MLflow

- [ ] Implement data drift detection

- [ ] Add A/B testing framework### Intermediate

- [ ] Connect to live customer database- 🔄 Add more features to the model

- [ ] Add email integration for retention campaigns- 🔄 Experiment with hyperparameters

- [ ] Deploy model as REST API- 🔄 Try different ML algorithms (XGBoost, LightGBM)

- 🔄 Add A/B testing for retention strategies

---

### Advanced

## 📄 License- 🚀 Deploy to cloud (AWS, GCP, Azure)

- 🚀 Add real-time monitoring dashboard

MIT License - Feel free to use this project for learning and development.- 🚀 Implement feedback loop (track retention success)

- 🚀 Build REST API alongside Streamlit

---- 🚀 Add user authentication

- 🚀 Create Docker container for deployment

## 🤝 Contributing

## 🎤 Talking Points for Interviews/Presentations

This is a learning/portfolio project. Feel free to fork and modify for your own use!

*"I built a Hybrid AI system that combines three AI technologies:*

---

1. ***Predictive AI**: I trained a RandomForest model using a complete MLOps pipeline with MLflow. The model predicts customer churn with 78% accuracy and is version-controlled in a model registry with production staging.*

## 📧 Contact

2. ***Explainable AI**: I integrated SHAP analysis so stakeholders can see exactly why each customer is at risk. This transparency is crucial for building trust with business teams.*

Built as an MLOps portfolio project demonstrating end-to-end machine learning system design.

3. ***Generative AI**: For high-risk customers, I use Google Gemini to automatically generate personalized retention strategies—including custom emails and action plans—based on the specific factors driving that customer's churn risk.*

---

*The result is a complete system that not only predicts problems but also provides actionable solutions, all accessible through a beautiful Streamlit dashboard that any team member can use."*

**⚡ Quick Commands:**

## 🛠️ Tech Stack

```bash

# Train model| Component | Technology | Purpose |

python train.py|-----------|-----------|---------|

| ML Framework | scikit-learn | Model training & preprocessing |

# Run app| MLOps | MLflow | Experiment tracking, model registry |

streamlit run app.py| Explainability | SHAP | Feature importance, model interpretation |

| Generative AI | Google Gemini | Retention strategy generation |

# Install dependencies| Frontend | Streamlit | Interactive dashboard |

pip install -r requirements.txt| Visualization | Plotly, Matplotlib | Charts and graphs |

```| Data | pandas | Data manipulation |

| Storage | SQLite | Lightweight database |

**🎯 Current Status:** ✅ Fully Operational - 78% Accuracy - Production Ready

## 📖 Additional Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **SHAP Tutorials**: https://shap.readthedocs.io/
- **Google Gemini API**: https://ai.google.dev/docs
- **Streamlit Gallery**: https://streamlit.io/gallery

## 🤝 Contributing

This is an educational project demonstrating MLOps + Hybrid AI concepts. Feel free to:
- Fork and enhance
- Add new features
- Create issues for bugs
- Share your implementations

## 📄 License

Educational project - free to use and modify.

---

<div align="center">

**🤖 Built with Predictive AI, Explainable AI, and Generative AI**

*Demonstrating the future of intelligent, human-centered ML systems*

</div>
