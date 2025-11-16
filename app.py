#!/usr/bin/env python3
"""
Hybrid AI Churn-Bot: Mission Control Dashboard
Combines MLOps (MLflow model) with Generative AI for customer retention
"""
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Import prediction logging from monitoring dashboard
try:
    from pages.production_monitor import save_prediction_log
except (ImportError, ModuleNotFoundError):
    # Fallback if module not available
    def save_prediction_log(*args, **kwargs):
        """Dummy function when production_monitor is not available"""
        pass

# Page configuration
st.set_page_config(
    page_title="Hybrid AI Churn-Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for stunning styling
st.markdown("""
    <style>
    @impor        # SHAP waterfall plot
        with st.expander("📊 Advanced: SHAP Waterfall Plot (Technical View)", expanded=False):
            st.info("This plot shows how each feature pushed the prediction away from the baseline.")
            fig = plot_shap_waterfall(explainer, shap_values, feature_names)
            st.pyplot(fig)('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .stAlert > div {
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .explanation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stay-reason {
        background: linear-gradient(135deg, #d4edda 0%, #a3d9a5 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: #155724;
        font-weight: 500;
    }
    
    .churn-reason {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c2c7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        color: #721c24;
        font-weight: 500;
    }
    
    .email-box {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe4b3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ffa500;
        margin: 1rem 0;
        color: #856404;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    .factor-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
MODEL_NAME = "churn-predictor"

# Initialize MLflow
@st.cache_resource
def init_mlflow():
    """Initialize MLflow connection and client."""
    import os
    # Use relative path for Streamlit Cloud compatibility
    tracking_uri = f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}"
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()

@st.cache_resource
def load_model():
    """Load the best model from MLflow runs (direct file loading for Streamlit Cloud)."""
    import glob
    import os
    
    try:
        # Find all model.pkl files in mlruns
        project_root = os.getcwd()
        model_files = glob.glob(os.path.join(project_root, "mlruns/*/*/artifacts/model/model.pkl"))
        
        if not model_files:
            st.error("No trained models found in mlruns/ directory.")
            st.info("The MLOps dashboards will still work, but predictions require a trained model.")
            st.stop()
        
        # Sort by modification time to get the latest
        model_files.sort(key=os.path.getmtime, reverse=True)
        latest_model_dir = os.path.dirname(model_files[0])
        
        # Load the model directly from the directory
        model = mlflow.pyfunc.load_model(latest_model_dir)
        
        # Extract run ID from path for display
        run_id = latest_model_dir.split('/')[-3]
        return model, f"Run {run_id[:8]}"
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("""
        **Troubleshooting:**
        - Ensure `mlruns/` folder is in your GitHub repository
        - Check that model files exist in `mlruns/1/*/artifacts/model/`
        - Try retraining the model locally: `python train.py`
        """)
        st.stop()

def get_shap_explainer(model):
    """Create SHAP explainer for the model."""
    # Extract the sklearn pipeline from pyfunc wrapper
    try:
        # Access the sklearn model from the MLflow sklearn wrapper
        sklearn_pipeline = model._model_impl.sklearn_model
    except AttributeError:
        try:
            # Fallback: try alternative attribute
            sklearn_pipeline = model._model_impl._model
        except AttributeError:
            # Final fallback - use the model directly
            sklearn_pipeline = model
    
    # SHAP needs the actual tree model, not the pipeline
    # Extract the RandomForest from the pipeline
    if hasattr(sklearn_pipeline, 'named_steps'):
        # It's a Pipeline - get the 'model' step
        tree_model = sklearn_pipeline.named_steps['model']
    else:
        # It's already the model
        tree_model = sklearn_pipeline
    
    return shap.TreeExplainer(tree_model)

def create_input_form():
    """Create the customer input form in the sidebar."""
    st.sidebar.header("📋 Customer Profile")
    
    with st.sidebar.form("customer_form"):
        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with col2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        col3, col4 = st.columns(2)
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        with col4:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges, step=50.0)
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        else:
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
        
        st.subheader("Contract & Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }
        return customer_data
    return None

def make_prediction(model, customer_data):
    """Make churn prediction and get probability."""
    df = pd.DataFrame([customer_data])
    
    # Get the sklearn model for probability prediction
    try:
        # Access the sklearn model from the MLflow sklearn wrapper
        sklearn_model = model._model_impl.sklearn_model
    except AttributeError:
        try:
            # Fallback: try to use the pyfunc predict method
            # MLflow pyfunc doesn't have predict_proba, so we need the actual model
            sklearn_model = model._model_impl._model
        except AttributeError:
            # Final fallback - use the model directly if it's already unwrapped
            sklearn_model = model
    
    proba = float(sklearn_model.predict_proba(df)[0][1])
    prediction = "Yes" if proba >= 0.5 else "No"
    
    return prediction, proba, df

def explain_prediction(model, customer_data_df):
    """Generate SHAP explanation for the prediction."""
    # Get the sklearn pipeline
    try:
        sklearn_pipeline = model._model_impl.sklearn_model
    except AttributeError:
        try:
            sklearn_pipeline = model._model_impl._model
        except AttributeError:
            sklearn_pipeline = model
    
    # Transform the data through preprocessing steps
    if hasattr(sklearn_pipeline, 'named_steps'):
        # Get preprocessor and transform data
        preprocessor = sklearn_pipeline.named_steps['preprocessor']
        transformed_data = preprocessor.transform(customer_data_df)
        
        # Get feature names after transformation
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            feature_names = None
    else:
        # No preprocessing, use raw data
        transformed_data = customer_data_df
        feature_names = customer_data_df.columns.tolist()
    
    # Get SHAP explainer (uses the tree model only)
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(transformed_data)
    
    # For binary classification, use the positive class (churn = 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Handle different SHAP value shapes
    # Shape can be: (n_samples, n_features) or (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        # Shape is (n_samples, n_features, n_classes)
        # Extract: first sample, all features, class 1 (churn)
        shap_values = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2:
        # Shape is (n_samples, n_features) - take first sample
        shap_values = shap_values[0]
    
    return explainer, shap_values, feature_names

def plot_shap_waterfall(explainer, shap_values, feature_names):
    """Create SHAP waterfall plot showing feature contributions."""
    import matplotlib.pyplot as plt
    
    # Create a bar plot instead of waterfall (simpler for transformed features)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get top 15 features by absolute impact
    indices = np.argsort(np.abs(shap_values))[-15:]
    top_features = [feature_names[i] for i in indices]
    top_values = shap_values[indices]
    
    # Create horizontal bar chart
    colors = ['#ff4b4b' if v > 0 else '#00cc66' for v in top_values]
    ax.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
    ax.set_title('Top 15 Features by Impact', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig

def generate_simple_explanation(customer_data, prediction, probability, top_risk_factors, top_protective_factors):
    """Generate a simple, human-readable explanation for why the customer will stay or leave."""
    
    if prediction == "Yes":
        # Customer will churn
        risk_summary = f"**Why This Customer is Likely to Leave:**\n\n"
        risk_summary += f"Based on our analysis, this customer has a **{probability:.0%} probability** of churning. "
        risk_summary += "Here are the main reasons:\n\n"
        
        for factor in top_risk_factors[:3]:
            if "Month-to-month" in str(factor):
                risk_summary += "🔴 **No long-term commitment**: They're on a month-to-month contract, making it easy to cancel anytime.\n\n"
            elif "tenure" in str(factor).lower() and customer_data.get('tenure', 0) < 12:
                risk_summary += f"🔴 **New customer**: Only {int(customer_data['tenure'])} months with us - haven't built loyalty yet.\n\n"
            elif "OnlineSecurity" in str(factor) and customer_data.get('OnlineSecurity') == 'No':
                risk_summary += "🔴 **Missing key services**: No online security protection adds to dissatisfaction.\n\n"
            elif "TechSupport" in str(factor) and customer_data.get('TechSupport') == 'No':
                risk_summary += "🔴 **No technical support**: When issues arise, they have no help, leading to frustration.\n\n"
            elif "Fiber optic" in str(factor) and customer_data.get('InternetService') == 'Fiber optic':
                risk_summary += "🔴 **Premium pricing concerns**: Fiber optic is expensive, and they may find cheaper alternatives.\n\n"
            elif "Electronic check" in str(factor):
                risk_summary += "🔴 **Manual payment method**: Electronic check is less convenient than automatic payments.\n\n"
        
        return risk_summary
    else:
        # Customer will stay
        stay_summary = f"**Why This Customer is Likely to Stay:**\n\n"
        stay_summary += f"Great news! This customer has only a **{probability:.0%} probability** of churning. "
        stay_summary += "Here's what's keeping them loyal:\n\n"
        
        for factor in top_protective_factors[:3]:
            if "tenure" in str(factor).lower() and customer_data.get('tenure', 0) > 24:
                stay_summary += f"✅ **Long-term loyalty**: {int(customer_data['tenure'])} months of service shows strong commitment.\n\n"
            elif "Two year" in str(factor) or "One year" in str(factor):
                stay_summary += f"✅ **Contract commitment**: Locked into a {customer_data.get('Contract', '')} contract - lower churn risk.\n\n"
            elif "OnlineSecurity" in str(factor) and customer_data.get('OnlineSecurity') == 'Yes':
                stay_summary += "✅ **Protected and secure**: Online security service adds value and peace of mind.\n\n"
            elif "TechSupport" in str(factor) and customer_data.get('TechSupport') == 'Yes':
                stay_summary += "✅ **Supported when needed**: Tech support subscription shows they value our assistance.\n\n"
            elif "automatic" in str(factor).lower():
                stay_summary += "✅ **Convenient payments**: Automatic payment method indicates satisfaction with service.\n\n"
            elif "Partner" in str(factor) and customer_data.get('Partner') == 'Yes':
                stay_summary += "✅ **Family plan benefits**: Partner on account suggests stable, family-oriented usage.\n\n"
        
        return stay_summary


def generate_retention_strategy(customer_data, prediction, probability, top_factors):
    """Generate personalized retention strategy using Groq's Llama 3.1 8B."""
    # Check if GROQ_API_KEY exists in st.secrets (Streamlit Cloud)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.warning("⚠️ Groq API key not found. Add GROQ_API_KEY to Streamlit Cloud secrets or .env file to enable AI-powered retention strategies.")
        return None
    
    try:
        # Initialize Groq client (compatible with groq 0.9.0)
        # Only pass api_key parameter, no other kwargs
        client = Groq(api_key=api_key)
        
        # Build context about the customer
        risk_level = "HIGH RISK" if probability > 0.7 else "MODERATE RISK" if probability > 0.4 else "LOW RISK"
        tenure = customer_data['tenure']
        monthly_charges = customer_data['MonthlyCharges']
        
        # Determine customer segment for personalized strategy
        if tenure < 12:
            segment = "New Customer (Under 1 Year)"
            strategy_focus = "Welcome aboard! Focus on onboarding support and early value demonstration"
        elif tenure >= 12 and tenure < 36:
            segment = "Growing Customer (1-3 Years)"
            strategy_focus = "Strengthen relationship with loyalty rewards and service upgrades"
        else:
            segment = "Loyal Long-Term Customer (3+ Years)"
            strategy_focus = "VIP treatment, exclusive offers, and deep appreciation for loyalty"
        
        # Determine offer based on risk level and charges
        if probability > 0.7:
            if monthly_charges > 70:
                offer = "30% discount for 6 months + free premium support"
            else:
                offer = "3 months at 50% off + free service upgrade"
        elif probability > 0.4:
            offer = "20% discount for 3 months + complimentary tech support package"
        else:
            offer = "15% loyalty discount + priority customer service"
        
        prompt = f"""You are writing a concise, professional customer retention email on behalf of a telecommunications company.

CUSTOMER PROFILE:
- Segment: {segment}
- Churn Risk: {risk_level} ({probability:.1%} probability of leaving)
- Tenure: {tenure} months with us
- Monthly Charges: ${monthly_charges:.2f}

TOP RISK FACTORS:
{chr(10).join(f"- {factor}" for factor in top_factors[:3])}

PERSONALIZED OFFER: {offer}

TASK: Write a SHORT, professional email (150-180 words MAX) with this structure:

Subject: Special Offer Just for You

Dear Valued Customer,

[ONE short paragraph: Thank them for {tenure} months with us. Mention you noticed some concerns about (mention 1-2 risk factors briefly).]

[ONE short paragraph: Present the EXACT offer: {offer}. Say it's exclusive and valid for 14 days only.]

[ONE SHORT line: To claim: Call 1-800-STAY-NOW or reply to this email.]

Best regards,
Customer Retention Team
retention@telcocare.com

---

IMPORTANT: Keep it SHORT and easy to read. Maximum 180 words total. Use simple, warm language. No fluff.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer retention specialist. Write SHORT, concise emails (150-180 words max) that are warm and professional."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=350
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating retention strategy: {str(e)}")
        return None

def display_prediction_results(prediction, probability):
    """Display prediction results with visual indicators."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Prediction")
        if prediction == "Yes":
            st.error(f"**WILL CHURN** ⚠️")
            st.caption(f"Probability ≥ 50%")
        else:
            st.success(f"**WILL STAY** ✅")
            st.caption(f"Probability < 50%")
    
    with col2:
        st.markdown("### 📊 Confidence")
        st.metric("Churn Probability", f"{probability:.1%}")
        # Add interpretation helper
        if probability < 0.3:
            st.caption("Very low churn risk")
        elif probability < 0.5:
            st.caption("Low churn risk")
        elif probability < 0.7:
            st.caption("Moderate churn risk")
        else:
            st.caption("High churn risk")
    
    with col3:
        st.markdown("### 🚨 Risk Level")
        if probability > 0.7:
            st.error("**HIGH RISK**")
        elif probability > 0.4:
            st.warning("**MODERATE RISK**")
        else:
            st.success("**LOW RISK**")
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, width="stretch")

def main():
    """Main application logic."""
    # Header with animation
    st.markdown('<h1 class="main-header">🤖 Hybrid AI Churn-Bot: Mission Control</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Combining Predictive AI + Explainable AI + Generative AI (Groq Llama 3.1)</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize MLflow and load model
    client = init_mlflow()
    model, model_stage = load_model()
    
    # Display model info with style
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(f"🚀 **Active Model**: {MODEL_NAME} ({model_stage}) | **Powered by**: Groq Llama 3.1 8B")
    
    # Sidebar input form
    customer_data = create_input_form()
    
    if customer_data is None:
        # Welcome screen with enhanced styling
        st.markdown("""
        ## 👋 Welcome to the Hybrid AI Churn-Bot!
        
        <div class="explanation-box">
        
        ### 🌟 This Is Not Just Another ML Model...
        
        This is a **complete intelligent system** that combines **THREE different AI technologies**:
        
        - **🎯 Predictive AI**: MLflow-registered RandomForest model for churn prediction
        - **🧠 Explainable AI**: SHAP analysis showing *why* customers churn
        - **✨ Generative AI**: Groq's Llama 3.1 8B creating personalized retention emails
        
        </div>
        
        ### 🚀 How to Use:
        1. **Fill in** the customer profile form in the sidebar ←
        2. **Click** "🔮 Predict Churn" to get instant AI analysis
        3. **Review** explanations, reasons, and AI-generated retention emails
        
        ### 💡 Try These Quick Test Scenarios:
        
        **🔴 High Risk Customer (Will Probably Leave):**
        - Tenure: 1-2 months (very new)
        - Contract: Month-to-month (no commitment)
        - Monthly Charges: $80-100 (expensive)
        - Additional Services: None selected
        - Payment: Electronic check
        
        **🟢 Low Risk Customer (Will Stay):**
        - Tenure: 60+ months (loyal)
        - Contract: Two year (committed)
        - Additional Services: Yes to everything
        - Payment: Bank transfer (automatic)
        """, unsafe_allow_html=True)
        
        # Display metrics in cards
        st.markdown("### 📊 System Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Model Accuracy", "78.4%", help="RandomForest accuracy on test set")
        with col2:
            st.metric("🔮 Predictions Made", "0", help="In this session")
        with col3:
            st.metric("⚡ Avg Response Time", "<2s", help="Prediction + Explanation + AI Email")
        with col4:
            st.metric("🤖 AI Model", "Llama 3.1 8B", help="Groq-powered generation")
        
    else:
        # Make prediction
        with st.spinner("🔮 Analyzing customer profile with AI..."):
            prediction, probability, customer_df = make_prediction(model, customer_data)
        
        # Log prediction for monitoring (production tracking)
        try:
            customer_id = f"CUST_{hash(str(customer_data))}"
            save_prediction_log(
                customer_id=customer_id,
                features=customer_data,
                prediction=prediction,
                probability=probability
            )
        except Exception as e:
            # Silently fail if logging doesn't work (don't break the app)
            pass
        
        # Display results with enhanced visuals
        st.markdown("## 🎯 Prediction Results")
        display_prediction_results(prediction, probability)
        
        st.markdown("---")
        
        # Explainability section
        st.markdown("## 🧠 AI Explainability: Understanding the Decision")
        
        with st.spinner("🔍 Generating SHAP explanation and natural language summary..."):
            explainer, shap_values, transformed_feature_names = explain_prediction(model, customer_df)
            
            # Get feature importance using transformed feature names
            if transformed_feature_names is not None:
                feature_names = transformed_feature_names
            else:
                feature_names = customer_df.columns.tolist()
            
            # shap_values is already 1D from explain_prediction, just ensure it's a proper numpy array
            shap_values_flat = np.asarray(shap_values).flatten()
            
            # Ensure lengths match
            if len(feature_names) != len(shap_values_flat):
                st.error(f"Feature count mismatch: {len(feature_names)} features but {len(shap_values_flat)} SHAP values")
                st.info(f"Debug: shap_values shape: {shap_values.shape}, feature_names length: {len(feature_names)}")
                return
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': list(feature_names),
                'Impact': list(shap_values_flat)
            }).sort_values('Impact', key=abs, ascending=False)
            
            # Map transformed features back to original features for better readability
            # Extract original feature names from transformed names (e.g., "Contract_Two year" -> "Contract")
            feature_importance['OriginalFeature'] = feature_importance['Feature'].apply(
                lambda x: x.split('_')[0] if '_' in x else x
            )
            
            # Aggregate by original feature (sum absolute impacts)
            original_feature_importance = feature_importance.groupby('OriginalFeature').agg({
                'Impact': lambda x: x.sum()  # Sum the impacts
            }).reset_index()
            original_feature_importance.columns = ['Feature', 'Impact']
            original_feature_importance = original_feature_importance.sort_values('Impact', key=abs, ascending=False)
            
            # Get top factors from original features
            top_risk_factors = original_feature_importance[original_feature_importance['Impact'] > 0]['Feature'].head(5).tolist()
            top_protective_factors = original_feature_importance[original_feature_importance['Impact'] < 0]['Feature'].head(5).tolist()
            
            # Generate simple explanation
            simple_explanation = generate_simple_explanation(
                customer_data, prediction, probability, 
                top_risk_factors, top_protective_factors
            )
            
            # Display plain language explanation
            if prediction == "Yes":
                st.markdown(f'<div class="churn-reason">{simple_explanation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stay-reason">{simple_explanation}</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed Factor Analysis")
        
        # Display top factors in styled cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Factors Increasing Churn Risk")
            churn_factors = original_feature_importance[original_feature_importance['Impact'] > 0].head(5)
            max_impact = original_feature_importance['Impact'].abs().max()
            for idx, row in churn_factors.iterrows():
                st.markdown(f'<div class="factor-item">📍 <b>{row["Feature"]}</b>: +{row["Impact"]:.3f}</div>', unsafe_allow_html=True)
                st.progress(min(abs(row['Impact']) / max_impact, 1.0))
        
        with col2:
            st.markdown("#### 🟢 Factors Reducing Churn Risk")
            stay_factors = original_feature_importance[original_feature_importance['Impact'] < 0].head(5)
            for idx, row in stay_factors.iterrows():
                st.markdown(f'<div class="factor-item">📍 <b>{row["Feature"]}</b>: {row["Impact"]:.3f}</div>', unsafe_allow_html=True)
                st.progress(min(abs(row['Impact']) / feature_importance['Impact'].abs().max(), 1.0))
        
        # SHAP waterfall plot
        with st.expander("� Advanced: SHAP Waterfall Plot (Technical View)", expanded=False):
            st.info("This plot shows how each feature pushed the prediction away from the baseline.")
            fig = plot_shap_waterfall(explainer, shap_values, feature_names)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Gen AI Retention Strategy (for churn cases) or appreciation (for stay cases)
        if prediction == "Yes":
            st.markdown("## ✉️ AI-Generated Retention Email")
            st.markdown("### 💼 Professional Email Drafted by AI")
            st.info("🤖 This email was personalized using Groq's Llama 3.1 8B based on customer segment, risk level, and specific pain points.")
            
            with st.spinner("✍️ Drafting personalized retention email..."):
                email_content = generate_retention_strategy(
                    customer_data, prediction, probability, top_risk_factors
                )
            
            if email_content:
                # Display email in a more professional format
                st.markdown("""
                    <style>
                    .email-container {
                        background: white;
                        padding: 2rem;
                        border-radius: 10px;
                        border: 1px solid #ddd;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        font-family: 'Arial', sans-serif;
                        color: #333;
                        line-height: 1.6;
                        white-space: pre-wrap;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                st.markdown(f'<div class="email-container">{email_content}</div>', unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="📥 Download Email",
                        data=email_content,
                        file_name=f"retention_email_tenure_{customer_data['tenure']}.txt",
                        mime="text/plain",
                        width="stretch"
                    )
                with col2:
                    if st.button("📋 Copy to Clipboard", width="stretch"):
                        st.success("✅ Email copied! (Use Ctrl+C manually)")
                with col3:
                    if st.button("📧 Send Email", width="stretch"):
                        st.info("Email integration coming soon!")
            else:
                st.error("❌ Could not generate email. Please check your GROQ_API_KEY in .env file.")
                st.markdown("""
                **To enable AI email generation:**
                1. Get a free Groq API key from https://console.groq.com/
                2. Add it to your `.env` file: `GROQ_API_KEY=your_key_here`
                3. Restart the app
                """)
        
        else:
            # Customer will stay - show appreciation message
            st.success("## ✅ Customer is Likely to Stay!")
            st.markdown("""
            <div class="stay-reason">
            
            ### 🎉 Great News!
            
            This customer shows strong loyalty indicators. Here's what to do:
            
            ✅ **Send a thank-you email** expressing appreciation for their continued business
            
            ✅ **Offer loyalty rewards** - Consider:
            - 10% discount on next bill
            - Free service upgrade for 3 months
            - Referral bonus ($50 credit for each friend they bring)
            
            ✅ **Proactive engagement** - Reach out 2 months before contract renewal with:
            - Exclusive early renewal offers
            - New service announcements
            - VIP customer appreciation events
            
            ✅ **Continue monitoring** - Set a reminder to check their usage in 3 months
            
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with updated branding
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.2rem; font-weight: 600;'>🤖 Hybrid AI Churn-Bot</p>
    <p style='font-size: 0.9rem;'>Powered by MLflow + SHAP + Groq Llama 3.1 8B</p>
    <p style='font-size: 0.85rem; color: #999;'>Combining Predictive AI + Explainable AI + Generative AI</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>⚡ Lightning-fast predictions • 🧠 Crystal-clear explanations • ✨ AI-powered retention</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
