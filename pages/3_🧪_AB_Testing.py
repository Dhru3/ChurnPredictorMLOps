#!/usr/bin/env python3
"""
A/B Testing Dashboard
Compare Champion vs Challenger models with statistical significance testing
"""
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import plotly.graph_objects as go
import plotly.express as px
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import mcnemar
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="A/B Testing", page_icon="🧪", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .champion-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
    }
    .challenger-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .winner-box {
        background: #ffd700;
        padding: 1rem;
        border-radius: 10px;
        border: 3px solid #ffaa00;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-comparison {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 A/B Testing Dashboard")
st.markdown("**Compare Champion vs Challenger models with statistical significance testing**")
st.markdown("---")

# Initialize MLflow
client = MlflowClient()

@st.cache_data
def load_test_data():
    """Load test dataset for evaluation"""
    import pandas as pd
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_file = PROJECT_ROOT / "Telco-Customer-Churn.csv"
    
    if not data_file.exists():
        return None, None
    
    df = pd.read_csv(data_file)
    
    # Sample test set (20% of data)
    test_df = df.sample(frac=0.2, random_state=42)
    
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn'].map({'No': 0, 'Yes': 1})
    
    return X_test, y_test

def get_model_predictions(model_uri, X_test):
    """Get predictions from a model"""
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(X_test)
        
        # Handle probability predictions
        if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
            # Binary classification probabilities
            pred_proba = predictions[:, 1]
            pred_class = (pred_proba > 0.5).astype(int)
        else:
            pred_class = predictions
            pred_proba = None
        
        return pred_class, pred_proba
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
    
    return metrics

def mcnemar_test(y_true, pred_champion, pred_challenger):
    """
    Perform McNemar's test to check if model performances are significantly different
    Returns: test statistic, p-value, and interpretation
    """
    # Create contingency table
    # Both correct (a), Champion correct & Challenger wrong (b)
    # Champion wrong & Challenger correct (c), Both wrong (d)
    
    both_correct = np.sum((pred_champion == y_true) & (pred_challenger == y_true))
    champion_correct = np.sum((pred_champion == y_true) & (pred_challenger != y_true))
    challenger_correct = np.sum((pred_champion != y_true) & (pred_challenger == y_true))
    both_wrong = np.sum((pred_champion != y_true) & (pred_challenger != y_true))
    
    # Contingency table
    table = [[both_correct, champion_correct],
             [challenger_correct, both_wrong]]
    
    # McNemar's test focuses on discordant pairs (b and c)
    result = mcnemar(table, exact=False, correction=True)
    
    return result.statistic, result.pvalue, table

# Load test data
X_test, y_test = load_test_data()

if X_test is not None and y_test is not None:
    st.success(f"✅ Loaded test set: {len(X_test)} samples")
    
    # Get all models from registry
    try:
        registered_models = client.search_registered_models()
        model_names = [rm.name for rm in registered_models]
        
        if len(model_names) >= 2:
            st.subheader("🎯 Select Models to Compare")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="champion-box"><h3>👑 CHAMPION MODEL</h3></div>', unsafe_allow_html=True)
                champion_model = st.selectbox(
                    "Select Champion Model",
                    model_names,
                    index=0,
                    key="champion"
                )
            
            with col2:
                st.markdown('<div class="challenger-box"><h3>🚀 CHALLENGER MODEL</h3></div>', unsafe_allow_html=True)
                challenger_model = st.selectbox(
                    "Select Challenger Model",
                    model_names,
                    index=min(1, len(model_names)-1),
                    key="challenger"
                )
            
            if champion_model and challenger_model:
                if champion_model == challenger_model:
                    st.warning("⚠️ Please select different models for comparison")
                else:
                    st.markdown("---")
                    
                    # Run comparison button
                    if st.button("🧪 Run A/B Test", type="primary", use_container_width=True):
                        with st.spinner("Running A/B test..."):
                            # Get model URIs
                            champion_uri = f"models:/{champion_model}/latest"
                            challenger_uri = f"models:/{challenger_model}/latest"
                            
                            # Get predictions
                            champion_pred, champion_proba = get_model_predictions(champion_uri, X_test)
                            challenger_pred, challenger_proba = get_model_predictions(challenger_uri, X_test)
                            
                            if champion_pred is not None and challenger_pred is not None:
                                # Calculate metrics
                                champion_metrics = calculate_metrics(y_test, champion_pred, champion_proba)
                                challenger_metrics = calculate_metrics(y_test, challenger_pred, challenger_proba)
                                
                                # Store in session state
                                st.session_state['ab_champion_metrics'] = champion_metrics
                                st.session_state['ab_challenger_metrics'] = challenger_metrics
                                st.session_state['ab_champion_pred'] = champion_pred
                                st.session_state['ab_challenger_pred'] = challenger_pred
                                st.session_state['ab_y_test'] = y_test
                                st.session_state['ab_test_complete'] = True
                                st.success("✅ A/B Test completed!")
                                st.rerun()
                    
                    # Display results if test has been run
                    if st.session_state.get('ab_test_complete', False):
                        st.markdown("---")
                        st.subheader("📊 Performance Comparison")
                        
                        champion_metrics = st.session_state['ab_champion_metrics']
                        challenger_metrics = st.session_state['ab_challenger_metrics']
                        champion_pred = st.session_state['ab_champion_pred']
                        challenger_pred = st.session_state['ab_challenger_pred']
                        y_test = st.session_state['ab_y_test']
                        
                        # Metrics comparison table
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown("### Metric")
                            for metric in champion_metrics.keys():
                                st.markdown(f"**{metric}**")
                        
                        with col2:
                            st.markdown("### 👑 Champion")
                            for value in champion_metrics.values():
                                st.markdown(f"{value:.4f}")
                        
                        with col3:
                            st.markdown("### 🚀 Challenger")
                            for value in challenger_metrics.values():
                                delta = value - list(champion_metrics.values())[list(challenger_metrics.keys()).index(list(challenger_metrics.keys())[list(champion_metrics.values()).index(value)])]
                                color = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
                                st.markdown(f"{value:.4f} {color} ({delta:+.4f})")
                        
                        # Visualization
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Radar chart
                            fig1 = go.Figure()
                            
                            metrics_list = list(champion_metrics.keys())
                            champion_values = list(champion_metrics.values())
                            challenger_values = list(challenger_metrics.values())
                            
                            fig1.add_trace(go.Scatterpolar(
                                r=champion_values,
                                theta=metrics_list,
                                fill='toself',
                                name='Champion',
                                line=dict(color='#11998e', width=2)
                            ))
                            
                            fig1.add_trace(go.Scatterpolar(
                                r=challenger_values,
                                theta=metrics_list,
                                fill='toself',
                                name='Challenger',
                                line=dict(color='#667eea', width=2)
                            ))
                            
                            fig1.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                title="Performance Radar Chart",
                                height=400,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Bar chart comparison
                            fig2 = go.Figure()
                            
                            x_labels = list(champion_metrics.keys())
                            
                            fig2.add_trace(go.Bar(
                                x=x_labels,
                                y=champion_values,
                                name='Champion',
                                marker=dict(color='#11998e')
                            ))
                            
                            fig2.add_trace(go.Bar(
                                x=x_labels,
                                y=challenger_values,
                                name='Challenger',
                                marker=dict(color='#667eea')
                            ))
                            
                            fig2.update_layout(
                                title="Metric Comparison",
                                yaxis_title="Score",
                                height=400,
                                barmode='group',
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # McNemar's Test
                        st.markdown("---")
                        st.subheader("📈 Statistical Significance Test (McNemar's Test)")
                        
                        test_stat, p_value, table = mcnemar_test(y_test, champion_pred, challenger_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Test Statistic", f"{test_stat:.4f}")
                        
                        with col2:
                            st.metric("P-Value", f"{p_value:.4f}")
                        
                        with col3:
                            significant = p_value < 0.05
                            st.metric("Significant?", "YES ✅" if significant else "NO ❌")
                        
                        # Interpretation
                        st.markdown("#### 📋 Test Interpretation")
                        
                        if significant:
                            # Determine which model is better
                            champion_acc = champion_metrics['Accuracy']
                            challenger_acc = challenger_metrics['Accuracy']
                            
                            if challenger_acc > champion_acc:
                                st.markdown(f"""
                                <div class="winner-box">
                                    <h3>🎉 CHALLENGER WINS! 🚀</h3>
                                    <p>The Challenger model performs <b>significantly better</b> than the Champion (p-value: {p_value:.4f} < 0.05)</p>
                                    <p><b>Recommendation:</b> Promote Challenger to Champion and deploy to production!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="winner-box">
                                    <h3>👑 CHAMPION RETAINS TITLE!</h3>
                                    <p>The Champion model performs <b>significantly better</b> than the Challenger (p-value: {p_value:.4f} < 0.05)</p>
                                    <p><b>Recommendation:</b> Keep Champion in production, archive Challenger</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info(f"""
                            **No Significant Difference** (p-value: {p_value:.4f} ≥ 0.05)
                            
                            The models perform similarly from a statistical perspective. Consider:
                            - Keeping the Champion if computational efficiency is similar
                            - Evaluating other factors: interpretability, deployment complexity, inference speed
                            - Running additional tests on different data splits
                            """)
                        
                        # Contingency table
                        st.markdown("#### 🔢 Contingency Table")
                        
                        contingency_df = pd.DataFrame(
                            table,
                            columns=['Challenger Correct', 'Challenger Wrong'],
                            index=['Champion Correct', 'Champion Wrong']
                        )
                        
                        st.dataframe(contingency_df, use_container_width=True)
                        
                        st.markdown("""
                        **How to read:**
                        - **Both Correct**: Both models predicted correctly
                        - **Champion Correct / Challenger Wrong**: Only Champion got it right
                        - **Challenger Correct / Champion Wrong**: Only Challenger got it right
                        - **Both Wrong**: Both models failed
                        
                        McNemar's test focuses on the *discordant pairs* (off-diagonal elements) to determine significance.
                        """)
                        
                        # Confusion matrices
                        st.markdown("---")
                        st.subheader("🎯 Confusion Matrices")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 👑 Champion")
                            cm_champion = confusion_matrix(y_test, champion_pred)
                            
                            fig3 = px.imshow(
                                cm_champion,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Stay', 'Churn'],
                                y=['Stay', 'Churn'],
                                color_continuous_scale='Greens',
                                text_auto=True
                            )
                            fig3.update_layout(height=350)
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 🚀 Challenger")
                            cm_challenger = confusion_matrix(y_test, challenger_pred)
                            
                            fig4 = px.imshow(
                                cm_challenger,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Stay', 'Churn'],
                                y=['Stay', 'Churn'],
                                color_continuous_scale='Purples',
                                text_auto=True
                            )
                            fig4.update_layout(height=350)
                            st.plotly_chart(fig4, use_container_width=True)
                        
                        # Clear test button
                        st.markdown("---")
                        if st.button("🔄 Clear Results and Run New Test", use_container_width=True):
                            st.session_state.clear()
                            st.rerun()
        
        elif len(model_names) == 1:
            st.warning("⚠️ Only one model found in registry. Train and register at least 2 models for A/B testing.")
        else:
            st.info("ℹ️ No registered models found. Register models using the Model Registry to enable A/B testing.")
    
    except Exception as e:
        st.error(f"Error loading models: {e}")

else:
    st.error("❌ Test data not found! Ensure `Telco-Customer-Churn.csv` exists in the project root.")
    st.markdown("""
    ### How to Fix
    1. Make sure `Telco-Customer-Churn.csv` is in your project directory
    2. This file should be the same one used for training
    3. The A/B test will automatically create a test split from this data
    """)
