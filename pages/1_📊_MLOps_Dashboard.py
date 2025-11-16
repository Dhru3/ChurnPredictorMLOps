#!/usr/bin/env python3
"""
MLOps Model Comparison Dashboard
Compare all trained models, track performance, and analyze hyperparameters
"""
import streamlit as st
import mlflow
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="MLOps Dashboard", page_icon="📊", layout="wide")

# MLflow setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
client = MlflowClient()

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .winner-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 MLOps Model Comparison Dashboard")
st.markdown("**Compare all trained models and select the champion**")
st.markdown("---")

# Get all experiments
try:
    experiments = client.search_experiments()
    experiment_names = [exp.name for exp in experiments if exp.name != "Default"]
    
    if not experiment_names:
        st.warning("⚠️ No experiments found. Run `train.py` first to create experiments.")
        st.stop()
    
    selected_experiment = st.selectbox("📁 Select Experiment", experiment_names, index=0)
    experiment = client.get_experiment_by_name(selected_experiment)
    
    if experiment:
        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_accuracy DESC"],
            max_results=20
        )
        
        if not runs:
            st.info("ℹ️ No model runs found in this experiment. Train a model first!")
            st.stop()
        
        # Create comparison dataframe
        comparison_data = []
        for run in runs:
            metrics = run.data.metrics
            params = run.data.params
            
            comparison_data.append({
                "Run ID": run.info.run_id[:8],
                "Date": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                "Accuracy": metrics.get('test_accuracy', 0),
                "Precision": metrics.get('test_precision', 0),
                "Recall": metrics.get('test_recall', 0),
                "F1-Score": metrics.get('test_f1', 0),
                "ROC-AUC": metrics.get('test_roc_auc', 0),
                "n_estimators": params.get('n_estimators', 'N/A'),
                "max_depth": params.get('max_depth', 'N/A'),
                "min_samples_split": params.get('min_samples_split', 'N/A'),
                "Status": run.info.status
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Key Metrics Section
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🏆 Total Models", len(df), help="Total models trained")
        with col2:
            st.metric("🎯 Best Accuracy", f"{df['Accuracy'].max():.1%}", help="Highest accuracy achieved")
        with col3:
            st.metric("📊 Best Precision", f"{df['Precision'].max():.1%}", help="Highest precision for churn class")
        with col4:
            st.metric("🔍 Best Recall", f"{df['Recall'].max():.1%}", help="Highest recall for churn class")
        with col5:
            st.metric("⚡ Best F1", f"{df['F1-Score'].max():.1%}", help="Highest F1-score")
        
        st.markdown("---")
        
        # Model Comparison Table
        st.subheader("📋 All Trained Models")
        st.markdown("*Rows highlighted in green show best performance for each metric*")
        
        # Style the dataframe
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_max]
        
        styled_df = df.style.apply(
            highlight_max, 
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Model Comparison CSV",
            data=csv,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Performance Evolution
        st.subheader("📈 Model Performance Evolution Over Time")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Accuracy'], 
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Precision'], 
            mode='lines+markers',
            name='Precision',
            line=dict(color='#f093fb', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Recall'], 
            mode='lines+markers',
            name='Recall',
            line=dict(color='#4facfe', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['F1-Score'], 
            mode='lines+markers',
            name='F1-Score',
            line=dict(color='#43e97b', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Model Metrics Over Time",
            xaxis_title="Training Date",
            yaxis_title="Score",
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Hyperparameter Analysis
        st.subheader("🔧 Hyperparameter Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # n_estimators vs Accuracy
            fig1 = go.Figure()
            
            # Convert to numeric for plotting
            df_numeric = df.copy()
            df_numeric['n_estimators'] = pd.to_numeric(df_numeric['n_estimators'], errors='coerce')
            df_numeric = df_numeric.dropna(subset=['n_estimators'])
            
            if len(df_numeric) > 0:
                fig1.add_trace(go.Scatter(
                    x=df_numeric['n_estimators'], 
                    y=df_numeric['Accuracy'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=df_numeric['Accuracy'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Accuracy"),
                        line=dict(width=2, color='white')
                    ),
                    text=df_numeric['Run ID'],
                    hovertemplate='<b>Run:</b> %{text}<br>' +
                                  '<b>n_estimators:</b> %{x}<br>' +
                                  '<b>Accuracy:</b> %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
                
                fig1.update_layout(
                    title="Impact of n_estimators on Accuracy",
                    xaxis_title="n_estimators",
                    yaxis_title="Accuracy",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No numeric n_estimators data available")
        
        with col2:
            # max_depth vs Accuracy
            fig2 = go.Figure()
            
            df_numeric2 = df.copy()
            df_numeric2['max_depth'] = pd.to_numeric(df_numeric2['max_depth'], errors='coerce')
            df_numeric2 = df_numeric2.dropna(subset=['max_depth'])
            
            if len(df_numeric2) > 0:
                fig2.add_trace(go.Scatter(
                    x=df_numeric2['max_depth'], 
                    y=df_numeric2['Accuracy'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=df_numeric2['Accuracy'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Accuracy"),
                        line=dict(width=2, color='white')
                    ),
                    text=df_numeric2['Run ID'],
                    hovertemplate='<b>Run:</b> %{text}<br>' +
                                  '<b>max_depth:</b> %{x}<br>' +
                                  '<b>Accuracy:</b> %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
                
                fig2.update_layout(
                    title="Impact of max_depth on Accuracy",
                    xaxis_title="max_depth",
                    yaxis_title="Accuracy",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No numeric max_depth data available")
        
        st.markdown("---")
        
        # Model Registry Status
        st.subheader("🏆 Model Registry Status")
        
        try:
            versions = client.search_model_versions(f"name='churn-predictor'")
            
            if versions:
                registry_data = []
                for version in versions:
                    # Get run info for this version
                    run_id = version.run_id
                    run = client.get_run(run_id)
                    
                    # Determine status
                    aliases = version.aliases if hasattr(version, 'aliases') else []
                    if 'champion' in aliases or version.current_stage == 'Production':
                        status = "🏆 CHAMPION (Production)"
                        status_color = "#28a745"
                    elif 'challenger' in aliases or version.current_stage == 'Staging':
                        status = "🥈 CHALLENGER (Staging)"
                        status_color = "#ffc107"
                    else:
                        status = "📦 ARCHIVED"
                        status_color = "#6c757d"
                    
                    registry_data.append({
                        "Version": version.version,
                        "Status": status,
                        "Run ID": run_id[:8],
                        "Accuracy": run.data.metrics.get('test_accuracy', 'N/A'),
                        "Created": datetime.fromtimestamp(int(version.creation_timestamp) / 1000).strftime("%Y-%m-%d %H:%M"),
                    })
                
                reg_df = pd.DataFrame(registry_data)
                
                # Display with custom styling
                for _, row in reg_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 2])
                    
                    with col1:
                        st.markdown(f"**Version {row['Version']}**")
                    with col2:
                        st.markdown(f"{row['Status']}")
                    with col3:
                        st.markdown(f"`{row['Run ID']}`")
                    with col4:
                        if isinstance(row['Accuracy'], float):
                            st.markdown(f"**{row['Accuracy']:.1%}**")
                        else:
                            st.markdown(f"**{row['Accuracy']}**")
                    with col5:
                        st.markdown(f"*{row['Created']}*")
                
            else:
                st.info("📭 No models in registry yet. Register a model by running the training script with model registration enabled.")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load model registry: {e}")
            st.info("Make sure you have registered at least one model version.")
        
        st.markdown("---")
        
        # Best Model Summary
        st.subheader("🌟 Champion Model Summary")
        
        best_run = df.loc[df['Accuracy'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Performance Metrics")
            st.metric("Accuracy", f"{best_run['Accuracy']:.3f}")
            st.metric("Precision", f"{best_run['Precision']:.3f}")
            st.metric("Recall", f"{best_run['Recall']:.3f}")
            st.metric("F1-Score", f"{best_run['F1-Score']:.3f}")
        
        with col2:
            st.markdown("### ⚙️ Hyperparameters")
            st.info(f"**n_estimators:** {best_run['n_estimators']}")
            st.info(f"**max_depth:** {best_run['max_depth']}")
            st.info(f"**min_samples_split:** {best_run['min_samples_split']}")
        
        with col3:
            st.markdown("### 📅 Details")
            st.success(f"**Run ID:** `{best_run['Run ID']}`")
            st.success(f"**Trained:** {best_run['Date']}")
            st.success(f"**Status:** {best_run['Status']}")
        
    else:
        st.error("❌ Experiment not found.")

except Exception as e:
    st.error(f"❌ Error loading MLflow data: {e}")
    st.info("""
    **Troubleshooting:**
    - Make sure you've run `train.py` at least once
    - Check that `mlflow.db` exists in the project root
    - Verify MLflow tracking URI is configured correctly
    """)
