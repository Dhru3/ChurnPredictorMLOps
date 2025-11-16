#!/usr/bin/env python3
"""Train and register the Telco churn model with MLflow."""
from __future__ import annotations

import joblib
import json
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.sklearn  # noqa: F401 - needed to register the sklearn flavor
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "telco_churn.csv"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "churn-experiments"
MODEL_NAME = "churn-predictor"
PIPELINE_PATH = PROJECT_ROOT / "churn_pipeline.pkl"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = df.copy()
    dataset["Churn"] = dataset["Churn"].map({"Yes": 1, "No": 0})
    dataset = dataset.dropna(subset=["Churn"])

    features = dataset.drop(columns=["customerID", "Churn"], errors="ignore")
    target = dataset["Churn"]
    return features, target


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def configure_mlflow() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_and_register() -> None:
    configure_mlflow()
    df = load_dataset()
    X, y = split_features_target(df)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    pipeline = build_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", pipeline.named_steps["model"].n_estimators)
        mlflow.log_param("max_depth", str(pipeline.named_steps["model"].max_depth))
        mlflow.log_param("min_samples_split", pipeline.named_steps["model"].min_samples_split)
        mlflow.log_param("min_samples_leaf", pipeline.named_steps["model"].min_samples_leaf)

        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Calculate and log all metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # For ROC-AUC, we need probability predictions
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics with consistent naming for dashboard
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(json.dumps(report, indent=2), "classification_report.json")

        input_example = X_test.iloc[:5]
        signature = infer_signature(input_example, pipeline.predict(input_example))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        print(f"Training complete:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Set 'champion' alias for the latest version (modern replacement for Production stage)
        client = mlflow.MlflowClient()
        model_version = model_info.registered_model_version
        client.set_registered_model_alias(MODEL_NAME, "champion", model_version)
        print(f"✅ Model registered as '{MODEL_NAME}' version {model_version} with 'champion' alias")
    
    # Save the final pipeline to a single file using joblib (for Streamlit deployment)
    print(f"\n💾 Saving pipeline to {PIPELINE_PATH}...")
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"✅ Model saved to churn_pipeline.pkl")


if __name__ == "__main__":
    train_and_register()
