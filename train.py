#!/usr/bin/env python3
"""Train and register the Telco churn model with MLflow."""
from __future__ import annotations

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
        mlflow.log_param("min_samples_leaf", pipeline.named_steps["model"].min_samples_leaf)

        pipeline.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        report = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
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

        print(f"Training complete. Accuracy: {accuracy:.4f}")
        
        # Set 'champion' alias for the latest version (modern replacement for Production stage)
        client = mlflow.MlflowClient()
        model_version = model_info.registered_model_version
        client.set_registered_model_alias(MODEL_NAME, "champion", model_version)
        print(f"✅ Model registered as '{MODEL_NAME}' version {model_version} with 'champion' alias")


if __name__ == "__main__":
    train_and_register()
