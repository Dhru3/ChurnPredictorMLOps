#!/usr/bin/env python3
"""
Extract the trained model from mlruns and save it as churn_pipeline.pkl
This is a one-time script to convert your existing trained model.
"""
import glob
import os
import joblib
import pickle

# Find the latest model in mlruns
project_root = os.getcwd()
model_files = glob.glob(os.path.join(project_root, "mlruns/*/*/artifacts/model/model.pkl"))

if not model_files:
    print("❌ No trained models found in mlruns/ directory.")
    print("Please run: python train.py")
    exit(1)

# Sort by modification time to get the latest
model_files.sort(key=os.path.getmtime, reverse=True)
latest_model_file = model_files[0]

print(f"📦 Found model at: {latest_model_file}")

# Load the model directly with pickle
print("⏳ Loading model...")
with open(latest_model_file, 'rb') as f:
    sklearn_pipeline = pickle.load(f)

# Save as churn_pipeline.pkl
output_path = "churn_pipeline.pkl"
print(f"💾 Saving pipeline to {output_path}...")
joblib.dump(sklearn_pipeline, output_path)

print(f"✅ Successfully saved model to {output_path}")
print(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print("\n🚀 Next steps:")
print("1. git add churn_pipeline.pkl")
print("2. git commit -m 'Add trained model pipeline'")
print("3. git push origin main")

