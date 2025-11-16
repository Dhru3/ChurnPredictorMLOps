# API Testing Guide

This directory contains sample test requests for the Churn Predictor API.

## Test Files

### 1. `test_request_high_risk.json`
**Profile**: New customer with high churn risk
- 1 month tenure (very new)
- Month-to-month contract (no commitment)
- High monthly charges ($85.50)
- Electronic check payment
- Fiber optic internet
- Minimal loyalty indicators

**Expected**: High churn probability (~70-90%)

### 2. `test_request_low_risk.json`
**Profile**: Established customer with low churn risk
- 72 months tenure (6 years!)
- Two-year contract (strong commitment)
- High monthly charges but proportional to services
- Bank auto-transfer (convenient)
- Full service package with protections
- Strong loyalty indicators

**Expected**: Low churn probability (~5-20%)

## How to Test

### Option 1: Interactive API Docs (Recommended)

1. Start the API server:
   ```bash
   uvicorn serve:app --reload
   ```

2. Open http://localhost:8000/docs

3. Click `/predict` → "Try it out"

4. Copy contents from a test file and paste into the Request body

5. Click "Execute"

### Option 2: cURL Command Line

**High Risk Customer:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_request_high_risk.json
```

**Low Risk Customer:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_request_low_risk.json
```

### Option 3: Python requests

```python
import requests
import json

# Load test data
with open('test_request_high_risk.json') as f:
    data = json.load(f)

# Make prediction
response = requests.post(
    'http://localhost:8000/predict',
    json=data
)

print(response.json())
# Output: {"churn_prediction": "Yes", "probability": 0.87}
```

## Response Format

All predictions return JSON in this format:

```json
{
  "churn_prediction": "Yes",
  "probability": 0.87
}
```

Where:
- `churn_prediction`: "Yes" or "No" (threshold: 0.5)
- `probability`: Float between 0.0 and 1.0 (rounded to 2 decimals)

## Understanding Results

### Churn Indicators (increase risk):
- ❌ Short tenure (<12 months)
- ❌ Month-to-month contract
- ❌ No online security
- ❌ No tech support
- ❌ Electronic check payment
- ❌ High monthly charges relative to tenure
- ❌ Fiber optic (often expensive, attracts switchers)

### Loyalty Indicators (decrease risk):
- ✅ Long tenure (>24 months)
- ✅ One or two-year contract
- ✅ Multiple services bundled
- ✅ Automatic payment methods
- ✅ Tech support subscription
- ✅ Device protection
- ✅ Partner and dependents (family plan)

## Creating Custom Tests

You can create your own test files following this template:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Credit card (automatic)",
  "MonthlyCharges": 50.0,
  "TotalCharges": 600.0
}
```

### Field Constraints

- `gender`: "Male" or "Female"
- `SeniorCitizen`: 0 or 1
- `Partner`: "Yes" or "No"
- `Dependents`: "Yes" or "No"
- `tenure`: number ≥ 0 (months)
- `PhoneService`: "Yes" or "No"
- `MultipleLines`: "Yes", "No", or "No phone service"
- `InternetService`: "DSL", "Fiber optic", or "No"
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: "Yes", "No", or "No internet service"
- `Contract`: "Month-to-month", "One year", or "Two year"
- `PaperlessBilling`: "Yes" or "No"
- `PaymentMethod`: "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"
- `MonthlyCharges`: number ≥ 0
- `TotalCharges`: number ≥ 0

## Batch Testing

To test multiple scenarios at once:

```bash
for file in test_request_*.json; do
    echo "Testing: $file"
    curl -s -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d @"$file" | jq
    echo ""
done
```

(Requires `jq` for pretty JSON output: `brew install jq`)
