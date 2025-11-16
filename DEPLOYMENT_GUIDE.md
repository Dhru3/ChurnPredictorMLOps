# 🚀 Deploying to Streamlit Cloud

## Quick Deployment Steps

### 1. Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Groq API key

### 2. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: ChurnPredictor MLOps"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ChurnPredictorMLOps.git
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `YOUR_USERNAME/ChurnPredictorMLOps`
4. Main file path: `app.py`
5. Click "Advanced settings"
6. Add secrets:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
7. Click "Deploy!"

### 4. Access Your App
Your app will be available at:
`https://YOUR_USERNAME-churnpredictomlops.streamlit.app`

---

## Environment Variables

### Required Secrets (.streamlit/secrets.toml for local, Streamlit Cloud secrets for production):

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a free Groq API key at: [console.groq.com](https://console.groq.com)

---

## File Structure for Deployment

```
ChurnPredictorMLOps/
├── app.py                          # Main Streamlit app
├── train.py                        # Model training script
├── model_validation.py             # Validation system
├── requirements.txt                # Python dependencies
├── Telco-Customer-Churn.csv        # Dataset
├── .streamlit/
│   ├── config.toml                # Streamlit configuration
│   └── secrets.toml               # API keys (DON'T COMMIT!)
├── pages/
│   ├── 1_📊_MLOps_Dashboard.py    # Model comparison
│   ├── 2_📡_Production_Monitor.py  # Production monitoring
│   └── 3_🧪_AB_Testing.py         # A/B testing
├── .github/workflows/
│   └── mlops.yml                  # CI/CD pipeline
└── mlruns/                        # MLflow tracking (auto-generated)
```

---

## Post-Deployment

### Test Your Deployed App
1. Open the deployed URL
2. Fill in customer details
3. Click "Predict Churn"
4. Verify predictions work
5. Check the MLOps Dashboard page
6. Test Production Monitor page
7. Try A/B Testing page

### Monitor Your App
- **Streamlit Cloud Dashboard**: Check app health and logs
- **Production Monitor Page**: View prediction logs and drift detection
- **MLOps Dashboard**: Compare model versions

---

## Troubleshooting

### App Won't Start
- Check Streamlit Cloud logs for errors
- Verify `requirements.txt` is complete
- Ensure secrets are configured correctly

### Missing Data
- Confirm `Telco-Customer-Churn.csv` is in the repository
- Check file permissions on Streamlit Cloud

### Model Not Loading
- Ensure MLflow database (`mlflow.db`) or runs are in the repo
- Or retrain the model by running `python train.py` before deployment

### GROQ_API_KEY Error
- Add the API key in Streamlit Cloud secrets
- Format: `GROQ_API_KEY = "gsk_..."`

---

## Optional: Custom Domain

1. Go to Streamlit Cloud settings
2. Click "Custom domain"
3. Follow instructions to add your domain

---

## Production Checklist

- [ ] Model trained and registered in MLflow
- [ ] `requirements.txt` is complete
- [ ] `.streamlit/secrets.toml` configured locally
- [ ] Streamlit Cloud secrets configured
- [ ] GitHub repository is public or Streamlit Cloud has access
- [ ] Dataset (`Telco-Customer-Churn.csv`) is in repository
- [ ] All pages work correctly locally
- [ ] App deployed successfully
- [ ] Test predictions on deployed app
- [ ] Check all dashboard pages
- [ ] Monitor logs for errors

---

## CI/CD Pipeline (GitHub Actions)

The included `.github/workflows/mlops.yml` provides:
- Automated testing on every push
- Model training and validation
- Staged deployments (staging → production)
- Drift monitoring

To enable:
1. Add GitHub repository secrets:
   - `MLFLOW_TRACKING_URI` (optional, for remote tracking)
   - `GROQ_API_KEY`
2. Push to `main` branch to trigger pipeline

---

## Support

For issues or questions:
1. Check Streamlit Community Forum
2. Review [Streamlit Docs](https://docs.streamlit.io)
3. Check [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

**Happy Deploying! 🚀**
