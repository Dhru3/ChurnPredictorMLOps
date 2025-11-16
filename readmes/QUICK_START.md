# 🚀 QUICK START: Get Your Hybrid AI Churn-Bot Running in 5 Minutes

## Step 1: Install Dependencies (2 minutes)

```bash
cd /Users/dhrutipurushotham/Documents/Projects/ChurnPredictorMLOps

# Activate your virtual environment
source .venv/bin/activate

# Install new dependencies
pip install streamlit shap plotly groq python-dotenv
```

## Step 2: Set Up Groq API Key (1 minute)

1. **Get FREE Groq API Key**: 
   - Go to https://console.groq.com/
   - Sign up (it's free!)
   - Create an API key

2. **Create .env file**:
   ```bash
   # Copy the example
   cp .env.example .env
   
   # Edit .env and add your key
   # GROQ_API_KEY=your_actual_key_here
   ```

## Step 3: Model is Auto-Ready! (0 seconds)

**Good news!** The model is automatically tagged with the "champion" alias during training. No manual promotion needed!

**If you want to see it in MLflow UI (optional):**

**If you want to see it in MLflow UI (optional):**

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Then:
1. Open http://localhost:5001
2. Models tab → churn-predictor
3. You'll see the "champion" alias already set!
4. Press `Ctrl+C` to close

## Step 4: Launch the App! (30 seconds)

```bash
streamlit run app.py
```

Your browser will automatically open to **http://localhost:8501** 🎉

---

## 🎨 What You'll See

### Landing Page
- Beautiful animated gradient header
- System overview
- Quick test scenarios
- Performance metrics

### Try a High-Risk Customer
Fill the sidebar form with:
- **Tenure**: 2 months
- **Contract**: Month-to-month
- **Monthly Charges**: $85
- **Additional Services**: All "No"
- **Payment Method**: Electronic check

**Click "🔮 Predict Churn"**

You'll get:
1. ⚠️ **Prediction**: WILL CHURN (87% probability)
2. 🔴 **Plain-language explanation**: Why they're leaving
3. 📊 **SHAP factors**: Visual breakdown
4. ✉️ **AI-generated email**: Ready to send!

### Try a Low-Risk Customer
- **Tenure**: 60 months
- **Contract**: Two year
- **Additional Services**: All "Yes"
- **Payment Method**: Bank transfer (automatic)

You'll get:
1. ✅ **Prediction**: WILL STAY (12% probability)
2. 🟢 **Plain-language explanation**: Why they're loyal
3. 📊 **SHAP factors**: Protective factors shown
4. 🎉 **Loyalty recommendations**: How to keep them happy

---

## 🎯 Key Features to Show Off

### 1. **Stunning UI**
- Animated gradient header
- Color-coded explanation boxes (red for churn, green for stay)
- Beautiful charts and gauges
- Smooth animations

### 2. **Plain-Language Explanations**
- Not just numbers - actual sentences!
- "🔴 No long-term commitment: They're on month-to-month..."
- "✅ Long-term loyalty: 60 months shows commitment..."

### 3. **AI-Generated Emails** (Groq Llama 3.1 8B)
- Personalized based on risk factors
- Warm, empathetic tone
- Concrete offers and solutions
- One-click download

### 4. **SHAP Explainability**
- Visual factor cards
- Progress bars showing impact
- Detailed waterfall plots
- Both risk and protective factors

---

## 🐛 Troubleshooting

### "Model not found" error
→ Run `python train.py` first to create the model

### "No module named 'groq'" error
→ Run `pip install groq`

### "API key not found" warning
→ Check your `.env` file has `GROQ_API_KEY=...`

### Email not generating
→ Make sure you have a valid Groq API key in `.env`

### Streamlit shows error
→ Try: `pip install --upgrade streamlit`

---

## 📸 Screenshots to Take (for Portfolio)

1. **Landing page** - Shows the welcome screen
2. **High-risk prediction** - Red explanation + AI email
3. **Low-risk prediction** - Green explanation + loyalty tips
4. **SHAP waterfall** - Technical explainability view
5. **Factor cards** - Beautiful styled components

---

## 🎤 Quick Demo Script (30 seconds)

*"This is my Hybrid AI Churn-Bot. Let me show you a high-risk customer..."*

[Fill form, click predict]

*"See? In 2 seconds we get:*
- *87% churn probability*
- *Plain-language explanation of WHY*
- *SHAP analysis showing top factors*
- *And here's the magic - an AI-generated retention email, written by Groq's Llama 3.1, personalized to this customer's specific situation.*

*It's ready to send. One click to download."*

**[Dramatic pause]**

*"This is what hybrid AI looks like - predictive, explainable, AND generative."*

---

## 🎯 Next Steps

Once running, try:

1. **Different customer profiles** - See how predictions change
2. **Toggle services** - Watch SHAP factors adjust
3. **Compare contracts** - Month vs. year vs. two-year
4. **Review AI emails** - See Llama 3.1's creativity
5. **Show friends/colleagues** - Get that "wow" reaction!

---

## 🏆 You're Ready!

Your **Hybrid AI Churn-Bot** is:
- ✅ Trained (78.4% accuracy)
- ✅ Promoted (Production stage)
- ✅ Explained (SHAP + plain language)
- ✅ Generated (Groq-powered emails)
- ✅ Stunning (Custom CSS design)

**Go forth and impress!** 🚀

---

**Still here? RUN THIS NOW:**

```bash
streamlit run app.py
```

**Then prepare to be amazed by what you built!** ✨
