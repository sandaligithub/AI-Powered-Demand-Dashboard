# 🚀 AI Demand Forecast Dashboard

> ML-powered demand forecasting with product & store filters, custom dataset uploads, and AI-generated business insights — built with Dash & Plotly.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Dash](https://img.shields.io/badge/Dash-2.x-informational?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-purple)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📸 Features

| Feature | Description |
|---|---|
| 📊 **Overview Tab** | Historical demand trend & revenue charts |
| 🤖 **Models Tab** | Compare Random Forest, Gradient Boosting & Linear Regression |
| 🔮 **Forecast Tab** | 30–120 day demand forecasts with model accuracy metrics |
| ⚠️ **Risk Alerts** | Overstock / Understock / Normal day classification |
| 📁 **Upload Dataset** | Plug in your own CSV data and get instant forecasts |
| ✨ **AI Insights** | LLM-generated executive summary, risks & recommendations |

---

## 🗂️ Project Structure

```
demand-forecast-dashboard/
│
├── app.py                  # Main Dash application
├── requirements.txt        # Python dependencies
├── Procfile                # For Heroku deployment
├── render.yaml             # For Render deployment
├── .gitignore              # Files to exclude from git
├── README.md               # This file
│
└── dataset/
    ├── sales.csv           # Order-level sales data
    ├── products.csv        # Product catalogue
    └── stores.csv          # Store information
```

---

## 📋 Dataset Schema

Your CSV files must follow this schema:

**`sales.csv`**
```
order_id, order_date, product_id, store_id, quantity, revenue
```

**`products.csv`**
```
product_id, product_name, category, price
```

**`stores.csv`**
```
store_id, store_name, region, city
```

---

## ⚙️ ML Models Used

- **Random Forest Regressor** — ensemble of decision trees, handles non-linearity well
- **Gradient Boosting Regressor** — sequential boosting for high accuracy
- **Linear Regression** — fast baseline model

**Features engineered per day:**
`day_of_week`, `month`, `trend`, `lag_1`, `lag_7`, `lag_14`, `rolling_mean_7`

The best model is selected automatically based on lowest **MAE** on the test split (last 20% of data).

---

## 📐 Model Accuracy Metrics

Each forecast displays live accuracy for the selected model:

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error in demand units |
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **R² Score** | Variance explained (1.0 = perfect, 0 = no predictive power) |
| **Accuracy %** | R² expressed as a percentage for quick readability |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/demand-forecast-dashboard.git
cd demand-forecast-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Open `app.py` and set your Groq API key at line 22:

```python
GROQ_API_KEY = "your-groq-api-key-here"
```

> Get a free key at [console.groq.com](https://console.groq.com)

### 4. Run the app

```bash
python app.py
```

Visit `http://localhost:8050` in your browser.

---

## 🌐 Deployment

### Render (Recommended — Free)

1. Add `server = app.server` in `app.py` after app initialization
2. Push to GitHub
3. Go to [render.com](https://render.com) → New Web Service → connect repo
4. Set start command: `gunicorn app:server`
5. Add `GROQ_API_KEY` in Render's Environment Variables

### Heroku

```bash
heroku create your-app-name
heroku config:set GROQ_API_KEY=your-key
git push heroku main
```

### Local Network

```python
# In app.py, change the last line to:
app.run(host="0.0.0.0", port=8050, debug=False)
```

---

## 🔐 Security Note

Never commit your API key to GitHub. For production, use environment variables:

```python
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
```

---

## 📦 Requirements

```
dash
pandas
numpy
plotly
scikit-learn
groq
gunicorn
```

---

## 🤖 AI Insights

Powered by **LLaMA 3.1 8B** via Groq API. For each forecast, the AI generates:

- 📝 Executive summary
- 💡 Key demand insights
- ⚠️ Risk factors
- ✅ Inventory recommendations
- 📦 Action: `BUY_MORE` / `REDUCE` / `MAINTAIN`
- 🎯 Confidence level: `HIGH` / `MEDIUM` / `LOW`

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙌 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push and open a Pull Request

## Dataset
The dataset is hosted on Google Drive and is **automatically downloaded** 
when you run the script for the first time.

No manual download needed — just run `python your_script.py`
