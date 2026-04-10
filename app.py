# ============================================================
# 💎 PREMIUM AI Demand Forecasting Dashboard — ENHANCED
# Features: Product/Store filters, Dataset Upload, AI Insights
# ============================================================

import warnings
import io
import os
import base64
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from groq import Groq


warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# AUTO-DOWNLOAD DATASET FROM GOOGLE DRIVE
# ─────────────────────────────────────────────
import gdown
import zipfile

DATA_DIR = 'dataset'
DRIVE_FILE_ID = "PASTE_YOUR_FILE_ID_HERE"   # ← Replace this with your actual Google Drive File ID

if not os.path.exists(DATA_DIR):
    print("📥 Downloading dataset from Google Drive...")
    zip_path = "dataset.zip"
    gdown.download(f"https://drive.google.com/uc?id=1dzgnKYhbP3cU6xgiEvFcz84DPR-uNzgC&confirm=t", zip_path, quiet=False)
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(".")
    os.remove(zip_path)
    print("✅ Dataset ready!")

# ─────────────────────────────────────────────
# HARDCODED API KEY
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # <-- Replace with your actual Groq API key


# ─────────────────────────────────────────────
# LOAD DEFAULT DATA
# ─────────────────────────────────────────────
def load_and_process(sales_df=None, products_df=None, stores_df=None):
    if sales_df is None:
        sales_df = pd.read_csv(f'{DATA_DIR}/sales.csv', parse_dates=['order_date'])
    if products_df is None:
        products_df = pd.read_csv(f'{DATA_DIR}/products.csv')
    if stores_df is None:
        stores_df = pd.read_csv(f'{DATA_DIR}/stores.csv')

    sales_full = sales_df.merge(products_df, on='product_id', how='left').merge(stores_df, on='store_id', how='left')

    daily = sales_df.groupby('order_date').agg(
        demand=('quantity', 'sum'),
        revenue=('revenue', 'sum'),
        orders=('order_id', 'count')
    ).reset_index().sort_values('order_date')

    daily['day_of_week'] = daily['order_date'].dt.dayofweek
    daily['month'] = daily['order_date'].dt.month
    daily['year'] = daily['order_date'].dt.year
    daily['trend'] = range(len(daily))

    for lag in [1, 7, 14]:
        daily[f'lag_{lag}'] = daily['demand'].shift(lag)

    daily['rolling_mean_7'] = daily['demand'].shift(1).rolling(7).mean()
    daily.dropna(inplace=True)

    return daily, sales_full


def load_and_process_filtered(sales_df, products_df, stores_df, product_ids=None, store_ids=None):
    """Load data filtered by product and store selections."""
    filtered = sales_df.copy()
    if product_ids:
        filtered = filtered[filtered['product_id'].isin(product_ids)]
    if store_ids:
        filtered = filtered[filtered['store_id'].isin(store_ids)]

    if filtered.empty:
        return None, None

    daily = filtered.groupby('order_date').agg(
        demand=('quantity', 'sum'),
        revenue=('revenue', 'sum'),
        orders=('order_id', 'count')
    ).reset_index().sort_values('order_date')

    daily['day_of_week'] = daily['order_date'].dt.dayofweek
    daily['month'] = daily['order_date'].dt.month
    daily['year'] = daily['order_date'].dt.year
    daily['trend'] = range(len(daily))

    for lag in [1, 7, 14]:
        daily[f'lag_{lag}'] = daily['demand'].shift(lag)

    daily['rolling_mean_7'] = daily['demand'].shift(1).rolling(7).mean()
    daily.dropna(inplace=True)

    if len(daily) < 20:
        return None, None

    return daily, None


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
FEATURES = ['day_of_week', 'month', 'trend', 'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7']


def train_models(daily):
    split = int(len(daily) * 0.8)
    train = daily[:split]
    test = daily[split:]

    X_train, y_train = train[FEATURES], train['demand']
    X_test, y_test = test[FEATURES], test['demand']

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }

    best_model = min(results, key=lambda x: results[x]['mae'])
    return results, best_model


# ─────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────
def forecast(model, df, days=90):
    data = df.copy()
    preds = []

    for _ in range(days):
        last = data.iloc[-1]
        date = last['order_date'] + pd.Timedelta(days=1)

        row = {
            'order_date': date,
            'day_of_week': date.dayofweek,
            'month': date.month,
            'year': date.year,
            'trend': len(data),
            'lag_1': last['demand'],
            'lag_7': data['demand'].iloc[-7],
            'lag_14': data['demand'].iloc[-14],
            'rolling_mean_7': data['demand'].tail(7).mean()
        }

        X = pd.DataFrame([row])[FEATURES]
        pred = model.predict(X)[0]
        noise = np.random.normal(0, df['demand'].std())
        row['demand'] = max(0, pred + noise)

        preds.append(row)
        data = pd.concat([data, pd.DataFrame([row])])

    return pd.DataFrame(preds)


# ─────────────────────────────────────────────
# AI INSIGHTS via Groq
# ─────────────────────────────────────────────
def get_ai_insights(daily, future, growth, best_model_name, model_metrics,
                    context_label="All Products / All Stores", api_key=None):

    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError("No API key provided.")

    client = Groq(api_key=key)

    demand_stats = {
        "historical_mean": round(float(daily['demand'].mean()), 2),
        "historical_std":  round(float(daily['demand'].std()),  2),
        "historical_max":  int(daily['demand'].max()),
        "historical_min":  int(daily['demand'].min()),
        "forecast_mean":   round(float(future['demand'].mean()), 2),
        "forecast_max":    int(future['demand'].max()),
        "forecast_min":    int(future['demand'].min()),
        "growth_pct":      round(growth, 2),
        "forecast_days":   len(future),
        "total_historical_days": len(daily),
        "best_model":  best_model_name,
        "model_r2":    round(model_metrics['r2'],   3),
        "model_mae":   round(model_metrics['mae'],  2),
        "model_rmse":  round(model_metrics['rmse'], 2),
        "context":     context_label
    }

    prompt = f"""
You are a senior supply chain and retail analytics consultant.

Analyze this demand forecasting data and provide actionable business insights.

Context: {demand_stats['context']}
Forecast Period: {demand_stats['forecast_days']} days
Best ML Model: {demand_stats['best_model']} (R2: {demand_stats['model_r2']}, MAE: {demand_stats['model_mae']})

Historical Demand: Mean={demand_stats['historical_mean']}, Std={demand_stats['historical_std']}, Max={demand_stats['historical_max']}, Min={demand_stats['historical_min']}
Forecast Demand: Mean={demand_stats['forecast_mean']}, Max={demand_stats['forecast_max']}, Min={demand_stats['forecast_min']}
Growth Trend: {demand_stats['growth_pct']}%

Return ONLY valid JSON (no markdown, no explanation):
{{
  "summary": "2-3 sentence executive summary",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "risks": ["risk 1", "risk 2"],
  "recommendations": ["action 1", "action 2", "action 3"],
  "inventory_action": "BUY_MORE or REDUCE or MAINTAIN",
  "confidence": "HIGH or MEDIUM or LOW",
  "confidence_reason": "one sentence explanation"
}}
"""

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a business analytics expert."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
    )

    text = response.choices[0].message.content.strip()

    # Safety: remove accidental markdown
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


# ─────────────────────────────────────────────
# BUILD DEFAULT DATA
# ─────────────────────────────────────────────
daily, sales_full = load_and_process()
results, best_model_name = train_models(daily)

raw_sales    = pd.read_csv(f'{DATA_DIR}/sales.csv',    parse_dates=['order_date'])
raw_products = pd.read_csv(f'{DATA_DIR}/products.csv')
raw_stores   = pd.read_csv(f'{DATA_DIR}/stores.csv')

product_options = [{'label': f"Product {pid}", 'value': pid} for pid in sorted(raw_sales['product_id'].unique())]
store_options   = [{'label': f"Store {sid}",   'value': sid} for sid in sorted(raw_sales['store_id'].unique())]


# ─────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Premium Forecast Dashboard"

CARD_STYLE = {
    "padding": "20px",
    "borderRadius": "12px",
    "background": "#1e293b",
    "color": "white",
    "flex": "1",
    "textAlign": "center"
}

TAB_STYLE = {
    'backgroundColor': '#1e293b',
    'color': '#94a3b8',
    'borderBottom': '2px solid #334155',
    'padding': '12px 20px',
    'fontWeight': '600',
    'fontSize': '14px'
}

TAB_SELECTED_STYLE = {
    'backgroundColor': '#0f172a',
    'color': '#38bdf8',
    'borderTop': '2px solid #38bdf8',
    'borderBottom': 'none',
    'padding': '12px 20px',
    'fontWeight': '700',
    'fontSize': '14px'
}

app.layout = html.Div(
    style={
        "background": "#0f172a", "color": "white",
        "padding": "20px", "minHeight": "100vh",
        "fontFamily": "system-ui, sans-serif"
    },
    children=[
        html.H1("🚀 AI Demand Forecast Dashboard",
                style={
                    "textAlign": "center", "marginBottom": "8px", "fontSize": "28px",
                    "background": "linear-gradient(90deg, #38bdf8, #818cf8)",
                    "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"
                }),

        html.P("ML-Powered · Product & Store Filters · Custom Dataset Upload · AI Insights",
               style={"textAlign": "center", "color": "#64748b", "marginBottom": "24px", "fontSize": "13px"}),

        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="📊 Overview",       value="overview", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="🤖 Models",         value="models",   style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="🔮 Forecast",       value="forecast", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="⚠️ Risk Alerts",    value="risk",     style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="📁 Upload Dataset", value="upload",   style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ]),

        html.Div(id="content", style={"marginTop": "20px"})
    ]
)


# ─────────────────────────────────────────────
# TAB ROUTER
# ─────────────────────────────────────────────
@app.callback(Output("content", "children"), Input("tabs", "value"))
def render(tab):

    # ── OVERVIEW ────────────────────────────────────────────────────────────────
    if tab == "overview":
        return html.Div([
            html.Div([
                html.Div([
                    html.P("Total Demand", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{int(daily['demand'].sum()):,}", style={"margin": 0, "color": "#38bdf8"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("Avg Daily Demand", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{int(daily['demand'].mean()):,}", style={"margin": 0, "color": "#a78bfa"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("Peak Demand", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{int(daily['demand'].max()):,}", style={"margin": 0, "color": "#34d399"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("Data Points", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{len(daily):,}", style={"margin": 0, "color": "#fb923c"})
                ], style=CARD_STYLE),
            ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),

            dcc.Graph(
                figure=go.Figure([
                    go.Scatter(
                        x=daily['order_date'], y=daily['demand'], name="Demand",
                        line=dict(color='#38bdf8', width=2),
                        fill='tozeroy', fillcolor='rgba(56,189,248,0.1)'
                    )
                ]).update_layout(
                    template="plotly_dark", title="📈 Historical Demand Trend",
                    paper_bgcolor='#1e293b', plot_bgcolor='#1e293b'
                )
            ),

            dcc.Graph(
                figure=go.Figure([
                    go.Bar(x=daily['order_date'], y=daily['revenue'],
                           name="Revenue", marker_color='#818cf8')
                ]).update_layout(
                    template="plotly_dark", title="💰 Revenue Over Time",
                    paper_bgcolor='#1e293b', plot_bgcolor='#1e293b'
                )
            )
        ])

    # ── MODELS ──────────────────────────────────────────────────────────────────
    elif tab == "models":
        return html.Div([
            html.H3("Model Performance Comparison", style={"marginBottom": "16px", "color": "#38bdf8"}),
            dash_table.DataTable(
                data=[{
                    "Model": k,
                    "MAE":   round(v['mae'],  2),
                    "RMSE":  round(v['rmse'], 2),
                    "R²":    round(v['r2'],   3),
                    "Best?": "✅ Best" if k == best_model_name else ""
                } for k, v in results.items()],
                columns=[{"name": i, "id": i} for i in ["Model", "MAE", "RMSE", "R²", "Best?"]],
                style_cell={"background": "#1e293b", "color": "white", "textAlign": "center",
                             "padding": "12px", "border": "1px solid #334155"},
                style_header={"background": "#334155", "fontWeight": "bold", "color": "#38bdf8"},
                style_data_conditional=[
                    {'if': {'filter_query': '{Best?} = "✅ Best"'},
                     'backgroundColor': '#1a3a2a', 'color': '#34d399'}
                ]
            ),
            html.Br(),
            dcc.Graph(
                figure=go.Figure([
                    go.Bar(
                        x=list(results.keys()),
                        y=[v['r2'] for v in results.values()],
                        name="R² Score",
                        marker_color=['#34d399' if k == best_model_name else '#334155' for k in results]
                    )
                ]).update_layout(
                    template="plotly_dark", title="Model R² Comparison",
                    paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
                    yaxis=dict(range=[0, 1])
                )
            )
        ])

    # ── FORECAST ─────────────────────────────────────────────────────────────────
    elif tab == "forecast":
        return html.Div([

            # Filter row
            html.Div([
                html.Div([
                    html.Label("🏪 Filter by Store",
                               style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="store-filter",
                        options=store_options,
                        value=[],
                        multi=True,
                        placeholder="All Stores",
                        style={"color": "black"}
                    )
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("📦 Filter by Product",
                               style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="product-filter",
                        options=product_options,
                        value=[],
                        multi=True,
                        placeholder="All Products",
                        style={"color": "black"}
                    )
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("🤖 Model",
                               style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="model",
                        options=[{"label": k, "value": k} for k in results],
                        value=best_model_name,
                        style={"color": "black"}
                    )
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

            # Slider
            html.Label("📅 Forecast Horizon (Days)",
                       style={"color": "#94a3b8", "fontSize": "12px"}),
            dcc.Slider(30, 120, 30, value=90, id="days",
                       marks={30: "30d", 60: "60d", 90: "90d", 120: "120d"}),
            html.Br(),

            # Model Accuracy Panel
            html.Div(id="model-accuracy-panel", style={"marginBottom": "20px"}),

            # AI Insights panel (no API key input — key is hardcoded)
            html.Div([
                html.Div([
                    html.Button("✨ Generate AI Insights", id="ai-btn",
                                style={
                                    "background": "linear-gradient(135deg, #38bdf8, #818cf8)",
                                    "color": "white", "border": "none", "padding": "10px 24px",
                                    "borderRadius": "8px", "cursor": "pointer",
                                    "fontWeight": "700", "fontSize": "14px"
                                }),
                    html.Span(" Powered by AI — business recommendations based on your forecast data",
                              style={"color": "#64748b", "fontSize": "12px", "marginLeft": "10px"})
                ])
            ], style={
                "marginBottom": "16px", "background": "#1e293b",
                "padding": "16px", "borderRadius": "10px", "border": "1px solid #334155"
            }),

            # AI insights output area
            html.Div(id="ai-insights-panel", style={"marginBottom": "20px"}),

            # Forecast chart
            dcc.Loading(dcc.Graph(id="graph"), type="circle", color="#38bdf8"),
            html.Br(),

            # Insight stat cards
            html.Div(id="insights", style={"marginTop": "20px"})
        ])

    # ── RISK ALERTS ───────────────────────────────────────────────────────────────
    elif tab == "risk":
        model = results[best_model_name]['model']
        future = forecast(model, daily, 90)

        mean = daily['demand'].mean()
        std  = daily['demand'].std()

        future['risk'] = 'Normal'
        future.loc[future['demand'] > mean + 0.5 * std, 'risk'] = 'Overstock'
        future.loc[future['demand'] < mean - 0.5 * std, 'risk'] = 'Understock'

        overstock  = (future['risk'] == 'Overstock').sum()
        understock = (future['risk'] == 'Understock').sum()
        normal     = (future['risk'] == 'Normal').sum()

        return html.Div([
            html.Div([
                html.Div([
                    html.P("⚠️ Overstock Days",  style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(str(overstock),  style={"margin": 0, "color": "#ff4d4f"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("🟡 Understock Days", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(str(understock), style={"margin": 0, "color": "#faad14"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("✅ Normal Days",     style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(str(normal),     style={"margin": 0, "color": "#34d399"})
                ], style=CARD_STYLE),
            ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),

            dash_table.DataTable(
                data=future[['order_date', 'demand', 'risk']].round(1).to_dict('records'),
                columns=[{"name": i, "id": i} for i in ['order_date', 'demand', 'risk']],
                page_size=15,
                style_cell={"background": "#1e293b", "color": "white",
                             "textAlign": "center", "border": "1px solid #334155"},
                style_header={"background": "#334155", "fontWeight": "bold", "color": "#38bdf8"},
                style_data_conditional=[
                    {'if': {'filter_query': '{risk} = "Overstock"'},
                     'backgroundColor': '#3b0000', 'color': '#ff4d4f'},
                    {'if': {'filter_query': '{risk} = "Understock"'},
                     'backgroundColor': '#3b2500', 'color': '#faad14'},
                    {'if': {'filter_query': '{risk} = "Normal"'},
                     'backgroundColor': '#002b1a', 'color': '#34d399'}
                ]
            )
        ])

    # ── UPLOAD DATASET ────────────────────────────────────────────────────────────
    elif tab == "upload":
        return html.Div([
            html.H3("📁 Upload Custom Dataset",
                    style={"color": "#38bdf8", "marginBottom": "8px"}),
            html.P("Upload your own CSV files to get the same forecasting features. "
                   "Files must match the schema below.",
                   style={"color": "#94a3b8", "marginBottom": "20px"}),

            # Schema reference
            html.Div([
                html.Div([
                    html.H4("sales.csv", style={"color": "#34d399", "marginBottom": "8px"}),
                    html.Code("order_id, order_date, product_id, store_id, quantity, revenue",
                              style={"color": "#e2e8f0", "fontSize": "12px"})
                ], style={**CARD_STYLE, "textAlign": "left"}),
                html.Div([
                    html.H4("products.csv", style={"color": "#a78bfa", "marginBottom": "8px"}),
                    html.Code("product_id, product_name, category, price",
                              style={"color": "#e2e8f0", "fontSize": "12px"})
                ], style={**CARD_STYLE, "textAlign": "left"}),
                html.Div([
                    html.H4("stores.csv", style={"color": "#fb923c", "marginBottom": "8px"}),
                    html.Code("store_id, store_name, region, city",
                              style={"color": "#e2e8f0", "fontSize": "12px"})
                ], style={**CARD_STYLE, "textAlign": "left"}),
            ], style={"display": "flex", "gap": "15px", "marginBottom": "24px"}),

            # Upload zones
            html.Div([
                html.Div([
                    html.Label("📄 Upload sales.csv",
                               style={"color": "#94a3b8", "marginBottom": "8px", "display": "block"}),
                    dcc.Upload(
                        id='upload-sales',
                        children=html.Div(['Drag & Drop or ', html.A('Select sales.csv')]),
                        style={"width": "100%", "height": "60px", "lineHeight": "60px",
                               "borderWidth": "2px", "borderStyle": "dashed",
                               "borderRadius": "8px", "textAlign": "center",
                               "borderColor": "#334155", "color": "#94a3b8"}
                    )
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("📄 Upload products.csv",
                               style={"color": "#94a3b8", "marginBottom": "8px", "display": "block"}),
                    dcc.Upload(
                        id='upload-products',
                        children=html.Div(['Drag & Drop or ', html.A('Select products.csv')]),
                        style={"width": "100%", "height": "60px", "lineHeight": "60px",
                               "borderWidth": "2px", "borderStyle": "dashed",
                               "borderRadius": "8px", "textAlign": "center",
                               "borderColor": "#334155", "color": "#94a3b8"}
                    )
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("📄 Upload stores.csv",
                               style={"color": "#94a3b8", "marginBottom": "8px", "display": "block"}),
                    dcc.Upload(
                        id='upload-stores',
                        children=html.Div(['Drag & Drop or ', html.A('Select stores.csv')]),
                        style={"width": "100%", "height": "60px", "lineHeight": "60px",
                               "borderWidth": "2px", "borderStyle": "dashed",
                               "borderRadius": "8px", "textAlign": "center",
                               "borderColor": "#334155", "color": "#94a3b8"}
                    )
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),

            html.Button("🚀 Process Uploaded Data", id="process-upload-btn",
                        style={
                            "background": "linear-gradient(135deg, #34d399, #059669)",
                            "color": "white", "border": "none", "padding": "12px 32px",
                            "borderRadius": "8px", "cursor": "pointer",
                            "fontWeight": "700", "fontSize": "14px", "marginBottom": "20px"
                        }),

            html.Div(id="upload-status")
        ])


# ─────────────────────────────────────────────
# FORECAST CALLBACK (with product/store filters + model accuracy)
# ─────────────────────────────────────────────
@app.callback(
    Output("graph", "figure"),
    Output("insights", "children"),
    Output("model-accuracy-panel", "children"),
    Input("model", "value"),
    Input("days", "value"),
    Input("product-filter", "value"),
    Input("store-filter", "value"),
)
def update_graph(model_name, days, product_ids, store_ids):

    if product_ids or store_ids:
        filtered_daily, _ = load_and_process_filtered(
            raw_sales, raw_products, raw_stores,
            product_ids or None, store_ids or None
        )
        if filtered_daily is None:
            empty_fig = go.Figure().update_layout(
                template="plotly_dark",
                title="❌ Not enough data for this filter combination",
                paper_bgcolor='#1e293b', plot_bgcolor='#1e293b'
            )
            empty_accuracy = html.P("No accuracy data available for selected filters.",
                                    style={"color": "#ff4d4f"})
            return empty_fig, html.P("No data available for selected filters.",
                                     style={"color": "#ff4d4f"}), empty_accuracy

        filt_results, filt_best = train_models(filtered_daily)
        use_daily = filtered_daily
        use_model_name = model_name if model_name in filt_results else filt_best
        use_model = filt_results[use_model_name]['model']
        use_metrics = filt_results[use_model_name]
    else:
        use_daily = daily
        use_model = results[model_name]['model']
        use_metrics = results[model_name]
        use_model_name = model_name

    future = forecast(use_model, use_daily, days)
    future['revenue'] = future['demand'] * use_daily['demand'].mean()

    growth = ((future['demand'].mean() - use_daily['demand'].mean()) / use_daily['demand'].mean()) * 100

    recommendation = "Stable demand — maintain current inventory"
    if growth > 5:
        recommendation = "Demand rising — increase stock 📈"
    elif growth < -5:
        recommendation = "Demand falling — reduce stock 📉"

    context_parts = []
    if product_ids:
        context_parts.append(f"Products: {', '.join(str(p) for p in product_ids)}")
    if store_ids:
        context_parts.append(f"Stores: {', '.join(str(s) for s in store_ids)}")
    context_str = " | ".join(context_parts) if context_parts else "All Products / All Stores"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=use_daily['order_date'], y=use_daily['demand'],
        name="Actual", line=dict(color='#38bdf8', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future['order_date'], y=future['demand'],
        name="Forecast", line=dict(color='#a78bfa', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=future['order_date'], y=future['revenue'],
        name="Revenue", line=dict(color='#34d399', width=1), yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"📊 Demand Forecast · {context_str}",
        paper_bgcolor='#1e293b', plot_bgcolor='#1e293b',
        yaxis=dict(title="Demand"),
        yaxis2=dict(title="Revenue", overlaying='y', side='right'),
        legend=dict(x=0, y=1)
    )

    insights = html.Div([
        html.Div([
            html.P("📈 Growth Trend", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
            html.H3(f"{growth:.2f}%", style={"margin": 0, "color": "#38bdf8"})
        ], style=CARD_STYLE),
        html.Div([
            html.P("📦 Avg Forecast Demand", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
            html.H3(f"{int(future['demand'].mean()):,}", style={"margin": 0, "color": "#a78bfa"})
        ], style=CARD_STYLE),
        html.Div([
            html.P("💰 Revenue Forecast", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
            html.H3(f"{int(future['revenue'].sum()):,}", style={"margin": 0, "color": "#34d399"})
        ], style=CARD_STYLE),
        html.Div([
            html.P("🤖 Recommendation", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
            html.P(recommendation, style={"margin": 0, "color": "#fb923c", "fontSize": "13px"})
        ], style=CARD_STYLE),
    ], style={"display": "flex", "gap": "15px"})

    # ── Model Accuracy Panel ──────────────────────────────────────────────────────
    mae_val  = round(use_metrics['mae'],  2)
    rmse_val = round(use_metrics['rmse'], 2)
    r2_val   = round(use_metrics['r2'],   3)

    # R² colour: green ≥ 0.8, yellow ≥ 0.5, red < 0.5
    r2_color = "#34d399" if r2_val >= 0.8 else ("#faad14" if r2_val >= 0.5 else "#ff4d4f")

    accuracy_panel = html.Div([
        html.Div(
            style={
                "background": "#1e293b", "borderRadius": "10px",
                "padding": "14px 20px", "border": "1px solid #334155",
                "marginBottom": "4px"
            },
            children=[
                html.Div([
                    html.Span("📐 Model Accuracy — ",
                              style={"color": "#94a3b8", "fontSize": "13px", "fontWeight": "600"}),
                    html.Span(use_model_name,
                              style={"color": "#38bdf8", "fontSize": "13px", "fontWeight": "700"}),
                ], style={"marginBottom": "12px"}),

                html.Div([
                    # MAE card
                    html.Div([
                        html.P("MAE", style={"margin": 0, "color": "#94a3b8", "fontSize": "11px",
                                             "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                        html.H4(f"{mae_val:,}", style={"margin": "4px 0 0 0", "color": "#fb923c",
                                                        "fontSize": "20px"}),
                        html.P("Mean Absolute Error", style={"margin": "2px 0 0 0",
                                                              "color": "#475569", "fontSize": "10px"})
                    ], style={**CARD_STYLE, "padding": "14px", "background": "#0f172a"}),

                    # RMSE card
                    html.Div([
                        html.P("RMSE", style={"margin": 0, "color": "#94a3b8", "fontSize": "11px",
                                              "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                        html.H4(f"{rmse_val:,}", style={"margin": "4px 0 0 0", "color": "#a78bfa",
                                                         "fontSize": "20px"}),
                        html.P("Root Mean Sq. Error", style={"margin": "2px 0 0 0",
                                                              "color": "#475569", "fontSize": "10px"})
                    ], style={**CARD_STYLE, "padding": "14px", "background": "#0f172a"}),

                    # R² card
                    html.Div([
                        html.P("R² Score", style={"margin": 0, "color": "#94a3b8", "fontSize": "11px",
                                                   "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                        html.H4(f"{r2_val}", style={"margin": "4px 0 0 0", "color": r2_color,
                                                     "fontSize": "20px"}),
                        html.P("Variance Explained", style={"margin": "2px 0 0 0",
                                                             "color": "#475569", "fontSize": "10px"})
                    ], style={**CARD_STYLE, "padding": "14px", "background": "#0f172a"}),

                    # Accuracy % card (derived from R²)
                    html.Div([
                        html.P("Accuracy", style={"margin": 0, "color": "#94a3b8", "fontSize": "11px",
                                                   "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                        html.H4(f"{max(0, r2_val * 100):.1f}%",
                                style={"margin": "4px 0 0 0", "color": r2_color, "fontSize": "20px"}),
                        html.P("Based on R² Score", style={"margin": "2px 0 0 0",
                                                            "color": "#475569", "fontSize": "10px"})
                    ], style={**CARD_STYLE, "padding": "14px", "background": "#0f172a"}),

                ], style={"display": "flex", "gap": "12px"})
            ]
        )
    ])

    return fig, insights, accuracy_panel


# ─────────────────────────────────────────────
# AI INSIGHTS CALLBACK
# ─────────────────────────────────────────────
@app.callback(
    Output("ai-insights-panel", "children"),
    Input("ai-btn", "n_clicks"),
    State("model", "value"),
    State("days", "value"),
    State("product-filter", "value"),
    State("store-filter", "value"),
    prevent_initial_call=True
)
def generate_ai_insights(n_clicks, model_name, days, product_ids, store_ids):
    if not n_clicks:
        return html.Div()

    try:
        if product_ids or store_ids:
            filtered_daily, _ = load_and_process_filtered(
                raw_sales, raw_products, raw_stores,
                product_ids or None, store_ids or None
            )
            if filtered_daily is None:
                return html.Div("⚠️ Not enough data for AI insights with current filters.",
                                style={"color": "#ff4d4f"})
            use_daily = filtered_daily
            filt_results, filt_best = train_models(filtered_daily)
            use_model_dict = filt_results.get(model_name, filt_results[filt_best])
        else:
            use_daily = daily
            use_model_dict = results[model_name]

        future = forecast(use_model_dict['model'], use_daily, days)
        growth = ((future['demand'].mean() - use_daily['demand'].mean()) / use_daily['demand'].mean()) * 100

        context_parts = []
        if product_ids:
            context_parts.append(f"Products {product_ids}")
        if store_ids:
            context_parts.append(f"Stores {store_ids}")
        context_label = " | ".join(context_parts) or "All Products / All Stores"

        ai_data = get_ai_insights(
            use_daily, future, growth,
            model_name, use_model_dict,
            context_label
            # api_key uses GROQ_API_KEY hardcoded in get_ai_insights
        )

        inv_colors  = {"BUY_MORE": "#34d399", "REDUCE": "#ff4d4f", "MAINTAIN": "#faad14"}
        conf_colors = {"HIGH": "#34d399",      "MEDIUM": "#faad14", "LOW":      "#ff4d4f"}

        inv_action = ai_data.get("inventory_action", "MAINTAIN")
        confidence = ai_data.get("confidence",       "MEDIUM")

        return html.Div([
            html.Div(
                style={"background": "#1e293b", "borderRadius": "12px",
                       "padding": "20px", "border": "1px solid #334155"},
                children=[
                    html.Div([
                        html.Span("✨ AI Business Insights",
                                  style={"fontWeight": "700", "color": "#38bdf8", "fontSize": "16px"}),
                        html.Span("  ·  Inventory Action: ",
                                  style={"color": "#94a3b8", "fontSize": "13px", "marginLeft": "12px"}),
                        html.Span(inv_action,
                                  style={"color": inv_colors.get(inv_action, "white"), "fontWeight": "700"}),
                        html.Span("  ·  Confidence: ",
                                  style={"color": "#94a3b8", "fontSize": "13px", "marginLeft": "12px"}),
                        html.Span(confidence,
                                  style={"color": conf_colors.get(confidence, "white"), "fontWeight": "700"}),
                    ], style={"marginBottom": "12px"}),

                    html.P(ai_data.get("summary", ""),
                           style={"color": "#e2e8f0", "marginBottom": "16px", "lineHeight": "1.6"}),

                    html.Div([
                        html.Div([
                            html.H5("💡 Key Insights",
                                    style={"color": "#a78bfa", "marginBottom": "8px"}),
                            html.Ul([html.Li(i, style={"color": "#cbd5e1", "marginBottom": "4px"})
                                     for i in ai_data.get("key_insights", [])])
                        ], style={"flex": "1"}),
                        html.Div([
                            html.H5("⚠️ Risks",
                                    style={"color": "#fb923c", "marginBottom": "8px"}),
                            html.Ul([html.Li(r, style={"color": "#cbd5e1", "marginBottom": "4px"})
                                     for r in ai_data.get("risks", [])])
                        ], style={"flex": "1"}),
                        html.Div([
                            html.H5("✅ Recommendations",
                                    style={"color": "#34d399", "marginBottom": "8px"}),
                            html.Ul([html.Li(r, style={"color": "#cbd5e1", "marginBottom": "4px"})
                                     for r in ai_data.get("recommendations", [])])
                        ], style={"flex": "1"}),
                    ], style={"display": "flex", "gap": "20px"}),

                    html.P(f"📊 {ai_data.get('confidence_reason', '')}",
                           style={"color": "#64748b", "fontSize": "12px",
                                  "marginTop": "12px", "fontStyle": "italic"})
                ]
            )
        ])

    except Exception as e:
        return html.Div([
            html.P(f"⚠️ AI insights unavailable: {str(e)}",
                   style={"color": "#ff4d4f", "padding": "12px", "background": "#1e293b",
                          "borderRadius": "8px", "border": "1px solid #ff4d4f"})
        ])


# ─────────────────────────────────────────────
# UPLOAD CALLBACK
# ─────────────────────────────────────────────
def parse_csv(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))


@app.callback(
    Output("upload-status", "children"),
    Input("process-upload-btn", "n_clicks"),
    State("upload-sales",    "contents"),
    State("upload-products", "contents"),
    State("upload-stores",   "contents"),
    prevent_initial_call=True
)
def process_upload(n_clicks, sales_content, products_content, stores_content):
    if not sales_content:
        return html.Div("❌ Please upload at least sales.csv", style={"color": "#ff4d4f"})

    try:
        up_sales = parse_csv(sales_content)
        up_sales['order_date'] = pd.to_datetime(up_sales['order_date'])

        up_products = parse_csv(products_content) if products_content else raw_products
        up_stores   = parse_csv(stores_content)   if stores_content   else raw_stores

        up_daily, _ = load_and_process(up_sales, up_products, up_stores)
        up_results, up_best = train_models(up_daily)
        up_model  = up_results[up_best]['model']
        up_future = forecast(up_model, up_daily, 90)
        up_future['revenue'] = up_future['demand'] * up_daily['demand'].mean()
        growth = ((up_future['demand'].mean() - up_daily['demand'].mean()) / up_daily['demand'].mean()) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=up_daily['order_date'], y=up_daily['demand'],
                                 name="Actual", line=dict(color='#38bdf8', width=2)))
        fig.add_trace(go.Scatter(x=up_future['order_date'], y=up_future['demand'],
                                 name="Forecast", line=dict(color='#a78bfa', width=2, dash='dash')))
        fig.update_layout(template="plotly_dark", title="📊 Uploaded Dataset Forecast",
                          paper_bgcolor='#1e293b', plot_bgcolor='#1e293b')

        return html.Div([
            html.Div([
                html.Div([
                    html.P("Rows Loaded", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{len(up_sales):,}", style={"margin": 0, "color": "#34d399"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("Best Model", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(up_best, style={"margin": 0, "color": "#38bdf8", "fontSize": "16px"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("R² Score", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{up_results[up_best]['r2']:.3f}",
                            style={"margin": 0, "color": "#a78bfa"})
                ], style=CARD_STYLE),
                html.Div([
                    html.P("Growth Trend", style={"margin": 0, "color": "#94a3b8", "fontSize": "12px"}),
                    html.H3(f"{growth:.1f}%", style={"margin": 0, "color": "#fb923c"})
                ], style=CARD_STYLE),
            ], style={"display": "flex", "gap": "15px", "marginBottom": "20px"}),

            dcc.Graph(figure=fig),
            html.Br(),

            dash_table.DataTable(
                data=[{
                    "Model": k,
                    "MAE":   round(v['mae'],  2),
                    "RMSE":  round(v['rmse'], 2),
                    "R²":    round(v['r2'],   3),
                    "Best?": "✅" if k == up_best else ""
                } for k, v in up_results.items()],
                columns=[{"name": i, "id": i} for i in ["Model", "MAE", "RMSE", "R²", "Best?"]],
                style_cell={"background": "#1e293b", "color": "white",
                             "textAlign": "center", "border": "1px solid #334155"},
                style_header={"background": "#334155", "fontWeight": "bold", "color": "#38bdf8"}
            )
        ])

    except Exception as e:
        return html.Div([
            html.P(f"❌ Error processing upload: {str(e)}",
                   style={"color": "#ff4d4f", "padding": "16px", "background": "#1e293b",
                          "borderRadius": "8px", "border": "1px solid #ff4d4f"})
        ])


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
