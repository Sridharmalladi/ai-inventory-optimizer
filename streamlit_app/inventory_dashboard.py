import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.forecasting import predict_demand
from src.optimizer import optimize_inventory
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Inventory Optimizer", layout="wide")

# Optional: show current working directory for debug
# st.text(f"📁 Current working directory: {os.getcwd()}")

# Sidebar navigation
page = st.sidebar.radio("Go to", [
    "📊 EDA & Modeling",
    "📉 Category & Seasonal Insights",
    "📈 Optimization Dashboard",
    "📄 Executive Summary Report"
])

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("🚨 Model file not found! Please make sure 'models/xgb_model.pkl' exists.")
        return None

model = load_model()

# Load data
@st.cache_data
def load_processed_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "final_model_data.csv")
        return pd.read_csv(data_path, parse_dates=['Date'])
    except FileNotFoundError:
        st.error("🚨 Data file not found! Please ensure 'data/processed_data/final_model_data.csv' is in your repo.")
        return pd.DataFrame()

df = load_processed_data()

# Cache RF model training
@st.cache_resource
def train_rf(X_train, y_train):
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    return model_rf

# --- PAGE 1: EDA & MODELING ---
if page == "📊 EDA & Modeling" and not df.empty and model is not None:
    st.markdown("""
    <style>
        .big-font { font-size: 32px !important; font-weight: 600; color: #2e86de; }
        .section { background-color: #f8f9fa; padding: 20px; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-font">🔍 Exploratory Data Analysis & Model Development</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="section">
        <ul>
            <li><b>EDA Highlights</b>:<br>Weekly seasonality and promotion-driven demand spikes observed.</li>
            <li><b>Model Experimentation</b>:<br>Random Forest (baseline) and XGBoost (final model).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    features = ['DayOfWeek', 'IsWeekend', 'IsPromo', 'RollingDemand7', 'RollingDemand14',
                'Lag_1', 'Discount', 'Inventory_Level']
    X = df[features]
    y = df['Units Sold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model metrics from experimentation
    st.markdown("### 📈 Model Performance")
    st.markdown("""
    <div class="section">
    <table>
        <tr><th style='text-align:left'>Model</th><th>RMSE</th><th>MAE</th></tr>
        <tr><td>Random Forest</td><td>114.07</td><td>89.84</td></tr>
        <tr><td><b>XGBoost</b></td><td><b>80.22</b></td><td><b>61.99</b></td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    rf_model = train_rf(X_train, y_train)

    try:
        y_pred_rf = rf_model.predict(X_test)
    except Exception as e:
        st.error(f"RF prediction failed: {e}")
        y_pred_rf = [0] * len(X_test)

    try:
        y_pred_xgb = model.predict(X_test)
    except Exception as e:
        st.error(f"XGBoost prediction failed: {e}")
        y_pred_xgb = [0] * len(X_test)

    sample_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf}).iloc[:200]
    sample_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb}).iloc[:200]

    st.markdown("### 📊 Side-by-Side Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Random Forest")
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        ax_rf.scatter(sample_rf['Predicted'], sample_rf['Actual'], color='#ffa502', edgecolor='black')
        ax_rf.plot([0, sample_rf.max().max()], [0, sample_rf.max().max()], 'r--', label='Ideal Fit')
        ax_rf.set_xlabel("Predicted Demand")
        ax_rf.set_ylabel("Actual Units Sold")
        ax_rf.set_title("RF Predictions")
        ax_rf.legend()
        st.pyplot(fig_rf)

    with col2:
        st.markdown("#### ⚡ XGBoost")
        fig_xgb, ax_xgb = plt.subplots(figsize=(6, 4))
        ax_xgb.scatter(sample_xgb['Predicted'], sample_xgb['Actual'], color='#00cec9', edgecolor='black')
        ax_xgb.plot([0, sample_xgb.max().max()], [0, sample_xgb.max().max()], 'r--', label='Ideal Fit')
        ax_xgb.set_xlabel("Predicted Demand")
        ax_xgb.set_ylabel("Actual Units Sold")
        ax_xgb.set_title("XGBoost Predictions")
        ax_xgb.legend()
        st.pyplot(fig_xgb)

# --- PAGE 2: CATEGORY & SEASONAL INSIGHTS ---
elif page == "📉 Category & Seasonal Insights" and not df.empty:
    st.markdown("""
    <style>
        .section-header { font-size: 28px; font-weight: 600; color: #2d3436; margin-top: 10px; }
        .styled-box { background-color: #f4f6f7; padding: 20px; border-radius: 10px; box-shadow: 0 0 8px rgba(0,0,0,0.03); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🗓️ Category Sales by Season</div>', unsafe_allow_html=True)

    df['Season'] = df['Date'].dt.month % 12 // 3 + 1
    season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    df['Season'] = df['Season'].map(season_names)

    selected_season = st.radio("Select Season", ["Winter", "Spring", "Summer", "Fall"], horizontal=True)
    season_df = df[df['Season'] == selected_season]

    category_summary = season_df.groupby('Category')['Units Sold'].mean().sort_values()

    st.markdown(f"### 📊 Avg Units Sold by Category – `{selected_season}`")
    with st.container():
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(category_summary.index, category_summary.values, color='#0984e3', edgecolor='black')
        ax.set_xlabel("Avg Units Sold")
        ax.set_title(f"📦 Category Performance in {selected_season}")
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

# --- PAGE 3: OPTIMIZATION DASHBOARD ---
elif page == "📈 Optimization Dashboard" and not df.empty and model is not None:
    st.markdown("""
    <style>
        .dashboard-title { font-size: 28px; font-weight: 600; color: #27ae60; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="dashboard-title">📦 Inventory Optimization Dashboard</div>', unsafe_allow_html=True)

    stores = sorted(df['Store ID'].unique())
    selected_store = st.selectbox("Select Store", stores)

    df_store = df[df['Store ID'] == selected_store].copy()
    latest_date = df_store['Date'].max()
    df_latest = df_store[df_store['Date'] == latest_date].copy()

    capacity = st.slider("Total Inventory Capacity", min_value=100, max_value=2500, step=100)

    if 'Predicted Demand' not in df_latest.columns:
        features = ['DayOfWeek', 'IsWeekend', 'IsPromo', 'RollingDemand7', 'RollingDemand14', 'Lag_1', 'Discount', 'Inventory_Level']
        try:
            df_latest['Predicted Demand'] = predict_demand(model, df_latest[features])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            df_latest['Predicted Demand'] = 0

    df_optimize = df_latest[['Product ID', 'Predicted Demand']].sort_values(by='Predicted Demand', ascending=False).head(10).reset_index(drop=True)
    result_df = optimize_inventory(df_optimize, total_capacity=capacity)

    result_df = result_df.merge(df_latest[['Product ID', 'Units Sold']], on='Product ID', how='left')
    result_df['Error'] = result_df['Units Sold'] - result_df['Predicted Demand']
    result_df['Gap'] = result_df['Units Sold'] - result_df['Allocated Stock']

    st.subheader(f"📋 Store {selected_store} – Allocation Summary")
    st.dataframe(result_df.rename(columns={
        "Predicted Demand": "Predicted",
        "Allocated Stock": "Allocated",
        "Units Sold": "Actual",
        "Error": "Forecast Error",
        "Gap": "Allocation Gap"
    }).style.format({
        "Predicted": "{:.0f}",
        "Allocated": "{:.0f}",
        "Actual": "{:.0f}",
        "Forecast Error": "{:+.0f}",
        "Allocation Gap": "{:+.0f}"
    }))

    n = len(result_df)
    fig, ax = plt.subplots(figsize=(min(14, n * 1.5), 5))
    x = np.arange(n)
    width = 0.25

    ax.bar(x - width, result_df['Predicted Demand'], width, label='Predicted', color='#74b9ff', edgecolor='black')
    ax.bar(x, result_df['Allocated Stock'], width, label='Allocated', color='#55efc4', edgecolor='black')
    ax.bar(x + width, result_df['Units Sold'], width, label='Actual', color='#a29bfe', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(result_df['Product ID'], rotation=45)
    ax.set_ylabel("Units")
    ax.set_title("Predicted vs Allocated vs Actual")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig)

# --- PAGE 4: EXECUTIVE REPORT ---
elif page == "📄 Executive Summary Report":
    st.title("📄 Executive Summary: AI-Powered Inventory Optimization")

    st.markdown("""
    ### 🧩 Problem Statement
    Retail chains often struggle with matching supply to fluctuating demand, resulting in overstocking, lost sales, or excess logistics costs.

    ### 🎯 Objective
    Build an AI-powered system that forecasts product-level demand and optimally allocates inventory per store to improve fulfillment and minimize surplus.

    ### 📊 Dataset Overview
    - **Source**: Internal retail data from multiple stores over time
    - **Rows**: 60,000+
    - **Features**: Date, Store ID, Product ID, Promotion, Discount, Category, Inventory Level, Units Sold, and more

    ### 🔍 Model Selection & Evaluation
    We experimented with two models:
    - Random Forest (baseline)
    - XGBoost (final model of choice due to better accuracy)

    #### 📈 Model Metrics:
    | Model           | RMSE   | MAE   | Accuracy |
    |------------------|--------|--------|-----------|
    | Random Forest    | 114.07 | 89.84  | 46%       |
    | **XGBoost**      | **80.22** | **61.99** | **57%**    |

    ### 🔧 Accuracy Boosting Techniques
    - Applied **log-normal transformation** on skewed numeric features
    - Created **product-level models** to capture variation across items
    - Engineered new temporal & rolling window features (e.g., Lag_1, RollingDemand7)

    ### 📌 Feature Impact Analysis (SHAP)
    SHAP values revealed the most influential and least impactful features:
    - **High Impact**: `Inventory_Level`, `RollingDemand7`, `Lag_1`
    - **Low Impact**: `IsWeekend`, `IsPromo`, `Discount`

    ### 🚀 How Can This Be Improved?
    While XGBoost showed superior accuracy, it was limited to ~57% due to the **absence of external context** in the feature set. Retail demand is influenced by many real-world variables that aren't present in the dataset. For example:

    - **Weather patterns** (rain, snow, extreme heat) that affect store foot traffic
    - **Local events and holidays** unique to regions
    - **Competitor promotions** and pricing
    - **Economic signals** (e.g., inflation, unemployment)

    Incorporating APIs like OpenWeather, Google Trends, or regional event calendars could introduce these variables and potentially boost accuracy to 70%+.

    ### ✅ Conclusion
    This AI system enables smarter inventory planning by accurately forecasting demand and dynamically adjusting allocations. It reduces stockouts and surplus, and with further feature enrichment, can become a highly adaptive real-time decision system.
    """)

st.caption("Built by Sridhar Malladi • AI Inventory Optimization Framework 🚚📦")
