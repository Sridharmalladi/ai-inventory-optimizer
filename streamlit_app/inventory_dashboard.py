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
    st.title("🔍 Exploratory Data Analysis & Model Development")

    st.markdown("""
    This page presents key insights extracted from the historical retail dataset:

    - **EDA Highlights**:
      - Strong weekly seasonality in demand.
      - Promotions and holidays drive spikes.
      - Discounts show a moderate positive correlation with sales.

    - **Model Experimentation**:
      - Random Forest (baseline)
      - XGBoost (final choice)
    """)

    features = ['DayOfWeek', 'IsWeekend', 'IsPromo', 'RollingDemand7', 'RollingDemand14',
                'Lag_1', 'Discount', 'Inventory_Level']
    X = df[features]
    y = df['Units Sold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model metrics from experimentation
    st.markdown("### 📈 Model Performance (from experimentation)")
    st.markdown("""
    | Model           | RMSE   | MAE   |
    |------------------|--------|--------|
    | Random Forest    | 114.07 | 89.84  |
    | **XGBoost**      | **80.22** | **61.99** |
    """)

    # Model selector for scatter plot
    selected_model = st.radio("Select Model for Visual Comparison", ["Random Forest", "XGBoost"], horizontal=True)

    rf_model = train_rf(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = model.predict(X_test)

    sample_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf}).iloc[:200]
    sample_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb}).iloc[:200]

    if selected_model == "Random Forest":
        st.markdown("### 🔍 Predicted vs Actual – Random Forest")
        with st.expander("Show RF Scatter Plot"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(sample_rf['Predicted'], sample_rf['Actual'], color='darkorange', label='RF Prediction')
            ax.plot([0, sample_rf.max().max()], [0, sample_rf.max().max()], 'r--', label='Ideal Fit')
            ax.set_xlabel("Predicted Demand")
            ax.set_ylabel("Actual Units Sold")
            ax.legend()
            st.pyplot(fig)
    else:
        st.markdown("### 🔍 Predicted vs Actual – XGBoost")
        with st.expander("Show XGBoost Scatter Plot"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(sample_xgb['Predicted'], sample_xgb['Actual'], color='teal', label='XGB Prediction')
            ax.plot([0, sample_xgb.max().max()], [0, sample_xgb.max().max()], 'r--', label='Ideal Fit')
            ax.set_xlabel("Predicted Demand")
            ax.set_ylabel("Actual Units Sold")
            ax.legend()
            st.pyplot(fig)

# --- PAGE 2: CATEGORY & SEASONAL INSIGHTS ---
elif page == "📉 Category & Seasonal Insights" and not df.empty:
    st.title("🗓️ Category Sales by Season")

    df['Season'] = df['Date'].dt.month % 12 // 3 + 1
    season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    df['Season'] = df['Season'].map(season_names)

    selected_season = st.radio("Select Season", ["Winter", "Spring", "Summer", "Fall"], horizontal=True)
    season_df = df[df['Season'] == selected_season]

    category_summary = season_df.groupby('Category')['Units Sold'].mean().sort_values()

    st.markdown(f"### 📊 Avg Units Sold by Category – `{selected_season}`")
    fig, ax = plt.subplots(figsize=(10, 5))
    category_summary.plot(kind='barh', color='steelblue', ax=ax)
    ax.set_xlabel("Avg Units Sold")
    ax.set_title(f"Category Performance in {selected_season}")
    ax.set_xlim(category_summary.min() * 0.9, category_summary.max() * 1.1)
    st.pyplot(fig)

# --- PAGE 3: OPTIMIZATION DASHBOARD ---
elif page == "📈 Optimization Dashboard" and not df.empty and model is not None:
    st.title("📦 Inventory Optimization Dashboard")

    stores = sorted(df['Store ID'].unique())
    selected_store = st.selectbox("Select Store", stores)

    df_store = df[df['Store ID'] == selected_store].copy()
    latest_date = df_store['Date'].max()
    df_latest = df_store[df_store['Date'] == latest_date].copy()

    capacity = st.slider("Total Inventory Capacity", min_value=100, max_value=2500, step=100)

    if 'Predicted Demand' not in df_latest.columns:
        features = ['DayOfWeek', 'IsWeekend', 'IsPromo', 'RollingDemand7', 'RollingDemand14', 'Lag_1', 'Discount', 'Inventory_Level']
        df_latest['Predicted Demand'] = predict_demand(model, df_latest[features])

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

    ax.bar(x - width, result_df['Predicted Demand'], width, label='Predicted', color='skyblue')
    ax.bar(x, result_df['Allocated Stock'], width, label='Allocated', color='lightgreen')
    ax.bar(x + width, result_df['Units Sold'], width, label='Actual', color='mediumpurple')

    ax.set_xticks(x)
    ax.set_xticklabels(result_df['Product ID'], rotation=45)
    ax.set_ylabel("Units")
    ax.set_title("Predicted vs Allocated vs Actual")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# --- PAGE 4: EXECUTIVE REPORT ---
elif page == "📄 Executive Summary Report":
    st.title("📄 Executive Summary: AI-Powered Inventory Optimization")

    st.markdown("""
    ## Problem Statement
    Retail chains often struggle with matching supply to fluctuating demand, resulting in overstocking, lost sales, or excess logistics costs.

    ## Objective
    Build an AI-powered system that forecasts product-level demand and optimally allocates inventory per store to improve fulfillment and minimize surplus.

    ## Dataset Overview
    - **Source**: Internal retail data from multiple stores over time
    - **Rows**: 60,000+
    - **Features**: Date, Store ID, Product ID, Promotion, Discount, Category, Inventory Level, Units Sold, and more

    ## Model Selection & Evaluation
    We experimented with two models:
    - Random Forest (baseline)
    - XGBoost (final model of choice due to better accuracy)

    ### Model Metrics:
    | Model           | RMSE   | MAE   | Accuracy |
    |------------------|--------|--------|-----------|
    | Random Forest    | 114.07 | 89.84  | 46%       |
    | **XGBoost**      | **80.22** | **61.99** | **57%**    |

    ## Accuracy Boosting Techniques
    - Applied **log-normal transformation** on skewed numeric features
    - Created **product-level models** to capture variation across items
    - Engineered new temporal & rolling window features (e.g., Lag_1, RollingDemand7)

    ## Feature Impact Analysis (SHAP)
    SHAP values revealed the most influential and least impactful features:
    - **High Impact**: `Inventory_Level`, `RollingDemand7`, `Lag_1`
    - **Low Impact**: `IsWeekend`, `IsPromo`, `Discount`

    ## How Can This Be Improved?
    While XGBoost showed superior accuracy, it was limited to ~57% due to the **absence of external context** in the feature set. Retail demand is influenced by many real-world variables that aren't present in the dataset. For example:

    - **Weather patterns** (rain, snow, extreme heat) that affect store foot traffic
    - **Local events and holidays** unique to regions
    - **Competitor promotions** and pricing
    - **Economic signals** (e.g., inflation, unemployment)

    Incorporating APIs like OpenWeather, Google Trends, or regional event calendars could introduce these variables and potentially boost accuracy to 70%+.

    ## Conclusion
    This AI system enables smarter inventory planning by accurately forecasting demand and dynamically adjusting allocations. It reduces stockouts and surplus, and with further feature enrichment, can become a highly adaptive real-time decision system.
    """)

st.caption("Built by Sridhar Malladi • AI Inventory Optimization Framework 🚚📦")

