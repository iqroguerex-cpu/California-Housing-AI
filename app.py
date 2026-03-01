import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="California Housing AI",
    page_icon="🏠",
    layout="wide"
)

# =========================
# DARK MODE TOGGLE
# =========================
dark_mode = st.sidebar.toggle("🌗 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        section[data-testid="stSidebar"] {
            background-color: #161B22;
        }
        div[data-testid="stMetricValue"] {
            color: #00FFAA;
        }
        </style>
    """, unsafe_allow_html=True)
    plt.style.use("dark_background")
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)
    plt.style.use("default")

# =========================
# LOAD DATA
# =========================
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

# =========================
# TRAIN / TEST SPLIT
# =========================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# =========================
# SCALING
# =========================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================
# MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =========================
# HEADER
# =========================
st.title("🏠 California Housing Price Prediction")
st.caption("Linear Regression • Scikit-Learn Dataset • ML Dashboard")

st.divider()

# =========================
# METRIC DISPLAY
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE ($100k units)", f"{mae:.3f}")
col3.metric("RMSE ($100k units)", f"{rmse:.3f}")

st.info("Target values are in units of $100,000")

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "📈 Predictions", "📉 Residual Analysis", "📥 Download"]
)

# =========================
# TAB 1 — DATA
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Feature Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

# =========================
# TAB 2 — PREDICTIONS
# =========================
with tab2:
    st.subheader("Actual vs Predicted Prices")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, ax=ax2)
    ax2.set_xlabel("Actual Price")
    ax2.set_ylabel("Predicted Price")
    st.pyplot(fig2)

    st.subheader("Feature Importance (Linear Coefficients)")
    coef_df = pd.DataFrame({
        "Feature": housing.feature_names,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    st.dataframe(coef_df)

# =========================
# TAB 3 — RESIDUALS
# =========================
with tab3:
    residuals = y_test - y_pred

    st.subheader("Residual Plot")

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.4, ax=ax3)
    ax3.axhline(0, color="red")
    ax3.set_xlabel("Predicted Price")
    ax3.set_ylabel("Residuals")
    st.pyplot(fig3)

    st.subheader("Residual Distribution")

    fig4, ax4 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax4)
    st.pyplot(fig4)

# =========================
# TAB 4 — DOWNLOAD
# =========================
with tab4:
    results_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred,
        "Residual": y_test - y_pred
    })

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="california_housing_predictions.csv",
        mime="text/csv"
    )
