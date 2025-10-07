"""
HW1: Simple Linear Regression demo using Streamlit

This version annotates the code with explicit CRISP-DM step labels so that each block of code is clearly marked which
CRISP-DM phase it implements. Keep the same interactive behavior as before:
- Sidebar exposes only 3 adjustable parameters: n_points, a, noise_var.
- When noise_var == 0, no noise and no outliers are added (all points lie exactly on y = a*x + b).
- Top-5 outliers table uses DataFrame index as point ID.

How to run:
1. pip install streamlit scikit-learn pandas numpy matplotlib
2. streamlit run HW1_simple_linear_regression_streamlit.py

"""

# -----------------------------
# IMPORTS
# -----------------------------
# CRISP-DM: Data Preparation (libraries & env setup)
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="HW1-1 Interactive Linear Regression (CRISP-DM)", layout="wide")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
# CRISP-DM: Business Understanding -> define what user can change to explore the business/scenario
st.sidebar.title("Configuration")
st.sidebar.caption("Only adjustable parameters shown")

n_points = st.sidebar.slider("Number of data points (n)", min_value=100, max_value=1000, value=500, step=10)
a = st.sidebar.slider("Coefficient 'a' (y = a*x + b + noise)", min_value=-10.0, max_value=10.0, value=2.0, step=0.01)
noise_var = st.sidebar.slider("Noise Variance (var)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)

# -----------------------------
# FIXED SETTINGS (experiment design)
# -----------------------------
# CRISP-DM: Data Understanding -> define data domain/ranges used in the experiments
b = 4.0  # fixed intercept (kept constant to match example)
x_min, x_max = 0.0, 10.0
outlier_frac = 0.03  # fixed small fraction of outliers added only when noise_var > 0
seed = 42

# -----------------------------
# CRISP-DM STEP 1: BUSINESS UNDERSTANDING
# -----------------------------
# The goal is to demonstrate linear regression fitting (estimate 'a' and 'b') and show how noise and outliers
# affect the fitted model. We expose only three parameters for pedagogical clarity: n_points, a, noise variance.

# -----------------------------
# CRISP-DM STEP 2: DATA UNDERSTANDING
# -----------------------------
# Generate synthetic data from y = a * x + b + noise. We examine data distributions visually below.
rng = np.random.RandomState(seed)
X = rng.uniform(x_min, x_max, size=n_points)

# Interpret provided noise_var as variance; compute standard deviation
noise_std = float(np.sqrt(max(0.0, noise_var)))
if noise_std == 0.0:
    # CRISP-DM: Data Preparation -> when variance is zero, produce exact deterministic values (no randomness)
    noise = np.zeros(n_points)
else:
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)

# create responses
y = a * X + b + noise

# -----------------------------
# CRISP-DM STEP 3: DATA PREPARATION
# -----------------------------
# Add outliers only when noise_var > 0. When noise_var == 0 we intentionally do NOT add outliers so every
# data point lies exactly on the same theoretical line y = a*x + b.
if noise_var > 0 and outlier_frac > 0:
    n_out = int(np.floor(outlier_frac * n_points))
    if n_out > 0:
        out_idxs = rng.choice(n_points, size=n_out, replace=False)
        outlier_scale = 10.0 * (1.0 + noise_std / (1.0 + noise_std))
        y[out_idxs] = y[out_idxs] + rng.normal(loc=0.0, scale=outlier_scale, size=n_out)
else:
    out_idxs = np.array([], dtype=int)

# compose DataFrame; index = point ID
# CRISP-DM: Data Understanding / Preparation -> keep original indices for traceability
df = pd.DataFrame({"x": X, "y": y})

# -----------------------------
# CRISP-DM STEP 4: MODELING
# -----------------------------
# Fit OLS linear regression on the full dataset (mirrors the screenshot behavior)
model = LinearRegression()
model.fit(df[["x"]], df["y"])

df["y_pred"] = model.predict(df[["x"]])
df["residuals"] = df["y"] - df["y_pred"]

# -----------------------------
# CRISP-DM STEP 5: EVALUATION
# -----------------------------
# Compute evaluation metrics on the dataset used for fitting (for this interactive demo)
mse = mean_squared_error(df["y"], df["y_pred"]) 
r2 = r2_score(df["y"], df["y_pred"]) 

# Visualize results and outliers
st.title("HW1-1: Interactive Linear Regression Visualizer")
st.markdown("### Generated Data and Linear Regression")

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["x"], df["y"], s=28, alpha=0.6, label="Generated Data")

    # Plot fitted line (Model result)
    xs = np.linspace(x_min, x_max, 300)
    ys_fit = model.coef_[0] * xs + model.intercept_
    ax.plot(xs, ys_fit, linewidth=2.5, color='red', label="Linear Regression")

    # Identify and highlight top-5 outliers by absolute residual
    top_outliers = df.reindex(df["residuals"].abs().sort_values(ascending=False).index).head(5)

    if len(top_outliers) > 0:
        ax.scatter(top_outliers["x"], top_outliers["y"], s=130, facecolors='none', edgecolors='purple', linewidths=2, label="Top outliers")
        for idx, row in top_outliers.iterrows():
            # label with original point ID (DataFrame index)
            offset = 1.2 if row["residuals"] >= 0 else -1.8
            ax.text(row["x"], row["y"] + offset, f"Outlier {int(idx)}", color='purple', fontsize=10, ha='center')

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Linear Regression with Outliers")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.markdown("### Model Coefficients")
    st.markdown(f"**Coefficient (a) [estimated]:** {model.coef_[0]:.2f}")
    st.markdown(f"**Intercept (b) [estimated]:** {model.intercept_:.2f}")
    st.markdown("---")
    st.markdown("**Performance (on full data)**")
    st.write({"MSE": float(mse), "R2": float(r2)})

# Show top-5 outliers table with point IDs as index
st.markdown("---")
st.subheader("Top 5 Outliers")
if len(top_outliers) > 0:
    top5 = top_outliers.loc[top_outliers.index].copy()
    top5 = top5[["x", "y", "residuals"]].round({"x":4, "y":4, "residuals":4})
    top5.index.name = "point"
    # CRISP-DM: Evaluation -> present findings (outliers) in tabular form for inspection
    st.table(top5)
else:
    st.write("No outliers to show (noise variance = 0 or dataset too small).")

# -----------------------------
# CRISP-DM STEP 6: DEPLOYMENT
# -----------------------------
# Provide download of generated data and explanatory notes; the Streamlit app itself is a lightweight deployment
st.markdown("---")
st.caption(
    "Notes: intercept b=4.0 (fixed). Outliers are only added when Noise Variance > 0.\n"
    "When Noise Variance is 0, no random noise nor outliers are added so all points lie exactly on the theoretical line y = a*x + b."
)

# CSV download includes the DataFrame index so point IDs are preserved
csv = df[["x", "y"]].to_csv(index=True)
st.download_button("Download dataset (CSV)", data=csv, file_name="synthetic_linear_regression.csv", mime="text/csv")
