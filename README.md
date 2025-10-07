#  HW1: Simple Linear Regression (CRISP-DM)

An interactive **Simple Linear Regression Visualizer** built with **Streamlit**, demonstrating the full **CRISP-DM** (Cross Industry Standard Process for Data Mining) workflow.  
This project allows users to explore the effect of data noise and sample size on linear regression results.

---

##  Features

- Adjustable parameters via Streamlit sidebar:
  - `Number of data points (n)`
  - `Coefficient a (y = a*x + b + noise)`
  - `Noise Variance (var)`
- Real-time visualization of:
  - Generated data points and regression line
  - Top-5 outliers with point IDs
  - Model coefficients (`a`, `b`)
  - Evaluation metrics (MSE, R²)
- CSV dataset download button
- Fully annotated **CRISP-DM process** within the code

---

##  CRISP-DM Mapping in Code

| Step | Description | Location in Code |
|------|--------------|-----------------|
| **1. Business Understanding** | Define project goal: demonstrate how noise and data size affect linear regression performance. | Sidebar configuration and initial docstring |
| **2. Data Understanding** | Generate synthetic data (`y = a*x + b + noise`), inspect variable distributions. | Data generation section |
| **3. Data Preparation** | Handle noise and outlier generation; clean data when `noise_var = 0`. | Outlier creation and DataFrame construction |
| **4. Modeling** | Fit a Linear Regression model using scikit-learn. | Model fitting block |
| **5. Evaluation** | Display model performance (MSE, R²) and visualize results. | Visualization and metrics output |
| **6. Deployment** | Streamlit app for interactive exploration and CSV download. | App layout, plotting, and `st.download_button` |

---

##  Installation & Run

$streamlit run hw1.py

Demo Site
The application is deployed on Streamlit Cloud and can be accessed here: [https://aiotda.streamlit.app/](https://aiothwgit-6cuzabuvhtzvmcxhexdy2g.streamlit.app/)

