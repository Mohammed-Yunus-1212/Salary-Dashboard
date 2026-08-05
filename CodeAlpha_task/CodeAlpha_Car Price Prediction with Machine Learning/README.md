# 🚗 Car Price Prediction Machine Learning Web Application

A professional machine learning web application built with **Python**, **Streamlit**, and **Scikit-learn** that predicts the selling price of used cars based on vehicle attributes.

---

## 🌟 Features

- **Automated Model Training & Selection**: Trains Linear Regression, Decision Tree, and Random Forest Regressors, evaluates R², MAE, and RMSE metrics, automatically picks the top model, and persists it using **Joblib**.
- **Clean & Modern UI**: Built with a clean white theme (`#ffffff` / `#f8fafc`), responsive wide layout, and rounded card containers.
- **Two-Column Input Form**: Easy-to-use input form with dropdowns and numerical controls.
- **Instant Valuation Result**: Displays predicted resale value formatted in Lakhs (`₹ XX.XX Lakhs`).
- **Feature Importance**: Interactive Plotly bar chart displaying key price drivers.
- **Interactive EDA & Dataset Explorer**: Expandable sections for dataset statistics, data types, missing values, and Plotly visualizations.

---

## 🛠 Project Structure

```
CodeAlpha_Car Price Prediction with Machine Learning/
│
├── app.py              # Main Streamlit Web Application
├── model.pkl           # Best Trained Machine Learning Model (Saved via Joblib)
├── car data.csv        # Used Car Dataset
├── requirements.txt    # Required Python Packages
└── README.md           # Project Documentation
```

---

## 🚀 Quick Start & Installation

1. **Clone or Open Workspace**:
   ```bash
   cd "CodeAlpha_Car Price Prediction with Machine Learning"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

---

## 👨‍💻 Developed By

**Mohammed Yunus**  
*Python • Streamlit • Scikit-learn • Plotly*
