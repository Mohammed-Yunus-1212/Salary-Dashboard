from __future__ import annotations

from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Machine Learning Imports
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# -----------------------------------------------------------------------------
# 1. Page Configuration & Clean White Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path(__file__).with_name("car data.csv")
MODEL_PATH = Path(__file__).with_name("model.pkl")
CURRENT_YEAR = datetime.now().year

# Clean White Design System CSS
CLEAN_WHITE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean Light Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    p, span, label, div {
        color: #334155;
    }

    /* Container Cards */
    .clean-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    /* Header Banner */
    .header-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }

    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .header-desc {
        font-size: 1.05rem;
        color: #64748b;
        margin: 0;
    }

    /* Centered Predict Button */
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-1px);
    }

    /* Form Label Styling */
    div[data-testid="stWidgetLabel"] label p,
    div[data-testid="stWidgetLabel"] p,
    label p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Success Card Styling */
    .result-container {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .result-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #166534;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .result-price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #15803d;
        margin-top: 0.2rem;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CLEAN_WHITE_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Data Loading & Feature Engineering
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    
    # Domain Feature Engineering for Age & Usage
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df["Driven_kms_log"] = np.log1p(df["Driven_kms"])
    df["Owner_Category"] = df["Owner"].map(
        {0: "1st Owner", 1: "2nd Owner", 2: "3rd Owner", 3: "4th+ Owner"}
    ).fillna("1st Owner")
    return df


if not DATA_PATH.exists():
    st.error("❌ Dataset `car data.csv` not found in workspace.")
    st.stop()

df = load_data(DATA_PATH)


# -----------------------------------------------------------------------------
# 3. Model Training & Persistence with Joblib
# -----------------------------------------------------------------------------
@st.cache_resource
def train_and_select_best_model(df_data: pd.DataFrame):
    features = [
        "Car_Name",
        "Year",
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Owner_Category",
        "Car_Age",
        "Driven_kms_log",
    ]
    target = "Selling_Price"

    X = df_data[features]
    y = df_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = ["Year", "Present_Price", "Driven_kms", "Owner", "Car_Age", "Driven_kms_log"]
    categorical_features = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission", "Owner_Category"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    candidate_models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    best_r2 = -float("inf")
    best_name = None
    best_pipeline = None
    best_metrics = {}
    eval_summary = {}

    for name, regressor in candidate_models.items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", regressor)]
        )
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        eval_summary[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipeline
            best_metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}

    # Save the best model using Joblib
    joblib.dump(best_pipeline, MODEL_PATH)

    return best_name, best_pipeline, best_metrics, eval_summary


best_model_name, best_pipeline, best_metrics, model_eval_summary = (
    train_and_select_best_model(df)
)


# -----------------------------------------------------------------------------
# 4. Header Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <div class="header-title">🚗 Car Price Prediction</div>
        <div class="header-desc">
            Estimate the resale market value of used cars using Machine Learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 5. Input Form (Two Columns with Smart Defaults)
# -----------------------------------------------------------------------------
st.markdown("### 📋 Vehicle Details")

with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        car_name = st.selectbox("Car Name", sorted(df["Car_Name"].unique()))
        
        car_subset = df[df["Car_Name"] == car_name]
        default_year = int(car_subset["Year"].median()) if not car_subset.empty else 2015
        default_price = float(car_subset["Present_Price"].median()) if not car_subset.empty else 5.0
        default_kms = int(car_subset["Driven_kms"].median()) if not car_subset.empty else 30000

        default_fuel = car_subset["Fuel_Type"].mode()[0] if not car_subset.empty else "Petrol"
        default_selling = car_subset["Selling_type"].mode()[0] if not car_subset.empty else "Dealer"
        default_transmission = car_subset["Transmission"].mode()[0] if not car_subset.empty else "Manual"

        year = st.number_input(
            "Year",
            min_value=int(df["Year"].min()),
            max_value=CURRENT_YEAR,
            value=default_year,
            step=1,
            help="Manufacturing year of the vehicle."
        )
        present_price = st.number_input(
            "Present Price (in Lakhs)",
            min_value=0.1,
            max_value=120.0,
            value=round(default_price, 2),
            step=0.1,
        )
        driven_kms = st.number_input(
            "Driven Kilometers",
            min_value=100,
            max_value=600000,
            value=default_kms,
            step=1000,
        )

    with col2:
        fuel_options = sorted(df["Fuel_Type"].unique())
        fuel_index = fuel_options.index(default_fuel) if default_fuel in fuel_options else 0
        fuel_type = st.selectbox("Fuel Type", fuel_options, index=fuel_index)

        selling_options = sorted(df["Selling_type"].unique())
        selling_index = selling_options.index(default_selling) if default_selling in selling_options else 0
        selling_type = st.selectbox("Selling Type", selling_options, index=selling_index)

        trans_options = sorted(df["Transmission"].unique())
        trans_index = trans_options.index(default_transmission) if default_transmission in trans_options else 0
        transmission = st.selectbox("Transmission", trans_options, index=trans_index)

        owner = st.selectbox("Owner", [0, 1, 2, 3], help="0 = 1st Owner, 1 = 2nd Owner, 2 = 3rd Owner, 3 = 4th+ Owner")

st.markdown("<br>", unsafe_allow_html=True)

# Centered Predict Button
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    predict_clicked = st.button("Predict Price", width="stretch")


# -----------------------------------------------------------------------------
# 6. Prediction Result with Age Depreciation & Owner Sensitivity
# -----------------------------------------------------------------------------
if predict_clicked:
    car_age_input = max(CURRENT_YEAR - year, 0)
    driven_kms_log_input = np.log1p(driven_kms)
    owner_category_map = {0: "1st Owner", 1: "2nd Owner", 2: "3rd Owner", 3: "4th+ Owner"}
    owner_cat_input = owner_category_map.get(owner, "1st Owner")

    input_data = pd.DataFrame(
        [
            {
                "Car_Name": car_name,
                "Year": year,
                "Present_Price": present_price,
                "Driven_kms": driven_kms,
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
                "Owner": owner,
                "Owner_Category": owner_cat_input,
                "Car_Age": car_age_input,
                "Driven_kms_log": driven_kms_log_input,
            }
        ]
    )

    # Base raw prediction from ML pipeline
    raw_pred = float(best_pipeline.predict(input_data)[0])

    # 1. Owner Sensitivity Factor (7% drop per previous owner)
    owner_discount = max(1.0 - 0.07 * owner, 0.75)
    
    # 2. Year / Car Age Depreciation Scaling Factor (~4.5% annual loss after 2 yrs)
    age_multiplier = max(1.0 - 0.045 * max(car_age_input - 2, 0), 0.20) if car_age_input > 2 else 0.95

    # Combined valuation adjustment
    adjusted_pred = raw_pred * owner_discount * age_multiplier

    # 3. Domain Depreciation Ceiling (Cannot exceed original showroom price minus annual age loss)
    max_realistic_price = max(present_price * (1.0 - 0.045 * max(car_age_input, 1)), 0.05)
    predicted_val = min(adjusted_pred, max_realistic_price)
    predicted_val = max(predicted_val, 0.05)  # Floor price at ₹ 5,000

    st.markdown(
        f"""
        <div class="result-container">
            <div class="result-title">Predicted Selling Price</div>
            <div class="result-price">₹ {predicted_val:.2f} Lakhs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Model Performance Section
# -----------------------------------------------------------------------------
st.markdown("### 📊 Model Performance Metrics")

perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric("Best Model", best_model_name)

with perf_col2:
    st.metric("R² Score", f"{best_metrics['R2']:.4f}")

with perf_col3:
    st.metric("MAE", f"{best_metrics['MAE']:.3f} Lakhs")

with perf_col4:
    st.metric("RMSE", f"{best_metrics['RMSE']:.3f} Lakhs")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. Feature Importance Section
# -----------------------------------------------------------------------------
st.markdown("### 🔥 Feature Importance")

regressor = best_pipeline.named_steps["regressor"]
preprocessor = best_pipeline.named_steps["preprocessor"]

if hasattr(regressor, "feature_importances_"):
    raw_feature_names = preprocessor.get_feature_names_out()
    
    # Formatted feature names for clear visualization
    clean_feature_names = []
    for f in raw_feature_names:
        clean_f = f.replace("num__", "").replace("cat__", "")
        clean_f = clean_f.replace("Car_Age", "Vehicle Age (Years)")
        clean_f = clean_f.replace("Year", "Manufacturing Year")
        clean_f = clean_f.replace("Driven_kms_log", "Mileage Log (Kms)")
        clean_f = clean_f.replace("Owner_Category_", "Owner: ")
        clean_f = clean_f.replace("Fuel_Type_", "Fuel: ")
        clean_f = clean_f.replace("Selling_type_", "Seller: ")
        clean_f = clean_f.replace("Transmission_", "Trans: ")
        clean_feature_names.append(clean_f)

    importance_df = (
        pd.DataFrame(
            {
                "Feature": clean_feature_names,
                "Importance": regressor.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=True)
        .tail(10)
    )

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top Drivers Influencing Price Predictions",
        template="plotly_white",
    )
    fig_imp.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=380
    )
    st.plotly_chart(fig_imp, width="stretch")
else:
    st.info("Feature importance chart is available for Tree-based models.")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 9. Dataset Preview Expander
# -----------------------------------------------------------------------------
with st.expander("📂 View Dataset"):
    st.markdown("#### First 10 Rows")
    st.dataframe(df.head(10), width="stretch")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("**Dataset Shape**")
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

    with info_col2:
        st.markdown("**Missing Values**")
        missing = df.isnull().sum().to_frame(name="Missing Count")
        st.dataframe(missing, width="stretch")

    with info_col3:
        st.markdown("**Data Types**")
        dtypes = df.dtypes.astype(str).to_frame(name="Data Type")
        st.dataframe(dtypes, width="stretch")


# -----------------------------------------------------------------------------
# 10. Exploratory Data Analysis Expander
# -----------------------------------------------------------------------------
with st.expander("📊 Exploratory Data Analysis"):
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        # Selling Price Distribution
        fig_price = px.histogram(
            df,
            x="Selling_Price",
            nbins=30,
            color_discrete_sequence=["#2563eb"],
            title="Selling Price Distribution",
            template="plotly_white",
        )
        fig_price.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_price, width="stretch")

        # Transmission Distribution
        fig_trans = px.pie(
            df,
            names="Transmission",
            title="Transmission Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig_trans.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_trans, width="stretch")

        # Correlation Heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        fig_corr = px.imshow(
            numeric_df.corr(),
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Correlation Heatmap",
            template="plotly_white",
        )
        fig_corr.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_corr, width="stretch")

        # Year vs Selling Price
        fig_year = px.scatter(
            df,
            x="Year",
            y="Selling_Price",
            color="Fuel_Type",
            hover_name="Car_Name",
            title="Year vs Selling Price",
            template="plotly_white",
        )
        fig_year.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_year, width="stretch")

    with chart_col2:
        # Fuel Type Distribution
        fig_fuel = px.bar(
            df["Fuel_Type"].value_counts().reset_index(),
            x="Fuel_Type",
            y="count",
            color="Fuel_Type",
            title="Fuel Type Distribution",
            template="plotly_white",
        )
        fig_fuel.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_fuel, width="stretch")

        # Selling Type Distribution
        fig_sell = px.bar(
            df["Selling_type"].value_counts().reset_index(),
            x="Selling_type",
            y="count",
            color="Selling_type",
            title="Selling Type Distribution",
            template="plotly_white",
        )
        fig_sell.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_sell, width="stretch")

        # Present Price vs Selling Price
        fig_pvss = px.scatter(
            df,
            x="Present_Price",
            y="Selling_Price",
            color="Transmission",
            hover_name="Car_Name",
            title="Present Price vs Selling Price",
            template="plotly_white",
        )
        fig_pvss.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_pvss, width="stretch")


# -----------------------------------------------------------------------------
# 11. Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-container">
        <strong>Developed by Mohammed Yunus</strong><br>
        Python • Streamlit • Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
