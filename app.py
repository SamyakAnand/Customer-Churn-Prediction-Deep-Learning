import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="🌀 Customer Churn Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Caching Model & Preprocessors
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load and return the trained Keras model."""
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    """Generic loader for pickled objects."""
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model("model.h5")
label_encoder_gender = load_pickle("label_encoder_gender.pkl")
onehot_encoder_geo = load_pickle("onehot_encoder_geo.pkl")
scaler = load_pickle("scaler.pkl")

# ------------------------------------------------------------------------------
# Preprocessing Function
# ------------------------------------------------------------------------------
def preprocess(
    geography: str,
    gender: str,
    age: int,
    balance: float,
    credit_score: float,
    estimated_salary: float,
    tenure: int,
    num_of_products: int,
    has_cr_card: int,
    is_active_member: int
) -> np.ndarray:
    """Encode, scale, and return a feature vector ready for prediction."""
    # Base numeric & label-encoded features
    df = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )

    # One-hot encode Geography
    geo_arr = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
    df_geo = pd.DataFrame(geo_arr, columns=geo_cols)

    # Combine & scale
    full_df = pd.concat([df, df_geo], axis=1)
    return scaler.transform(full_df)

# ------------------------------------------------------------------------------
# Sidebar: User Inputs
# ------------------------------------------------------------------------------
st.sidebar.header("Customer Profile")
geography = st.sidebar.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
age = st.sidebar.slider("Age", min_value=18, max_value=92, value=35)
balance = st.sidebar.number_input("Balance", min_value=0.0, format="%.2f")
credit_score = st.sidebar.number_input("Credit Score", min_value=300.0, max_value=850.0, format="%.1f")
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, format="%.2f")
tenure = st.sidebar.slider("Tenure (years)", min_value=0, max_value=10, value=3)
num_of_products = st.sidebar.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.sidebar.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", options=[0, 1])

# ------------------------------------------------------------------------------
# Main: Display Inputs & Predict
# ------------------------------------------------------------------------------
st.title("💡 Customer Churn Prediction Dashboard")
st.markdown("Adjust the sidebar controls and click **Predict** to see churn risk.")

# Show input summary
with st.expander("🔍 Input Summary"):
    summary = {
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Balance": balance,
        "Credit Score": credit_score,
        "Estimated Salary": estimated_salary,
        "Tenure": tenure,
        "Products": num_of_products,
        "Has Credit Card": has_cr_card,
        "Active Member": is_active_member,
    }
    st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]))

# Predict button
if st.button("Predict Churn"):
    try:
        features = preprocess(
            geography, gender, age, balance,
            credit_score, estimated_salary,
            tenure, num_of_products,
            has_cr_card, is_active_member
        )
        proba = model.predict(features)[0][0]          # numpy.float32

        # Convert to integer percentage (0–100)
        churn_pct = int(proba * 100)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Churn Probability")
            st.progress(churn_pct)                     # now an int
            st.metric(label="Probability", value=f"{churn_pct}%")

        with col2:
            if proba > 0.5:
                st.error("⚠️ Likely to churn")
            else:
                st.success("✅ Unlikely to churn")
    except Exception as e:
        st.exception(f"Prediction failed: {e}")