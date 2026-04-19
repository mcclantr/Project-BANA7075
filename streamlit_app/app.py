
# =============================================================================
# Streamlit App — Columbus Airbnb Price Predictor
# BANA 7075 Group Project
#
# Model: Random Forest (R² = 0.7547)
#
# Note on model selection:
# An earlier model achieved R² = 0.9945 but included "price_per_person"
# as a feature (price / accommodates). This created a circular dependency
# since price is what we are trying to predict. That inflated performance
# metrics but made the model unusable in a real UI where price is unknown.
# This version removes price_per_person for honest, real-world predictions.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "rf_model_improved.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "model_columns.pkl"), "rb") as f:
        columns = pickle.load(f)
    with open(os.path.join(base_dir, "dropdown_values.pkl"), "rb") as f:
        dropdowns = pickle.load(f)
    return model, columns, dropdowns

model, columns, dropdowns = load_model()

# ── App Header ─────────────────────────────────────────────────
st.title("🏠 Columbus Airbnb Price Predictor")
st.markdown("Enter your listing details below to get a recommended nightly price based on Columbus Airbnb market data.")
st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📋 Listing Details")
    accommodates  = st.slider("Number of Guests", 1, 16, 2)
    bedrooms      = st.slider("Bedrooms", 0, 10, 1)
    beds          = st.slider("Beds", 1, 10, 1)
    bathrooms     = st.slider("Bathrooms", 0.5, 8.0, 1.0, step=0.5)
    room_type     = st.selectbox("Room Type", ["Entire home/apt"] + dropdowns["room_types"])
    neighbourhood = st.selectbox("Neighbourhood", dropdowns["neighbourhoods"])

with right_col:
    st.subheader("📊 Listing Performance")
    minimum_nights    = st.slider("Minimum Nights", 1, 30, 2)
    number_of_reviews = st.slider("Number of Reviews", 0, 500, 50)
    host_is_superhost = st.checkbox("Superhost", value=False)
    instant_bookable  = st.checkbox("Instant Bookable", value=False)

    # Note: review_scores_rating is passed as the training median (4.5)
    # since it has low predictive power (7.6% feature importance) and
    # a new listing would not have a rating yet
    review_scores_rating = 4.5

st.markdown("---")

# ── Prediction ─────────────────────────────────────────────────
if st.button("💰 Predict Price", use_container_width=True):

    # Calculate engineered feature
    review_density = number_of_reviews / (minimum_nights + 1)

    # Build input dataframe with all model columns initialized to 0
    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    # Fill numeric features
    input_data["accommodates"]         = accommodates
    input_data["bathrooms"]            = bathrooms
    input_data["bedrooms"]             = bedrooms
    input_data["beds"]                 = beds
    input_data["minimum_nights"]       = minimum_nights
    input_data["number_of_reviews"]    = number_of_reviews
    input_data["review_scores_rating"] = review_scores_rating
    input_data["host_is_superhost"]    = int(host_is_superhost)
    input_data["instant_bookable"]     = int(instant_bookable)
    input_data["review_density"]       = review_density

    # Handle room type one-hot encoding
    # "Entire home/apt" is the baseline (dropped during training)
    # so it is represented by all room type columns being 0
    if room_type != "Entire home/apt":
        room_col = f"room_type_{room_type}"
        if room_col in columns:
            input_data[room_col] = 1

    # Handle neighbourhood one-hot encoding
    hood_col = f"neighbourhood_cleansed_{neighbourhood}"
    if hood_col in columns:
        input_data[hood_col] = 1

    # Predict log price and convert back to dollars
    # expm1 is the inverse of log1p used during training
    predicted_price = np.expm1(model.predict(input_data)[0])

    # ── Display Results ────────────────────────────────────────
    st.success(f"### 💵 Recommended Nightly Price: ${predicted_price:.2f}")

    st.markdown("#### How does this compare to the Columbus market?")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Your Predicted Price",        f"${predicted_price:.2f}")
    with metric_col2:
        st.metric("Columbus Avg (Entire Home)",  "~$150")
    with metric_col3:
        st.metric("Columbus Avg (Private Room)", "~$75")

    st.markdown("---")
    st.caption("Prediction based on Columbus Airbnb listings data. Model: Random Forest (R² = 0.7547)")

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This tool predicts Airbnb nightly prices for Columbus, Ohio listings using a
Random Forest model trained on real Airbnb data.

**Model Performance:**
- R² Score: 0.7547
- RMSE: 0.2864 (log scale)
- MAE: 0.2205 (log scale)

**Why not the highest R² model?**
An earlier model scored R² = 0.9945 but used *price_per_person*
(price ÷ accommodates) as a feature. Since price is what we are
trying to predict, this is data leakage — the model was essentially
using price to predict price. This inflated performance metrics but
made the model unusable in practice where price is unknown.
This model removes that feature for honest, deployable predictions.

**Note on review score rating:**
Rating is set to the dataset median (4.5) since new listings
would not have a rating yet and it has low predictive power
(7.6% feature importance).

**Built by:** BANA 7075 Group Project
**Data:** Columbus, OH Airbnb Listings
""")
