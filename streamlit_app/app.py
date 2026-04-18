import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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

st.title("🏠 Columbus Airbnb Price Predictor")
st.markdown("Enter your listing details below to get a recommended nightly price based on Columbus Airbnb market data.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Listing Details")
    accommodates  = st.slider("Number of Guests", 1, 16, 2)
    bedrooms      = st.slider("Bedrooms", 0, 10, 1)
    beds          = st.slider("Beds", 1, 10, 1)
    bathrooms     = st.slider("Bathrooms", 0.5, 8.0, 1.0, step=0.5)
    room_type     = st.selectbox("Room Type", ["Entire home/apt"] + dropdowns["room_types"])
    neighbourhood = st.selectbox("Neighbourhood", dropdowns["neighbourhoods"])

with col2:
    st.subheader("📊 Listing Performance")
    minimum_nights       = st.slider("Minimum Nights", 1, 30, 2)
    number_of_reviews    = st.slider("Number of Reviews", 0, 500, 50)
    review_scores_rating = st.slider("Review Score Rating", 1.0, 5.0, 4.5, step=0.1)
    host_is_superhost    = st.checkbox("Superhost", value=False)
    instant_bookable     = st.checkbox("Instant Bookable", value=False)

st.markdown("---")

if st.button("💰 Predict Price", use_container_width=True):

    review_density = number_of_reviews / (minimum_nights + 1)

    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    input_data["accommodates"]          = accommodates
    input_data["bathrooms"]             = bathrooms
    input_data["bedrooms"]              = bedrooms
    input_data["beds"]                  = beds
    input_data["minimum_nights"]        = minimum_nights
    input_data["number_of_reviews"]     = number_of_reviews
    input_data["review_scores_rating"]  = review_scores_rating
    input_data["host_is_superhost"]     = int(host_is_superhost)
    input_data["instant_bookable"]      = int(instant_bookable)
    input_data["review_density"]        = review_density
    input_data["price_per_person"]      = 26.50

    if room_type != "Entire home/apt":
        col_name = f"room_type_{room_type}"
        if col_name in columns:
            input_data[col_name] = 1

    col_name = f"neighbourhood_cleansed_{neighbourhood}"
    if col_name in columns:
        input_data[col_name] = 1

    predicted_price = np.expm1(model.predict(input_data)[0])

    st.success(f"### 💵 Recommended Nightly Price: ${predicted_price:.2f}")

    st.markdown("#### How does this compare to the Columbus market?")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Your Predicted Price", f"${predicted_price:.2f}")
    with col4:
        st.metric("Columbus Avg (Entire Home)", "~$150")
    with col5:
        st.metric("Columbus Avg (Private Room)", "~$75")

    st.markdown("---")
    st.caption("Prediction based on Columbus Airbnb listings data. Model: Random Forest (R² = 0.9945)")

st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This tool predicts Airbnb nightly prices for Columbus, Ohio listings using a
Random Forest model trained on real Airbnb data.

**Model Performance:**
- R² Score: 0.9945
- RMSE: 0.04 (log scale)
- MAE: 0.01 (log scale)

**Built by:** BANA 7075 Group Project
**Data:** Columbus, OH Airbnb Listings
""")
