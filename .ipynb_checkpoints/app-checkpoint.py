
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Title
st.title("🏠 Paris Housing Price Predictor")

# Instructions
st.markdown("Enter the features of the house below to predict the **price** (in €).")

# Input fields
squareMeters = st.number_input("Square Meters", min_value=10, max_value=100000, value=75000)
numberOfRooms = st.number_input("Number of Rooms", min_value=1, max_value=50, value=5)
floors = st.slider("Number of Floors", min_value=1, max_value=10, value=2)
cityCode = st.selectbox("City Code", list(range(10)))
cityPartRange = st.slider("City Part Range", min_value=1, max_value=10, value=5)
numPrevOwners = st.slider("Number of Previous Owners", min_value=0, max_value=10, value=1)
made = st.slider("Year Built", min_value=1900, max_value=2024, value=2000)
basement = st.checkbox("Has Basement?")
attic = st.checkbox("Has Attic?")
garage = st.checkbox("Has Garage?")
hasGuestRoom = st.checkbox("Has Guest Room?")
hasStormProtector = st.checkbox("Has Storm Protector?")
hasStorageRoom = st.checkbox("Has Storage Room?")
hasYard = st.checkbox("Has Yard?")
hasPool = st.checkbox("Has Pool?")
isNewBuilt = st.checkbox("Is Newly Built?")


# Derived feature
price_per_m2 = 100  # Default rate as seen from EDA

# Assemble input
input_data = pd.DataFrame([{
    "squareMeters": squareMeters,
    "numberOfRooms": numberOfRooms,
    "floors": floors,
    "cityCode": cityCode,
    "cityPartRange": cityPartRange,
    "numPrevOwners": numPrevOwners,
    "made": made,
    "basement": int(basement),
    "attic": int(attic),
    "garage": int(garage),
    "hasGuestRoom": int(hasGuestRoom),
    "hasStormProtector": int(hasStormProtector),
    "hasStorageRoom": int(hasStorageRoom),
    "hasYard": int(hasYard),
    "hasPool": int(hasPool),
    "isNewBuilt": int(isNewBuilt),
    "price_per_m2": price_per_m2
}])

# Load model and scaler (replace with actual saved model if available)
model = RandomForestRegressor()
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")
expected_cols = joblib.load("feature_columns.joblib")

# Reorder columns to match training
input_data = input_data[expected_cols]

# Predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

prediction = model.predict(input_data)[0]
st.subheader(f"💶 Estimated Price: €{prediction:,.2f}")
