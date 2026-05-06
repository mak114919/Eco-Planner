import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('urban_model.joblib')

st.set_page_config(page_title="Eco-Urban Planner AI", layout="wide")

st.title("🏙️ Eco-Urban Planner: Heat-Resilient AI")
st.markdown("Adjust the zoning parameters below to see the impact on surface temperature.")

# Sidebar Inputs
st.sidebar.header("Zoning Parameters")
density = st.sidebar.slider("Building Density (%)", 0, 100, 50)
ndvi = st.sidebar.slider("Vegetation Index (NDVI)", 0.0, 1.0, 0.3)
albedo = st.sidebar.slider("Surface Albedo (Reflectivity)", 0.1, 0.9, 0.2)
svf = st.sidebar.slider("Sky View Factor (Openness)", 0.0, 1.0, 0.5)

# Prediction Logic
input_data = np.array([[ndvi, albedo, density, svf]])
prediction = model.predict(input_data)[0]

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted Surface Temperature", value=f"{prediction:.2f} °C")
    
    if prediction > 35:
        st.error("⚠️ Warning: Extreme Heat Zone Detected")
    elif prediction > 30:
        st.warning("🟠 Caution: Moderate Thermal Stress")
    else:
        st.success("🟢 Safe: Thermally Resilient Design")

with col2:
    st.subheader("AI Advisor Recommendations")
    if prediction > 32:
        st.write(f"- **Increase NDVI**: Adding more green space could lower temp by ~{(prediction-32)*0.8:.1f}°C.")
        st.write(f"- **Cool Pavements**: Increasing Albedo to {albedo + 0.2:.2f} is recommended.")
    else:
        st.write("Your current design meets sustainability goals. No further mitigation required.")