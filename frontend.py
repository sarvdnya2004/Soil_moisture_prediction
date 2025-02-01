# Import necessary libraries
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_filename = "finalized_model (1).sav"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Streamlit app configuration
st.title("Soil Moisture Detection")
st.write("This app predicts soil moisture levels based on input features.")

# Input feature descriptions
st.header("Enter the feature values:")
temp = st.number_input("Temperature (e.g., 25.3):", value=0.0, step=0.1)
humidity = st.number_input("Humidity (e.g., 50.7):", value=0.0, step=0.1)
pressure = st.number_input("Pressure (e.g., 1012):", value=0.0, step=0.1)
wind_speed = st.number_input("Wind Speed (e.g., 3.2):", value=0.0, step=0.1)
dew_point = st.number_input("Dew Point (e.g., 12.4):", value=0.0, step=0.1)
solar_radiation = st.number_input("Solar Radiation (e.g., 300):", value=0.0, step=1.0)
soil_temp = st.number_input("Soil Temperature (e.g., 18.5):", value=0.0, step=0.1)

# Input data as a NumPy array
input_data = np.array([[temp, humidity, pressure, wind_speed, dew_point, solar_radiation, soil_temp]])

# Feature scaling using StandardScaler
scaler = StandardScaler()
scaled_input_data = scaler.fit_transform(input_data)

# Predict soil moisture
if st.button("Predict"):
    prediction = model.predict(scaled_input_data)
    st.subheader("Prediction Result:")
    st.write(f"The predicted soil moisture level is: {prediction[0]:.2f}")

# Additional information
st.markdown("""
### About
This application uses a machine learning model to predict soil moisture levels. 
The model was trained using features like temperature, humidity, pressure, and other environmental factors.
""")