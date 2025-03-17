import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
MODEL_PATH = "svm_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please ensure the model is present in the correct path.")
    st.stop()

st.set_page_config(page_title="Stress Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Stress Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter your physiological details to predict stress levels</h4>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About This App")
st.sidebar.info("This app predicts stress levels based on physiological parameters using a trained ML model.")

# Two-column layout for input fields
col1, col2 = st.columns(2)

# Column 1 inputs with default value set to zero
with col1:
    snoring_range = st.text_input("Snoring Range", "0.0")
    respiration_rate = st.text_input("Respiration Rate", "0.0")
    body_temp = st.text_input("Body Temperature (F)", "0.0")
    limb_movement = st.text_input("Limb Movement Rate", "0.0")

# Column 2 inputs with default value set to zero
with col2:
    blood_oxygen = st.text_input("Blood Oxygen Levels (%)", "0.0")
    eye_movement = st.text_input("Eye Movement Rate", "0.0")
    sleep_hours = st.text_input("Sleep Hours", "0.0")
    heart_rate = st.text_input("Heart Rate (BPM)", "0.0")

# Button to Predict Stress Level
if st.button("Predict Stress Level"):
    try:
        # Default to 0.0 if the input is empty or invalid
        snoring_range = float(snoring_range) if snoring_range else 0.0
        respiration_rate = float(respiration_rate) if respiration_rate else 0.0
        body_temp = float(body_temp) if body_temp else 0.0
        limb_movement = float(limb_movement) if limb_movement else 0.0
        blood_oxygen = float(blood_oxygen) if blood_oxygen else 0.0
        eye_movement = float(eye_movement) if eye_movement else 0.0
        sleep_hours = float(sleep_hours) if sleep_hours else 0.0
        heart_rate = float(heart_rate) if heart_rate else 0.0

        # Create the features array with the correct column names
        features = pd.DataFrame([[
            snoring_range, respiration_rate, body_temp, limb_movement,
            blood_oxygen, eye_movement, sleep_hours, heart_rate
        ]], columns=['snoring_range', 'respiration_rate', 'body_temp', 'limb_movement',
                     'blood_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate'])

        # Rename columns to match model's training feature names based on error message clues
        features = features.rename(columns={
            'blood_oxygen': 'bo',   # 'bo' for blood oxygen
            'body_temp': 't',       # 't' for body temperature
            'eye_movement': 'rem',  # 'rem' for eye movement
            'heart_rate': 'hr',     # 'hr' for heart rate
            'limb_movement': 'lm',  # 'lm' for limb movement
            'respiration_rate': 'rr',  # 'rr' for respiration rate
            'snoring_range': 'sr',  # 'sr' for snoring range
            'sleep_hours': 'sh'     # 'sh' for sleep hours
        })

        # Model prediction
        prediction = model.predict(features)[0]

        # Display the prediction result
        st.markdown(f"""
        <div style="background-color:#4CAF50;padding:10px;border-radius:10px;">
            <h3 style="text-align:center;color:white;">Predicted Stress Level: {prediction}</h3>
        </div>
        """, unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Error: {str(e)}. Please ensure all fields contain valid numeric values.")
