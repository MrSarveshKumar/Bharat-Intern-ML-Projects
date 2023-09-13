import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained machine learning model from model.pkl
model_file = 'model.pkl'
with open(model_file, 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app title
st.title("Wine Quality Prediction")

# Input form for user to enter wine features
st.header("Input Features")
fixed_acidity = st.slider("Fixed Acidity", 4.6, 16.0, 8.31)
volatile_acidity = st.slider("Volatile Acidity", 0.12, 1.58, 0.53)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.29)
residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.2)
chlorides = st.slider("Chlorides", 0.012, 0.611, 0.076)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 13)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 38)
density = st.slider("Density", 0.990, 1.039, 0.9967)
pH = st.slider("pH", 2.74, 4.01, 3.31)
sulphates = st.slider("Sulphates", 0.33, 2.0, 0.66)
alcohol = st.slider("Alcohol", 8.4, 14.9, 9.55)

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# Make predictions
prediction = model.predict(user_input)

# Display the predicted quality
st.subheader("Predicted Wine Quality")
rounded_prediction = round(prediction[0])
st.success(f"The predicted wine quality is: {rounded_prediction}")

st.markdown("****")

st.write("NOTE: This is only for Educational Purpose")
st.write("<span style='font-size: 15px;'>Founder: *Sarvesh Kumar*</span>", unsafe_allow_html=True)
