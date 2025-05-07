# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load ML model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# App Title
st.title("SRF Property Prediction App")
st.markdown("Upload your dataset or manually input waste composition to predict SRF fuel characteristics.")

# Sidebar for input method
option = st.sidebar.radio("Choose Input Method", ["Upload File", "Manual Input"])

def preprocess_input(df):
    # Add any required preprocessing steps here
    return df

def predict(df):
    return model.predict(df)

# File Upload Option
if option == "Upload File":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(data)

        processed_data = preprocess_input(data)
        predictions = predict(processed_data)

        data['Predicted HHV'] = predictions
        st.subheader("Prediction Results")
        st.dataframe(data)

        fig = px.histogram(data, x="Predicted HHV", nbins=20, title="Predicted HHV Distribution")
        st.plotly_chart(fig)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "srf_predictions.csv", "text/csv")

# Manual Input Option
elif option == "Manual Input":
    st.subheader("Enter Waste Composition")
    plastic = st.slider("Plastic (%)", 0, 100, 10)
    paper = st.slider("Paper (%)", 0, 100, 10)
    organic = st.slider("Organic (%)", 0, 100, 10)
    moisture = st.slider("Moisture (%)", 0, 100, 20)
    ash = st.slider("Ash (%)", 0, 100, 15)

    input_dict = {
        "Plastic": [plastic],
        "Paper": [paper],
        "Organic": [organic],
        "Moisture": [moisture],
        "Ash": [ash]
    }

    input_df = pd.DataFrame(input_dict)
    st.write("Input Data:")
    st.dataframe(input_df)

    if st.button("Predict"):
        processed_data = preprocess_input(input_df)
        prediction = predict(processed_data)
        st.success(f"Predicted HHV: {prediction[0]:.2f} MJ/kg")

