import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.sav')

st.title("Advertising Sales Prediction App")
st.write("Enter the advertising budget (in thousands of dollars) for each channel to predict Sales.")

# Input fields
tv_budget = st.number_input("TV Budget ($1000s)", min_value=0.0, value=100.0, step=1.0)
radio_budget = st.number_input("Radio Budget ($1000s)", min_value=0.0, value=20.0, step=1.0)
newspaper_budget = st.number_input("Newspaper Budget ($1000s)", min_value=0.0, value=15.0, step=1.0)

# Predict button
if st.button("Predict Sales"):
    # Create a DataFrame matching the model's feature names
    input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], 
                              columns=['TV', 'Radio', 'Newspaper'])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales: {prediction:.2f} (in thousands of units)")

print("app.py has been created successfully!")

# Create requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("streamlit\nscikit-learn\njoblib\npandas\n")

print("requirements.txt has been created successfully!")
