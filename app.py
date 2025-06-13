# ["bed", "bath", "house_size"]
# ["price"]
import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")

model = joblib.load("model.pkl")

st.title("House Price Prediction")

st.divider()
bed = st.number_input("Number of Bedrooms",value=2, step = 1)
bath = st.number_input("Number of Bathrooms",value=1, step = 1)
size = st.number_input("House Size (in sqft)",value=1000, step = 50)
x = [bed, bath, size]

st.divider()

predict = st.button("Predict Price")

st.divider()

if predict:
    st.balloons()
    x1 = np.array(x).reshape(1, -1)
    x_array = scaler.transform(x1)
    predict = model.predict(x_array.reshape(1, -1))
    st.subheader(f"Predicted Price: ${predict[0]:,.2f}")