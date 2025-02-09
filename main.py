import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load("kmeans-model-new.pkl")
scaler = joblib.load("scaler.pkl")
def pre_input(quantity, sales, cost):
    data = pd.DataFrame([[quantity, sales, cost]], columns=['Quantity', 'Sales', 'Cost'])
    scaled_data = scaler.transform(data)
    return scaled_data

st.title('Reseller Segmentation')

quantity = st.number_input('Enter Quantity Sold', min_value=0)
sales = st.number_input('Enter Sales Amount ($)', min_value=0.0)
cost = st.number_input('Enter Cost ($)', min_value=0.0)

if st.button('Predict Cluster'):
    input_data = pre_input(quantity, sales, cost)
    cluster = model.predict(input_data)[0]
    st.success(f'The seller belongs to **Cluster {cluster}**')
