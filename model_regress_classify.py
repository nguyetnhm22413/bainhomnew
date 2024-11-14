# -*- coding: utf-8 -*-
"""model_regress_classify.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nUI4HgqV9bHn5OPnLS8qnCeVQBJLk5BJ
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Function to perform frequency encoding
def frequency_encoding(df, column):
    frequency_map = df[column].value_counts(normalize=True)
    df[column + '_freq_encoded'] = df[column].map(frequency_map)
    return df

# Streamlit title
st.title("Delivery Prediction App")

# New data to predict
instances_to_predict = pd.DataFrame({
    'Type': ['DEBIT', 'DEBIT'],
    'Days for shipment (scheduled)': [5, 6],
    'Delivery Status': ['Advance shipping', 'Advance shipping'],
    'Category Id': [17, 17],
    'Category Name': ['Cleats', 'Cleats'],
    'Customer City': ['Los Angeles', 'Los Angeles'],
    'Customer Country': ['EE. UU.', 'EE. UU.'],
    'Customer Segment': ['Corporate', 'Corporate'],
    'Customer State': ['NY', 'NY'],
    'Latitude': [17.24253835, 17.24253835],
    'Longitude': [-65.03704823, -65.03704823],
    'Order City': ['Bikaner', 'Bikaner'],
    'Order Country': ['India', 'India'],
    'Order Item Product Price': [327.75, 327.75],
    'Order Item Quantity': [2, 2],
    'Order Status': ['COMPLETE', 'COMPLETE'],
    'Product Card Id': [1360, 1360],
    'Product Price': [327.75, 327.75],
    'Order Region': ['Southeast Asia', 'Southeast Asia'],
    'Market': ['Asia', 'Asia']
})

# Columns to be frequency encoded
columns_to_encode = ['Type', 'Delivery Status', 'Category Name', 'Customer City', 'Customer Country', 'Customer Segment',
                     'Customer State', 'Order City', 'Order Country', 'Order Status', 'Order Region', 'Market']

# Perform frequency encoding for each column
for column in columns_to_encode:
    instances_to_predict = frequency_encoding(instances_to_predict, column)

# Drop the original categorical columns
instances_to_predict.drop(columns=columns_to_encode, inplace=True)

# Standardize the features
scaler = StandardScaler()
instances_to_predict_scaled = scaler.fit_transform(instances_to_predict)

# Simulated training data and labels
X_train = np.random.rand(10, instances_to_predict_scaled.shape[1])
y_train = np.random.randint(0, 2, 10)  # Random binary labels

# Train the model
gb_model = RandomForestClassifier()
gb_model.fit(X_train, y_train)

# Predict whether delivery is late or not
predictions = gb_model.predict(instances_to_predict_scaled)

# Display the predictions in Streamlit
st.subheader("Predictions for Late Delivery:")
st.write("Case 1: Predicted Late Delivery:", "Yes" if predictions[0] == 1 else "No")
st.write("Case 2: Predicted Late Delivery:", "Yes" if predictions[1] == 1 else "No")

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Function to perform frequency encoding
def frequency_encoding(df, column):
    frequency_map = df[column].value_counts(normalize=True)
    df[column + '_freq_encoded'] = df[column].map(frequency_map)
    return df

# Streamlit title
st.title("Delivery Prediction App")

# New data to predict
instances_to_predict = pd.DataFrame({
    'Delivery Status': ['Advance shipping', 'Advance shipping'],
    'Category Id': [17, 17],
    'Customer City': ['Los Angeles', 'Los Angeles'],
    'Customer Country': ['EE. UU.', 'EE. UU.'],
    'Customer Segment': ['Corporate', 'Corporate'],
    'Customer State': ['NY', 'NY'],
    'Order City': ['Bikaner', 'Bikaner'],
    'Order Country': ['India', 'India'],
    'Order Item Product Price': [327.75, 327.75],
    'Order Item Quantity': [2, 2],
    'Order Status': ['COMPLETE', 'COMPLETE'],
    'Product Card Id': [1360, 1360],
    'Product Price': [327.75, 327.75],
    'Order Region': ['Southeast Asia', 'Southeast Asia'],
    'Market': ['Asia', 'Asia']
})

# Columns to be frequency encoded
columns_to_encode = ['Delivery Status','Category Id','Customer City','Customer Country','Customer Segment',
                     'Customer State','Order City','Order Country','Order Status','Order Region','Market']

# Perform frequency encoding for each column
for column in columns_to_encode:
    instances_to_predict = frequency_encoding(instances_to_predict, column)

# Drop the original categorical columns
instances_to_predict.drop(columns=columns_to_encode, inplace=True)

# Standardize the features
scaler = StandardScaler()
instances_to_predict_scaled = scaler.fit_transform(instances_to_predict)

# Simulated training data and labels
# Replace these with your actual training data and labels
X_train = np.random.rand(10, instances_to_predict_scaled.shape[1])
y_train = np.random.randint(0, 2, 10)  # Random binary labels (0 or 1)

# Train the model using LinearRegression with Polynomial Features
gb_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
gb_model.fit(X_train, y_train)

# Predict whether delivery is late or not (binary classification: 1 for late, 0 for on time)
predictions = gb_model.predict(instances_to_predict_scaled)

# Display the predictions in Streamlit
st.subheader("Predictions for Late Delivery:")
st.write("Case 1: Predicted Sales:", predictions[0])
st.write("Case 2: Predicted Sales:", predictions[1])