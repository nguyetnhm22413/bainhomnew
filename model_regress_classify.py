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

# User input for data
st.subheader("Enter the details for delivery prediction:")

# Input fields for each feature
type_input = st.selectbox("Type", ['DEBIT', 'CREDIT'])
days_for_shipment_input = st.number_input("Days for shipment (scheduled)", min_value=0, value=5)
delivery_status_input = st.selectbox("Delivery Status", ['Advance shipping', 'Standard shipping', 'On-time'])
category_id_input = st.number_input("Category Id", min_value=0, value=17)
category_name_input = st.text_input("Category Name", 'Cleats')
customer_city_input = st.text_input("Customer City", 'Los Angeles')
customer_country_input = st.text_input("Customer Country", 'EE. UU.')
customer_segment_input = st.selectbox("Customer Segment", ['Corporate', 'Consumer', 'Home office'])
customer_state_input = st.text_input("Customer State", 'NY')
latitude_input = st.number_input("Latitude", value=17.24253835)
longitude_input = st.number_input("Longitude", value=-65.03704823)
order_city_input = st.text_input("Order City", 'Bikaner')
order_country_input = st.text_input("Order Country", 'India')
order_item_product_price_input = st.number_input("Order Item Product Price", min_value=0.0, value=327.75)
order_item_quantity_input = st.number_input("Order Item Quantity", min_value=1, value=2)
order_status_input = st.selectbox("Order Status", ['COMPLETE', 'PENDING'])
product_card_id_input = st.number_input("Product Card Id", min_value=0, value=1360)
product_price_input = st.number_input("Product Price", min_value=0.0, value=327.75)
order_region_input = st.text_input("Order Region", 'Southeast Asia')
market_input = st.text_input("Market", 'Asia')

# Collect user inputs into a DataFrame
user_input_data = pd.DataFrame({
    'Type': [type_input],
    'Days for shipment (scheduled)': [days_for_shipment_input],
    'Delivery Status': [delivery_status_input],
    'Category Id': [category_id_input],
    'Category Name': [category_name_input],
    'Customer City': [customer_city_input],
    'Customer Country': [customer_country_input],
    'Customer Segment': [customer_segment_input],
    'Customer State': [customer_state_input],
    'Latitude': [latitude_input],
    'Longitude': [longitude_input],
    'Order City': [order_city_input],
    'Order Country': [order_country_input],
    'Order Item Product Price': [order_item_product_price_input],
    'Order Item Quantity': [order_item_quantity_input],
    'Order Status': [order_status_input],
    'Product Card Id': [product_card_id_input],
    'Product Price': [product_price_input],
    'Order Region': [order_region_input],
    'Market': [market_input]
})

# Columns to be frequency encoded
columns_to_encode = ['Type', 'Delivery Status', 'Category Name', 'Customer City', 'Customer Country', 'Customer Segment',
                     'Customer State', 'Order City', 'Order Country', 'Order Status', 'Order Region', 'Market']

# Perform frequency encoding for each column
for column in columns_to_encode:
    user_input_data = frequency_encoding(user_input_data, column)

# Drop the original categorical columns
user_input_data.drop(columns=columns_to_encode, inplace=True)

# Standardize the features
scaler = StandardScaler()
user_input_data_scaled = scaler.fit_transform(user_input_data)

# Simulated training data and labels (replace with actual data for production)
X_train = np.random.rand(10, user_input_data_scaled.shape[1])
y_train = np.random.randint(0, 2, 10)  # Random binary labels (0 or 1)

# Train the model
gb_model = RandomForestClassifier()
gb_model.fit(X_train, y_train)

# Predict whether delivery is late or not
predictions = gb_model.predict(user_input_data_scaled)

# Display the predictions in Streamlit
st.subheader("Predictions for Late Delivery:")
st.write("Predicted Late Delivery:", "Yes" if predictions[0] == 1 else "No")

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
