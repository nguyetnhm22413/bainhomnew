import joblib
import streamlit as st

# Hiển thị kết quả trong Streamlit
st.title("Random Forest Model")

# Tải mô hình đã lưu
rf_model_loaded = joblib.load('best_model.pkl')

# Nhập dữ liệu từ người dùng
st.subheader("Enter data for prediction:")
type = st.text_input("Type")
days_for_shipment_scheduled = st.number_input("Days for Shipment Scheduled", min_value=0)
delivery_status = st.text_input("Delivery Status")
category_id = st.number_input("Category ID", min_value=0)
customer_city = st.text_input("Customer City")
# Thêm các input tương ứng với các trường dữ liệu cần dự đoán

# Dự đoán
if st.button("Predict"):
    # Tạo DataFrame cho các giá trị nhập từ người dùng
    user_data = pd.DataFrame([[type, days_for_shipment_scheduled, delivery_status, category_id, customer_city]],
                             columns=['type', 'days_for_shipment_scheduled', 'delivery_status', 'category_id', 'customer_city'])
    
    # Tiền xử lý dữ liệu nhập vào (nếu cần)
    # user_data_scaled = scaler.transform(user_data)  # Nếu mô hình cần tiền xử lý dữ liệu, hãy thêm bước này
    
    # Dự đoán
    prediction = rf_model_loaded.predict(user_data)
    
    # Hiển thị dự đoán
    st.subheader("Prediction Result")
    st.write(f"Prediction: {prediction[0]}")
