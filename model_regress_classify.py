import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import joblib
import streamlit as st
import requests
from io import StringIO

sheet_id = '1L8HOtCvDeGdtLOmWPKrF-5YtkR1ubX-4lnMcaoPZQdU'
sheet_name = 'Preprocessing data Export'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))
    st.write(df)
else:
    st.write("Lỗi khi tải dữ liệu")

df_copy=df[['type',"days_for_shipment_scheduled","delivery_status","late_delivery_risk","category_id",
            "customer_city","customer_country","customer_segment","customer_state","latitude","longitude","order_country","order_city",
            "order_item_product_price","order_item_quantity","order_status","product_card_id","product_price",
            "shipping_date_dateorders","shipping_mode","late_days","order_date_dateorders","order_region","market"]]
target1 = 'late_delivery_risk'
if target1 not in df_1.columns:
    raise ValueError(f"The target variable '{target1}' is not present in the dataset.")
    
X = df_copy.drop(target1, axis=1)
y = df_copy[target1]

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
X = X_balanced
y = y_balanced

# Chia dữ liệu thành train và test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tiền xử lý dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Các mô hình cần thử
models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression()),
    ("KNeighbors", KNeighborsClassifier())
]

# Tạo một dictionary để lưu kết quả
output = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

# Huấn luyện và đánh giá từng mô hình
for model_name, model in models:
    model.fit(X_train, y_train)
    accuracy_scores = cross_val_score(model, X_test, y_test, scoring='accuracy')
    precision_scores = cross_val_score(model, X_test, y_test, scoring='precision')
    recall_scores = cross_val_score(model, X_test, y_test, scoring='recall')
    f1_scores = cross_val_score(model, X_test, y_test, scoring='f1')
    
    # Lưu kết quả cho mỗi mô hình
    output['Model'].append(model_name)
    output['Accuracy'].append(np.mean(accuracy_scores))
    output['Precision'].append(np.mean(precision_scores))
    output['Recall'].append(np.mean(recall_scores))
    output['F1-score'].append(np.mean(f1_scores))

# Chuyển kết quả thành DataFrame
output_df = pd.DataFrame(output)

# Tìm mô hình tốt nhất theo F1-score
best_model_index = output_df['F1-score'].idxmax()
best_model_name = output_df['Model'][best_model_index]
best_model = models[best_model_index][1]

# Huấn luyện lại mô hình tốt nhất trên toàn bộ dữ liệu
best_model.fit(X_train, y_train)

# Lưu mô hình tốt nhất vào file
joblib.dump(best_model, 'best_model.pkl')

# Hiển thị kết quả trong Streamlit
st.title("Model Evaluation and Selection")

st.subheader("Model Performance Results")
st.write(output_df)

st.subheader(f"Best Model: {best_model_name}")
st.write(f"F1-Score: {output_df['F1-score'][best_model_index]:.4f}")

# Load and make predictions with the best model
best_model_loaded = joblib.load('best_model.pkl')

# Dự đoán mẫu dữ liệu (giả sử bạn có data nhập vào từ người dùng)
instances_to_predict = pd.DataFrame({
    # Thêm dữ liệu cần dự đoán từ người dùng ở đây
})

# Tiền xử lý dữ liệu nhập vào
instances_to_predict_scaled = scaler.transform(instances_to_predict)

# Dự đoán
predictions = best_model_loaded.predict(instances_to_predict_scaled)

# Hiển thị dự đoán
st.subheader("Predictions")
st.write(predictions)
