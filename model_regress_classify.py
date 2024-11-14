import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Giả định rằng X và y là dữ liệu đã được xử lý và sẵn sàng
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Các mô hình sẽ thử nghiệm
models = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression()),
    ("KNeighbors", KNeighborsClassifier())
]

# Tạo dictionary để lưu kết quả
output = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

# Đánh giá mô hình với các chỉ số khác nhau
for model_name, model in models:
    accuracy_scores = cross_val_score(model, X_test, y_test, scoring='accuracy')
    precision_scores = cross_val_score(model, X_test, y_test, scoring='precision')
    recall_scores = cross_val_score(model, X_test, y_test, scoring='recall')
    f1_scores = cross_val_score(model, X_test, y_test, scoring='f1')

    output['Model'].append(model_name)
    output['Accuracy'].append(np.mean(accuracy_scores))
    output['Precision'].append(np.mean(precision_scores))
    output['Recall'].append(np.mean(recall_scores))
    output['F1-score'].append(np.mean(f1_scores))

# Chuyển kết quả thành DataFrame và hiển thị trong Streamlit
output_df = pd.DataFrame(output)
st.write("Model Comparison:", output_df)

# Chọn mô hình tốt nhất dựa trên F1-score hoặc một chỉ số khác
best_model_name = output_df.loc[output_df['F1-score'].idxmax(), 'Model']
best_model = dict(models)[best_model_name]

# Huấn luyện lại mô hình tốt nhất trên toàn bộ dữ liệu huấn luyện
best_model.fit(X_train, y_train)

# Lưu mô hình tốt nhất
joblib.dump(best_model, 'best_model.pkl')

# Hiển thị mô hình tốt nhất
st.write(f"Best Model: {best_model_name} with F1-score: {output_df['F1-score'].max()}")

# Cho phép tải mô hình tốt nhất qua Streamlit
st.download_button(label="Download Best Model", data=open('best_model.pkl', 'rb').read(), file_name='best_model.pkl')
