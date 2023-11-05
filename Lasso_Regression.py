import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV

# Đọc dữ liệu từ tệp CSV
file_path = 'D:\\hoctap\\Nam4\\Datamiling\\Dushanbe_house.csv'
df = pd.read_csv(file_path)

# Kiểm tra và xử lý dữ liệu null
print("\nTrước khi loại bỏ dữ liệu null: ", df.isnull().sum())
df.dropna(axis=0, inplace=True)
print("\nSau khi loại bỏ dữ liệu null: ", df.isnull().sum())

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
columns = df.columns
Inputs = df[columns[:-1]]
outputs = df[columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(Inputs, outputs, test_size=0.25, random_state=42)

# Thiết lập các giá trị alpha cần kiểm tra
parameters = {'alpha': np.logspace(-3, 2, 100)}  # Các giá trị alpha cần thử

lasso_model = Lasso()

# Tìm giá trị alpha tốt nhất thông qua Randomized Search
lasso_random = RandomizedSearchCV(lasso_model, parameters, n_iter=20, cv=5, random_state=42)
lasso_random.fit(X_train, y_train)

print("Alpha tốt nhất từ Randomized Search:", lasso_random.best_params_)

# Huấn luyện mô hình với alpha tốt nhất
lasso_best = Lasso(alpha=lasso_random.best_params_['alpha'])
lasso_best.fit(X_train, y_train)
y_pred_best = lasso_best.predict(X_test)

# Vẽ biểu đồ giá trị dự đoán và thực tế
plt.figure(figsize=(15, 8))
plt.plot([i for i in range(len(y_test))], y_test, label='Giá trị thực')
plt.plot([i for i in range(len(y_test))], y_pred_best, label='Giá trị dự đoán')
plt.legend()
plt.show()
