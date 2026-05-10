import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# Завантаження даних (Diabetes dataset часто використовується для прикладів регресії)
diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2] # беремо одну ознаку для візуалізації
y = diabetes.target
# Розбиття на вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# 1. Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)
# 2. Поліноміальна регресія ступеня 2
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_train_poly, y_train)
y_pred_poly = poly_regressor.predict(X_test_poly)
# Візуалізація
plt.scatter(X_test, y_test, color='black', label='Фактичні дані')
# Побудова лінії лінійної регресії
plt.plot(X_test, y_pred_linear, color='blue', linewidth=3, label='Лінійна')
# Побудова кривої поліноміальної регресії
X_grid = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_grid_poly = poly_regressor.predict(poly_features.transform(X_grid))
plt.plot(X_grid, y_grid_poly, color='red', linewidth=3, label='Поліноміальна')
plt.title('Порівняння регресій на реальних даних')
plt.xlabel('Ознака')
plt.ylabel('Цільовий показник')
plt.legend()
plt.show()
print("Метрики Лінійної регресії:")
print("MSE:", round(mean_squared_error(y_test, y_pred_linear), 2))
print("R2:", round(r2_score(y_test, y_pred_linear), 2))
print("\nМетрики Поліноміальної регресії:")
print("MSE:", round(mean_squared_error(y_test, y_pred_poly), 2))
print("R2:", round(r2_score(y_test, y_pred_poly), 2))
