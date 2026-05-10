import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split # Оновлено замість cross_validation
# Шлях до файлу з даними
input_file = r'C:\Users\sofia\Downloads\data_singlevar_regr.txt'
# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Розбиття на навчальну та тестову вибірки (80% на 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Створення та навчання регресора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# Прогнозування результатів
y_test_pred = regressor.predict(X_test)
# Візуалізація результатів
plt.scatter(X_test, y_test, color='green', label='Actual data')
plt.plot(X_test, y_test_pred, color='black', linewidth=4, label='Regression line')
plt.title('Linear Regression: Single Variable')
plt.xlabel('Input Variable')
plt.ylabel('Target Variable')
plt.legend()
plt.show()
# Вивід метрик якості
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
