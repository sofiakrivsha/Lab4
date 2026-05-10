import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
# Шлях до файлу з багатовимірними даними
input_file = r'C:\Users\sofia\Downloads\data_multivar_regr.txt'
# Завантаження даних 
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Розбиття на навчальну та тестову вибірки 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Створення та навчання багатовимірного регресора 
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# Прогнозування результату 
y_test_pred = regressor.predict(X_test)
# Вивід коефіцієнтів моделі 
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)
# Обчислення метрик якості 
print("\nPerformance metrics:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
# Тестування на нових даних (приклад) 
# Припустимо, ми хочемо спрогнозувати результат для конкретного набору вхідних значень
test_data = [3.45, 5.21, 1.98]
print(f"\nPredicted output for {test_data}:", round(regressor.predict([test_data])[0], 2))
