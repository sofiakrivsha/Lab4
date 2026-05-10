import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
# Шлях до файлу з даними
input_file = r'C:\Users\sofia\Downloads\data_multivar_regr.txt'
# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Для візуалізації візьмемо лише перший стовпець даних
X_single = X[:, 0].reshape(-1, 1)
# Розподіл на навчальну та тестову вибірки
num_training = int(0.8 * len(X_single))
X_train, y_train = X_single[:num_training], y[:num_training]
X_test, y_test = X_single[num_training:], y[num_training:]
# 1. Лінійна регресія
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# 2. Поліноміальна регресія (ступінь 2)
polynomial = PolynomialFeatures(degree=2)
X_train_transformed = polynomial.fit_transform(X_train)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
# Візуалізація результатів
plt.figure()
plt.scatter(X_single, y, color='green', s=10, label='Дані')
# Побудова лінії лінійної регресії
X_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
y_line = regressor.predict(X_range)
plt.plot(X_range, y_line, color='red', linewidth=2, label='Лінійна регресія')
# Побудова кривої поліноміальної регресії
X_range_poly = polynomial.transform(X_range)
y_poly = poly_linear_model.predict(X_range_poly)
plt.plot(X_range, y_poly, color='blue', linewidth=3, label='Поліноміальна регресія')
plt.title('Порівняння лінійної та поліноміальної регресії')
plt.legend()
plt.show()
# Перевірка на конкретному значенні
datapoint = [[7.75]]
print(f"Прогноз (лінійна):", regressor.predict(datapoint)[0])
print(f"Прогноз (поліноміальна):", poly_linear_model.predict(polynomial.transform(datapoint))[0])
