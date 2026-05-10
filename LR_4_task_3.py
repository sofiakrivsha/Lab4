import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import pickle
from sklearn.model_selection import train_test_split
# Шлях до файлу з даними
input_file = r'C:\Users\sofia\Downloads\data_multivar_regr.txt'
# Завантаження та підготовка даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Створення та навчання моделі
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# Шлях для збереження моделі
output_model_file = r'C:\Users\sofia\Downloads\model.pkl'
# Збереження моделі у файл
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
print(f"Модель успішно збережена у файл: {output_model_file}")
# Завантаження моделі з файлу
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)
# Використання завантаженої моделі для прогнозу
y_test_pred = regressor_model.predict(X_test)
# Вивід результатів для перевірки
print("\nМетрики завантаженої моделі:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
