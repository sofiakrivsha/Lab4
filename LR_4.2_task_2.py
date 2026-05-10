import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Вхідні дані за варіантом 9
x = np.array([1.5, 2.3, 3.1, 3.9, 4.7, 5.5]).reshape((-1, 1))
y = np.array([2.6, 3.1, 4.8, 5.2, 5.9, 6.8])
# Побудова моделі за методом найменших квадратів
model = LinearRegression()
model.fit(x, y)
# Отримання результатів розрахунку
r_sq = model.score(x, y)
a = model.coef_[0]
b = model.intercept_
print(f"Коефіцієнт детермінації (R2): {r_sq:.4f}")
print(f"Рівняння регресії: y = {a:.3f}x + {b:.3f}")
# Прогноз для побудови графіку
y_pred = model.predict(x)
# Візуалізація
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Експериментальні точки (Вар 9)')
plt.plot(x, y_pred, color='red', linewidth=2, label=f'Лінія регресії: y = {a:.2f}x + {b:.2f}')
plt.title('Лінійна регресія за методом найменших квадратів')
plt.xlabel('Параметр x')
plt.ylabel('Параметр y')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
