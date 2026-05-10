import numpy as np
import matplotlib.pyplot as plt
# 1. Вектори даних зі скріншота 
x_points = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y_points = np.array([3.2, 3, 1, 1.8, 1.9])
# 2. Заповнення матриці X (Вандермонда) для полінома 4-го ступеня 
# Матриця має вигляд: [1, x, x^2, x^3, x^4]
X_matrix = np.vander(x_points, 5, increasing=True)
# 3. Отримання коефіцієнтів інтерполяційного полінома 
# Розв'язуємо систему лінійних рівнянь X * A = Y
a_coeffs = np.linalg.solve(X_matrix, y_points)
print("Коефіцієнти полінома (a0, a1, a2, a3, a4):")
print(np.round(a_coeffs, 4))
# 4. Визначення функції полінома 
def polynomial_func(x):
    return sum(a * (x**i) for i, a in enumerate(a_coeffs))
# 5. Визначення значень у проміжних точках 0.2 та 0.5 
points_to_check = [0.2, 0.5]
for pt in points_to_check:
    print(f"Значення у точці x = {pt}: {polynomial_func(pt):.4f}")
# 6. Побудова графіка 
x_range = np.linspace(0, 0.8, 100)
y_range = [polynomial_func(x) for x in x_range]
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, color='blue', label='Інтерполяційний поліном (4 ст.)')
plt.scatter(x_points, y_points, color='red', label='Вузли інтерполяції')
plt.scatter(points_to_check, [polynomial_func(pt) for pt in points_to_check], 
            color='green', marker='x', s=100, label='Проміжні точки (0.2, 0.5)')
plt.title('Інтерполяція поліномом 4-го ступеня')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
