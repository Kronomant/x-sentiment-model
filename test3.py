import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Dados fornecidos
x_data = np.array([19, 3, 2, 11, 9, 14, 4, 17, 5, 1, 6, 20])
y_data = np.array([8, 18, 25, 8, 13, 9, 17, 6, 20, 42, 13, 7])

# Função de modelo
def quadratic_model(x, a, b, c):
    return a + b * x + c * x**2

# Ajuste de curva
params, covariance = curve_fit(quadratic_model, x_data, y_data)

# Parâmetros ajustados
a, b, c = params

# Criando uma curva usando os parâmetros ajustados
x_curve = np.linspace(min(x_data), max(x_data), 100)
y_curve = quadratic_model(x_curve, a, b, c)

# Plotando os dados e a curva ajustada
plt.scatter(x_data, y_data, label='Dados')
plt.plot(x_curve, y_curve, color='red', label='Regressão Quadrática')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()