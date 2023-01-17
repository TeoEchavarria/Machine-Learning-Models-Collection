#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Enero 2023

@author: Mateo Echavarria Sierra
"""

# Regresión polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red") #scatter = Nube de puntos
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


#Observamos las graficas al cambiar el grado del modelo de prediccion Polinomica
for i in range(10):
    # Ajustar la regresión polinómica con el dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = i+1)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    
    # Visualización de los resultados del Modelo Polinómico
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1)
    plt.scatter(X, y, color = "red")
    plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
    plt.title(f"Modelo de Regresión Polinómica (grado {i+1})")
    plt.xlabel("Posición del empleado")
    plt.ylabel("Sueldo (en $)")
    plt.show()
    

# Predicción de nuestros modelos
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))






