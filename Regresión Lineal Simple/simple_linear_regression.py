#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Enero 2023

@author: Mateo Echavarria
"""

# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
"""test_size = tamaño del conjunto test, el sobrante sera el conjunto de entrenamiento"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
#Adecuamos el modelo con los datos de entrenamiento
regression.fit(X_train, y_train)

# Predecir el conjunto de test
"""Tomamos los datos a predecir y luego los compararemos con el y_test"""
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

#EVALUACION DE % DE ERROR
error_aprox = np.mean([ abs((y_pred[i]-y_test[i])/y_test[i]) for i in range(len(X_test))])
print(f' Error aproximado entre y_pred & y_test = {round(error_aprox, 4)*100} %')
