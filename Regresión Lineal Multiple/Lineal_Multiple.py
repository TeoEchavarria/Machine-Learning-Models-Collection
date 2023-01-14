# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:02:12 2022

@author: Mateo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("Data/50_Startups.csv")
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values  # variable a predecir el profit

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=float)

# Evitar variables ficticias - Repeticion de datos cuando hacemos lo de Dummies 
X = X[:, 1:]


# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)


# Ajustar modelo : REGRESION LINEAL MULTIPLE (modelo de entrenamiento)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de resultado con el conjunto de testeo

y_prediction = regression.predict(X_test)

#EVALUACION DE % DE ERROR
error_aprox = np.mean([ abs((y_prediction[i]-y_test[i])/y_test[i]) for i in range(len(X_test))])
print(f' Error aproximado entre y_prediction & y_test = {round(error_aprox, 4)*100} %')


# ----- ELIMINACION HACIA ATRAS (ELIMINACION DE VARIABLES PARA MEJORAR PREDICCIONES -----
import statsmodels.api as sm
# agregamos una columna de 1's representando la variable dependiente
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 

nivel_significacion = 0.05  # si una de las variables tiene un nivel de significacion mayor a este SE ELIMINA


X_optimo = X[:, [0,1,2,3,4,5]]
regresion_ordinal_list_squars = sm.OLS(endog= y, exog = X_optimo).fit()

#me muestra la informacion interna y desde ahí podemos ver que variables son realmente significativas
regresion_ordinal_list_squars.summary()

#observamos el P>|t| valor y si vemos que es mas alto que 0.5 significa que dicho dato no es relevante para la prediccion del modelo

X_optimo = X[:, [0,1,3,4,5]]
regresion_ordinal_list_squars = sm.OLS(endog= y, exog = X_optimo).fit()
regresion_ordinal_list_squars.summary()

X_optimo = X[:, [0,3,4,5]]
regresion_ordinal_list_squars = sm.OLS(endog= y, exog = X_optimo).fit()
regresion_ordinal_list_squars.summary()

X_optimo = X[:, [0,3,5]]
regresion_ordinal_list_squars = sm.OLS(endog= y, exog = X_optimo).fit()
regresion_ordinal_list_squars.summary()


"""X_optimo = X[:, [0,3]]
regresion_ordinal_list_squars = sm.OLS(endog= y, exog = X_optimo).fit()
regresion_ordinal_list_squars.summary()"""


# volvemos a dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_optimo, y, test_size = 0.2, random_state = 0)
# REGRESION LINEAL MULTIPLE (modelo de entrenamiento)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de resultado con el conjunto de testeo

y_prediction_optimal = regression.predict(X_test)

#EVALUACION DE % DE ERROR
error_aprox = np.mean([ abs((y_prediction_optimal[i]-y_test[i])/y_test[i]) for i in range(len(X_test))])
print(f' Error aproximado entre y_prediction_optimal & y_test = {round(error_aprox, 4)*100} %')

#Visualización de datos
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')


# Agregamos los puntos en el plano 3D
ax1.scatter(X_train[:, 1], X_train[:, 2], y_train, c='g', marker='o')
plot_y = sorted([i for i in regression.predict(X_train)])
ax1.plot(sorted(X_train[:, 1]), sorted(X_train[:, 2]), plot_y, c="r")
# Mostramos el gráfico
plt.show()
