# Algoritmo de Clasificación Logística

*Dataset* = Archivo de Excel que contiene cantidad de datos clientes *(ID, Género, Edad, Estimación Salarial)* y una columna que dice si compro o no el producto. 
 
![Entreno - Logistica](https://user-images.githubusercontent.com/63327224/214622690-7bc58d63-2405-44e9-83db-4e8ab6c14254.png)
![Test Logistica](https://user-images.githubusercontent.com/63327224/214622682-f387e078-a701-4e5c-a8db-8772acb2f262.png)

Dentro de los gráficos observamos como hay una clara división entre los clientes que compraron y los que no, y el hecho de como el algoritmo de Regrasión Logística logro marcar de forma muy precisa dichas áreas para predecir el conjunto de testeo.

|           | Positivo | Negativo |
|-----------|----------|----------|
| Positivo  | 65       | 3        |
| Negativo  | 8        | 24       |

Vemos de igual forma como de los 100 clientes analizados en el grupo de testeo, al hacer una matriz de confusión para evaluar los falsos positivos y los falsos negativos. Nos arroja solo 11 de estos.
Siendo si un alto porcentaje de error 11% más observemos que aquellos puntos que se salen de nuestra predicción son en su mayoría casos aislados, de igual forma si se desea tener una precisión adicional podemos usar el otro algoritmo de K - Nearest Neighbors.
