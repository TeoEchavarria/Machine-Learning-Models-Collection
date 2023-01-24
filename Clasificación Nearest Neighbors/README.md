# Algoritmo de Clasificación K - Nearest Neighbors

*Dataset* = Archivo de Excel que contiene cantidad de datos clientes *(ID, Género, Edad, Estimación Salarial)* y una columna que dice si compro o no el producto. 
 
![Entreno KNN](https://user-images.githubusercontent.com/63327224/214205484-bfd3c9e5-9d5c-4fbe-8c64-d8ba54fb18c0.png)
![Test KNN](https://user-images.githubusercontent.com/63327224/214205485-c2037018-b5eb-47dd-85c8-c282e33700b8.png)

Dentro de los gráficos observamos como hay una clara división entre los clientes que compraron y los que no, y el hecho de como el algoritmo de K - Nearest Neighbors logro marcar de forma muy precisa dichas áreas para predecir el conjunto de testeo.

|           | Positivo | Negativo |
|-----------|----------|----------|
| Positivo  | 64       | 4        |
| Negativo  | 3        | 29       |

Vemos de igual forma como de los 100 clientes analizados en el grupo de testeo, al hacer una matriz de confusión para evaluar los falsos positivos y los falsos negativos. Nos arroja solo 7 de estos.

Continuando con el análisis, observemos que podemos tomar a nuestro público objetivo, o en este caso al que más se vio involucrado en la compra de nuestro producto, a aquellos clientes con un salario mayor a los 30.000 y menor a los 90.000; con una edad aproximada de 27 y 44, tal como se ve en las gráficas.

![image](https://user-images.githubusercontent.com/63327224/214212531-b35c9fdc-de38-49ea-8c66-72d1689427fb.png)
![image](https://user-images.githubusercontent.com/63327224/214212518-fdc60fcb-c78c-401e-9ba8-05e88d51b0e8.png)
