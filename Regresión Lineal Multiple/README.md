# Regresión Lineal Multiple

*Dataset* = Archivo de Excel que contiene cantidad de datos sobre 50 Startups con relación a lo invertido en: **R&D Spend, Administration, Marketing Spend y State**. Esto con relación al **Beneficio** obtenido.

![Figure 2023-01-14 170425](https://user-images.githubusercontent.com/63327224/212572293-ccf5be63-9ff6-4c00-ba52-b51589c37770.png)
![Figure 2023-01-14 172348](https://user-images.githubusercontent.com/63327224/212572297-918608f7-c669-48c6-8909-9529ef250c5d.png)

Estos son los resultados obtenidos realizando una ***Regresión Lineal Multiple*** tomando una parte del Dataset como el conjunto de entrenamiento y otra para el testeo correspondiente. 

Observemos primeramente como es que algunos datos como lo son **Administration y State** están ilustrados en la gráfica, ni en el modelo mismo dentro del archivo de Python. La razón de esto es que el $P$ valor de estas variables es muy alto, lo cual significa que no es significativo para realizar una correcta predicción con el modelo, alterándolo y empeorándolo.

Viendo la mejora que se le aplica al modelo al eliminar estas dos variables en los porcentajes de error obtenidos en los dos casos. Siendo estos $6.45$% tomando todas las variables y $5.86$% usando solo **R&D Spend, Marketing Spend y Beneficio**

La correlación que se halla entre la inversión en **R&D Spend, Marketing Spend** y el **Beneficio** obtenido, es positiva. Dado que como se ve en los resultados, las Startups que más ingresaron dinero en estos dos campos fueron las que más beneficios generaron. He de aclarar de igual forma que aunque el $P$ valor obtenido de la variable del Marketing Spend fue un poco mayor al $P$ valor esperado $(0.5)$, deje este dentro del modelo para conservar el nombre de *Regresión Lineal Múltiple*.

Concluyendo así que teniendo en cuenta los ámbitos evaluados en este modelo, la variable *R&D Spend* es la más significativa y con más peso en este modelo.
