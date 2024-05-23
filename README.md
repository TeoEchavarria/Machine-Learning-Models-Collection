# Modelos de Regresión

Esta carpeta contiene una serie de modelos de regresión implementados como objetos, diseñados para facilitar tanto el entrenamiento como la evaluación de modelos en machine learning para predecir valores continuos.

## Diseño Orientado a Objetos

Cada modelo de regresión está encapsulado en una clase que maneja todos los aspectos necesarios para realizar predicciones efectivas. Estas clases están diseñadas para ser intuitivas y fáciles de usar, permitiendo a los usuarios enfocarse en la experimentación y evaluación de diferentes enfoques sin preocuparse por los detalles de bajo nivel del procesamiento de datos o la optimización de modelos.

## Características de las Clases

- **Limpieza de Datos**: Métodos integrados para la preparación y limpieza de datos que aceptan parámetros configurables según las necesidades del proyecto.
- **Búsqueda de Hiperparámetros**: Funcionalidad para automatizar la búsqueda de los mejores parámetros para cada modelo, garantizando una optimización efectiva.
- **Evaluación de Modelos**: Métodos para calcular métricas de error utilizando conjuntos de datos de prueba, permitiendo una evaluación rápida y efectiva del rendimiento del modelo.
- **Pruebas con Nuevos Datos**: Capacidad para introducir nuevos conjuntos de datos de prueba para evaluar la generalización del modelo.
- **Predicción Directa**: Funciones para realizar predicciones con nuevos datos de entrada, simplificando el proceso de despliegue de modelos en producción.

## Modelos Incluidos

- **Regresión**: Modelos destinados a predecir valores continuos.
- **Clasificación**: Modelos para clasificar entradas en categorías predefinidas.
- **Agrupación**: Técnicas para agrupar un conjunto de objetos de forma que los objetos de un mismo grupo sean más similares entre sí que los de otros grupos.
- **Reglas de asociación**: Modelos para descubrir relaciones interesantes entre variables en grandes bases de datos.
- **Aprendizaje por refuerzo**: Modelos que aprenden su comportamiento para maximizar una noción de recompensa acumulativa.
- **Procesamiento del lenguaje natural**: Modelos centrados en el procesamiento y análisis del lenguaje humano.
- **Aprendizaje profundo**: Modelos que utilizan redes neuronales profundas para aprender de grandes cantidades de datos.
### Uso

Cada modelo incluido tiene un script asociado que demuestra cómo instanciar y utilizar la clase del modelo. Se proporcionan ejemplos de cómo configurar los parámetros, limpiar los datos, realizar la búsqueda de hiperparámetros, entrenar el modelo, y finalmente, cómo evaluarlo y hacer predicciones. Se recomienda revisar estos scripts para entender el flujo completo de trabajo con cada modelo.

```python
from models import SimpleLinearRegression

# Model instantiation
model = SimpleLinearRegression()

# Data cleaning and preparation
clean_data = model.prepare_data(raw_data)

# Hyperparameter tuning
model.tune_parameters(clean_data)

# Training
model.train(clean_data)

# Evaluation
error = model.evaluate(test_data)

# Prediction
predictions = model.predict(new_data)
```