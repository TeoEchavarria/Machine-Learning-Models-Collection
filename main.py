# Regresion Lineal
from Regresion.simple_linear_regression import SimpleLinearRegressionModel

if __name__ == "__main__":
    # Linear Regression
    model = SimpleLinearRegressionModel('DataSets/Salary_Data.csv', x_columns=[0], y_column=1)
    model.train_model()
    model.plot_training_results("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)", "Años de Experiencia", "Sueldo (en USD$)")
    model.plot_testing_results("Sueldo vs Años de Experiencia (Conjunto de testing)", "Años de Experiencia", "Sueldo (en USD$)")