# Regresion Lineal
from Regresion.simple_linear_regression import SimpleLinearRegressionModel
from Regresion.multiple_linear_regression import MultipleLinearRegressionModel

if __name__ == "__main__":
    # Linear Regression
    # model = SimpleLinearRegressionModel('DataSets/Salary_Data.csv', x_columns=[0], y_column=1)
    # model.train_model()
    # model.plot_training_results("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)", "Años de Experiencia", "Sueldo (en USD$)")
    # model.plot_testing_results("Sueldo vs Años de Experiencia (Conjunto de testing)", "Años de Experiencia", "Sueldo (en USD$)")
    
    # Linear Multiple Regression
    model = MultipleLinearRegressionModel('DataSets/50_Startups.csv', x_columns=[0,1,2,3], y_column=4, categorical_features_indices=[3])
    model.train_model()
    model_summary = model.backward_elimination_with_plot()