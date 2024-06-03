# Regresion Lineal
from Regresion.simple_linear_regression import SimpleLinearRegressionModel
from Regresion.multiple_linear_regression import MultipleLinearRegressionModel
from Regresion.polynomial_regression import PolynomialRegression

if __name__ == "__main__":
    # Linear Regression
    # linear_reg = SimpleLinearRegressionModel('DataSets/Salary_Data.csv', x_columns=[0], y_column=1)
    # linear_reg.train_model()
    # linear_reg.plot_training_results("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)", "Años de Experiencia", "Sueldo (en USD$)")
    # linear_reg.plot_testing_results("Sueldo vs Años de Experiencia (Conjunto de testing)", "Años de Experiencia", "Sueldo (en USD$)")
    
    # Linear Multiple Regression
    # linear_multiple_reg = MultipleLinearRegressionModel('DataSets/50_Startups.csv', x_columns=[0,1,2,3], y_column=4, categorical_features_indices=[3])
    # linear_multiple_reg.train_model()
    # linear_multiple_reg = model.backward_elimination_with_plot()
    
    # Polynomial Regression
    poly_reg = PolynomialRegression('DataSets/Position_Salaries.csv', degree=4)
    poly_reg.prepare_data()
    poly_reg.train_model()
    poly_reg.plot_results()

    print(poly_reg.predict(6.5))