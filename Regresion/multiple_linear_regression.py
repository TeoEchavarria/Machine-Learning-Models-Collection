from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from utils.data_processing import DataProcessing

class MultipleLinearRegressionModel(DataProcessing):
    def __init__(self, data_path, x_columns, y_column, categorical_features_indices, test_size=0.2, random_state=0):
        super().__init__(data_path, test_size, random_state)
        self.load_data(x_columns, y_column)
        self.encode_categorical_features(categorical_features_indices)
        self.split_data()
        self.regression = LinearRegression()

    def train_model(self):
        self.regression.fit(self.X_train, self.y_train)

    def predict(self):
        return self.regression.predict(self.X_test)

    def backward_elimination_with_plot(self, significance_level=0.07):
        # Adding a column of ones for the intercept
        self.X_train = self.X_train.astype(float)  
        self.y_train = self.y_train.astype(float)

        X = np.append(arr=np.ones((self.X_train.shape[0], 1)).astype(float), values=self.X_train, axis=1)
        numVars = X.shape[1]
        selected_vars = list(range(numVars))
        iterations = 0

        while len(selected_vars) > 1:
            regressor_OLS = sm.OLS(endog=self.y_train, exog=X[:, selected_vars]).fit()
            max_p_value = max(regressor_OLS.pvalues)
            if max_p_value > significance_level:
                max_p_var_index = np.argmax(regressor_OLS.pvalues)
                if selected_vars[max_p_var_index] == 0:  # Avoid removing the intercept
                    break
                selected_vars.pop(max_p_var_index)  # Remove the variable with the highest p-value
            else:
                break
            iterations += 1

        return regressor_OLS.summary()