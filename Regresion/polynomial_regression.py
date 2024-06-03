import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from utils.data_processing import DataProcessing  # Assuming the class is in a module named data_processing

class PolynomialRegression:
    def __init__(self, data_path, degree=2):
        self.data_processor = DataProcessing(data_path, test_size=0.2, random_state=0)
        self.degree = degree
        self.poly_model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=self.degree)

    def prepare_data(self):
        self.data_processor.load_data(x_columns=slice(1,2), y_columns=2)
        self.data_processor.split_data()

    def train_model(self):
        X_poly_train = self.poly_features.fit_transform(self.data_processor.X_train)
        self.poly_model.fit(X_poly_train, self.data_processor.y_train)

    def plot_results(self, title="Training Results", xlabel="Input Variable", ylabel="Output Variable"):
        X_grid = np.arange(min(self.data_processor.X), max(self.data_processor.X), 0.1)
        X_grid = X_grid.reshape(-1, 1)
        X_grid_poly = self.poly_features.fit_transform(X_grid)
        self.data_processor.plot_model2(self.poly_model, title, xlabel, ylabel, X_grid_poly, X_grid)

    def predict(self, value):
        value_poly = self.poly_features.fit_transform(np.array([[value]]))
        return self.poly_model.predict(value_poly)