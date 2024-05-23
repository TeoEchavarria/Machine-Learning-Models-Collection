from sklearn.linear_model import LinearRegression
from utils.data_processing import DataProcessing

class SimpleLinearRegressionModel(DataProcessing):
    def __init__(self, data_path, x_columns, y_column, test_size=1/3, random_state=0):
        super().__init__(data_path, test_size, random_state)
        self.load_data(x_columns, y_column)
        self.split_data()
        self.regression = LinearRegression()
    
    def train_model(self):
        self.regression.fit(self.X_train, self.y_train)
        
    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.regression.predict(X)

    def plot_training_results(self, title="Training Results", xlabel="Input Variable", ylabel="Output Variable"):
        self.plot_model(self.regression, title, xlabel, ylabel, X_train=self.X_train)

    def plot_testing_results(self, title="Testing Results", xlabel="Input Variable", ylabel="Output Variable"):
        self.plot_model(self.regression, title, xlabel, ylabel, X_train=self.X_test)