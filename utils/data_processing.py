import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class DataProcessing:
    def __init__(self, data_path, test_size=1/3, random_state=0):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, x_columns, y_columns):
        self.dataset = pd.read_csv(self.data_path)
        self.X = self.dataset.iloc[:, x_columns].values  # Use iloc for integer-location based indexing
        if isinstance(y_columns, list):
            self.y = self.dataset.iloc[:, y_columns].values
        else:
            self.y = self.dataset.iloc[:, [y_columns]].values  # Ensure y is always a 2D array even for a single column

    def encode_categorical_features(self, categorical_features_indices):
        if categorical_features_indices:
            # Codificar variables categóricas y evitar la trampa de las variables ficticias
            ct = ColumnTransformer(
                [('one_hot_encoder', OneHotEncoder(), categorical_features_indices)],
                remainder='passthrough')
            self.X = ct.fit_transform(self.X)
            # Eliminamos la primera columna de cada grupo de dummies
            self.X = self.X[:, 1:]
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def scale_data(self):
        sc_x = StandardScaler()
        self.X_train = sc_x.fit_transform(self.X_train)
        self.X_test = sc_x.transform(self.X_test)
        
    def plot_model(self, regression, title, xlabel, ylabel, X_grid=None):
        plt.scatter(self.X, self.y, color = "red")
        if X_grid is not None:
            plt.plot(X_grid, regression.predict(X_grid), color = "blue")
        else:
            plt.plot(self.X, regression.predict(self.X), color = "blue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()