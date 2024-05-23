import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataCleanerAndTrainer:
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

    def load_data(self, x_columns, y_column):
        self.dataset = pd.read_csv(self.data_path)
        self.X = self.dataset[x_columns].values
        self.y = self.dataset[y_column].values

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def scale_data(self):
        sc_x = StandardScaler()
        self.X_train = sc_x.fit_transform(self.X_train)
        self.X_test = sc_x.transform(self.X_test)