import pandas as pd


class SourceData:
    """
    Encapsulation class to hold paths for the three CSV files.
    """
    def __init__(self, train_csv, test_csv, validation_csv):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.validation_csv = validation_csv

    def get_train_data(self):
        return pd.read_csv(self.train_csv)

    def get_test_data(self):
        return pd.read_csv(self.test_csv)

    def get_validation_data(self):
        return pd.read_csv(self.validation_csv)