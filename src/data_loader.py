import os
import pandas as pd


class DataLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_datasets(self):

        datasets = {}

        for file in os.listdir(self.dataset_path):

            if file.endswith(".csv"):

                file_path = os.path.join(self.dataset_path, file)

                dataset_name = file.replace(".csv", "")

                df = pd.read_csv(file_path)

                datasets[dataset_name] = df

        return datasets