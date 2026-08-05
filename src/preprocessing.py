import pandas as pd


class Preprocessor:

    def __init__(self):
        self.remove_columns = [
            "name",
            "version",
            "name.1"
        ]

    def clean_dataset(self, df):

        df = df.copy()

        # Remove metadata columns
        df.drop(columns=self.remove_columns,
                inplace=True,
                errors="ignore")

        # Convert bug column into binary labels
        df["bug"] = (df["bug"] > 0).astype(int)

        return df