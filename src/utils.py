import pandas as pd


def dataset_summary(datasets):

    summary = []

    for name, df in datasets.items():

        summary.append({

            "Dataset": name,

            "Rows": df.shape[0],

            "Columns": df.shape[1],

            "Missing Values": df.isnull().sum().sum()

        })

    summary = pd.DataFrame(summary)

    return summary