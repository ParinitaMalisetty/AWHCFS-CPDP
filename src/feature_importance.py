import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class FeatureImportanceExtractor:

    def __init__(self):

        self.model = RandomForestClassifier(
            random_state=42,
            n_estimators=200
        )

    def extract(self, dataset_name, df):

        X = df.drop("bug", axis=1)
        y = df["bug"]

        self.model.fit(X, y)

        importance = pd.DataFrame({

            "Feature": X.columns,
            "Importance": self.model.feature_importances_

        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        os.makedirs(
            "results/feature_importance",
            exist_ok=True
        )

        importance.to_csv(
            f"results/feature_importance/{dataset_name}.csv",
            index=False
        )

        return importance