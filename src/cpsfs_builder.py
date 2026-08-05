import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class CPSFSBuilder:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    def build(self, training_datasets):

        feature_scores = {}

        for dataset_name, df in training_datasets.items():

            X = df.drop("bug", axis=1)
            y = df["bug"]

            self.model.fit(X, y)

            importance = self.model.feature_importances_

            importance = importance / importance.sum()

            for feature, score in zip(X.columns, importance):

                feature_scores[feature] = (
                    feature_scores.get(feature, 0)
                    + score
                )

        ranking = pd.DataFrame({

            "Feature": feature_scores.keys(),

            "Consensus Score": feature_scores.values()

        })

        ranking = ranking.sort_values(
            by="Consensus Score",
            ascending=False
        ).reset_index(drop=True)

        ranking["Rank"] = ranking.index + 1

        return ranking

    def get_top_features(self, training_datasets, k=10):

        ranking = self.build(training_datasets)

        return ranking.head(k)["Feature"].tolist()