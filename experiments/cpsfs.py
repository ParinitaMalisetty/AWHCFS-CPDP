import os
import pandas as pd


class CPSFS:

    def __init__(self, folder="results/feature_importance"):

        self.folder = folder

    def build_consensus(self):

        feature_scores = {}

        csv_files = [
            f for f in os.listdir(self.folder)
            if f.endswith(".csv")
        ]

        print(f"\nProcessing {len(csv_files)} feature importance files...")

        for file in csv_files:

            df = pd.read_csv(
                os.path.join(self.folder, file)
            )

            # Normalize importance values (0 to 1)
            total = df["Importance"].sum()

            df["Normalized Importance"] = (
                df["Importance"] / total
            )

            for _, row in df.iterrows():

                feature = row["Feature"]

                score = row["Normalized Importance"]

                feature_scores[feature] = (
                    feature_scores.get(feature, 0) + score
                )

        consensus = pd.DataFrame({

            "Feature": list(feature_scores.keys()),

            "Consensus Score": list(feature_scores.values())

        })

        consensus = consensus.sort_values(
            by="Consensus Score",
            ascending=False
        ).reset_index(drop=True)

        consensus["Rank"] = consensus.index + 1

        consensus.to_csv(
            "results/cpsfs_feature_ranking.csv",
            index=False
        )

        return consensus

    def get_top_features(self, k=10):

        ranking = self.build_consensus()

        return ranking.head(k)["Feature"].tolist()