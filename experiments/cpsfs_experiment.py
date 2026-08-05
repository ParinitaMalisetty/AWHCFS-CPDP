import pandas as pd

from experiments.cross_project import CrossProjectExperiment
from experiments.cpsfs import CPSFS


class CPSFSExperiment:

    def __init__(self):

        self.cross = CrossProjectExperiment()
        self.cpsfs = CPSFS()

    def run(self, datasets):

        feature_sizes = [5, 10, 15, 20]

        all_results = []

        for k in feature_sizes:

            print(f"\nRunning CPSFS Top-{k}")

            features = self.cpsfs.get_top_features(k)

            results = self.cross.run(
                datasets["ant-1.7"],
                datasets["camel-1.0"],
                features=features
            )

            for model, metrics in results.items():

                row = {

                    "Method": "CPSFS",

                    "Top Features": k,

                    "Train": "ant-1.7",

                    "Test": "camel-1.0",

                    "Model": model

                }

                row.update(metrics)

                all_results.append(row)

        df = pd.DataFrame(all_results)

        df.to_csv(
            "results/cpsfs_comparison.csv",
            index=False
        )

        return df