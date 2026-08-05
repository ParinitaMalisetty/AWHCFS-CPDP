import pandas as pd

from whcfs_builder_fixed import WHCFSBuilder
from src.models import Models
from src.evaluation import evaluate


class LOPOWHCFSExperiment:

    def __init__(self):

        self.whcfs = WHCFSBuilder()
        self.models = Models().get_models()

    def run(self, datasets):

        results = []

        total_projects = len(datasets)

        for i, test_project in enumerate(datasets.keys(), start=1):

            print("\n" + "=" * 80)
            print(f"LOPO WHCFS {i}/{total_projects}")
            print(f"Test Project : {test_project}")

            # ----------------------------
            # Build Training Set
            # ----------------------------

            training = {}

            for name, df in datasets.items():

                if name != test_project:
                    training[name] = df

            train_df = pd.concat(
                training.values(),
                ignore_index=True
            )

            test_df = datasets[test_project]

            print(f"Training Size : {train_df.shape}")
            print(f"Testing Size  : {test_df.shape}")

            # ----------------------------
            # HCFS Feature Selection
            # ----------------------------

            top_features = self.whcfs.get_top_features(
                training,
                k=10
            )

            print("\nTop Features")
            print(top_features)

            X_train = train_df[top_features]
            y_train = train_df["bug"]

            X_test = test_df[top_features]
            y_test = test_df["bug"]

            # ----------------------------
            # Train Models
            # ----------------------------

            for model_name, model in self.models.items():

                print(f"Training {model_name}")

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                metrics = evaluate(
                    y_test,
                    predictions
                )

                row = {
                    "Test Project": test_project,
                    "Method": "WHCFS",
                    "Model": model_name,
                    "Features": len(top_features)
                }

                row.update(metrics)

                results.append(row)

        results = pd.DataFrame(results)

        results.to_csv(
            "results/lopo_whcfs_results.csv",
            index=False
        )

        print("\n" + "=" * 80)
        print("WHCFS Experiment Completed")
        print("=" * 80)

        print(f"Total Experiments : {len(results)}")
        print("\nResults saved to:")
        print("results/lopo_whcfs_results.csv")

        return results