import os
import pandas as pd

from src.cpsfs_builder import CPSFSBuilder
from src.models import Models
from src.evaluation import evaluate


class LOPOExperiment:

    def __init__(self):

        self.builder = CPSFSBuilder()

        os.makedirs("results/lopo_rankings", exist_ok=True)

    def run(self, datasets):

        all_results = []

        total_projects = len(datasets)

        for index, test_dataset in enumerate(datasets.keys(), start=1):

            print("\n" + "=" * 80)
            print(f"LOPO {index}/{total_projects}")
            print(f"Test Project : {test_dataset}")

            ####################################################
            # Build Training Set
            ####################################################

            training = {}

            for name, df in datasets.items():

                if name != test_dataset:

                    training[name] = df

            train_df = pd.concat(
                training.values(),
                ignore_index=True
            )

            test_df = datasets[test_dataset]

            print(f"Training Size : {train_df.shape}")
            print(f"Testing Size  : {test_df.shape}")

            ####################################################
            # Build CPSFS Ranking
            ####################################################

            ranking = self.builder.build(training)

            ranking.to_csv(
                f"results/lopo_rankings/{test_dataset}_ranking.csv",
                index=False
            )

            top_features = ranking.head(10)["Feature"].tolist()

            print("\nTop Features")

            print(top_features)

            ####################################################
            # Prepare Data
            ####################################################

            X_train = train_df[top_features]

            y_train = train_df["bug"]

            X_test = test_df[top_features]

            y_test = test_df["bug"]

            ####################################################
            # Train Models
            ####################################################

            models = Models().get_models()

            for model_name, model in models.items():

                print(f"Training {model_name}")

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(X_test)

                metrics = evaluate(
                    y_test,
                    predictions
                )

                row = {

                    "Test Project": test_dataset,

                    "Method": "LOPO-CPSFS",

                    "Model": model_name,

                    "Feature Count": len(top_features),

                    "Selected Features": ", ".join(top_features),

                    "Train Size": len(train_df),

                    "Test Size": len(test_df)

                }

                row.update(metrics)

                all_results.append(row)

        ####################################################
        # Save Results
        ####################################################

        results_df = pd.DataFrame(all_results)

        results_df.to_csv(
            "results/lopo_cpsfs_results.csv",
            index=False
        )

        print("\n" + "=" * 80)
        print("LOPO Experiment Completed")
        print("=" * 80)

        print(f"Total Experiments : {len(results_df)}")

        print("\nResults saved to:")
        print("results/lopo_cpsfs_results.csv")

        print("\nFeature rankings saved to:")
        print("results/lopo_rankings/")

        return results_df