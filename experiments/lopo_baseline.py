import pandas as pd

from src.models import Models
from src.evaluation import evaluate


class LOPOBaselineExperiment:

    def __init__(self):
        pass

    def run(self, datasets):

        all_results = []

        total_projects = len(datasets)

        for index, test_dataset in enumerate(datasets.keys(), start=1):

            print("\n" + "=" * 80)
            print(f"LOPO BASELINE {index}/{total_projects}")
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

            ####################################################
            # Use ALL FEATURES
            ####################################################

            features = [
                col for col in train_df.columns
                if col != "bug"
            ]

            X_train = train_df[features]
            y_train = train_df["bug"]

            X_test = test_df[features]
            y_test = test_df["bug"]

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

                    "Method": "LOPO-Baseline",

                    "Model": model_name,

                    "Feature Count": len(features),

                    "Train Size": len(train_df),

                    "Test Size": len(test_df)

                }

                row.update(metrics)

                all_results.append(row)

        results_df = pd.DataFrame(all_results)

        results_df.to_csv(
            "results/lopo_baseline_results.csv",
            index=False
        )

        print("\nBaseline experiment completed.")

        return results_df