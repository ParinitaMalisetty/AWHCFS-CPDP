import pandas as pd

from src.awhcfs_builder import AWHCFSBuilder
from src.models import Models
from src.evaluation import evaluate


class LOPOAWHCFSExperiment:

    def __init__(self):

        self.builder = AWHCFSBuilder()

        self.models = Models().get_models()

    def run(self, datasets):

        results = []

        weight_results = []

        total_projects = len(datasets)

        for index, test_project in enumerate(datasets.keys(), start=1):

            print("\n" + "=" * 80)
            print(f"LOPO AWHCFS {index}/{total_projects}")
            print(f"Test Project : {test_project}")

            ####################################################
            # Build Training Set
            ####################################################

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

            ####################################################
            # Adaptive Weighted HCFS
            ####################################################

            top_features, weights = self.builder.get_top_features(
                training,
                k=10
            )

            print("\nTop Features")
            print(top_features)

            print("\nAdaptive Weights")
            print(f"RF Weight     : {weights['RF Weight']:.4f}")
            print(f"MI Weight     : {weights['MI Weight']:.4f}")
            print(f"ANOVA Weight  : {weights['ANOVA Weight']:.4f}")

            print("\nVariances")
            print(f"RF Variance   : {weights['RF Variance']:.6f}")
            print(f"MI Variance   : {weights['MI Variance']:.6f}")
            print(f"ANOVA Variance: {weights['ANOVA Variance']:.6f}")

            ####################################################
            # Save Adaptive Weights
            ####################################################

            weight_results.append({

            "Test Project": test_project,

            "RF Variance": round(weights["RF Variance"],6),

            "MI Variance": round(weights["MI Variance"],6),

            "ANOVA Variance": round(weights["ANOVA Variance"],6),

            "RF Weight": round(weights["RF Weight"],4),

            "MI Weight": round(weights["MI Weight"],4),

            "ANOVA Weight": round(weights["ANOVA Weight"],4)

            })

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

            for model_name, model in self.models.items():

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

                    "Test Project": test_project,

                    "Method": "AWHCFS",

                    "Model": model_name,

                    "Feature Count": len(top_features),

                    "Train Size": len(train_df),

                    "Test Size": len(test_df)

                }

                row.update(metrics)

                results.append(row)

        ####################################################
        # Save Results
        ####################################################

        results_df = pd.DataFrame(results)

        results_df.to_csv(
            "results/lopo_awhcfs_results.csv",
            index=False
        )

        ####################################################
        # Save Adaptive Weights
        ####################################################

        weights_df = pd.DataFrame(weight_results)

        weights_df.to_csv(
            "results/adaptive_weights.csv",
            index=False
        )

        ####################################################
        # Finish
        ####################################################

        print("\n" + "=" * 80)
        print("AWHCFS Experiment Completed")
        print("=" * 80)

        print(f"Total Experiments : {len(results_df)}")

        print("\nResults saved to:")
        print("results/lopo_awhcfs_results.csv")

        print("\nAdaptive weights saved to:")
        print("results/adaptive_weights.csv")

        return results_df