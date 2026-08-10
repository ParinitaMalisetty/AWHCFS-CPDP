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
        all_selected_features = []

        total_projects = len(datasets)

        for index, test_project in enumerate(
            datasets.keys(),
            start=1
        ):

            print("\n" + "=" * 80)
            print(
                f"LOPO AWHCFS {index}/{total_projects}"
            )
            print(
                f"Test Project : {test_project}"
            )

            training = {}

            for name, df in datasets.items():

                if name != test_project:
                    training[name] = df

            train_df = pd.concat(
                training.values(),
                ignore_index=True
            )

            test_df = datasets[test_project]

            print(
                f"Training Size : {train_df.shape}"
            )
            print(
                f"Testing Size  : {test_df.shape}"
            )

            top_features, weights = (
                self.builder.get_top_features(
                    training,
                    k=10
                )
            )

            all_selected_features.extend(
                top_features
            )

            print("\nAdaptive Weights")

            print(
                f"RF Weight     : "
                f"{weights['RF Weight']:.4f}"
            )

            print(
                f"MI Weight     : "
                f"{weights['MI Weight']:.4f}"
            )

            print(
                f"ANOVA Weight  : "
                f"{weights['ANOVA Weight']:.4f}"
            )

            print("\nTop Features")
            print(top_features)

            print("\nVariances")

            print(
                f"RF Variance   : "
                f"{weights['RF Variance']:.6f}"
            )

            print(
                f"MI Variance   : "
                f"{weights['MI Variance']:.6f}"
            )

            print(
                f"ANOVA Variance: "
                f"{weights['ANOVA Variance']:.6f}"
            )

            weight_results.append({

                "Test Project": test_project,

                "RF Variance": round(
                    weights["RF Variance"],
                    6
                ),

                "MI Variance": round(
                    weights["MI Variance"],
                    6
                ),

                "ANOVA Variance": round(
                    weights["ANOVA Variance"],
                    6
                ),

                "RF Weight": round(
                    weights["RF Weight"],
                    4
                ),

                "MI Weight": round(
                    weights["MI Weight"],
                    4
                ),

                "ANOVA Weight": round(
                    weights["ANOVA Weight"],
                    4
                )

            })

            X_train = train_df[
                top_features
            ]

            y_train = train_df["bug"]

            X_test = test_df[
                top_features
            ]

            y_test = test_df["bug"]

            for model_name, model in (
                self.models.items()
            ):

                print(
                    f"Training {model_name}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )

                metrics = evaluate(
                    y_test,
                    predictions
                )

                row = {

                    "Test Project": test_project,

                    "Method": "AWHCFS",

                    "Model": model_name,

                    "Feature Count": len(
                        top_features
                    ),

                    "Train Size": len(
                        train_df
                    ),

                    "Test Size": len(
                        test_df
                    )

                }

                row.update(metrics)

                results.append(row)

        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            "results/lopo_awhcfs_results.csv",
            index=False
        )

        weights_df = pd.DataFrame(
            weight_results
        )

        weights_df.to_csv(
            "results/adaptive_weights.csv",
            index=False
        )

        feature_frequency = pd.Series(
            all_selected_features
        ).value_counts()

        awhcfs_ranking = (
            feature_frequency
            .reset_index()
        )

        awhcfs_ranking.columns = [
            "Feature",
            "Selection Count"
        ]

        awhcfs_ranking[
            "Selection Percentage"
        ] = (
            awhcfs_ranking[
                "Selection Count"
            ]
            / total_projects
        ) * 100

        awhcfs_ranking[
            "AWHCFS Rank"
        ] = range(
            1,
            len(awhcfs_ranking) + 1
        )

        awhcfs_ranking = awhcfs_ranking[
            [
                "Feature",
                "Selection Count",
                "Selection Percentage",
                "AWHCFS Rank"
            ]
        ]

        awhcfs_ranking.to_csv(
            "results/awhcfs_feature_frequency.csv",
            index=False
        )

        print("\n" + "=" * 80)
        print(
            "AWHCFS Experiment Completed"
        )
        print("=" * 80)

        print(
            f"Total Experiments : "
            f"{len(results_df)}"
        )

        print("\nResults saved to:")
        print(
            "results/lopo_awhcfs_results.csv"
        )

        print("\nAdaptive weights saved to:")
        print(
            "results/adaptive_weights.csv"
        )

        print(
            "\nAWHCFS feature frequency saved to:"
        )
        print(
            "results/awhcfs_feature_frequency.csv"
        )

        print(
            "\nOverall AWHCFS Feature Ranking:"
        )

        print(
            awhcfs_ranking.head(15).to_string(
                index=False
            )
        )

        return results_df