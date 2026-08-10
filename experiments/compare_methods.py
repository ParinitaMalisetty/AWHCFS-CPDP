import os
import pandas as pd
import numpy as np

from scipy.stats import friedmanchisquare


RESULT_FILES = {
    "Baseline": "results/lopo_baseline_results.csv",
    "HCFS": "results/lopo_hcfs_results.csv",
    "WHCFS": "results/lopo_whcfs_results.csv",
    "AWHCFS": "results/lopo_awhcfs_results.csv",
}


METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "MCC",
]


def load_results():

    all_results = []

    for method, path in RESULT_FILES.items():

        if not os.path.exists(path):

            print(
                f"WARNING: {path} not found."
            )

            continue

        df = pd.read_csv(path)

        df["Method"] = method

        all_results.append(df)

        print(
            f"Loaded {method}: "
            f"{len(df)} rows"
        )

    if not all_results:

        raise FileNotFoundError(
            "No experiment result files were found."
        )

    return pd.concat(
        all_results,
        ignore_index=True
    )


def find_available_metrics(df):

    return [
        metric
        for metric in METRICS
        if metric in df.columns
    ]


def create_summary(df, metrics):

    rows = []

    for method in [
        "Baseline",
        "HCFS",
        "WHCFS",
        "AWHCFS"
    ]:

        method_df = df[
            df["Method"] == method
        ]

        if method_df.empty:
            continue

        for metric in metrics:

            rows.append({

                "Method": method,

                "Metric": metric,

                "Mean": method_df[
                    metric
                ].mean(),

                "Std": method_df[
                    metric
                ].std(),

                "Min": method_df[
                    metric
                ].min(),

                "Max": method_df[
                    metric
                ].max(),

            })

    return pd.DataFrame(rows)


def create_pivot(summary):

    pivot = summary.pivot(
        index="Method",
        columns="Metric",
        values="Mean"
    )

    return pivot


def create_mean_std_table(summary):

    rows = []

    for method in [
        "Baseline",
        "HCFS",
        "WHCFS",
        "AWHCFS"
    ]:

        method_df = summary[
            summary["Method"] == method
        ]

        if method_df.empty:
            continue

        row = {
            "Method": method
        }

        for _, record in method_df.iterrows():

            metric = record["Metric"]

            row[
                metric
            ] = (
                f"{record['Mean']:.4f} "
                f"± {record['Std']:.4f}"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_improvement(df, metrics):

    methods = [
        "Baseline",
        "HCFS",
        "WHCFS",
        "AWHCFS"
    ]

    baseline = df[
        df["Method"] == "Baseline"
    ]

    rows = []

    for method in methods[1:]:

        current = df[
            df["Method"] == method
        ]

        row = {
            "Method": method
        }

        for metric in metrics:

            baseline_mean = baseline[
                metric
            ].mean()

            current_mean = current[
                metric
            ].mean()

            absolute_change = (
                current_mean
                - baseline_mean
            )

            if baseline_mean != 0:

                percentage_change = (
                    absolute_change
                    / baseline_mean
                ) * 100

            else:

                percentage_change = np.nan

            row[
                f"{metric} Change"
            ] = absolute_change

            row[
                f"{metric} Improvement %"
            ] = percentage_change

        rows.append(row)

    return pd.DataFrame(rows)


def create_project_level_table(
    df,
    metric="Balanced Accuracy"
):

    pivot = df.pivot_table(
        index=[
            "Test Project"
        ],
        columns="Method",
        values=metric,
        aggfunc="mean"
    )

    return pivot.reset_index()


def create_model_level_table(
    df,
    metric="Balanced Accuracy"
):

    pivot = df.pivot_table(
        index="Model",
        columns="Method",
        values=metric,
        aggfunc="mean"
    )

    return pivot.reset_index()


def calculate_method_wins(
    df,
    metric="Balanced Accuracy"
):

    project_scores = df.pivot_table(
        index=[
            "Test Project",
            "Model"
        ],
        columns="Method",
        values=metric,
        aggfunc="mean"
    )

    methods = [
        method
        for method in [
            "Baseline",
            "HCFS",
            "WHCFS",
            "AWHCFS"
        ]
        if method in project_scores.columns
    ]

    winner_counts = {
        method: 0
        for method in methods
    }

    for _, row in project_scores.iterrows():

        values = row[
            methods
        ].dropna()

        if values.empty:
            continue

        best_value = values.max()

        winners = values[
            values == best_value
        ].index

        for winner in winners:

            winner_counts[
                winner
            ] += 1

    return pd.DataFrame(
        [
            {
                "Method": method,
                "Wins": winner_counts[method]
            }
            for method in methods
        ]
    )


def friedman_test(
    df,
    metric="Balanced Accuracy"
):

    pivot = df.pivot_table(
        index=[
            "Test Project",
            "Model"
        ],
        columns="Method",
        values=metric,
        aggfunc="mean"
    )

    required_methods = [
        "Baseline",
        "HCFS",
        "WHCFS",
        "AWHCFS"
    ]

    if not all(
        method in pivot.columns
        for method in required_methods
    ):

        return None

    complete = pivot[
        required_methods
    ].dropna()

    if len(complete) < 2:

        return None

    statistic, p_value = (
        friedmanchisquare(
            complete["Baseline"],
            complete["HCFS"],
            complete["WHCFS"],
            complete["AWHCFS"]
        )
    )

    return {

        "Metric": metric,

        "Samples": len(complete),

        "Friedman Statistic": statistic,

        "p-value": p_value

    }


def main():

    os.makedirs(
        "results/comparison",
        exist_ok=True
    )

    print("\n" + "=" * 80)
    print(
        "BASELINE vs HCFS vs WHCFS vs AWHCFS"
    )
    print("=" * 80)

    df = load_results()

    metrics = find_available_metrics(df)

    print("\nAvailable Metrics:")

    print(metrics)

    # --------------------------------------------------
    # 1. Overall summary
    # --------------------------------------------------

    summary = create_summary(
        df,
        metrics
    )

    summary.to_csv(
        "results/comparison/"
        "method_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # 2. Mean ± Standard Deviation
    # --------------------------------------------------

    mean_std = create_mean_std_table(
        summary
    )

    mean_std.to_csv(
        "results/comparison/"
        "method_mean_std.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    print(
        mean_std.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 3. Mean performance pivot
    # --------------------------------------------------

    pivot = create_pivot(
        summary
    )

    pivot.to_csv(
        "results/comparison/"
        "method_mean_comparison.csv"
    )

    # --------------------------------------------------
    # 4. Improvement over baseline
    # --------------------------------------------------

    improvement = calculate_improvement(
        df,
        metrics
    )

    improvement.to_csv(
        "results/comparison/"
        "improvement_over_baseline.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 80)

    print(
        improvement.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 5. Project-level comparison
    # --------------------------------------------------

    project_table = create_project_level_table(
        df,
        "Balanced Accuracy"
    )

    project_table.to_csv(
        "results/comparison/"
        "project_balanced_accuracy.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. Model-level comparison
    # --------------------------------------------------

    model_table = create_model_level_table(
        df,
        "Balanced Accuracy"
    )

    model_table.to_csv(
        "results/comparison/"
        "model_balanced_accuracy.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("MODEL-LEVEL BALANCED ACCURACY")
    print("=" * 80)

    print(
        model_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 7. Method wins
    # --------------------------------------------------

    wins = calculate_method_wins(
        df,
        "Balanced Accuracy"
    )

    wins.to_csv(
        "results/comparison/"
        "method_wins.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("METHOD WINS")
    print("=" * 80)

    print(
        wins.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 8. Friedman statistical test
    # --------------------------------------------------

    statistical_results = []

    for metric in [
        "Balanced Accuracy",
        "MCC",
        "F1"
    ]:

        if metric not in metrics:
            continue

        result = friedman_test(
            df,
            metric
        )

        if result is not None:

            statistical_results.append(
                result
            )

    statistics_df = pd.DataFrame(
        statistical_results
    )

    statistics_df.to_csv(
        "results/comparison/"
        "friedman_tests.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("FRIEDMAN TEST")
    print("=" * 80)

    if not statistics_df.empty:

        print(
            statistics_df.to_string(
                index=False
            )
        )

    else:

        print(
            "Not enough complete samples "
            "for statistical testing."
        )

    # --------------------------------------------------
    # 9. Best method for each metric
    # --------------------------------------------------

    best_rows = []

    for metric in metrics:

        metric_means = (
            summary[
                summary["Metric"] == metric
            ]
            .set_index("Method")["Mean"]
        )

        if metric_means.empty:
            continue

        best_method = (
            metric_means.idxmax()
        )

        best_value = (
            metric_means.max()
        )

        best_rows.append({

            "Metric": metric,

            "Best Method": best_method,

            "Mean Score": best_value

        })

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        "results/comparison/"
        "best_method_by_metric.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("BEST METHOD BY METRIC")
    print("=" * 80)

    print(
        best_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED")
    print("=" * 80)

    print("\nResults saved under:")

    print(
        "results/comparison/"
    )

    print("\nGenerated files:")

    print(
        "  method_summary.csv"
    )

    print(
        "  method_mean_std.csv"
    )

    print(
        "  method_mean_comparison.csv"
    )

    print(
        "  improvement_over_baseline.csv"
    )

    print(
        "  project_balanced_accuracy.csv"
    )

    print(
        "  model_balanced_accuracy.csv"
    )

    print(
        "  method_wins.csv"
    )

    print(
        "  friedman_tests.csv"
    )

    print(
        "  best_method_by_metric.csv"
    )


if __name__ == "__main__":
    main()