import pandas as pd

from scipy.stats import wilcoxon


def rank_biserial(statistic, n):
    """
    Rank-Biserial Correlation
    """

    total_rank = n * (n + 1) / 2

    return 1 - (2 * statistic) / total_rank


baseline = pd.read_csv(
    "results/lopo_baseline_results.csv"
)

whcfs = pd.read_csv(
    "results/lopo_whcfs_results.csv"
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "MCC"
]

rows = []

for model in baseline["Model"].unique():

    base = baseline[
        baseline["Model"] == model
    ]

    ours = whcfs[
        whcfs["Model"] == model
    ]

    print(f"\n{model}")

    for metric in metrics:

        statistic, p = wilcoxon(
            base[metric],
            ours[metric]
        )

        r = rank_biserial(
            statistic,
            len(base)
        )

        value = abs(r)

        if value < 0.10:
            effect = "Negligible"

        elif value < 0.30:
            effect = "Small"

        elif value < 0.50:
            effect = "Medium"

        else:
            effect = "Large"

        rows.append({

            "Model": model,

            "Metric": metric,

            "Statistic": statistic,

            "P-Value": round(p, 6),

            "Rank-Biserial": round(r, 4),

            "Effect Size": effect

        })

results = pd.DataFrame(rows)

print("\n")
print(results)

results.to_csv(
    "results/rank_biserial_results.csv",
    index=False
)

print("\nResults saved to results/rank_biserial_results.csv")