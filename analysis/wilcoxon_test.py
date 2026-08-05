import pandas as pd

from scipy.stats import wilcoxon


# Load Results
baseline = pd.read_csv("results/lopo_baseline_results.csv")
whcfs = pd.read_csv("results/lopo_whcfs_results.csv")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "MCC"
]

results = []

models = baseline["Model"].unique()

for model in models:

    base_model = baseline[
        baseline["Model"] == model
    ]

    whcfs_model = whcfs[
        whcfs["Model"] == model
    ]

    print(f"\n{model}")

    for metric in metrics:

        stat, p = wilcoxon(
            base_model[metric],
            whcfs_model[metric]
        )

        results.append({
            "Model": model,
            "Metric": metric,
            "Statistic": round(stat, 4),
            "P-Value": round(p, 6),
            "Significant": "Yes" if p < 0.05 else "No"
        })

results = pd.DataFrame(results)

print("\n")
print(results)

results.to_csv(
    "results/wilcoxon_results.csv",
    index=False
)

print("\nWilcoxon test completed.")
print("Results saved to results/wilcoxon_results.csv")