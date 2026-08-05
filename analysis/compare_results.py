import pandas as pd

# -------------------------------
# Load Results
# -------------------------------

baseline = pd.read_csv(
    "results/lopo_baseline_results.csv"
)

whcfs = pd.read_csv(
    "results/lopo_whcfs_results.csv"
)

wilcoxon = pd.read_csv(
    "results/wilcoxon_results.csv"
)

effect = pd.read_csv(
    "results/rank_biserial_results.csv"
)

# -------------------------------
# Metrics to Compare
# -------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "MCC"
]

rows = []

# -------------------------------
# Build Comparison
# -------------------------------

for model in baseline["Model"].unique():

    base_model = baseline[
        baseline["Model"] == model
    ]

    whcfs_model = whcfs[
        whcfs["Model"] == model
    ]

    for metric in metrics:

        baseline_mean = base_model[metric].mean()

        whcfs_mean = whcfs_model[metric].mean()

        improvement = whcfs_mean - baseline_mean

        p = wilcoxon[
            (wilcoxon["Model"] == model) &
            (wilcoxon["Metric"] == metric)
        ]["P-Value"].values[0]

        effect_size = effect[
            (effect["Model"] == model) &
            (effect["Metric"] == metric)
        ]["Effect Size"].values[0]

        rows.append({

            "Model": model,

            "Metric": metric,

            "Baseline Mean": round(baseline_mean,4),

            "WHCFS Mean": round(whcfs_mean,4),

            "Improvement": round(improvement,4),

            "P-Value": p,

            "Effect Size": effect_size

        })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    "results/final_comparison_table.csv",
    index=False
)

print(comparison)

print("\nFinal comparison table saved.")