import pandas as pd

baseline = pd.read_csv("results/lopo_baseline_results.csv")
cpsfs = pd.read_csv("results/lopo_cpsfs_results.csv")

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
    "MCC"
]

summary = []

models = baseline["Model"].unique()

for model in models:

    base = baseline[baseline["Model"] == model]
    cps = cpsfs[cpsfs["Model"] == model]

    row = {"Model": model}

    for metric in metrics:

        base_mean = base[metric].mean()
        cps_mean = cps[metric].mean()

        row[f"Baseline {metric}"] = round(base_mean, 4)
        row[f"CPSFS {metric}"] = round(cps_mean, 4)
        row[f"Improvement {metric}"] = round(cps_mean - base_mean, 4)

    summary.append(row)

summary = pd.DataFrame(summary)

summary.to_csv(
    "results/model_comparison.csv",
    index=False
)

print(summary)