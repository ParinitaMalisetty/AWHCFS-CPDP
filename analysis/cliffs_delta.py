import pandas as pd


def cliffs_delta(x, y):

    greater = 0
    smaller = 0

    for xi in x:
        for yi in y:

            if xi > yi:
                greater += 1

            elif xi < yi:
                smaller += 1

    delta = (greater - smaller) / (len(x) * len(y))

    return delta


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

    for metric in metrics:

        delta = cliffs_delta(
            base[metric].values,
            ours[metric].values
        )

        value = abs(delta)

        if value < 0.147:
            interpretation = "Negligible"

        elif value < 0.33:
            interpretation = "Small"

        elif value < 0.474:
            interpretation = "Medium"

        else:
            interpretation = "Large"

        rows.append({

            "Model": model,

            "Metric": metric,

            "Cliffs Delta": round(delta, 4),

            "Effect Size": interpretation

        })

results = pd.DataFrame(rows)

print(results)

results.to_csv(
    "results/cliffs_delta_results.csv",
    index=False
)

print("\nResults saved to results/cliffs_delta_results.csv")