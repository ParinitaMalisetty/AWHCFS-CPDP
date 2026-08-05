import pandas as pd
from experiments.cross_project import CrossProjectExperiment


class ExperimentRunner:

    def __init__(self):
        self.cross = CrossProjectExperiment()

    def run(self, datasets):

        all_results = []

        dataset_names = [
        "ant-1.7",
        "camel-1.0",
        "jedit-3.2"
    ]

        total = 0

        for train_name in dataset_names:

            for test_name in dataset_names:

                if train_name == test_name:
                    continue

                print(f"\n{train_name} --> {test_name}")

                results = self.cross.run(
                    datasets[train_name],
                    datasets[test_name]
                )

                for model, metrics in results.items():

                    row = {
                        "Train": train_name,
                        "Test": test_name,
                        "Model": model
                    }

                    row.update(metrics)

                    all_results.append(row)

                total += 1
                print(f"Completed {total}")

        return pd.DataFrame(all_results)