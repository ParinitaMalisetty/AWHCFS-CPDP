import os
import pandas as pd

from src.models import Models
from src.explainability import Explainability


class SHAPExperiment:

    def __init__(self):

        self.models = Models().get_models()
        self.xai = Explainability()

    def run(self, datasets):

        print("\n" + "=" * 80)
        print("GLOBAL SHAP EXPLAINABILITY")
        print("=" * 80)

        train_df = pd.concat(
            datasets.values(),
            ignore_index=True
        )

        X = train_df.drop(columns=["bug"])
        y = train_df["bug"]

        all_importance = []

        for model_name, model in self.models.items():

            print(f"\nTraining {model_name}")

            model.fit(X, y)

            try:

                importance = self.xai.explain(
                    model,
                    X,
                    model_name
                )

                all_importance.append(
                    importance
                )

            except Exception as e:

                print(f"\nSkipping {model_name}")
                print(e)

        if len(all_importance) == 0:

            print("No SHAP results generated.")

            return None

        merged = all_importance[0]

        for df in all_importance[1:]:

            merged = merged.merge(
                df,
                on="Feature",
                how="outer"
            )

        merged = merged.fillna(0)

        model_columns = merged.columns[1:]

        merged["Average"] = merged[
            model_columns
        ].mean(axis=1)

        merged = merged.sort_values(
            by="Average",
            ascending=False
        )

        os.makedirs(
            "results/shap",
            exist_ok=True
        )

        merged.to_csv(
            "results/shap/shap_importance.csv",
            index=False
        )

        print("\n" + "=" * 80)
        print("SHAP Experiment Completed")
        print("=" * 80)

        print("\nResults saved to:")
        print("results/shap/shap_importance.csv")

        return merged