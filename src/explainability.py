import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Explainability:

    def __init__(self, output_dir="results/shap"):

        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def explain(self, model, X, model_name):

        print(f"\nGenerating SHAP explanation for {model_name}")

        X_sample = X.sample(
            n=min(500, len(X)),
            random_state=42
        )

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            values = shap_values[1]
        else:
            values = shap_values

        if values.ndim == 3:
            values = values[:, :, 1]

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            values,
            X_sample,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                f"{model_name}_summary.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        importance = np.mean(
            np.abs(values),
            axis=0
        )

        importance_df = pd.DataFrame({
            "Feature": X_sample.columns,
            model_name: importance
        })

        importance_df = importance_df.sort_values(
            by=model_name,
            ascending=False
        )

        importance_df.to_csv(
            os.path.join(
                self.output_dir,
                f"{model_name}_importance.csv"
            ),
            index=False
        )

        print(f"{model_name} SHAP completed.")

        return importance_df