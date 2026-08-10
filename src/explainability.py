import os
import shap
import xgboost as xgb
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

        if model_name == "XGBoost":
            values = self._explain_xgboost(
                model,
                X_sample
            )
        else:
            values = self._explain_tree_model(
                model,
                X_sample
            )

        if hasattr(values, "values"):
            values = values.values

        if values.ndim == 3:
            values = values[:, :, 1]

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

    def _explain_tree_model(self, model, X_sample):

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X_sample
        )

        if isinstance(shap_values, list):
            return shap_values[1]

        return shap_values

    def _explain_xgboost(self, model, X_sample):

        print(
            "Using native XGBoost TreeSHAP..."
        )

        booster = model.get_booster()

        dmatrix = xgb.DMatrix(
            X_sample,
            feature_names=list(
                X_sample.columns
            )
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True
        )

        if contributions.ndim == 3:
            values = contributions[:, 0, :-1]
        else:
            values = contributions[:, :-1]

        print(
            "Native XGBoost TreeSHAP completed."
        )

        return values