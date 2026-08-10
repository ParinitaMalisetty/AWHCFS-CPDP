import os
import pandas as pd

from src.xai_validation import XAIValidator


class XAIValidationExperiment:

    def __init__(self):
        self.validator = XAIValidator()

    def run(self, awhcfs_features):

        print("\n" + "=" * 80)
        print("AWHCFS vs SHAP VALIDATION")
        print("=" * 80)

        shap_path = (
            "results/shap/shap_importance.csv"
        )

        if not os.path.exists(shap_path):

            raise FileNotFoundError(
                "SHAP importance file not found: "
                + shap_path
            )

        shap_df = pd.read_csv(
            shap_path
        )

        print("\nNormalizing SHAP importance...")

        normalized = (
            self.validator.normalize_shap(
                shap_df
            )
        )

        print("\nSHAP Consensus Ranking:")
        print(
            normalized[
                [
                    "Feature",
                    "SHAP Consensus",
                    "SHAP Rank"
                ]
            ].head(10)
        )

        print("\nComparing with AWHCFS...")

        comparison = (
            self.validator.compare_with_awhcfs(
                normalized,
                awhcfs_features
            )
        )

        statistics = (
            self.validator.calculate_statistics(
                comparison
            )
        )

        overlap5 = (
            self.validator.top_k_overlap(
                awhcfs_features,
                normalized,
                k=5
            )
        )

        overlap10 = (
            self.validator.top_k_overlap(
                awhcfs_features,
                normalized,
                k=10
            )
        )

        self.validator.save_results(
            normalized,
            comparison,
            statistics,
            overlap5,
            overlap10
        )

        print("\n" + "=" * 80)
        print("XAI VALIDATION RESULTS")
        print("=" * 80)

        print(
            f"\nSpearman Correlation : "
            f"{statistics['Spearman Correlation']:.4f}"
        )

        print(
            f"Spearman p-value    : "
            f"{statistics['Spearman p-value']:.4f}"
        )

        print(
            f"\nKendall Correlation  : "
            f"{statistics['Kendall Correlation']:.4f}"
        )

        print(
            f"Kendall p-value     : "
            f"{statistics['Kendall p-value']:.4f}"
        )

        print(
            f"\nTop-5 Overlap       : "
            f"{overlap5['Overlap Count']}/5 "
            f"({overlap5['Overlap Percentage']:.1f}%)"
        )

        print(
            f"Top-10 Overlap      : "
            f"{overlap10['Overlap Count']}/10 "
            f"({overlap10['Overlap Percentage']:.1f}%)"
        )

        print("\nCommon Top-10 Features:")

        print(
            overlap10["Common Features"]
        )

        print("\nFiles saved:")

        print(
            "results/shap/"
            "normalized_shap_consensus.csv"
        )

        print(
            "results/shap/"
            "awhcfs_shap_comparison.csv"
        )

        print(
            "results/shap/"
            "awhcfs_shap_statistics.csv"
        )

        print(
            "results/shap/"
            "awhcfs_shap_topk_overlap.csv"
        )

        return {
            "normalized": normalized,
            "comparison": comparison,
            "statistics": statistics,
            "overlap5": overlap5,
            "overlap10": overlap10
        }