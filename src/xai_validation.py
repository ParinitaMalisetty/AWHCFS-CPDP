import os
import pandas as pd
from scipy.stats import spearmanr, kendalltau


class XAIValidator:

    def __init__(self, output_dir="results/shap"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def normalize_shap(self, shap_df):

        result = shap_df.copy()

        model_columns = [
            col for col in result.columns
            if col != "Feature"
        ]

        for col in model_columns:

            max_value = result[col].max()

            if max_value > 0:
                result[col] = (
                    result[col] / max_value
                )

            else:
                result[col] = 0.0

        result["SHAP Consensus"] = result[
            model_columns
        ].mean(axis=1)

        result = result.sort_values(
            by="SHAP Consensus",
            ascending=False
        ).reset_index(drop=True)

        result["SHAP Rank"] = (
            result.index + 1
        )

        return result

    def compare_with_awhcfs(
        self,
        shap_df,
        awhcfs_features
    ):

        shap_features = shap_df[
            ["Feature", "SHAP Consensus", "SHAP Rank"]
        ].copy()

        awhcfs_df = pd.DataFrame({
            "Feature": awhcfs_features,
            "AWHCFS Rank": range(
                1,
                len(awhcfs_features) + 1
            )
        })

        comparison = shap_features.merge(
            awhcfs_df,
            on="Feature",
            how="outer"
        )

        comparison["SHAP Rank"] = comparison[
            "SHAP Rank"
        ].fillna(len(shap_features) + 1)

        comparison["AWHCFS Rank"] = comparison[
            "AWHCFS Rank"
        ].fillna(len(awhcfs_features) + 1)

        comparison["Rank Difference"] = (
            comparison["AWHCFS Rank"]
            - comparison["SHAP Rank"]
        )

        comparison = comparison.sort_values(
            by="SHAP Rank"
        ).reset_index(drop=True)

        return comparison

    def calculate_statistics(
        self,
        comparison
    ):

        valid = comparison[
            ["AWHCFS Rank", "SHAP Rank"]
        ].dropna()

        spearman, spearman_p = spearmanr(
            valid["AWHCFS Rank"],
            valid["SHAP Rank"]
        )

        kendall, kendall_p = kendalltau(
            valid["AWHCFS Rank"],
            valid["SHAP Rank"]
        )

        results = {
            "Spearman Correlation": spearman,
            "Spearman p-value": spearman_p,
            "Kendall Correlation": kendall,
            "Kendall p-value": kendall_p
        }

        return results

    def top_k_overlap(
        self,
        awhcfs_features,
        shap_df,
        k=10
    ):

        awhcfs_top = set(
            awhcfs_features[:k]
        )

        shap_top = set(
            shap_df
            .sort_values(
                by="SHAP Consensus",
                ascending=False
            )
            .head(k)["Feature"]
        )

        intersection = (
            awhcfs_top & shap_top
        )

        overlap = len(intersection)

        percentage = (
            overlap / k
        ) * 100

        return {
            "K": k,
            "Overlap Count": overlap,
            "Overlap Percentage": percentage,
            "Common Features": sorted(
                intersection
            )
        }

    def save_results(
        self,
        shap_df,
        comparison,
        statistics,
        overlap5,
        overlap10
    ):

        shap_df.to_csv(
            os.path.join(
                self.output_dir,
                "normalized_shap_consensus.csv"
            ),
            index=False
        )

        comparison.to_csv(
            os.path.join(
                self.output_dir,
                "awhcfs_shap_comparison.csv"
            ),
            index=False
        )

        statistics_df = pd.DataFrame([
            statistics
        ])

        statistics_df.to_csv(
            os.path.join(
                self.output_dir,
                "awhcfs_shap_statistics.csv"
            ),
            index=False
        )

        overlap_df = pd.DataFrame([
            {
                "K": overlap5["K"],
                "Overlap Count": overlap5[
                    "Overlap Count"
                ],
                "Overlap Percentage": overlap5[
                    "Overlap Percentage"
                ],
                "Common Features": ", ".join(
                    overlap5["Common Features"]
                )
            },
            {
                "K": overlap10["K"],
                "Overlap Count": overlap10[
                    "Overlap Count"
                ],
                "Overlap Percentage": overlap10[
                    "Overlap Percentage"
                ],
                "Common Features": ", ".join(
                    overlap10["Common Features"]
                )
            }
        ])

        overlap_df.to_csv(
            os.path.join(
                self.output_dir,
                "awhcfs_shap_topk_overlap.csv"
            ),
            index=False
        )