# #import pandas as pd
# # from experiments import baseline
# # from experiments.lopo_baseline import LOPOBaselineExperiment
# # from experiments.lopo_cpdp import LOPOExperiment
# # from experiments.lopo_baseline import LOPOBaselineExperiment
# # from experiments.lopo_whcfs import LOPOWHCFSExperiment
# # from src.config import DATASET_PATH
# # from src.data_loader import DataLoader
# # from src.preprocessing import Preprocessor
# # from experiments.lopo_awhcfs import LOPOAWHCFSExperiment
# # from experiments.baseline import BaselineExperiment
# # from experiments.cross_project import CrossProjectExperiment
# # from src.feature_importance import FeatureImportanceExtractor
# # from experiments.cpsfs import CPSFS
# # from experiments.cpsfs_experiment import CPSFSExperiment

# from sklearn import datasets

# from experiments.lopo_baseline import LOPOBaselineExperiment
# from experiments.lopo_awhcfs import LOPOAWHCFSExperiment
# from experiments.xai_validation_experiment import XAIValidationExperiment
# from src.config import DATASET_PATH
# from src.data_loader import DataLoader
# from src.preprocessing import Preprocessor
# from experiments.shap_experiment import SHAPExperiment
# def load_and_preprocess():

#     loader = DataLoader(DATASET_PATH)
#     datasets = loader.load_datasets()

#     preprocessor = Preprocessor()

#     cleaned = {}

#     for name, df in datasets.items():

#         if df.shape[1] != 24:
#             continue

#         cleaned[name] = preprocessor.clean_dataset(df)

#     return cleaned


# # def save_baseline(results):

# #     rows = []

# #     for model, metrics in results.items():

# #         row = {"Model": model}
# #         row.update(metrics)

# #         rows.append(row)

# #     df = pd.DataFrame(rows)

# #     df.to_csv(
# #         "results/baseline_results.csv",
# #         index=False
# #     )

# #     return df


# # def save_cpsfs_results(results, train, test, feature_count):

# #     rows = []

# #     for model, metrics in results.items():

# #         row = {
# #             "Method": "CPSFS",
# #             "Train": train,
# #             "Test": test,
# #             "Feature Count": feature_count,
# #             "Model": model
# #         }

# #         row.update(metrics)

# #         rows.append(row)

# #     df = pd.DataFrame(rows)

# #     df.to_csv(
# #         "results/cpsfs_results.csv",
# #         index=False
# #     )

# #     return df


# def main():

#     ####################################################
#     # Load Data
#     ####################################################

#     datasets = load_and_preprocess()

#     print("\nDatasets Loaded Successfully")
#     print(datasets.keys())
#     # baseline = LOPOBaselineExperiment()
#     # baseline_results = baseline.run(datasets)

#     # print("\nBaseline Completed")
#     # print(baseline_results.head())

#     # awhcfs = LOPOAWHCFSExperiment()
#     # awhcfs_results = awhcfs.run(datasets)

#     # print("\nAWHCFS Completed")
#     # print(awhcfs_results.head())
    
#     shap_exp = SHAPExperiment()

#     shap_results = shap_exp.run(datasets)

#     print("\nSHAP Completed")
#     print(shap_results.head())
    
#     # experiment = LOPOExperiment()
#     # results = experiment.run(datasets)
#     # print("First 5 results:")
#     # print(results.head())
#     # print("\nLOPO Experiment completed successfully.")
#     # print("\nResults saved to results/lopo_cpsfs_results.csv")

# #     ####################################################
# #     # Baseline
# #     ####################################################

# #     print("\n" + "=" * 70)
# #     print("BASELINE")
# #     print("=" * 70)

# #     baseline = BaselineExperiment()

# #     baseline_results = baseline.run(
# #         datasets["ant-1.7"]
# #     )

# #     baseline_df = save_baseline(
# #         baseline_results
# #     )

# #     print(baseline_df)

# #     ####################################################
# #     # Feature Importance
# #     ####################################################

# #     print("\n" + "=" * 70)
# #     print("FEATURE IMPORTANCE")
# #     print("=" * 70)

# #     extractor = FeatureImportanceExtractor()

# #     for dataset_name, df in datasets.items():

# #         print(f"Processing {dataset_name}")

# #         extractor.extract(
# #             dataset_name,
# #             df
# #         )

# #     print("\nFeature importance extraction completed.")

# #     ####################################################
# #     # CPSFS Ranking
# #     ####################################################

# #     print("\n" + "=" * 70)
# #     print("CPSFS FEATURE RANKING")
# #     print("=" * 70)

# #     cpsfs = CPSFS()

# #     ranking = cpsfs.build_consensus()

# #     print(ranking)

# #     top_features = cpsfs.get_top_features(10)

# #     print("\nTop 10 Features")

# #     print(top_features)

# #     ####################################################
# #     # Cross Project using CPSFS
# #     ####################################################

# #     print("\n" + "=" * 70)
# #     print("CPSFS CROSS PROJECT")
# #     print("=" * 70)

# #     cross = CrossProjectExperiment()

# #     cpsfs_results = cross.run(

# #         datasets["ant-1.7"],

# #         datasets["camel-1.0"],

# #         features=top_features

# #     )

# #     cpsfs_df = save_cpsfs_results(

# #         cpsfs_results,

# #         "ant-1.7",

# #         "camel-1.0",

# #         len(top_features)

# #     )

# #     print(cpsfs_df)

# #     print("\nProject completed successfully.")

# #     print("\nFiles generated:")

# #     print("results/baseline_results.csv")

# #     print("results/cpsfs_feature_ranking.csv")

# #     print("results/cpsfs_results.csv")

# #     print("results/feature_importance/")

# #     print("\n" + "="*70)
# #     print("CPSFS COMPARISON")
# #     print("="*70)

# #     experiment = CPSFSExperiment()

# #     comparison = experiment.run(datasets)

# #     print(comparison)
    
# if __name__ == "__main__":
#     main()

import os
import pandas as pd

from src.config import DATASET_PATH
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor

from experiments.lopo_baseline import LOPOBaselineExperiment
from experiments.lopo_awhcfs import LOPOAWHCFSExperiment
from experiments.shap_experiment import SHAPExperiment
from experiments.xai_validation_experiment import XAIValidationExperiment


def load_and_preprocess():

    loader = DataLoader(DATASET_PATH)
    datasets = loader.load_datasets()

    preprocessor = Preprocessor()

    cleaned = {}

    for name, df in datasets.items():

        if df.shape[1] != 24:
            continue

        cleaned[name] = preprocessor.clean_dataset(df)

    return cleaned


def main():

    os.makedirs(
        "results",
        exist_ok=True
    )

    os.makedirs(
        "results/shap",
        exist_ok=True
    )

    datasets = load_and_preprocess()

    print("\nDatasets Loaded Successfully")
    print(datasets.keys())

    print("\n" + "=" * 80)
    print("LOPO BASELINE")
    print("=" * 80)

    baseline = LOPOBaselineExperiment()

    baseline_results = baseline.run(
        datasets
    )

    print("\nBaseline Completed")

    print(
        baseline_results.head()
    )

    print("\n" + "=" * 80)
    print("LOPO AWHCFS")
    print("=" * 80)

    awhcfs = LOPOAWHCFSExperiment()

    awhcfs_results = awhcfs.run(
        datasets
    )

    print("\nAWHCFS Completed")

    print(
        awhcfs_results.head()
    )

    print("\n" + "=" * 80)
    print("GLOBAL SHAP EXPLAINABILITY")
    print("=" * 80)

    shap_exp = SHAPExperiment()

    shap_results = shap_exp.run(
        datasets
    )

    print("\nSHAP Completed")

    if shap_results is not None:

        print(
            shap_results.head()
        )

    awhcfs_ranking_path = (
        "results/awhcfs_feature_frequency.csv"
    )

    if not os.path.exists(
        awhcfs_ranking_path
    ):

        print(
            "\nAWHCFS feature frequency file "
            "was not found."
        )

        print(
            "Skipping XAI validation."
        )

        return

    awhcfs_ranking = pd.read_csv(
        awhcfs_ranking_path
    )

    awhcfs_features = (
        awhcfs_ranking[
            "Feature"
        ]
        .tolist()
    )

    print("\n" + "=" * 80)
    print("AWHCFS vs SHAP VALIDATION")
    print("=" * 80)

    xai_validation = (
        XAIValidationExperiment()
    )

    xai_results = xai_validation.run(
        awhcfs_features
    )

    print("\nXAI Validation Completed")

    print("\n" + "=" * 80)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 80)

    print("\nGenerated Results:")

    print(
        "results/lopo_baseline_results.csv"
    )

    print(
        "results/lopo_awhcfs_results.csv"
    )

    print(
        "results/adaptive_weights.csv"
    )

    print(
        "results/awhcfs_feature_frequency.csv"
    )

    print(
        "results/shap/shap_importance.csv"
    )

    print(
        "results/shap/normalized_shap_consensus.csv"
    )

    print(
        "results/shap/awhcfs_shap_comparison.csv"
    )

    print(
        "results/shap/awhcfs_shap_statistics.csv"
    )

    print(
        "results/shap/awhcfs_shap_topk_overlap.csv"
    )


if __name__ == "__main__":
    main()