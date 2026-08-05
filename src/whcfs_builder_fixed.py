import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler


class WHCFSBuilder:

    RF_WEIGHT = 0.50
    MI_WEIGHT = 0.30
    ANOVA_WEIGHT = 0.20

    def __init__(self):
        pass

    ####################################################
    # Random Forest Scores
    ####################################################

    def rf_scores(self, X, y):

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)

        return pd.DataFrame({

            "Feature": X.columns,

            "RF Score": model.feature_importances_

        })

    ####################################################
    # Mutual Information Scores
    ####################################################

    def mi_scores(self, X, y):

        scores = mutual_info_classif(
            X,
            y,
            random_state=42
        )

        return pd.DataFrame({

            "Feature": X.columns,

            "MI Score": scores

        })

    ####################################################
    # ANOVA Scores
    ####################################################

    def anova_scores(self, X, y):

        scores, _ = f_classif(X, y)

        return pd.DataFrame({

            "Feature": X.columns,

            "ANOVA Score": scores

        })

    ####################################################
    # Weighted Hybrid Consensus
    ####################################################

    def build(self, datasets):

        train = pd.concat(
            datasets.values(),
            ignore_index=True
        )

        X = train.drop(columns=["bug"])
        y = train["bug"]

        rf = self.rf_scores(X, y)
        mi = self.mi_scores(X, y)
        anova = self.anova_scores(X, y)

        ranking = rf.merge(mi, on="Feature")
        ranking = ranking.merge(anova, on="Feature")

        ####################################################
        # Normalize Scores
        ####################################################

        scaler = MinMaxScaler()

        ranking[["RF Score"]] = scaler.fit_transform(
            ranking[["RF Score"]]
        )

        ranking[["MI Score"]] = scaler.fit_transform(
            ranking[["MI Score"]]
        )

        ranking[["ANOVA Score"]] = scaler.fit_transform(
            ranking[["ANOVA Score"]]
        )

        ####################################################
        # Weighted Score
        ####################################################

        ranking["Weighted Score"] = (

            self.RF_WEIGHT * ranking["RF Score"]

            +

            self.MI_WEIGHT * ranking["MI Score"]

            +

            self.ANOVA_WEIGHT * ranking["ANOVA Score"]

        )

        ranking = ranking.sort_values(

            by="Weighted Score",

            ascending=False

        )

        ranking["Final Rank"] = range(

            1,

            len(ranking) + 1

        )

        print("\nUsing Score-Based WHCFS")
        print(f"RF Weight     : {self.RF_WEIGHT}")
        print(f"MI Weight     : {self.MI_WEIGHT}")
        print(f"ANOVA Weight  : {self.ANOVA_WEIGHT}")

        print("\nTop Features")

        print(

            ranking[
                ["Feature", "Weighted Score", "Final Rank"]
            ].head(10)

        )

        ranking.to_csv(

            "results/latest_weighted_ranking.csv",

            index=False

        )

        return ranking

    ####################################################
    # Top-k Features
    ####################################################

    def get_top_features(self, datasets, k=10):

        ranking = self.build(datasets)

        return ranking.head(k)["Feature"].tolist()