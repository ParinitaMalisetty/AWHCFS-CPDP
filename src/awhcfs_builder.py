import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import MinMaxScaler

class AWHCFSBuilder:

    def rf_scores(self, X, y):
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return pd.DataFrame({
            "Feature": X.columns,
            "RF Score": model.feature_importances_
        })

    def mi_scores(self, X, y):
        scores = mutual_info_classif(X, y, random_state=42)
        return pd.DataFrame({
            "Feature": X.columns,
            "MI Score": scores
        })

    def anova_scores(self, X, y):
        scores, _ = f_classif(X, y)
        return pd.DataFrame({
            "Feature": X.columns,
            "ANOVA Score": scores
        })

    def build(self, datasets):

        train = pd.concat(datasets.values(), ignore_index=True)

        X = train.drop(columns=["bug"])
        y = train["bug"]

        rf = self.rf_scores(X, y)
        mi = self.mi_scores(X, y)
        anova = self.anova_scores(X, y)

        ranking = rf.merge(mi, on="Feature")
        ranking = ranking.merge(anova, on="Feature")

        scaler = MinMaxScaler()

        ranking["RF Score"] = scaler.fit_transform(ranking[["RF Score"]])
        ranking["MI Score"] = scaler.fit_transform(ranking[["MI Score"]])
        ranking["ANOVA Score"] = scaler.fit_transform(ranking[["ANOVA Score"]])

        rf_var = ranking["RF Score"].var()
        mi_var = ranking["MI Score"].var()
        anova_var = ranking["ANOVA Score"].var()

        total_var = rf_var + mi_var + anova_var

        rf_weight = rf_var / total_var
        mi_weight = mi_var / total_var
        anova_weight = anova_var / total_var
        ranking["Adaptive Score"] = (
            rf_weight * ranking["RF Score"] +
            mi_weight * ranking["MI Score"] +
            anova_weight * ranking["ANOVA Score"]
        )

        ranking = ranking.sort_values(
            by="Adaptive Score",
            ascending=False
        )

        ranking["Final Rank"] = range(
            1,
            len(ranking) + 1
        )

        ranking.to_csv(
            "results/latest_weighted_ranking.csv",
            index=False
        )

        weights = {
            "RF Weight": rf_weight,
            "MI Weight": mi_weight,
            "ANOVA Weight": anova_weight,
            "RF Variance": rf_var,
            "MI Variance": mi_var,
            "ANOVA Variance": anova_var
        }

        return ranking, weights

    def get_top_features(self, datasets, k=10):

        ranking, weights = self.build(datasets)

        features = ranking.head(k)["Feature"].tolist()

        print("\nAdaptive Variance Weights")
        print(f"RF Weight     : {weights['RF Weight']:.4f}")
        print(f"MI Weight     : {weights['MI Weight']:.4f}")
        print(f"ANOVA Weight  : {weights['ANOVA Weight']:.4f}")

        print("\nTop Features")
        print(
            ranking[
                ["Feature", "Adaptive Score", "Final Rank"]
            ].head(k)
        )

        return features, weights