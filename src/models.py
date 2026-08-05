from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier


class Models:

    def __init__(self):

        self.models = {

            "Random Forest": RandomForestClassifier(
                random_state=42
            ),

            "XGBoost": XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),

            "LightGBM": LGBMClassifier(
                random_state=42,
                verbose=-1
            ),

            "CatBoost": CatBoostClassifier(
                random_state=42,
                verbose=False
            )

        }

    def get_models(self):

        return self.models