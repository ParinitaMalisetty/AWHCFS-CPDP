from sklearn.model_selection import train_test_split

from src.models import Models
from src.evaluation import evaluate


class BaselineExperiment:

    def __init__(self):
        self.models = Models().get_models()

    def run(self, df):

        X = df.drop("bug", axis=1)
        y = df["bug"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        results = {}

        for model_name, model in self.models.items():

            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            results[model_name] = evaluate(
                y_test,
                predictions
            )

        return results