from src.models import Models
from src.evaluation import evaluate


class CrossProjectExperiment:

    def __init__(self):
        self.models = Models().get_models()

    def run(self, train_df, test_df, features=None):

        # Target variables
        y_train = train_df["bug"]
        y_test = test_df["bug"]

        # Feature selection
        if features is None:
            X_train = train_df.drop("bug", axis=1)
            X_test = test_df.drop("bug", axis=1)
        else:
            X_train = train_df[features]
            X_test = test_df[features]

        results = {}

        for model_name, model in self.models.items():

            print(f"Training {model_name}...")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            metrics = evaluate(y_test, predictions)

            results[model_name] = metrics

        return results