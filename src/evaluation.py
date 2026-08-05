from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import balanced_accuracy_score


def evaluate(y_true, y_pred):

    results = {

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(y_true, y_pred),

        "Recall": recall_score(y_true, y_pred),

        "F1": f1_score(y_true, y_pred),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        )

    }

    return results