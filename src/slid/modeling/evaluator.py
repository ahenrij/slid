"""Evaluate module"""

from typing import Dict
from sklearn import metrics
from slid.utils import utils


def compute_metrics(y_pred, y_test) -> Dict[str, float]:
    """Computes performance metrics.

    Parameters:
    - y_test    : list of true labels (1 for flaky, 0 for safe).
    - y_pred  : list of prediction value between 0.0 and 1.0.

    Output:
    - result: dictionnary of the performance metrics.
    """
    report = metrics.classification_report(
        y_true=y_test, y_pred=y_pred, output_dict=True, zero_division=0
    )

    # unnest dictionnary
    result = utils.flatten(report)

    # correct key names
    result = {k.replace("-", "_"): v for k, v in result.items()}
    result = {k.replace(" ", "_"): v for k, v in result.items()}

    return result


def f1_score(y_pred, y_test) -> float:
    """Compute weighted average f1 score."""
    return metrics.f1_score(y_true=y_test, y_pred=y_pred, average="binary")  # type: ignore
