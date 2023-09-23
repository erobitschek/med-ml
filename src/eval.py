import json
from typing import Dict, List

from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve


def evaluate_predictions(
    predictions: List[int], true_labels: List[int]
) -> Dict[str, float]:
    """Evaluate predictions using various metrics."""

    auc = roc_auc_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Compute ROC curve points
    fpr, tpr, _ = roc_curve(true_labels, predictions)

    return {
        "AUC": auc,
        "Balanced Accuracy": balanced_acc,
        "F1 Score": f1,
        "FPR": fpr,
        "TPR": tpr,
    }


def save_evaluation_summary(
    metric_dict: Dict[str, float],
    run_folder: str,
    filename: str = "evaluation_summary.json",
) -> None:
    """
    Save evaluation metrics to a summary file.
    """
    with open(f"{run_folder}/{filename}", "w") as f:
        metric_dict = {k: str(v) for k, v in metric_dict.items()}
        f.write(json.dumps(metric_dict, indent=4))
