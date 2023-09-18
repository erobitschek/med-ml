from typing import List, Dict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, roc_curve


def evaluate_predictions(
    predictions: List[int], true_labels: List[int]
) -> Dict[str, float]:
    """Evaluate predictions using various metrics."""

    # Compute metrics
    auc = roc_auc_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Compute ROC curve points
    fpr, tpr, _ = roc_curve(true_labels, predictions)

    metrics = {
        "AUC": auc,
        "Balanced Accuracy": balanced_acc,
        "F1 Score": f1,
        "FPR": fpr,
        "TPR": tpr,
    }

    return metrics


def save_evaluation_summary(
    true_labels: List[int],
    predicted_probs: List[float],
    run_folder: str,
    filename: str = "evaluation_summary.txt",
) -> None:
    """
    Compute and save evaluation metrics to a summary file.

    Args:
    - true_labels (List[int]): The true labels.
    - predicted_probs (List[float]): Predicted probabilities for the positive class.
    - run_folder (str): The directory where the evaluation summary will be saved.

    Returns:
    None
    """

    # Compute metrics
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    auc = roc_auc_score(true_labels, predicted_probs)
    balanced_acc = balanced_accuracy_score(
        true_labels, [1 if p > 0.5 else 0 for p in predicted_probs]
    )
    f1 = f1_score(true_labels, [1 if p > 0.5 else 0 for p in predicted_probs])

    # Write metrics to a file
    with open(f"{run_folder}/{filename}", "w") as f:
        f.write("metric: {}\n".format("value"))
        f.write("ROC AUC: {}\n".format(auc))
        f.write("Balanced Accuracy: {}\n".format(balanced_acc))
        f.write("F1 Score: {}\n".format(f1))
