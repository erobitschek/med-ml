import json
import logging
from typing import Dict, Union

import numpy.typing as npt
import torch
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from configs.config_scaffold import RunConfig
from data import DataSplit
from predict import save_predictions_to_file


def evaluate_predictions(
    predictions: npt.ArrayLike, true_labels: npt.ArrayLike
) -> Dict[str, float]:
    """Evaluate predictions using various metrics."""

    auc = roc_auc_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    # Compute ROC curve points
    fpr, tpr, _ = roc_curve(true_labels, predictions)

    return {
        "auc": auc,
        "balanced_accuracy": balanced_acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
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


def run_eval(
    predictions: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    true_labels: DataSplit,
    run_dir: str,
    logger: logging.Logger,
) -> None:
    """
    Saves the predictions and probabilities (typically from the test set) and evaluates the model, saving a summary to the run_directory.

    Parameters
    ----------
    predictions : Array of predictions.
    probabilities : Array of probabilities.
    true_labels : DataSplit with true labels.
    run_dir : Directory to save metadata to.
    logger : Logger object.
    """
    logger.info(
        f"The first 5 predictions and their probabilities are: {predictions[:5], probabilities[:5]}"
    )
    logger.info(f"Saving predictions to {run_dir}")
    save_predictions_to_file(
        predictions=predictions,
        probabilities=probabilities,
        run_folder=run_dir,
        filename=f"predictions.txt",
    )
    logger.info(f"Evaluating model predictions")
    assert len(predictions) == len(
        true_labels
    ), "Predictions and true labels must be the same length."
    evaluation = evaluate_predictions(predictions=predictions, true_labels=true_labels)
    save_evaluation_summary(
        metric_dict=evaluation,
        run_folder=run_dir,
        filename=f"evaluation_summary.json",
    )
    logger.info(f"Saved evaluation summary.")
