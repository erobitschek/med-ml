from typing import List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn


def predict_from_torch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_probabilities: bool = False,
) -> List[Union[int, float]]:
    """
    Use the provided model to predict labels for data in the data_loader.

    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model to use for predictions.
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the data to predict on.
    device : torch.device
        The device (CPU or GPU) to which the model and data should be moved before prediction.
    return_probabilities : bool, optional
        If True, returns the probability of the positive class, otherwise returns binary labels.

    Returns
    -------
    List[Union[int, float]]
        List of predicted labels or probabilities.

    Example
    -------
    >>> model = LogisticRegression(input_dim=10)
    >>> data_loader = DataLoader(dataset, batch_size=32)
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> predictions = predict(model, data_loader, device)
    """

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()

            if return_probabilities:
                predictions = torch.sigmoid(outputs).cpu().numpy()
            else:
                predictions = (outputs > 0.5).long().cpu().numpy()

            all_predictions.extend(predictions)

    return all_predictions


def save_predictions_to_file(
    predictions: List[int],
    run_folder: Union[str, Path],
    filename: str,
    probabilities: Optional[List[float]] = None,
) -> None:
    """
    Save predictions and optionally their corresponding probabilities to a file in the specified directory.

    Parameters
    ----------
    predictions : List[int]
        List of predicted labels.
    run_folder : Union[str, Path]
        Directory where the predictions should be saved.
    filename : str
        Name of the file to save the predictions to.
    probabilities : Optional[List[float]], optional
        List of predicted probabilities corresponding to the labels. If provided, each line in the
        output file will be in the format 'label,probability'.

    Example
    -------
    >>> predictions = [1, 0, 1, 0]
    >>> probabilities = [0.8, 0.2, 0.7, 0.1]
    >>> save_predictions_to_file(predictions, "./output/", "results.txt", probabilities)
    """

    output_path = Path(run_folder) / filename
    with open(output_path, "w") as f:
        for idx, label in enumerate(predictions):
            if probabilities is not None:
                f.write(f"{label},{probabilities[idx]}\n")
            else:
                f.write(f"{label}\n")
