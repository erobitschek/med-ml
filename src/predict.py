from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def predict_from_torch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Use the provided model to predict labels for data in the data_loader.

    Args:
        model: The trained PyTorch model to use for predictions.
        data_loader: DataLoader containing the data to predict on.
        device: The device (CPU or GPU) to which the model and data should be moved before prediction.
    Returns:
        Arrays of predicted labels and probabilities.

    Example:
        >>> model = LogisticRegression(input_dim=10)
        >>> data_loader = DataLoader(dataset, batch_size=32)
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> predictions, probabilities = predict(model, data_loader, device)
    """

    model.eval()
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_features, _ in data_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)

            # Use softmax for multi-class probabilities
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probabilities.extend(probabilities)

            # Get the indices of max probabilities
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def save_predictions_to_file(
    predictions: npt.ArrayLike,
    run_folder: Union[str, Path],
    filename: str,
    probabilities: Optional[npt.ArrayLike] = None,
) -> None:
    """Save predictions and optionally their corresponding probabilities to a file in the specified directory.

    Args:
        predictions: Predicted labels.
        run_folder: Where the predictions should be saved.
        filename: Name of the file to save the predictions to.
        probabilities: List of predicted probabilities corresponding
            to the labels. If provided, each line in the output file will be in the format 'label,probability'.
    """

    output_path = Path(run_folder) / filename
    with open(output_path, "w") as f:
        for idx, label in enumerate(predictions):
            if probabilities is None:
                f.write(f"{label}\n")
            else:
                if hasattr(probabilities[idx], "__iter__"):
                    probas_str = ",".join(
                        [str(p) for p in probabilities[idx]]
                    )  # for more than one class
                else:
                    probas_str = str(probabilities[idx])  # for single class
                f.write(f"{label},{probas_str}\n")
