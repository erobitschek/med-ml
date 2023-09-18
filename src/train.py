import torch as torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable, Dict
import numpy.typing as npt
import logging
import yaml
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from utils import save_model


def train_simple_model(
    x_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    x_test: npt.ArrayLike,
    y_test: npt.ArrayLike,
    model: Callable,
    param_grid: Optional[Dict] = None,
) -> Callable:
    """
    Function to train a sklearn model on the specified data. If param grid is used, the model hyperparameter is selected
    using 5 fold CV.

    Parameters:
    - x_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - x_test (array-like): Testing features.
    - y_test (array-like): Testing labels.
    - model (Callable): The machine learning model to train.
    - param_grid (Optional[Dict]): Hyperparameters for grid search (default: None).

    Returns:
    - Trained sci-kit learn model.
     The function prints evaluation metrics on the test set for an initial preview of model performance.

    """
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()

    if not param_grid:
        model.fit(x_train, y_train)
    else:
        clf = GridSearchCV(model, param_grid, cv=5)
        clf.fit(x_train, y_train)
        model = clf.best_estimator_
        print(clf.best_params_)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_pred, y_test)
    bal_acc = balanced_accuracy_score(y_pred, y_test)
    print(f"Test Acc: {acc}")
    print(f"Balanced Acc: {bal_acc}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1: {f1_score(y_test, y_pred, average='weighted')}")

    if max(y_test) == 1:
        y_pred_prob = model.predict_proba(x_test)
        auroc = roc_auc_score(y_true=y_test, y_score=y_pred_prob[:, 1])
        auprc = average_precision_score(y_test, y_pred_prob[:, 1])
        print(f"AUROC: {auroc}")
        print(f"AUPRC: {auprc}")

    return model


def train_pytorch_model(
    model,
    train_loader,
    val_loader,
    train_dir: str,
    train_logger: logging.Logger,
    device="cpu",
    criterion=nn.BCELoss(),
    optimizer="adam",
    num_epochs=100,
    learning_rate=0.001,
    start_epoch: int = 0,
    patience: int = 20,
    checkpoint_freq: int = 10,
    save_best: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Train a PyTorch model using the given data.

    Parameters:
    - model: class of PyTorch model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - train_dir (str): Directory to save training artifacts.
    - train_logger (logging.Logger): Logger for training progress.
    - device (str): Device to use for training (default: "cpu").
    - criterion: Loss criterion (default: nn.BCELoss()).
    - optimizer (str): Optimizer choice, either "adam" or "sgd" (default: "adam").
    - num_epochs (int): Number of training epochs (default: 100).
    - learning_rate (float): Learning rate for the optimizer (default: 0.001).
    - start_epoch (int): Starting epoch (default: 0).
    - patience (int): Patience for early stopping (default: 20).
    - checkpoint_freq (int): Frequency to save model checkpoints (default: 10).
    - save_best (bool): Save the best model (default: True).
    - save_path (Optional[str]): Path to save the best model (default: None).

    Returns:
    - None

    This function trains a PyTorch model using the provided data loaders and logs training progress and loss. It supports
    early stopping and model checkpointing.

    """
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(
            "Optimizer not supported. Please choose between 'adam' and 'sgd'."
        )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels.float())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_train_loss = running_loss / len(train_loader)
        train_losses.append(average_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_labels.float())
                running_val_loss += loss.item()

        average_val_loss = running_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        if epoch % checkpoint_freq == checkpoint_freq - 1:
            train_logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f}"
            )

        # Check for early stopping
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_without_improvement = 0
            if save_best:
                if not save_path:
                    raise ValueError(
                        "Please provide a save_path to save the best model."
                    )
                save_model(model, run_folder=save_path, only_weights=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if train_logger:
                    train_logger.info("Early stopping triggered.")
                else:
                    print("Early stopping triggered.")
                break

    if epochs_without_improvement < patience:
        print(
            f"Early stopping (patience {patience}) not triggered, so it's possible that the model did not converge.",
            "Try adjusting the hyperparameters to find the best model.",
        )

    with open(train_dir / "loss.yaml", "w") as f:
        yaml.dump({"train_loss": train_losses, "val_loss": val_losses}, f)
