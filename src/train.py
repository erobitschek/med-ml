import logging
from typing import Callable, Dict, Optional

import numpy as np
import numpy.typing as npt
import torch as torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils import save_model


def train_simple_model(
    run_dir: str,
    x_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    x_test: npt.ArrayLike,
    y_test: npt.ArrayLike,
    model: Callable,
    param_grid: Optional[Dict] = None,
    x_val: Optional[npt.ArrayLike] = None,
    y_val: Optional[npt.ArrayLike] = None,
) -> Callable:
    """
    Function to train a sklearn model on the specified data. If param grid is used, the model hyperparameter is selected
    using 5 fold CV.

    Parameters:
    - run_dir (str): Directory to save outputs
    - x_train (array-like): Training features.
    - y_train (array-like): Training labels.
    - x_test (array-like): Testing features.
    - y_test (array-like): Testing labels.
    - model (Callable): The machine learning model to train.
    - param_grid (Optional[Dict]): Hyperparameters for grid search 
    - x_val (Optional[array-like]): Validation features
    - y_val (Optional[array-like]): Validation labels 

    Returns:
    - Trained sci-kit learn model.
     The function prints evaluation metrics on the test set for an initial preview of model performance.

    """
    
    if x_val is None:
        x_train = x_train.squeeze()
        x_test = x_test.squeeze()
    else:
        print("combine training and validation data")
        x_train = np.vstack((x_train, x_val)).squeeze()
        y_train = np.hstack((y_train, y_val))

    if not param_grid:
        model.fit(x_train, y_train)
    else:
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        grid.fit(x_train, y_train)
        model = grid.best_estimator_
        print(grid.best_params_)
        with open(f"{run_dir}/grid_search.yaml", "w") as f:
            yaml.dump({f"best_params": grid.best_params_}, f)

    return model


def train_pytorch_model(
    model,
    train_loader,
    val_loader,
    train_dir: str,
    logger: logging.Logger,
    device="cpu",
    criterion=nn.BCELoss(),
    optimizer="adam",
    num_epochs=100,
    learning_rate = float, 
    start_epoch: int = 0,
    patience: int = 20,
    checkpoint_freq: int = 10,
    save_path: str = None,
) -> None:
    """
    Train a PyTorch model using the given data.

    Parameters:
    - model: class of PyTorch model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - train_dir (str): Directory to save training artifacts.
    - logger (logging.Logger): Logger for training progress.
    - device (str): Device to use for training 
    - criterion: Loss criterion 
    - optimizer (str): Optimizer choice, either "adam" or "sgd"
    - num_epochs (int): Number of training epochs
    - learning_rate (float): Learning rate for the optimizer 
    - start_epoch (int): Starting epoch 
    - patience (int): Patience for early stopping 
    - checkpoint_freq (int): Frequency to save model checkpoints
    - save_path (str): Path to save the best model

    Returns:
    - None

    This function trains a PyTorch model using the provided data loaders and logs training progress and loss. It supports
    early stopping and model checkpointing.

    """
    if save_path is None:
        save_path = train_dir
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(
            f"Optimizer {optimizer} not supported. Please choose between 'adam' and 'sgd'."
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
                batch_features= batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_labels.float())
                running_val_loss += loss.item()

        average_val_loss = running_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        if epoch % checkpoint_freq == checkpoint_freq - 1:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {average_train_loss:.4f} | Val Loss: {average_val_loss:.4f}"
            )

        # Check for early stopping
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_without_improvement = 0
            save_model(model, run_folder=save_path, only_weights=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if logger:
                    logger.info("Early stopping triggered.")
                else:
                    print("Early stopping triggered.")
                break

    if epochs_without_improvement < patience:
        print(
            f"Early stopping (patience {patience}) not triggered, so it's possible that the model did not converge.",
            "Try adjusting the hyperparameters to find the best model.",
        )

    with open(f"{train_dir}/loss.yaml", "w") as f:
        yaml.dump({"train_loss": train_losses, "val_loss": val_losses}, f)
