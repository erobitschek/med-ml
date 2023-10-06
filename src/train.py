import os
from logging import Logger
from typing import Callable, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import torch as torch
import torch.nn as nn
import torch.optim as optim
import yaml
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from configs.config_scaffold import RunConfig
from data import DataSplit
from utils import save_model
ModelType = Callable[..., skLogisticRegression]

def train_simple_model(
    run_dir: str,
    x_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    model: ModelType,
    param_grid: Optional[Dict] = None,
    x_val: Optional[npt.ArrayLike] = None,
    y_val: Optional[npt.ArrayLike] = None,
) -> ModelType:
    """Trains a scikit-learn model using the provided data. If a parameter grid is provided, it performs
    hyperparameter selection using 5-fold cross-validation.

    Args:
        run_dir: Directory to save outputs.
        x_train: Training feature data.
        y_train: Training target data.
        model: Untrained machine learning model.
        param_grid: Parameters to search over for hyperparameter tuning.
        x_val: Validation feature data.
        y_val: Validation target data.

    Returns:
        Trained scikit-learn model.
    """

    if x_val is None:
        x_train = x_train.squeeze()
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
    logger: Logger,
    device="cpu",
    criterion=nn.BCELoss(),
    optimizer="adam",
    num_epochs=100,
    learning_rate=float,
    start_epoch: int = 0,
    patience: int = 20,
    checkpoint_freq: int = 10,
    save_path: str = None,
) -> None:
    """Trains a PyTorch model using the provided data loaders. The function logs the training progress and loss. It also
    offers functionalities such as early stopping and model checkpointing.

    Args:
        model: PyTorch model class instance to be trained.
        train_loader: DataLoader object for training data.
        val_loader: DataLoader object for validation data.
        train_dir: Directory to save training artifacts.
        logger: Logger for capturing training progress.
        device: Device on which the model should be trained.
        criterion: Loss function to use.
        optimizer: Optimizer choice ("adam" or "sgd").
        num_epochs: Total number of epochs for training.
        learning_rate: Learning rate for the optimizer.
        start_epoch: Epoch to begin training. Useful for resuming training.
        patience: Patience parameter for early stopping.
        checkpoint_freq: Frequency for saving model checkpoints.
        save_path: Directory path to save the best model weights.
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
                batch_features = batch_features.to(device)
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


def gridsearch_lgbm(
    run_dir: str,
    config: RunConfig,
    train_set: DataSplit,
    model: lgb.LGBMClassifier,
    logger: Logger,
) -> lgb.Booster:
    """Run a grid search for a LightGBM model. The best parameters will be saved to a YAML file.

    Args:
        run_dir: Directory where results and metadata will be saved.
        config: Configuration object for the run.
        train_set: Data split object containing training data.
        val_set: Data split object containing validation data.
        model: Untrained LGBM model or similar API.
        logger: Logging object.
        save_model: If True, save the trained model to disk.

    Returns:
        lgb.Booster: Model post grid search.

    Raises:
        ValueError: If grid search is attempted without a specified parameter grid.
    """
    if config.model.param_grid is None:
        raise ValueError(
            "Provide a parameter grid in the config file in order to run a grid search."
        )

    logger.info("Performing a grid search for the best parameters...")
    grid = GridSearchCV(estimator=model, param_grid=config.model.param_grid, cv=5)
    grid.fit(train_set.x, train_set.y)
    model = grid.best_estimator_
    logger.info(grid.best_params_)
    with open(f"{run_dir}/grid_search.yaml", "w") as f:
        yaml.dump({f"best_params": grid.best_params_}, f)
    return model


def train_lgbm(
    run_dir: str,
    config: RunConfig,
    train_set: DataSplit,
    val_set: DataSplit,
    model: lgb.LGBMClassifier,
    logger: Logger,
    model_path: str,
) -> lgb.Booster:
    """Train a LightGBM model. Save it to the run directory.

    Early stopping is enabled if a validation set is provided and patience is not None in the config.

    Args:
        run_dir: Directory where results and metadata will be saved.
        config: Configuration object for the run.
        train_set: Data split object containing training data.
        val_set: Data split object containing validation data.
        model: Untrained LGBM model or similar API.
        logger: Logging object.

    Returns:
        lgb.Booster: Trained model.

    """

    logger.info(f"Training a {type(model)} model...")

    callbacks = []
    eval_set = None

    if val_set:
        eval_set = [(val_set.x, val_set.y)]

        if config.model.patience is not None:
            logger.info(f"Early stopping enabled w/ {config.model.patience} patience.")
            callbacks.append(lgb.early_stopping(stopping_rounds=config.model.patience))

    model = model.fit(
        train_set.x,
        train_set.y,
        eval_set=eval_set,
        callbacks=callbacks,
    )

    logger.info(
        f"Training finished. Model type trained: {type(model)}. Best iteration: {model.booster_.best_iteration}."
    )

    dump(model, model_path)
    logger.info(f"Model saved to .pkl file")

    return model
