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
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from configs.config_scaffold import RunConfig
from data import DataSplit
from utils import save_model


def train_simple_model(
    run_dir: str,
    x_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    model: Callable,
    param_grid: Optional[Dict] = None,
    x_val: Optional[npt.ArrayLike] = None,
    y_val: Optional[npt.ArrayLike] = None,
) -> Callable:
    """
    Trains a scikit-learn model using the provided data. If a parameter grid is provided, it performs hyperparameter
    selection using 5-fold cross-validation.

    Args:
        run_dir (str): Directory to save outputs.
        x_train (npt.ArrayLike): Training feature data.
        y_train (npt.ArrayLike): Training target data.
        x_test (npt.ArrayLike): Testing feature data.
        y_test (npt.ArrayLike): Testing target data.
        model (Callable): Untrained machine learning model.
        param_grid (Optional[Dict], optional): Parameters to search over for hyperparameter tuning. Defaults to None.
        x_val (Optional[npt.ArrayLike], optional): Validation feature data. Defaults to None.
        y_val (Optional[npt.ArrayLike], optional): Validation target data. Defaults to None.

    Returns:
        Callable: Trained scikit-learn model.

    The function also outputs model evaluation metrics on the test set for an initial overview of model performance.
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
    """
    Trains a PyTorch model using the provided data loaders. The function logs the training progress and loss. It also
    offers functionalities such as early stopping and model checkpointing.

    Args:
        model: PyTorch model class instance to be trained.
        train_loader: DataLoader object for training data.
        val_loader: DataLoader object for validation data.
        train_dir (str): Directory to save training artifacts.
        logger (logging.Logger): Logger for capturing training progress.
        device (str, optional): Device on which the model should be trained. Defaults to "cpu".
        criterion (optional): Loss function to use. Defaults to nn.BCELoss().
        optimizer (str, optional): Optimizer choice ("adam" or "sgd"). Defaults to "adam".
        num_epochs (int, optional): Total number of epochs for training. Defaults to 100.
        learning_rate (float): Learning rate for the optimizer.
        start_epoch (int, optional): Epoch to begin training. Useful for resuming training. Defaults to 0.
        patience (int, optional): Patience parameter for early stopping. Defaults to 20.
        checkpoint_freq (int, optional): Frequency for saving model checkpoints. Defaults to 10.
        save_path (str, optional): Directory path to save the best model weights. Defaults to train_dir.

    Returns:
        None
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


def train_lgbm(
    run_dir: str,
    config: RunConfig,
    train_set: DataSplit,
    val_set: DataSplit,
    model: Callable,
    logger: Logger,
    save_model: bool = True,
) -> lgb.Booster:
    """
    Train a LightGBM model with optional grid search. If grid search is enabled, the best parameters
    will be saved to a YAML file.

    Args:
        run_dir (str): Directory where results and metadata will be saved.
        config (RunConfig): Configuration object for the run.
        train_set (DataSplit): Data split object containing training data.
        val_set (DataSplit): Data split object containing validation data.
        model (Callable): Untrained LGBM model or similar API.
        logger (Logger): Logging object.
        save_model (bool, optional): If True, save the trained model to disk.

    Returns:
        lgb.Booster: Trained model.

    Raises:
        ValueError: If grid search is attempted without a specified parameter grid.
    """
    if config.model.grid_search:
        try:
            logger.info("Performing a grid search for the best parameters...")
            grid = GridSearchCV(
                estimator=model, param_grid=config.model.param_grid, cv=5
            )
            grid.fit(train_set.x, train_set.y)
            model = grid.best_estimator_
            logger.info(grid.best_params_)
            with open(f"{run_dir}/grid_search.yaml", "w") as f:
                yaml.dump({f"best_params": grid.best_params_}, f)
            return model
        except ValueError:
            logger.info(
                "Provide a parameter grid in the config file in order to run a grid search."
            )
            return

    else:
        logger.info(f"Training a {type(model)} model...")

        callbacks = []

        eval_set = None
        if val_set:
            eval_set = [(val_set.x, val_set.y)]

            if config.model.patience is not None:
                logger.info(
                    f"Early stopping enabled w/ {config.model.patience} patience ."
                )
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=config.model.patience)
                )

        model = model.fit(
            train_set.x,
            train_set.y,
            eval_set=eval_set,
            callbacks=callbacks,
        )

    return model
