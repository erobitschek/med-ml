import datetime
import importlib.util
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from configs.config_scaffold import ModelType, RunConfig


def load_config(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    return config


def set_seed():
    """Make the execution deterministic before model training."""

    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def remove_dir_contents(dir_path: str) -> None:
    """Removes all contents from a specified directory."""
    print(f"Removing contents of {dir_path}")
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def get_path(
    dataset_name: str, model_name: str, run_name: str, training: bool = False
) -> str:
    """Constructs a directory path based on dataset name, model name, and run name.

    Args:
        dataset_name: Name of the dataset.
        model_name: Name of the model.
        run_name: Name of the analysis run.
        training: If True, appends a "training" subfolder to the path.

    Returns:
        Constructed directory path.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    path = os.path.join(
        parent_directory, "out", "results", dataset_name, model_name, run_name
    )
    if training:
        path = os.path.join(path, "training")
    return path


def setup_output_dir(
    dataset_name: str, model_name: str, run_name: str, training: bool = False
) -> str:
    """Sets up the output directory. If directory already exists, prompts user for overwrite. If it doesn't, creates it.

    Args:
        dataset_name: Name of the dataset.
        model_name: Name of the model.
        run_name: Name of the run.
        training: If True, appends a "training" subfolder to the path.

    Returns:
        Path to the set up directory.
    """

    path = get_path(dataset_name, model_name, run_name, training=training)

    if os.path.exists(path):
        yn = input(f"Warning: Run already exists. Overwrite it? [y/N]")
        if yn.lower() == "y":
            remove_dir_contents(path)
        else:
            print("Aborting run.")
            sys.exit(1)

    else:
        Path(path).mkdir(parents=True)

    return path


# TODO: This function may be redundant / the flow it enables may be redundant.
def setup_training_dir(
    dataset_name: str, model_name: str, run_name: str, train_mode: str
) -> str:
    """Sets up the training directory based on the training mode.

    Args:
        dataset_name: Name of the dataset.
        model_name: Name of the model.
        run_name: Name of the run.
        train_mode: Mode of training ('train', 'resume', or 'load').

    Returns:
        Path to the training directory.
    """
    if train_mode == "train":
        path = setup_output_dir(dataset_name, model_name, run_name, training=True)

    elif train_mode in ["resume", "load"]:
        path = get_path(dataset_name, model_name, run_name, training=True)
        if not os.path.exists(path):
            raise FileNotFoundError("Training directory {path} not found")

    return path


def setup_logger(run_folder: str, log_file: str = "run.log", level=logging.INFO):
    """Configures and returns a logger to log messages to the console and a file."""

    log_file_path = os.path.join(run_folder, log_file)

    # Define the logger and set the logging level (e.g., DEBUG, INFO, ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized on {datetime.datetime.now()}")
    logger.info("Starting logging...")

    return logger


def save_model(model: nn.Module, run_folder: str, only_weights: bool = True) -> None:
    """Saves a PyTorch model to a specified directory.

    Args:
        model: PyTorch model to be saved.
        run_folder: Directory where the model should be saved.
        only_weights: If True, only saves model weights. Otherwise, saves the entire model.
    """
    if only_weights:
        torch.save(model.state_dict(), f"{run_folder}/weights.pth")
    else:
        torch.save(model, f"{run_folder}/model.pth")
        torch.save(model.state_dict(), f"{run_folder}/weights.pth")


def load_model(run_folder: str, model: Optional[nn.Module] = None) -> nn.Module:
    """Loads a PyTorch model or its weights from a specified directory.

    Args:
        run_folder: Directory from which the model or its weights should be loaded.
        model: If provided, updates this model's weights. If not provided, a full model is loaded.

    Returns:
        Loaded model.
    """
    if model:
        if not os.path.exists(f"{run_folder}/weights.pth"):
            raise FileNotFoundError("Weights file not found")
        model.load_state_dict(torch.load(f"{run_folder}/weights.pth"))
        return model
    else:
        if not os.path.exists(f"{run_folder}/model.pth"):
            raise FileNotFoundError("Model file not found")
        return torch.load(f"{run_folder}/model.pth")


def init_model(
    config: RunConfig,
    train_loader: torch.utils.data.dataloader.DataLoader,
    n_classes: int = 2,
) -> Tuple[Callable, torch.optim.Optimizer, torch.nn.modules.loss._Loss]:
    """Initialize and return a machine learning model, optimizer, and loss criterion based on the configuration.
    Note: Ensure all model constructors can accept the same or similar arguments

    Args:
        config: Configuration settings for the run, containing model details, optimizer type, loss criterion,
                learning rate, etc.
        train_loader: DataLoader for the training dataset. Used to infer input dimensions for the model.
        n_classes: Number of target classes in the dataset.

    Returns:
        Tuple containing the initialized model, optimizer, and loss criterion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = config.model.model_type

    if model_class in {
        ModelType.LOGREG_TORCH,
        ModelType.SIMPLE_CNN,
        ModelType.SIMPLE_RNN,
    }:
        input_dim = (
            train_loader.dataset.x.shape[1]
            if model_class == ModelType.LOGREG_TORCH
            else train_loader.dataset.x.shape
        )
        print(input_dim)
        model = model_class(input_dim=input_dim, n_class=n_classes).to(device)

    else:
        raise ValueError(f"Unsupported model: {model_class}")

    optimizer_class = config.model.optimizer
    optimizer = optimizer_class(model.parameters(), lr=config.model.learning_rate)

    criterion = config.model.loss_criterion

    print(f"Model type is: {model.__class__.__name__}")
    print(f"Optimizer type is: {optimizer.__class__.__name__}")
    print(f"Loss criterion is: {criterion().__class__.__name__}")

    return model, optimizer, criterion


def init_simple_model(
    config: RunConfig,
) -> Callable:
    """Initialize and return a machine learning model based on the configuration.

    Args:
        config: Configuration settings for the run, containing model details, optimizer type, loss criterion,
                learning rate, etc.
    Returns:
        The initialized model
    """
    model_class = config.model.model_type

    if model_class == ModelType.LOGREG_SKLEARN:
        model = model_class(max_iter=config.model.epochs)

    elif model_class == ModelType.LGBM_CLASSIFIER:
        model = model_class(**config.model.params)

    else:
        raise ValueError(f"Unsupported simple model: {model_class}")

    print(f"Model type is: {model.__class__.__name__}")

    return model
