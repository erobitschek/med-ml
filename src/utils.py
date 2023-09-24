import datetime
import importlib.util
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from configs.config_scaffold import TrainMode


def load_config(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    return config


def set_seed():
    """
    Use this function to make the execution deterministic before model training.
    """

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
    print(f"Removing contents of {dir_path}")
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def get_path(dataset_name: str, model_name: str, run_name: str, training: bool = False):
    """
    Function used to create the directory path for a given dataset - model - run_name configuration, or a subfolder of that directory for training.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    path = os.path.join(
        parent_directory, "out", "results", dataset_name, model_name, run_name)
    if training: 
        path = os.path.join(path, "training")
    return path


def setup_output_dir(dataset_name: str, model_name: str, run_name: str, training: bool = False) -> Path:
    """
    Function used to create an output directory for a given dataset - model - run_name configuration.

    Parameters
    ----------
    dataset_name: str
      Name of the dataset used for training.
    model_name: str
      Name of the model.
    run_name: str
      Name of the run.

    Returns
    -------
    Path to run directory
    """

    path = get_path(dataset_name, model_name, run_name, training=training)

    if os.path.exists(path):
        yn = input(f"Warning: Run already exists. Overwrite it? [y/N]")
        if yn.lower() == "y":
            remove_dir_contents(path)
        else:
            print('Aborting run.')
            sys.exit(1)

    else: 
        Path(path).mkdir(parents=True)

    return path


def setup_training_dir(
    dataset_name: str, model_name: str, run_name: str, train_mode: str
) -> str:
    if train_mode == "train":
        path = setup_output_dir(dataset_name, model_name, run_name, training = True)

    elif train_mode.isin(["resume", "load"]):
        path = get_path(dataset_name, model_name, run_name, training=True)
        if not os.path.exists(path):
            raise FileNotFoundError("Training directory {path} not found")

    return path

def setup_logger(run_folder: str, log_file: str = "run.log", level=logging.INFO):
    """
    Set up the logger.

    Args:
    - run_folder (str): Path to the folder where the logs should be saved.
    - log_file (str): Name of the file where the logs should be saved.
    - level (int): Logging level. By default, it's set to logging.INFO.

    Returns:
    - logger (logging.Logger): Configured logger.
    """
    log_file_path = os.path.join(run_folder, log_file)

    # Define the logger and set the logging level (e.g., DEBUG, INFO, ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create handlers for both the console and the log file
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    # Define the log format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized on {datetime.datetime.now()}")
    logger.info("Starting logging...")

    return logger


def save_model(model: nn.Module, run_folder: str, only_weights: bool = True):
    """
    Save a PyTorch model's state to a specified folder.
    This can save either just the weights or the whole model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to save.
    run_folder : str
        Folder path where the model's state will be saved.
    only_weights : bool, default=True
        If True, only the model's weights are saved.
        If False, the entire model and its weights are saved.

    Returns
    -------
    None
    """
    if only_weights:
        torch.save(model.state_dict(), f"{run_folder}/weights.pth")
    else:  # save the whole model and the weights too
        torch.save(model, f"{run_folder}/model.pth")
        torch.save(model.state_dict(), f"{run_folder}/weights.pth")


def load_model(run_folder: str, model: Optional[nn.Module] = None):
    """
    Load a PyTorch model's state from a specified folder.
    This can either load weights into an existing model or load an entire saved model.

    Parameters
    ----------
    run_folder : str
        Folder path from where the model's state will be loaded.
    model : Optional[nn.Module], default=None
        If provided, this model's weights are updated from the saved state.
        If not provided, a complete saved model is loaded.

    Returns
    -------
    nn.Module
        The loaded or updated PyTorch model.
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
