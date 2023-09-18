import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import random
import numpy as np
import numpy.typing as npt
import logging
import datetime
from typing import Optional


def set_seed():
    """
    Use this function to set a seed before model training.
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


def get_run_dir(dataset_name: str, model_name: str, run_name: str) -> Path:
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
    resume_training: bool
      if True, resume past training; if False, training from scratch

    Returns
    -------
    Path to run directory
    """

    path = (
        Path(__file__).parent.parent
        / "out"
        / "results"
        / dataset_name
        / model_name
        / run_name
    )

    try:
        path.mkdir(parents=True)
    except FileExistsError:
        yn = input(
            f"Warning: this run already exists. Do you want to overwrite it? [y/n]"
        )

        if yn.lower() == "y":
            for file in os.listdir(path):
                (path / file).unlink()

        else:
            print("Loading existing directory path")
            #sys.exit(1)

    return path


def get_training_dir(
    dataset_name: str, model_name: str, run_name: str, resume_training: bool = False
) -> Path:
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
    resume_training: bool
      if True, resume past training; if False, training from scratch

    Returns
    -------
    Path to run directory
    """

    path = (
        Path(__file__).parent.parent
        / "out"
        / "results"
        / dataset_name
        / model_name
        / run_name
        / "training"
    )

    try:
        path.mkdir(parents=True)
    except FileExistsError:
        if resume_training:
            # Case #1: resume training existing model
            yn = input(
                "Warning: this run already exists. Do you want to resume training? [y/n]"
            )

            if yn.lower() != "y":
                print("Abort run")
                sys.exit(1)

        else:
            # Case #2: rerun existing model from scratch (default case)
            yn = input(
                "Warning: this run already exists. Do you want to overwrite it? [y/n]"
            )

            if yn.lower() == "y":
                for file in os.listdir(path):
                    (path / file).unlink()

            else:
                print("Loading existing directory path")
                sys.exit(1)

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


def ensure_directory_exists(dir_path: str) -> None:
    """
    Ensure that the specified directory exists. If it doesn't, create it along with any necessary parent directories.

    Parameters
    ----------
    dir_path : str
        Path to the directory that needs to be checked and potentially created.

    Returns
    -------
    None

    Example
    -------
    >>> ensure_directory_exists('./plots/subfolder')
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


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
        assert os.path.exists(f"{run_folder}/weights.pth"), "Weights file not found"
        model.load_state_dict(torch.load(f"{run_folder}/weights.pth"))
        return model
    else:  # save the whole model and the weights too
        assert os.path.exists(f"{run_folder}/model.pth"), "Model file not found"
        loaded_model = torch.load(f"{run_folder}/model.pth")
        return loaded_model
