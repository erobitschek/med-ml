import numpy.typing as npt
import torch
import torch.nn as nn
import yaml

from configs.config_scaffold import TrainMode
from configs.experiment_config_example import RunConfig
from data import get_dataloaders
from eval import run_eval
from models import TorchLogisticRegression
from predict import predict_from_torch
from train import train_pytorch_model
from utils import load_model, setup_logger, setup_training_dir
from vis import plot_loss


def run_torch(
    config: RunConfig,
    run_dir: str,
    train_set: npt.ArrayLike,
    val_set: npt.ArrayLike,
    test_set: npt.ArrayLike,
    train_mode: str,
    model_eval: bool = True,
):
    """
    Trains, loads, and evaluates a model using PyTorch.

    Parameters:
    - config (Config): RunConfiguration object containing runtime settings and model parameters.
    - run_dir (str): Directory to save and retrieve models and logs.
    - train_set (DataSet): Training dataset object with attributes x and y.
    - val_set (DataSet): Validation dataset object with attributes x and y.
    - test_set (DataSet): Test dataset object with attributes x and y.
    - train_mode (str): Either "train" for training or "load" for loading pre-trained model.
    - model_eval (bool): If True, evaluate the model on the test set.

    Returns:
    - None

    Raises:
    - FileNotFoundError: If model weights are not found when train_mode is "load".

    Notes:
    - This function is intended for models using the PyTorch library.
    - Ensure the correct dependencies are imported when using different functionalities.
    """
    if val_set is None:
        raise ValueError(
            "A validation set is required for the PyTorch model implementation. Adjust split ratios."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")

    train_loader, test_loader, val_loader = get_dataloaders(
        dataset=config.dataset.name,
        train=train_set,
        test=test_set,
        val=val_set,
        batch_size=config.model.batch_size,
    )

    train_dir = setup_training_dir(
        dataset_name=config.dataset.name,
        model_name=config.model.name,
        run_name=config.run_name,
        train_mode=train_mode,
    )

    if train_mode == "train":  # TODO: implement 'resume' option
        input_dim = train_set.x.shape[1]  # (Number of features)
        model = TorchLogisticRegression(input_dim=input_dim).to(device)
        logger.info("Training pytorch implementation of model...")
        logger.info(f"Model type is: {type(model)}")

        train_pytorch_model(
            train_dir=train_dir,
            logger=logger,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer="adam",
            device=device,
            start_epoch=0,
            num_epochs=config.model.epochs,
            learning_rate=config.model.learning_rate,
            patience=config.model.patience,
            save_path=run_dir,
        )

        with open(f"{train_dir}/loss.yaml", "r") as loss_file:
            data = yaml.safe_load(loss_file)
        train_losses, val_losses = data["train_loss"], data["val_loss"]
        plot_loss(train_losses, val_losses, out_dir=train_dir)

    elif train_mode == "load":
        input_dim = train_set.x.shape[1]
        model = TorchLogisticRegression(input_dim=input_dim).to(device)
        model = load_model(run_dir, model=model)
        logger.info(f"Model weights loaded from previous training")

    elif train_mode == "resume":
        raise NotImplementedError("Resume training is not implemented yet.")

    if model_eval:
        logger.info(f"Predicting on test set...")
        predictions = predict_from_torch(
            model=model, data_loader=test_loader, device=device
        )
        probabilities = predict_from_torch(
            model=model,
            data_loader=test_loader,
            device=device,
            return_probabilities=True,
        )
        run_eval(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=test_set.y,
            run_dir=run_dir,
            logger=logger,
        )
