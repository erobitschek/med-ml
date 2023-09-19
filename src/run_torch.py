import torch
import torch.nn as nn
import yaml
from utils import set_seed, setup_logger
from data import get_dataloaders
from models import torchLogisticRegression
from predict import predict_from_torch, save_predictions_to_file
from eval import evaluate_predictions, save_evaluation_summary
from configs.experiment_config_example import Config
import numpy.typing as npt

def run_torch(config: Config, run_dir: str, train_set: npt.ArrayLike, val_set: npt.ArrayLike, test_set: npt.ArrayLike, train_mode: str, model_eval: bool=True):
    """
    Trains, loads, and evaluates a model using PyTorch.

    Parameters:
    - config (Config): Configuration object containing runtime settings and model parameters.
    - run_dir (str): Directory to save and retrieve models and logs.
    - train_set (DataSet): Training dataset object with attributes X and y.
    - val_set (DataSet): Validation dataset object with attributes X and y.
    - test_set (DataSet): Test dataset object with attributes X and y.
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
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")

    train_loader, test_loader, val_loader = get_dataloaders(
        dataset=config.dataset.name,
        train=train_set,
        test=test_set,
        val=val_set,
        batch_size=config.model.batch_size,
    )

    if train_mode == "train":  # FUTURE: implement 'resume' option
        from utils import get_training_dir
        from train import train_pytorch_model
        from vis import plot_loss

        train_dir = get_training_dir(
            dataset_name=config.dataset.name,
            model_name=config.model.name,
            run_name=config.run_name,
            resume_training=config.resume_training,
        )
        print("The log is created at: ", train_dir)
        train_logger = setup_logger(
            run_folder=train_dir, log_file=f"{config.run_name}_train.log"
        )
        input_dim = train_set.X.shape[1]  # (Number of features)
        model = torchLogisticRegression(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
        logger.info("Training pytorch implementation of model...")
        logger.info(f"Model type is: {type(model)}")
        num_epochs = config.model.epochs
        learning_rate = config.model.learning_rate

        train_pytorch_model(
            train_dir=train_dir,
            train_logger=train_logger,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer="adam",
            device=device,
            start_epoch=0,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=50,
            save_best=True,
            save_path=run_dir,
        )

        with open(f"{train_dir}/loss.yaml", "r") as loss_file:
            data = yaml.safe_load(loss_file)
        train_losses, val_losses = data["train_loss"], data["val_loss"]
        plot_loss(train_losses, val_losses, out_dir=train_dir)

    elif train_mode == "load":
        from utils import load_model

        input_dim = train_set.X.shape[1]
        model = torchLogisticRegression(input_dim=input_dim).to(device)
        model = load_model(run_dir, model=model)  # add weights to model
        logger.info(f"Model loaded from previous training")

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

        logger.info(
            f"The first 5 predictions and their probabilities are: {predictions[:5], probabilities[:5]}"
        )
        logger.info(f"Saving predictions to {run_dir}")
        save_predictions_to_file(
            predictions=predictions,
            probabilities=probabilities,
            run_folder=run_dir,
            filename=f"predictions.txt",
        )
        logger.info(f"Evaluating model predictions")
        evaluation = evaluate_predictions(
            predictions=predictions, true_labels=test_set.y
        )
        save_evaluation_summary(
            true_labels=test_set.y,
            predicted_probs=probabilities,
            run_folder=run_dir,
            filename=f"evaluation_summary.txt",
        )
        logger.info(f"Saved evaluation summary.")
