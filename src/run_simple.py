import os
from joblib import dump, load
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from utils import set_seed, setup_logger
from train import train_simple_model
from predict import save_predictions_to_file
from eval import evaluate_predictions, save_evaluation_summary
from configs.experiment_config_example import Config
import numpy.typing as npt
from typing import Optional


def run_simple(
    config: Config,
    run_dir: str,
    train_set: npt.ArrayLike,
    test_set: npt.ArrayLike,
    train_mode: str,
    val_set: Optional[npt.ArrayLike] = None,
    model_eval: bool = True,
):
    """
    Trains, loads, and evaluates a simple model using scikit-learn.

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
    - AssertionError: If model file is not found when train_mode is "load".

    Notes:
    - This function is intended for models using the scikit-learn library.
    """
    set_seed()
    logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")

    if train_mode == "train":  # FUTURE: implement 'resume' option
        logger.info("Training sklearn implementation of model...")
        model = train_simple_model(
            run_dir=run_dir,
            x_train=train_set.X,
            y_train=train_set.y,
            x_test=test_set.X,
            y_test=test_set.y,
            model=skLogisticRegression(max_iter=1000),
            param_grid=config.model.param_grid,
            x_val=val_set.X,
            y_val=val_set.y,
        )
        logger.info(f"Training finished. Model type trained: {type(model)}")
        dump(
            model,
            f"{run_dir}/{config.model.name}_{config.model.implementation}_model.joblib",
        )
        logger.info(f"Model saved to .joblib file")

    elif train_mode == "load":
        assert os.path.exists(
            f"{run_dir}/{config.model.name}_{config.model.implementation}_model.joblib"
        ), "Model file not found"
        model = load(
            f"{run_dir}/{config.model.name}_{config.model.implementation}_model.joblib"
        )
        logger.info(f"Model loaded from previous training")

    if model_eval:
        logger.info(f"Predicting on test set...")
        predictions, probabilities = (
            model.predict(test_set.X),
            model.predict_proba(test_set.X)[:, 1],
        )  # this assumes binary classification

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
