import os
from typing import Optional

import numpy.typing as npt
from joblib import dump, load
from sklearn.linear_model import LogisticRegression as skLogisticRegression

from configs.experiment_config_example import RunConfig
from eval import run_eval
from train import train_simple_model
from utils import setup_logger


def run_simple(
    config: RunConfig,
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
    - config (Config): RunConfiguration object containing runtime settings and model parameters.
    - run_dir (str): Directory to save and retrieve models and logs.
    - train_set (DataSet): Training dataset object with attributes x and y.
    - test_set (DataSet): Test dataset object with attributes x and y.
    - train_mode (str): Either "train" for training or "load" for loading pre-trained model.
    - val_set (DataSet): Validation dataset object with attributes x and y. (optional)
    - model_eval (bool): If True, evaluate the model on the test set.

    Returns:
    - None

    Notes:
    - This function is intended for testing simple, non-deep learning models.
    """
    logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")
    model_path = f"{run_dir}/{config.model.name}_{config.model.framework}_model.joblib"

    if train_mode == "train":
        logger.info("Training sklearn framework of model...")

        if val_set is None:
            model = train_simple_model(
                run_dir=run_dir,
                x_train=train_set.x,
                y_train=train_set.y,
                x_test=test_set.x,
                y_test=test_set.y,
                model=skLogisticRegression(max_iter=config.model.max_iter),
                param_grid=config.model.param_grid,
            )

        else:
            model = train_simple_model(
                run_dir=run_dir,
                x_train=train_set.x,
                y_train=train_set.y,
                x_test=test_set.x,
                y_test=test_set.y,
                model=skLogisticRegression(max_iter=config.model.max_iter),
                param_grid=config.model.param_grid,
                x_val=val_set.x,
                y_val=val_set.y,
            )

        logger.info(f"Training finished. Model type trained: {type(model)}")
        dump(model, model_path)
        logger.info(f"Model saved to .joblib file")

    elif train_mode == "load":
        if os.path.exists(model_path):
            model = load(model_path)
            logger.info(f"Model loaded from previous training")
        else:
            raise FileNotFoundError("Model file not found")

    elif train_mode == "resume":
        raise NotImplementedError("Resume training is not implemented yet.")

    if model_eval:
        logger.info(f"Predicting on test set...")
        predictions, probabilities = (
            model.predict(test_set.x),
            model.predict_proba(test_set.x)[:, 1],
        )  # this assumes binary classification

        run_eval(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=test_set.y,
            run_dir=run_dir,
            logger=logger,
        )
