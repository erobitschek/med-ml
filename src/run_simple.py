import os
from logging import Logger
from typing import Optional

import lightgbm as lgb
import numpy.typing as npt
from joblib import dump, load
from sklearn.linear_model import LogisticRegression as skLogisticRegression

from configs.config_scaffold import TrainMode
from configs.experiment_config_example import RunConfig
from eval import run_eval
from train import gridsearch_lgbm, train_lgbm, train_simple_model
from utils import setup_logger


def run_simple(
    config: RunConfig,
    run_dir: str,
    train_set: npt.ArrayLike,
    test_set: npt.ArrayLike,
    logger: Logger,
    train_mode: TrainMode = TrainMode.TRAIN,
    val_set: Optional[npt.ArrayLike] = None,
    model_eval: bool = True,
) -> None:
    """Trains, loads, and evaluates a simple model using scikit-learn.

    This function is intended for testing simple, non-neural network models.

    Args:
        config: RunConfiguration object containing runtime settings and model parameters.
        run_dir: Directory to save and retrieve models and logs.
        train_set: Training dataset object with attributes x and y.
        test_set: Test dataset object with attributes x and y.
        logger: Training log.
        train_mode: Either "train" for training or "load" for loading pre-trained model.
        val_set: Validation dataset object with attributes x and y.
        model_eval: If True, evaluate the model on the test set.
    """
    model_path = os.path.join(run_dir, "model.txt")

    if train_mode == TrainMode.TRAIN.name.lower():
        logger.info("Training sklearn framework of model...")

        if val_set is None:
            model = train_simple_model(
                run_dir=run_dir,
                x_train=train_set.x,
                y_train=train_set.y,
                model=skLogisticRegression(max_iter=config.model.epochs),
                param_grid=config.model.param_grid,
            )

        else:
            model = train_simple_model(
                run_dir=run_dir,
                x_train=train_set.x,
                y_train=train_set.y,
                model=skLogisticRegression(max_iter=config.model.epochs),
                param_grid=config.model.param_grid,
                x_val=val_set.x,
                y_val=val_set.y,
            )

        logger.info(f"Training finished. Model type trained: {type(model)}")
        dump(model, model_path)
        logger.info(f"Model saved to .joblib file")

    elif train_mode == TrainMode.LOAD.name.lower():
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")

        model = lgb.Booster(model_file=model_path)
        logger.info(f"Model loaded from previous training")

    elif train_mode == TrainMode.RESUME.name.lower():
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


def run_lgbm(
    config: RunConfig,
    run_dir: str,
    train_set: npt.ArrayLike,
    test_set: npt.ArrayLike,
    logger: Logger,
    train_mode: TrainMode = TrainMode.TRAIN,
    val_set: Optional[npt.ArrayLike] = None,
    model_eval: bool = True,
) -> None:
    """Trains, loads, or resumes an LGBM model based on the specified train_mode. Additionally,
    it evaluates the model on the test set if model_eval is True. If the grid search parameter 
    is True in the config, a grid search is run instead of training.

    Args:
        config: Configuration object for the run.
        run_dir: Directory where results and model will be saved or loaded.
        train_set: Training data.
        test_set: Test data for evaluation.
        logger: Logging object.
        train_mode: Either "train" for training or "load" for loading pre-trained model.
        val_set: Optional validation data.
        model_eval: If True, evaluates the model on the test set.

    Raises:
        FileNotFoundError: If trying to load a model that doesn't exist.
        NotImplementedError: If trying to resume a model which is not implemented yet.
    """
    model_path = os.path.join(run_dir, "model.pkl")

    if config.model.grid_search:
        model = gridsearch_lgbm(
            run_dir=run_dir,
            config=config,
            train_set=train_set,
            model=lgb.LGBMClassifier(**config.model.params),
            logger=logger,
        )

    if train_mode == TrainMode.TRAIN.name.lower():
        logger.info("Training lgbm framework of model...")

        model = train_lgbm(
            run_dir=run_dir,
            config=config,
            train_set=train_set,
            val_set=val_set,
            model=lgb.LGBMClassifier(**config.model.params),
            logger=logger,
            model_path=model_path
        )

    elif train_mode == TrainMode.LOAD.name.lower():
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")

        model = load(model_path)
        logger.info(f"Model loaded from previous training")

    elif train_mode == TrainMode.RESUME.name.lower():
        raise NotImplementedError("Resume training is not implemented yet.")

    if model_eval:
        logger.info(f"Predicting on test set...")
        probabilities = model.predict_proba(test_set.x)
        predictions = model.predict(test_set.x)

        run_eval(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=test_set.y,
            run_dir=run_dir,
            logger=logger,
        )
