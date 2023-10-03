import os
from logging import Logger
from typing import Optional

import lightgbm as lgb
import numpy.typing as npt
from joblib import dump, load
from sklearn.linear_model import LogisticRegression as skLogisticRegression

from configs.experiment_config_example import RunConfig
from eval import run_eval
from train import train_lgbm, train_simple_model
from utils import setup_logger


def run_simple(
    config: RunConfig,
    run_dir: str,
    train_set: npt.ArrayLike,
    test_set: npt.ArrayLike,
    logger: Logger,
    train_mode: str,
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
                model=skLogisticRegression(max_iter=config.model.epochs),
                param_grid=config.model.param_grid,
            )

        else:
            model = train_simple_model(
                run_dir=run_dir,
                x_train=train_set.x,
                y_train=train_set.y,
                x_test=test_set.x,
                y_test=test_set.y,
                model=skLogisticRegression(max_iter=config.model.epochs),
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


def run_lgbm(
    config: RunConfig,
    run_dir: str,
    train_set: npt.ArrayLike,
    test_set: npt.ArrayLike,
    logger: Logger,
    train_mode: str,
    val_set: Optional[npt.ArrayLike] = None,
    model_eval: bool = True,
) -> None:
    """
    Trains, loads, or resumes an LGBM model based on the specified train_mode. Additionally,
    it evaluates the model on the test set if model_eval is True.

    Args:
        config (RunConfig): Configuration object for the run.
        run_dir (str): Directory where results and model will be saved or loaded.
        train_set (npt.ArrayLike): Training data.
        test_set (npt.ArrayLike): Test data for evaluation.
        logger (Logger): Logging object.
        train_mode (str): Either "train" for training or "load" for loading pre-trained model.
        val_set (Optional[npt.ArrayLike], optional): Optional validation data.
        model_eval (bool, optional): If True, evaluates the model on the test set.

    Raises:
        FileNotFoundError: If trying to load a model that doesn't exist.
        NotImplementedError: If trying to resume a model which is not implemented yet.
    """
    model_path = f"{run_dir}/model.txt"

    if train_mode == "train":
        logger.info("Training lgbm framework of model...")

        model = train_lgbm(
            run_dir=run_dir,
            train_set=train_set,
            val_set=val_set,
            model=lgb.LGBMClassifier(**config.model.params),
            config=config,
            logger=logger,
        )

        logger.info(
            f"Training finished. Model type trained: {type(model)}. Best iteration: {model.booster_.best_iteration}."
        )
        model.booster_.save_model(model_path)
        logger.info(f"Model saved to .txt file")

    elif train_mode == "load":
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            logger.info(f"Model loaded from previous training")
        else:
            raise FileNotFoundError("Model file not found")

    elif train_mode == "resume":
        raise NotImplementedError("Resume training is not implemented yet.")

    if model_eval:
        logger.info(f"Predicting on test set...")
        probabilities = model.predict(
            test_set.x, num_iteration=model.booster_.best_iteration
        )
        predictions = probabilities > 0.5

        run_eval(
            predictions=predictions,
            probabilities=probabilities,
            true_labels=test_set.y,
            run_dir=run_dir,
            logger=logger,
        )
