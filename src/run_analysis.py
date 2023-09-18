import argparse
import sys
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Main pipeline for model processing.")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--setup", action="store_true", help="Execute setup related code."
    )
    parser.add_argument(
        "--get_data",
        action="store_true",
        help="Execute data loading and preprocessing code.",
    )
    parser.add_argument(
        "--model_train", action="store_true", help="Execute model training code."
    )
    parser.add_argument("--model_load", action="store_true", help="Load trained model.")
    parser.add_argument(
        "--model_eval",
        action="store_true",
        help="Execute model prediction code and evaluation code.",
    )
    parser.add_argument(
        "--load_eval_summary", action="store_true", help="Load model summary."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config

    from joblib import dump, load
    import pandas as pd
    from sklearn.linear_model import LogisticRegression as skLogisticRegression
    import torch
    import torch.nn as nn
    from utils import set_seed, get_run_dir, setup_logger, get_training_dir, load_model
    from models import torchLogisticRegression
    from train import train_simple_model, train_pytorch_model
    from predict import predict_from_torch, save_predictions_to_file
    from vis import plot_loss
    from eval import evaluate_predictions, save_evaluation_summary

    if args.setup:
        # resume_training = config.resume_training # TODO: add this functionality
        set_seed()
        print(config.run_name, config.model.name)

        run_dir = get_run_dir(
            run_name=config.run_name,
            dataset_name=config.dataset.name,
            model_name=config.model.name,
        )

        logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")

    if args.get_data:
        from data import (
            load_data,
            get_X_y,
            split_data,
            df_to_array,
            save_vars_to_pickle,
            get_dataloaders,
        )

        logger.info(f"Loading data")
        raw = load_data(config.dataset.path, filter_cols=["ID", "CODE", "SEX"])
        X, y = get_X_y(
            raw,
            target=config.dataset.target,
            threshold=config.dataset.feature_threshold,
            encoding=config.dataset.encoding,
        )
        logger.info(f"Data loaded")
        logger.info(f"Converting target df to array")

        y = df_to_array(y, feature_array=False)
        X, meta = df_to_array(X, feature_array=True)

        logger.info(f"Converting feature df to array")
        logger.info(f"Extracting row, column metadata from feature array")
        logger.info(f"Saving metadata to {run_dir}")

        save_vars_to_pickle(
            run_dir,
            items=[meta.ids, meta.feature_names],
            names=["individual_ids", "feature_names"],
        )

        split_ratios = config.dataset.split_ratios
        logger.info(f"The split ratios for the dataset are: {split_ratios}")
        train_set, test_set, val_set = split_data(
            X,
            y,
            train_size=split_ratios["train"],
            val_size=split_ratios["val"],
            test_size=split_ratios["test"],
        )

        logger.info(f"Training dataset shape: {train_set.X.shape}")
        logger.info(f"Validation dataset shape: {val_set.X.shape}")
        logger.info(f"Testing dataset shape: {test_set.X.shape}")

        logger.info(f"Getting dataloaders for modeling with pytorch:")
        train_loader, test_loader, val_loader = get_dataloaders(
            dataset=config.dataset.name,
            train=train_set,
            test=test_set,
            val=val_set,
            batch_size=config.model.batch_size,
        )

    if args.model_train:
        if config.model.implementation == "sklearn":
            logger.info("Training sklearn implementation of model...")
            model = train_simple_model(
                x_train=train_set.X,
                y_train=train_set.y,
                x_test=test_set.X,
                y_test=test_set.y,
                model=skLogisticRegression(max_iter=1000),
                param_grid={"C": [1, 10, 50, 100, 1000]},
            )
            logger.info(f"Training finished. Model type trained: {type(model)}")
            dump(
                model,
                f"{run_dir}/{config.model.name}_{config.model.implementation}_model.joblib",
            )
            logger.info(f"Model saved to .joblib file")

        elif config.model.implementation == "pytorch":
            # set up training directory and logger for more complex model
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

            # get data loaders for pytorch model
            train_loader, test_loader, val_loader = get_dataloaders(
                dataset=config.dataset.name,
                train=train_set,
                test=test_set,
                val=val_set,
                batch_size=config.model.batch_size,
            )

            input_dim = train_set.X.shape[
                1
            ]  # For example: X_train.shape[1] (Number of features)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torchLogisticRegression(input_dim=input_dim).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.model.learning_rate
            )
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

    if args.model_load:
        if config.model.implementation == "sklearn":
            model = load(
                f"{run_dir}/{config.model.name}_{config.model.implementation}_model.joblib"
            )
        elif config.model.implementation == "pytorch":
            input_dim = train_set.X.shape[1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torchLogisticRegression(input_dim=input_dim).to(device)
            model = load_model(run_dir, model=model)  # add weights to model

    if args.model_eval:
        logger.info(f"Predicting on test set...")
        if config.model.implementation == "sklearn":
            predictions, probabilities = (
                model.predict(test_set.X),
                model.predict_proba(test_set.X)[:, 1],
            )  # this assumes binary classification
        elif config.model.implementation == "pytorch":
            device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if args.load_eval_summary:
        print("Loading evaluation summary as eval_sum")
        eval_sum = pd.read_csv(f"{run_dir}/evaluation_summary.txt", sep=":")
        print(eval_sum)


if __name__ == "__main__":
    sys.exit(main())
