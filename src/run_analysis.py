import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Main pipeline for model processing.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--data_state",
        default="raw",
        help="Execute data loading and preprocessing code.",
    )
    parser.add_argument(
        "--train_mode",
        default="train", # can be "load" too
        help="Whether to train the model or load it.",
    )
    parser.add_argument(
        "--model_eval",
        default=True,
        help="Execute model prediction code and evaluation code.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from utils import load_config, set_seed, get_run_dir, setup_logger

    config = load_config(args.config)
    set_seed()

    run_dir = get_run_dir(
        run_name=config.run_name,
        dataset_name=config.dataset.name,
        model_name=config.model.name,
    )

    logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")

    if (
        args.data_state == "raw"
    ):  # FUTURE: implement loading from 'preprocessed' and 'split' data states
        from data import (
            load_data,
            get_X_y,
            split_data,
            df_to_array,
            save_vars_to_pickle,
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

        if config.model.implementation == "sklearn":
            from run_simple import run_simple

            run_simple(
                config=config,
                run_dir=run_dir,
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                train_mode=args.train_mode,
                model_eval=args.model_eval,
            )
        if config.model.implementation == "pytorch":
            from run_torch import run_torch

            run_torch(
                config=config,
                run_dir=run_dir,
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                train_mode=args.train_mode,
                model_eval=args.model_eval,
            )


if __name__ == "__main__":
    sys.exit(main())