import argparse
import sys

from configs.config_scaffold import (DataState, ModelFramework, RunConfig,
                                     TrainMode)
from data import (df_to_array, get_x_y, load_data, prep_data_for_modelling,
                  save_vars_to_pickle, split_data_train_test,
                  split_data_train_test_val)
from run_simple import run_simple
from run_torch import run_torch
from utils import (get_path, load_config, set_seed, setup_logger,
                   setup_output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Main pipeline for model processing.")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--data_state",
        choices=[state.name.lower() for state in DataState],
        default=DataState.RAW.name.lower(),
        help="Execute data loading and preprocessing code.",
    )
    parser.add_argument(
        "--train_mode",
        choices=[mode.name.lower() for mode in TrainMode],
        default=TrainMode.TRAIN.name.lower(),
        help="Whether to train the model or load it (e.g. for prediction/evaluation).",
    )
    parser.add_argument(
        "--model_eval",
        default=True,
        help="Execute model prediction code and evaluation code.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    set_seed()

    if args.train_mode == "train":
        # either create or overwrite the run directory
        run_dir = setup_output_dir(
            run_name=config.run_name,
            dataset_name=config.dataset.name,
            model_name=config.model.name,
        )
    elif args.train_mode == "load":
        # load the path to the previous run directory
        run_dir = get_path(
            run_name=config.run_name,
            dataset_name=config.dataset.name,
            model_name=config.model.name,
            training=False,
        )
    else:
        raise ValueError(
            f"Got train mode {args.train_mode}; supported values are: train or load"
        )

    print(f"Run dir is: {run_dir}")

    logger = setup_logger(run_folder=run_dir, log_file=f"{config.run_name}_run.log")

    # TODO: implement loading from 'preprocessed' and 'split' data states
    if args.data_state == "raw":
        train_set, test_set, val_set = prep_data_for_modelling(
            config=config, run_dir=run_dir, data_state=args.data_state, logger=logger
        )
    else:
        raise ValueError(
            f"Got data state {args.data_state}; supported values are: raw."
        )
    if config.model.framework == ModelFramework.SKLEARN:
        run_simple(
            config=config,
            run_dir=run_dir,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            train_mode=args.train_mode,
            model_eval=args.model_eval,
        )
    elif config.model.framework == ModelFramework.PYTORCH:
        run_torch(
            config=config,
            run_dir=run_dir,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            train_mode=args.train_mode,
            model_eval=args.model_eval,
        )
    else:
        supported_frameworks = [f.name for f in ModelFramework]
        raise ValueError(
            f"Got the framework {config.model.framework}; supported values are: {supported_frameworks}."
        )


if __name__ == "__main__":
    sys.exit(main())
