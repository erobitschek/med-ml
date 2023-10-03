from configs.config_scaffold import *

config = RunConfig(
    run_name="test_lightgbm",
    model=ModelConfig(
        name="gbm",
        learning_rate=0.01,
        batch_size=32,
        epochs=2000,
        framework=ModelFramework.LIGHTGBM,
        patience=10,
        grid_search=False,
        params={
            "objective": "binary",
            "n_estimators": 1000,
            "metric": "auc",  # can add 'binary_logloss' for log loss as well
            "boosting_type": "gbdt",
            "num_leaves": 50,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "is_unbalance": False,
        },
        param_grid={
            "learning_rate": [0.01, 0.05],
            "num_leaves": [31, 50],
        },
    ),
    dataset=DatasetConfig(
        name="synth_10000pts",
        project="synth_med_data",
        path="../data/synth_med_data/raw/synth_med_data_10000patients_05women_01target_015femalecodes_015malecodes.csv",
        split_ratios=SplitRatios(0.8, 0.1, 0.1),
        target="is_Female",
        feature_threshold=10,
    ),
)
