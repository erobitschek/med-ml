from configs.config_scaffold import *

config = RunConfig(
    run_name="test_lightgbm",
    model=ModelConfig(
        name="gbm",
        learning_rate=0.005,
        batch_size=32,
        epochs=500,
        framework="lightgbm",
        param_grid= None,
    ),
    dataset=DatasetConfig(
        name="synth_10000pts",
        path="../data/raw/synth_med_data_10000patients_05women_01target_015femalecodes_015malecodes.csv",
        split_ratios=SplitRatios(0.8, 0.1, 0.1),
        target="is_Female",
        feature_threshold=5,
    ),
)
