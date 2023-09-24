from configs.config_scaffold import *

config = RunConfig(
    run_name="test_logreg_torch",
    model=ModelConfig(
        name="logreg",
        learning_rate=0.005,
        batch_size=32,
        epochs=500,
        framework=ModelFramework.PYTORCH,
        patience=50,
        param_grid= None,
    ),
    dataset=DatasetConfig(
        name="synth_400pts",
        path="../data/raw/synth_med_data_400patients_05women_01target_015femalecodes_015malecodes.csv",
        split_ratios=SplitRatios(0.8, 0.1, 0.1),
        target="is_Female",
        feature_threshold=5,
    ),
)
