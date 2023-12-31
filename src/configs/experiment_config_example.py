from configs.config_scaffold import *

config = RunConfig(
    run_name="test_logreg_sklear_v2",
    model=ModelConfig(
        name="logreg",
        learning_rate=0.001,
        batch_size=32,
        epochs=1000,
        framework=ModelFramework.SKLEARN,
        param_grid={
            "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse regularization strength
            "penalty": ["l2"],
        },  # L2 regularization
    ),
    dataset=DatasetConfig(
        name="synth_400pts",
        project="synth_med_data",
        path="../data/synth_med_data/raw/synth_med_data_400patients_05women_01target_015femalecodes_015malecodes.csv",
        split_ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
        target="is_Female",
        class_names=["Male", "Female"],
        feature_threshold=5,
    ),
)
