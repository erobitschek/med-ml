from configs.config_scaffold import *

config = RunConfig(
    run_name="test_logreg_sklearn",
    model=ModelConfig(
        name="logreg",
        learning_rate=0.001,
        batch_size=32,
        epochs=500,
        framework=ModelFramework.SKLEARN,
        max_iter=1000,
        param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse regularization strength
                    'penalty': ['l2']}  # L2 regularization
    ),
    dataset=DatasetConfig(
        name="synth_400pts",
        path="../data/raw/synth_med_data_400patients_05women_01target_015femalecodes_015malecodes.csv",
        split_ratios=SplitRatios(train=0.8, val=0.1, test=0.1),
        target="is_Female",
        feature_threshold=5,
    ),
)
