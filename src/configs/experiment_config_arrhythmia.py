from configs.config_scaffold import *

config = RunConfig(
    run_name="arr_logreg_sklearn_v1",
    model=ModelConfig(
        name="logreg",
        learning_rate=0.01,
        epochs=10000,
        framework=ModelFramework.SKLEARN,
        param_grid={'C': [1, 10, 50, 100, 1000]},  # Inverse regularization strength
    ),
    dataset=DatasetConfig(
        name="mitbih",
        project="arrhythmia",
        state=DataState.SPLIT,
        path="../data/",
        split_ratios=SplitRatios(train=0.8, val=0.0, test=0.2),
        target="y_arrhythmia",
    ),
)
