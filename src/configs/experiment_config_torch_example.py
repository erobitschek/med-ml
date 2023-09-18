from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    learning_rate: float
    batch_size: int
    epochs: int
    implementation: str = "pytorch"
    dropout_rate: float = 0.5
    param_grid: dict = None


@dataclass
class DatasetConfig:
    name: str
    path: str
    target: str
    split_ratios: dict
    encoding: str = "binary"
    feature_threshold: int = 0
    shuffle: bool = True


@dataclass
class Config:
    run_name: str
    model: ModelConfig
    dataset: DatasetConfig
    resume_training: bool = False


config = Config(
    run_name="test_logreg_torch",
    model=ModelConfig(
        name="logreg",
        learning_rate=0.005,
        batch_size=32,
        epochs=500,
        implementation="pytorch",
        param_grid={"C": [1, 10, 50, 100, 1000]},
    ),
    dataset=DatasetConfig(
        name="synth_400pts",
        path="../data/raw/synth_med_data_400patients_05women_01target_015femalecodes_015malecodes.csv",
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        target="is_Female",
        feature_threshold=5,
    ),
)
