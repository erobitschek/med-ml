from dataclasses import dataclass
from enum import Enum, auto


class ModelFramework(Enum):
    SKLEARN = auto()  # good for standard simple models
    PYTORCH = auto()  # for more complex nn models / more control over training process
    LIGHTGBM = auto()  # for gradient boosting models


class FeatureEncoding(Enum):
    BINARY = auto()  # binary encoding
    COUNT = auto()  # count encoding


class TrainMode(Enum):
    TRAIN = auto()
    LOAD = auto()
    RESUME = auto()


class DataState(Enum):
    RAW = auto()  # data is in raw format
    PROCESSED = auto()  # data is in processed format
    SPLIT = auto()


@dataclass
class SplitRatios:
    """Must add to 1.0 val should be 0 if no validation set is desired."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self):
        # due to floating point precision, we round to 4 decimal places
        rounded_sum = round(self.train + self.val + self.test, 4)
        if not rounded_sum == 1.0:
            raise ValueError(
                f"Split ratios must add to 1.0. Currently: {self.train + self.val + self.test}"
            )


@dataclass
class ModelConfig:
    name: str
    learning_rate: float
    batch_size: int
    epochs: int
    framework: ModelFramework = ModelFramework.SKLEARN
    max_iter: int = 500
    dropout_rate: float = 0.5
    patience: int = 20
    param_grid: dict = None  # for grid search to find optimal model parameters


@dataclass
class DatasetConfig:
    name: str
    path: str
    target: str
    split_ratios: SplitRatios
    medcode_col: str = "CODE"
    id_col: str = "ID"
    encoding: FeatureEncoding = FeatureEncoding.BINARY
    feature_threshold: int = 0 # the minimum number of times a medical code occurs in the dataset
    shuffle: bool = True
    raw_dir: str = "../data/raw/"
    processed_dir: str = "../data/processed/"


@dataclass
class RunConfig:
    run_name: str
    model: ModelConfig
    dataset: DatasetConfig
    resume_training: bool = False
