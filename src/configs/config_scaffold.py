from dataclasses import dataclass, field
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ModelConfig:
    name: str
    learning_rate: float
    batch_size: int
    epochs: int = 500
    framework: ModelFramework = ModelFramework.SKLEARN
    dropout_rate: float = 0.5
    patience: int = None
    params: dict = None  # param dict for lgbm model
    param_grid: dict = None  # for grid search to find optimal model parameters
    grid_search: bool = False  # whether to perform the grid search

    # TODO: add post init that adds some parameters based on framework


@dataclass
class DatasetConfig:
    name: str
    project: str
    path: str
    target: str
    split_ratios: SplitRatios
    medcode_col: str = "CODE"
    id_col: str = "ID"
    encoding: FeatureEncoding = FeatureEncoding.BINARY
    feature_threshold: int = (
        0  # the minimum number of times a medical code occurs in the dataset
    )
    shuffle: bool = True
    raw_dir: str = field(init=False)
    processed_dir: str = field(init=False)
    split_dir: str = field(init=False)

    def __post_init__(self):
        # get directory paths for raw, processed and split data for the project
        self.raw_dir = f"../data/{self.project}/raw/"
        self.processed_dir = f"../data/{self.project}/processed/"
        self.split_dir = f"../data/{self.project}/split/"


@dataclass
class RunConfig:
    run_name: str
    model: ModelConfig
    dataset: DatasetConfig
    resume_training: bool = False
