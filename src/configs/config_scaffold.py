from dataclasses import dataclass, field, InitVar
from enum import Enum, auto
from typing import Callable, Optional, Type

import torch.nn as nn
import torch.optim as optim

from configs.models import ModelType

class ModelFramework(Enum):
    SKLEARN = auto()  # good for standard simple models
    PYTORCH = auto()  # for more complex nn models / more control over training process
    LIGHTGBM = auto()  # for gradient boosting models


class PyTorchLoss(Enum):
    BINARY_CROSS_ENTROPY = nn.BCELoss # commonly used for binary classification
    CROSS_ENTROPY = nn.CrossEntropyLoss # commonly used for multi-class classification
    MSE = nn.MSELoss # commonly used for regression


class PyTorchOptim(Enum):
    ADAM = optim.Adam 
    SGD = optim.SGD 
    RMS_PROP = optim.RMSprop 


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


class PermuteTransform:
    """Transform to permute the dimensions of a tensor. For use with pytorch models."""
    def __init__(self, *dims):
        self.dims = dims

    def __call__(self, tensor):
        return tensor.permute(*self.dims)


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
    batch_size: int = 32
    epochs: int = 500
    model_type: ModelType = ModelType.LOGREG_SKLEARN
    framework: ModelFramework = ModelFramework.SKLEARN
    loss_criterion: InitVar[Optional[str]] = "CROSS_ENTROPY"  # default value for pytorch models
    optimizer: InitVar[Optional[str]] = "ADAM"  # default value for pytorch models
    actual_loss_criterion: Optional[Type[nn.Module]] = field(default=None, init=False)
    actual_optimizer: Optional[Type[optim.Optimizer]] = field(default=None, init=False)
    data_transforms: list = field(default_factory=lambda: None) # list of transforms to apply to data
    dropout_rate: float = 0.5
    patience: int = None
    params: dict = None  # param dict for lgbm model
    param_grid: dict = None  # for grid search to find optimal model parameters
    grid_search: bool = False  # whether to perform the grid search

    def __post_init__(self, loss_criterion: str, optimizer: str):
        if self.framework == ModelFramework.PYTORCH:
            self.set_pytorch_components(loss_criterion, optimizer)
        else:
            # TODO: Implement logic for other frameworks if needed
            pass

    def set_pytorch_components(self, loss_criterion: str, optimizer: str):
        loss_class = PyTorchLoss[loss_criterion].value if loss_criterion else PyTorchLoss.CROSS_ENTROPY.value
        optimizer_class = PyTorchOptim[optimizer].value if optimizer else PyTorchOptim.ADAM.value

        object.__setattr__(self, 'actual_loss_criterion', loss_class) # this logic is needed to avoid dataclass immutability
        object.__setattr__(self, 'actual_optimizer', optimizer_class)


@dataclass
class DatasetConfig:
    name: str
    project: str
    path: str
    target: str
    split_ratios: SplitRatios
    state: DataState = DataState.RAW
    class_names: list = field(default_factory=lambda: ["0", "1"]) # assumes binary classification
    medcode_col: str = "CODE"
    id_col: str = "ID"
    encoding: FeatureEncoding = FeatureEncoding.BINARY
    feature_threshold: int = (
        0  # the minimum frequency of a medical code in the dataset
    )
    shuffle: bool = True
    raw_dir: str = field(init=False)
    processed_dir: str = field(init=False)
    split_dir: str = field(init=False)
    path_train: str = field(init=False)
    path_val: str = field(init=False)
    path_test: str = field(init=False)

    def __post_init__(self):
        # get directory paths for raw, processed and split data for the project
        self.raw_dir = f"../data/{self.project}/raw/"
        self.processed_dir = f"../data/{self.project}/processed/"
        self.split_dir = f"../data/{self.project}/split/"

        if self.state == DataState.SPLIT:
            self.path_train = f"{self.split_dir}{self.name}_train.csv"
            self.path_val = f"{self.split_dir}{self.name}_val.csv"
            self.path_test = f"{self.split_dir}{self.name}_test.csv"


@dataclass
class RunConfig:
    run_name: str
    model: ModelConfig
    dataset: DatasetConfig
    resume_training: bool = False
