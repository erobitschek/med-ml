import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from configs.config_scaffold import FeatureEncoding, RunConfig, SplitRatios

Array = Union[np.ndarray, pd.DataFrame]


@dataclass
class DatasetMeta:
    ids: list
    feature_names: list


@dataclass
class DataSplit:
    x: Array
    y: Array
    feature_names: Array = None
    ndropped: int = None


class TorchDataset(torch.utils.data.Dataset):
    """Extension of torch Dataset class, which can be passed to a DataLoader."""

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        dataset_name: str,
        transforms: List[Callable] = None,
    ):
        self.y = torch.tensor(y, dtype=torch.long)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.dataset_name = dataset_name
        self.transforms = transforms

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_idx = self.x[idx]

        if self.transforms:
            x_final = self.transforms(x_idx)
        else:
            x_final = x_idx

        return x_final, self.y[idx]


def load_data(path: str, filter_cols: list = None) -> pd.DataFrame:
    if filter_cols:
        return pd.read_csv(path)[filter_cols]
    else:
        return pd.read_csv(path)


def get_biosex_targetdf(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and encode biological sex information from data as a dataframe with one-hot encoding.

    Parameters
    ----------
    data : Input data containing 'ID' and 'SEX' columns.
    """
    pt_sex = data.drop_duplicates(subset="ID").reset_index(drop=True)[["ID", "SEX"]]
    pt_sex["is_Female"] = pt_sex["SEX"].map({"Female": 1, "Male": 0})
    pt_sex["is_Male"] = pt_sex["SEX"].map({"Female": 0, "Male": 1})
    pt_sex.set_index("ID", inplace=True)
    return pt_sex


def filter_codes_by_overall_freq(
    data: pd.DataFrame,
    code_col: str = "CODE",
    threshold: int = 0,
    write_kept_codes: bool = False,
    run_dir: str = None,
) -> pd.DataFrame:
    """
    Filter codes based on their frequency. Optionally writes the kept codes to a pickle file.

    Parameters
    ----------
    data : Data containing codes.
    code_col : Column name containing codes.
    threshold : Minimum frequency for code to be retained.
    write_kept_codes: Whether to write the kept codes to a pickle file.
    run_dir: Directory to write the pickle file to.
    """
    unique_codes = data[code_col].nunique()
    code_counts = data[code_col].value_counts()
    codes_to_keep = code_counts[code_counts > threshold].index
    filtered_data = data[data[code_col].isin(codes_to_keep)]
    filtered_data.reset_index(inplace=True, drop=True)
    if write_kept_codes:
        with open(f"{run_dir}/kept_codes.pkl", "wb") as f:
            pickle.dump(codes_to_keep, f)
    print(f"Out of {unique_codes}, {len(codes_to_keep)} pass the threshold.")
    return filtered_data


def encode_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode 'CODE' column of the dataframe.
    """
    one_hot = (
        pd.get_dummies(df, columns=["CODE"], prefix="", prefix_sep="")
        .groupby("ID")
        .max()
    )
    one_hot = one_hot.drop(columns="SEX", errors="ignore")
    return one_hot


def encode_codes_with_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode 'CODE' column and return count of each code.
    """
    one_hot = pd.get_dummies(df, columns=["CODE"], prefix="", prefix_sep="")
    code_counts = one_hot.groupby("ID").sum()
    code_counts = code_counts.drop(columns="SEX", errors="ignore")
    return code_counts


def get_x_y(
    data: pd.DataFrame,
    target: str = "is_Female",
    threshold: int = 0,
    encoding: str = "binary",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Filter the data based on code frequency, encode it, and extract features (x) and target (y) variables based on encoding method.

    Parameters
    ----------
    data : Input data.
    target : Name of the target variable.
    threshold : Frequency threshold for filtering codes.
    encoding : Encoding method, either "binary" or "counts".
    """
    filtered_data = filter_codes_by_overall_freq(
        data, code_col="CODE", threshold=threshold
    )
    target_df = get_biosex_targetdf(filtered_data)
    if encoding == FeatureEncoding.BINARY:
        print("Encoding feature presence.")
        features = encode_codes(filtered_data).astype(int)
    elif encoding == FeatureEncoding.COUNT:
        print("Encoding feature counts.")
        features = encode_codes_with_counts(filtered_data).astype(int)
    else: 
        raise ValueError(f"Encoding method {encoding} not supported.")

    assert features.index.equals(target_df.index), 'Feature and target var indices must match.'

    x = features
    y = target_df[target]

    return x, y


def df_to_array(x: pd.DataFrame, y: pd.Series) -> tuple[npt.ArrayLike, npt.ArrayLike, DatasetMeta]:
    """
    Convert x and y dataframes to arrays and return metadata for x.
    """
    x_values = x.values
    y_values = y.values
    meta = DatasetMeta(ids=x.index.tolist(), feature_names=x.columns.tolist())
    return x_values, y_values, meta


def save_vars_to_pickle(run_folder: str, data: Any, filename: str):
    with open(f"{run_folder}/{filename}.pkl", "wb") as f:
        pickle.dump(data, f)


def split_data_train_test(
    x: Array,
    y: Array,
    split_ratios: SplitRatios = SplitRatios(),
    random_state: int = 3,
) -> tuple[DataSplit, DataSplit]:
    """
    Split data into training and testing sets.
    """
    train_size, test_size = split_ratios.train, split_ratios.test

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    train = DataSplit(x=x_train, y=y_train)
    test = DataSplit(x=x_test, y=y_test)

    return train, test
    

def split_data_train_test_val(
    x: Array,
    y: Array,
    split_ratios: SplitRatios = SplitRatios(),
    random_state: int = 3,
) -> tuple[DataSplit, DataSplit, DataSplit]:
    """
    Split data into training, testing, and validation sets.
    """
    train_size, val_size, test_size = split_ratios.train, split_ratios.val, split_ratios.test

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=1 - train_size, random_state=random_state
    )
    ratio_val = val_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=1 - ratio_val, random_state=random_state
    )

    train = DataSplit(x=x_train, y=y_train)
    test = DataSplit(x=x_test, y=y_test)
    val = DataSplit(x=x_val, y=y_val)

    return train, test, val



def get_dataloaders(
    dataset: str, train: DataSplit, test: DataSplit, batch_size: int = 32, val: Optional[DataSplit] = None
) -> tuple[torch.utils.data.DataLoader, ...]:
    """
    Create dataloaders for training, testing, and optionally validation sets.
    """
    ds_train = TorchDataset(x=train.x, y=train.y, dataset_name=dataset)
    ds_test = TorchDataset(x=test.x, y=test.y, dataset_name=dataset)

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size)

    val_loader = None

    if val is not None:
        ds_val = TorchDataset(x=val.x, y=val.y, dataset_name=dataset)
        val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
    
    return train_loader, test_loader, val_loader

def prep_data_for_modelling(config: RunConfig, run_dir: str, data_state: str, logger: logging.Logger) -> tuple[DataSplit, DataSplit, Optional[DataSplit]]:
    """
    Uses the config to load the raw data, filter and preprocess it into arrays, save the metadata for the processed data, and split it into train, test, and optionally validation sets for modelling.

    Parameters
    ----------
    config : RunConfig object.
    run_dir : Directory to save metadata to.
    data_state : Whether to load raw or processed data.
    logger : Logger object.
    """
    logger.info(f"Loading {data_state} data")
    if not os.path.exists(config.dataset.path):
        raise FileNotFoundError(f"File {config.dataset.path} not found.")
    raw = load_data(config.dataset.path, filter_cols=["ID", "CODE", "SEX"])
    x, y = get_x_y(
        raw,
        target=config.dataset.target,
        threshold=config.dataset.feature_threshold,
        encoding=config.dataset.encoding,
    )
    logger.info(f"Data loaded")
    logger.info(f"Converting target df to array")

    x, y, meta = df_to_array(x, y)

    logger.info(f"Extracting row, column metadata from feature array")
    logger.info(f"Saving metadata to {run_dir}")

    save_vars_to_pickle(run_dir, meta.ids, "individual_ids")
    save_vars_to_pickle(run_dir, meta.feature_names, "feature_names")

    logger.info(
        f"The split ratios for the dataset are: {config.dataset.split_ratios}"
    )

    if config.dataset.split_ratios.val == 0:
        val_set = None
        train_set, test_set = split_data_train_test(
            x,
            y,
            split_ratios=config.dataset.split_ratios,
        )
        logger.info(
            f"Dataset shapes (train, test): {train_set.x.shape}, {test_set.x.shape}"
        )

    else:
        train_set, test_set, val_set = split_data_train_test_val(
            x,
            y,
            split_ratios=config.dataset.split_ratios,
        )
        logger.info(
            f"Dataset shapes (train, test, val): {train_set.x.shape}, {test_set.x.shape}, {val_set.x.shape}"
        )

    return train_set, test_set, val_set
