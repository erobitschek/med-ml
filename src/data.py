import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from typing import Optional, Union, List, Callable
import numpy.typing as npt
from sklearn.model_selection import train_test_split
import torch

Array = Union[np.ndarray, pd.DataFrame]


@dataclass
class DatasetMeta:
    ids: list
    feature_names: list


@dataclass
class Trn:
    X: Array
    y: Array
    feature_names: Array = None
    ndropped: int = None


@dataclass
class Val:
    X: Array
    y: Array
    feature_names: Array = None
    ndropped: int = None


@dataclass
class Tst:
    X: Array
    y: Array
    feature_names: Array = None
    ndropped: int = None


class torch_dataset(torch.utils.data.Dataset):
    """
    Wrapper around a torch Dataset. This can be passed to a DataLoader for training.
    """

    def __init__(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        dataset_name: str,
        transforms: List[Callable] = None,
    ):
        self.y = torch.tensor(y, dtype=torch.long)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.dataset_name = dataset_name
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_idx = self.X[idx]

        if self.transforms:
            X_final = self.transforms(X_idx)
        else:
            X_final = X_idx

        return X_final, self.y[idx]


def load_data(path: str, filter_cols: list = None):
    """
    Load CSV data from a given path, optionally filtering specific columns.

    Parameters
    ----------
    path : str
        Path to the CSV data source.
    filter_cols : list, optional
        List of columns to filter. If None, all columns are loaded.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    if filter_cols:
        return pd.read_csv(path)[filter_cols]
    else:
        return pd.read_csv(path)


def get_targetdf(data: pd.DataFrame):
    """
    Extract and encode gender information from data.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing 'ID' and 'SEX' columns.

    Returns
    -------
    pd.DataFrame
        A dataframe with one-hot encoding for gender.
    """
    pt_sex = data.drop_duplicates(subset="ID").reset_index(drop=True)[["ID", "SEX"]]
    pt_sex["is_Female"] = pt_sex["SEX"].map({"Female": 1, "Male": 0})
    pt_sex["is_Male"] = pt_sex["SEX"].map({"Female": 0, "Male": 1})
    pt_sex.set_index("ID", inplace=True)
    return pt_sex


def filter_codes_by_overall_freq(
    data: pd.DataFrame, code_col: str = "CODE", threshold: int = 0
):
    """
    Filter codes based on their frequency.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing codes.
    code_col : str
        Column name containing codes.
    threshold : int
        Minimum frequency for code to be retained.

    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        Codes that pass the threshold and the filtered data.
    """
    unique_codes = data[code_col].nunique()
    code_counts = data[code_col].value_counts()
    codes_to_keep = code_counts[code_counts > threshold].index
    filtered_data = data[data[code_col].isin(codes_to_keep)]
    filtered_data.reset_index(inplace=True, drop=True)
    print(f"Out of {unique_codes}, {len(codes_to_keep)} pass the threshold.")
    return codes_to_keep, filtered_data


def encode_codes(df: pd.DataFrame):
    """
    One-hot encode 'CODE' column of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'CODE' column.

    Returns
    -------
    pd.DataFrame
        One-hot encoded dataframe.
    """
    one_hot = (
        pd.get_dummies(df, columns=["CODE"], prefix="", prefix_sep="")
        .groupby("ID")
        .max()
    )
    one_hot = one_hot.drop(columns="SEX", errors="ignore")
    return one_hot


def encode_codes_with_counts(df: pd.DataFrame):
    """
    One-hot encode 'CODE' column and return count of each code.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'CODE' column.

    Returns
    -------
    pd.DataFrame
        Count of each one-hot encoded code.
    """
    one_hot = pd.get_dummies(df, columns=["CODE"], prefix="", prefix_sep="")
    code_counts = one_hot.groupby("ID").sum()
    code_counts = code_counts.drop(columns="SEX", errors="ignore")
    return code_counts


def get_X_y(
    data: pd.DataFrame,
    target: str = "is_Female",
    threshold: int = 0,
    encoding: str = "binary",
):
    """
    Extract features and target variables based on encoding method.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    target : str
        Name of the target variable.
    threshold : int
        Frequency threshold for filtering codes.
    encoding : str
        Encoding method, either "binary" or "counts".

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (X) and target (y) data.
    """
    codes_to_keep, filtered_data = filter_codes_by_overall_freq(
        data, code_col="CODE", threshold=threshold
    )
    pt_sex = get_targetdf(filtered_data)
    if encoding == "binary":
        print("Encoding feature presence.")
        features = encode_codes(filtered_data).astype(int)
    elif encoding == "counts":
        print("Encoding feature counts.")
        features = encode_codes_with_counts(filtered_data).astype(int)

    # assumes these are in the same order/share the same index!
    X = features
    y = pt_sex[target]

    return X, y


def df_to_array(df: pd.DataFrame, feature_array=True):
    """
    Convert a dataframe to an array and optionally return metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    feature_array : bool
        If True, return metadata for dataset.

    Returns
    -------
    Tuple[np.array, Optional[DatasetMeta]]
        Data array and optional metadata.
    """
    data_array = df.values
    if feature_array:
        meta = DatasetMeta(ids=df.index.tolist(), feature_names=df.columns.tolist())
        return data_array, meta
    return data_array


def save_vars_to_pickle(run_folder: str, items: list, names: list):
    """
    Save variables to pickle files in a given folder.

    Parameters
    ----------
    run_folder : str
        Folder path to save pickle files.
    items : list
        List of items to save.
    names : list
        Names for the pickle files.

    Returns
    -------
    None
    """
    for i in range(len(items)):
        with open(f"{run_folder}/{names[i]}.pkl", "wb") as f:
            pickle.dump(items[i], f)


def split_data(
    X: Array,
    y: Array,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 3,
):
    """
    Split data into training, testing, and optionally validation sets.

    Parameters
    ----------
    X : np.array
        Feature data.
    y : np.array
        Target data.
    train_size : float
        Proportion of data to use for training.
    val_size : float
        Proportion of data to use for validation.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Seed for random number generator.

    Returns
    -------
    Tuple[Trn, Tst, Optional[Val]]
        Data splits for training, testing, and optionally validation.
    """
    assert (
        train_size + val_size + test_size == 1.0
    ), "The sum of split ratios should be 1.0"
    if val_size == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        train = Trn(X=X_train, y=y_train)
        test = Tst(X=X_test, y=y_test)

        return train, test
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1 - train_size, random_state=random_state
        )
        ratio_val = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - ratio_val, random_state=random_state
        )

        train = Trn(X=X_train, y=y_train)
        test = Tst(X=X_test, y=y_test)
        val = Val(X=X_val, y=y_val)

        return train, test, val


def get_dataloaders(
    dataset: str, train: Trn, test: Tst, batch_size: int = 32, val: Optional[Val] = None
) -> torch.utils.data.DataLoader:
    """
    Create dataloaders for training, testing, and optionally validation sets.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    train : Trn
        Training data.
    test : Tst
        Testing data.
    batch_size : int
        Size of batches for dataloading.
    val : Optional[Val]
        Validation data.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]
        Dataloaders for training, testing, and optionally validation.
    """
    ds_train = torch_dataset(X=train.X, y=train.y, dataset_name=dataset)
    ds_test = torch_dataset(X=test.X, y=test.y, dataset_name=dataset)

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size)

    if val:
        ds_val = torch_dataset(X=val.X, y=val.y, dataset_name=dataset)
        val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
        return train_loader, test_loader, val_loader

    else:
        return train_loader, test_loader
