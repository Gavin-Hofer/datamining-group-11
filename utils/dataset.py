import numpy as np
import pandas as pd
from typing import Tuple

# %% Constants
DATA_FILE = "./data/YearPredictionMSD.txt"
TRAIN_IDX = np.arange(0, 463715, dtype=int)
TEST_IDX = np.arange(463715, 463715 + 51630, dtype=int)


# %% Dataset class
class Dataset(object):
    def __init__(self, data=None, labels=None):
        """Initialize the Dataset object"""
        self.labels = labels
        self.data = data

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        """Create a Dataset object from a pandas DataFrame"""
        obj = Dataset()
        obj.labels = df.iloc[:, 0].to_numpy(dtype=int)
        obj.data = df.iloc[:, 1:].to_numpy(dtype=float)
        return obj


# %% functions
def load_dataset(filename: str, train_size: int) -> Tuple[Dataset, Dataset]:
    """Load the Dataset and split into training and test sets"""
    raw_df = pd.read_csv(filename, header=None)
    df_train, df_test = _split_dataframe(raw_df, train_size)
    train = Dataset.from_dataframe(df_train)
    test = Dataset.from_dataframe(df_test)
    return train, test


def _split_dataframe(
    df: pd.DataFrame,
    split_idx: int
)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe at the specified index"""
    df1 = df.iloc[:split_idx]
    df2 = df.iloc[split_idx:]
    return df1, df2
