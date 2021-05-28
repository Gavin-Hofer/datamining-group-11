import numpy as np
import pandas as pd
from typing import Tuple

# %% Constants
DATA_FILE = "./data/YearPredictionMSD.txt"
TRAIN_IDX = np.arange(0, 463715, dtype=int)
TEST_IDX = np.arange(463715, 463715 + 51630, dtype=int)


# %% Dataset class
class Dataset(object):
    def __init__(self, df: pd.DataFrame):
        """Initialize the Dataset object"""
        self.years = df.iloc[:, 0].to_numpy(dtype=int)
        self.data = df.iloc[:, 1:].to_numpy(dtype=float)


# %% functions
def load_msd() -> Tuple[Dataset, Dataset]:
    """Load the Million Song Dataset and split into training and test sets"""
    raw_df = pd.read_csv(DATA_FILE, header=None)
    df_train = raw_df.iloc[TRAIN_IDX, :]
    df_test = raw_df.iloc[TEST_IDX, :]
    train = Dataset(df_train)
    test = Dataset(df_test)
    return train, test
