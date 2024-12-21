"""
Loads data for neural networks.

Slices features and target dataFrame into sequences.
"""


from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for fitting timeseries models.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seq_length_input: int = 288,
        seq_length_prediction: int = 96,
    ):
        """
        Timeseries dataset holding data for models.

        Args:
            X (pd.DataFrame): features
            y (pd.Series): target
            seq_length_input (int, optional): seq. len. of input. Defaults to 288.
            seq_length_prediction (int, optional): seq. len. of target. Defaults to 96.
        """
        self.X = torch.tensor(X.astype(np.float32).values).float()
        self.y = torch.tensor(y.astype(np.float32).values).float()

        self.seq_length_prediction = seq_length_prediction
        self.seq_length_input = seq_length_input

        assert isinstance(
            self.seq_length_prediction, int
        ), "sequence length prediction must be integer."
        assert isinstance(
            self.seq_length_input, int
        ), "sequence length input must be integer."
        assert (
            self.seq_length_prediction > 0
        ), "min prediction length must be larger than 0"
        assert self.seq_length_input > 0, "min input length must be larger than 0"
        assert (
            self.seq_length_input + 1 <= self.y.shape[0]
        ), "sequence length input + 1 for prediction must be smaller than len(y)"

        # number of full and parital chunks of target
        self.length = (
            self.X.shape[0] - self.seq_length_input + (self.seq_length_prediction - 1)
        ) // self.seq_length_prediction

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample for model.

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: X and y for model
        """
        # calculate true index in tensor
        real_idx_X = idx * self.seq_length_prediction
        real_idx_y = real_idx_X + self.seq_length_input

        # slice tensor to subset
        X = self.X[real_idx_X : real_idx_X + self.seq_length_input]
        y = self.y[real_idx_y : real_idx_y + self.seq_length_prediction]

        # apply padding to target, if target series is incomplete
        if (real_idx_y + self.seq_length_prediction) >= len(self.y):
            padding = self.y[-1].repeat(self.seq_length_prediction - y.shape[0])
            y = torch.cat((y, padding), 0)
        return X, y
