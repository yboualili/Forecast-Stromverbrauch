"""
Implementation of CNNs.

Supports classic CNN.
"""

import math

import torch
from torch import nn


class CNN(nn.Module):
    """
    CNN network.

    CNN with convolutional layer and pooling layer and final dropout layer.

    Based on:
    @book{goodfellowDeepLearning2016,
    title = {Deep Learning},
    author = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
    date = {2016},
    publisher = {{MIT Press}},
    keywords = {ObsCite}
    }
    """

    def __init__(
        self,
        num_features: int = 1,
        hidden_units: int = 32,
        batch_size: int = 128,
        kernel_size: int = 2,
        filters: int = 64,
        pool_size: int = 2,
        seq_length_input: int = 288,
        seq_length_output: int = 96,
        dropout: float = 0,
    ):
        """
        CNN network.

        CNN with Convolutional, Max Pooling, Dropout and Output layer.

        Args:
            num_features (int, optional): number of features. Defaults to 1.
            hidden_units (int, optional): number of hidden units. Defaults to 32.
            batch_size (int, optional): number of tensors in batch. Defaults to 128.
            kernel_size (int, optional): size of kernel. Defaults to 2.
            filters (int, optional): number of filters. Defaults to 64.
            pool_size (int, optional): size of window in pooling layayer. Defaults to 2.
            seq_length_input (int, optional): length of input. Defaults to 288.
            seq_length_output (int, optional): length of output. Defaults to 96.
            dropout (float, optional): degree of dropout. Defaults to 0.
        """
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.filters = filters
        self.seq_length_input = (seq_length_input,)
        self.seq_length_output = (seq_length_output,)
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.dropout = dropout

        # conv layers
        self.conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
        )

        # output dims for stride = 1
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        c_out = self.filters
        l_out = math.floor(self.seq_length_input[0] - (self.kernel_size - 1) - 1 + 1)

        self.pool = nn.MaxPool1d(kernel_size=self.pool_size)

        # output dims for stride = 1
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
        l_out = math.floor((l_out - 1 * (self.pool_size - 1) - 1) / self.pool_size + 1)

        # ffn layers
        self.hidden = nn.Linear(c_out * l_out, hidden_units)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(hidden_units, seq_length_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input tensor in forward pass.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        # permute for use in CNN (batch size, channels, seq len)
        x = x.permute((0, 2, 1))

        # convolutional layers
        x = self.relu(self.conv(x))
        x = self.pool(x)

        # ffn layers
        x = x.flatten(start_dim=1)
        x = self.relu(self.hidden(x))
        x = self.drop(x)
        x = self.out(x)

        return x
