"""
Implementation of LSTMs.

Supports classic LSTM.
"""

import torch
from torch import nn


class LSTM(nn.Module):
    """
    Implementation of a LSTM.

    Based on:
    @article{hochreiterLongShorttermMemory1997,
    title = {Long Short-Term Memory},
    author = {Hochreiter, Sepp and Schmidhuber, Jürgen},
    date = {1997-11-15},
    journaltitle = {Neural Computation},
    shortjournal = {Neural Computation},
    volume = {9},
    number = {8},
    pages = {1735--1780},
    issn = {0899-7667},
    doi = {10.1162/neco.1997.9.8.1735},
    }

    Args:
        nn (_type_): Module
    """

    def __init__(
        self,
        num_features: int = 1,
        hidden_units: int = 32,
        num_layers: int = 1,
        batch_size: int = 128,
        bias: bool = True,
        seq_length_input: int = 288,
        seq_length_output: int = 96,
        dropout: float = 0,
    ):
        """
        LSTM network.

        Simple LSTM followed by an output layer.

        Args:
            num_features (int, optional): number of features. Defaults to 1.
            hidden_units (int, optional): number of hidden units. Defaults to 32.
            num_layers (int, optional): number of layers in LSTM stack. Defaults to 1.
            batch_size (int, optional): number of tensors in batch. Defaults to 128.
            bias (bool, optional): add bias term. Defaults to True.
            seq_length_input (int, optional): length of input. Defaults to 288.
            seq_length_output (int, optional): length of output. Defaults to 96.
            dropout (float, optional): degree of dropout in stacked LSTM. Defaults to 0.
        """
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_features = num_features
        self.hidden_units = hidden_units

        # num_layers similar to stacking LSTMs
        self.num_layers = num_layers

        self.seq_length_input = seq_length_input
        self.seq_length_output = seq_length_output

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.seq_length_input * self.hidden_units,
            out_features=self.seq_length_output,
        )
        self._init_hidden(batch_size)

    def _init_hidden(self, batch_size: int) -> None:
        """
        Initialize / flush hidden state of h_t and c_t with zeros.

        Args:
            batch_size (int): batch size of input tensor
        """
        self.hidden = (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_units, requires_grad=True
            ).to(self.device),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_units, requires_grad=True
            ).to(self.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processs input tensor in forward pass.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        batch_size, _, _ = x.size()

        # flush hidden state for every batch
        # see: https://bit.ly/3A0DaAO
        self._init_hidden(batch_size)

        # LSTM layer using output instead of hidden state
        # see: https://stackoverflow.com/a/48305882/5755604
        out, self.hidden = self.lstm(x, self.hidden)
        x = out.contiguous().view(batch_size, -1)

        # output layer
        x = self.linear(x)

        return x
