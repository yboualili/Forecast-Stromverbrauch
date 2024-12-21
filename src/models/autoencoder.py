"""
Implementation of an Autoencoder.

Adds support for underfull Bottleneck encoder.
"""

from typing import Tuple

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Implementation of an Autoencoder.

    Inspired from:
    https://bit.ly/3xTcP54

    """

    def __init__(
        self,
        num_features: int = 4,
        bottleneck_capacity: int = 4,
        num_layers: int = 2,
        dropout: float = 0,
        activation: str = "ReLU",
    ):
        """
        Autoencoder.

        Autoencoder consists of encoder and decoder with hour-glass shape.
        Contains two dropout layers. Number of neurons is depend on the number of layers
        for the hidden layers and `bottleneck_capacity` for the bottleneck layer.

        Note:
        Autoencoder follows hourglass shape, thus increases capacity first, to learn
        richer representations.

        Example:
        input_dim = 736
        num_layers = 3
        bottleneck_capacity = 64

        Autoencoder(
        (encoder): Sequential(
            (0): Linear(in_features=736, out_features=767, bias=True)
            (1): ReLU()
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=767, out_features=453, bias=True)
            (4): ReLU()
            (5): Linear(in_features=453, out_features=184, bias=True)
            (6): ReLU()
            (7): Linear(in_features=184, out_features=64, bias=True)
        )
        (decoder): Sequential(
            (0): Linear(in_features=64, out_features=184, bias=True)
            (1): ReLU()
            (2): Linear(in_features=184, out_features=453, bias=True)
            (3): ReLU()
            (4): Linear(in_features=453, out_features=767, bias=True)
            (5): ReLU()
            (6): Dropout(p=0.5, inplace=False)
            (7): Linear(in_features=767, out_features=736, bias=True)
        )
        )
        Args:
            num_features (int, optional): number of features. Defaults to 4.
            bottleneck_capacity (int, optional): capacity of bottleneck layer.
            Defaults to 4.
            num_layers (int, optional): number of layers in encoder / decoder.
            Defaults to 2.
            dropout (float, optional): degree of dropout. Defaults to 0.
            activation (str, optional): activation function. Defaults to "ReLU".
        """
        super().__init__()

        # get activation function
        activation = getattr(nn, activation)
        # calculate capacities e. g., [310, 323, 190, 77, 30] for encoder
        encoder_capacity = [
            int((num_features / (2**2)) * i**1.3) for i in range(num_layers, 0, -1)
        ]
        encoder_capacity.insert(0, num_features)
        encoder_capacity.insert(num_layers + 2, bottleneck_capacity)

        decoder_capacity = encoder_capacity[::-1]

        # stack layers encoder
        layers_encoder = []
        for i in range(len(encoder_capacity) - 1):
            if i == 1:
                layers_encoder.append(nn.Dropout(dropout))
            layers_encoder.append(
                nn.Linear(encoder_capacity[i], encoder_capacity[i + 1])
            )
            if not i == len(encoder_capacity) - 2:
                layers_encoder.append(activation())  # type: ignore
        self.encoder = nn.Sequential(*layers_encoder)

        # stack layers decoder
        layers_decoder = []
        for i in range(len(decoder_capacity) - 1):
            layers_decoder.append(
                nn.Linear(decoder_capacity[i], decoder_capacity[i + 1])
            )
            if not i == len(decoder_capacity) - 2:
                layers_decoder.append(activation())  # type: ignore
            if i == num_layers - 1:
                layers_decoder.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers_decoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input tensor in forward pass.

        Args:
            x (torch.Tensor): input

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: reconstruction, latent_representation
        """
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent
