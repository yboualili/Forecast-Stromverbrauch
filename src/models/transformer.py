"""
Implementation of a Transformer.

Adapted from:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
"""

import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """
    Positional encoding of sequences.

    The authors of the original transformer paper describe very succinctly what
    the positional encoding layer does and why it is needed:

    "Since our model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we must inject some
    information about the relative or absolute position of the tokens in the
    sequence." (Vaswani et al, 2017)
    """

    def __init__(
        self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512
    ):
        """
        Positional encoding of sequences.

        Args:
            dropout (float, optional): max dropout. Defaults to 0.1.
            max_seq_len (int, optional): max lenth of input sequences. Defaults to 5000.
            d_model (int, optional): The dimension of the output of sub-layers in
            the model. Defaults to 512.
        """
        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values dependent
        # on position and i
        position = torch.arange(max_seq_len).unsqueeze(1)

        exp_input = torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)

        div_term = torch.exp(exp_input)
        # Returns a new tensor with the exponential of the elements of exp_input

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        # register that pe is not a model parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processs input tensor in forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        add = self.pe[: x.size(1), :].squeeze(1)

        x = x + add

        return self.dropout(x)


class Transformer(nn.Module):
    """
    Implementation of a Transformer.

    Based on:
    @inproceedings{vaswaniAttentionAllYou2017,
    title = {Attention Is All You Need},
    booktitle = {Advances in {{Neural Information Processing Systems}}},
    author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit,
    Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin,
    Illia},
    date = {2017},
    volume = {30},
    eprint = {1706.03762},
    eprinttype = {arxiv},
    publisher = {{Curran Associates, Inc.}},
    location = {{Long Beach, CA}},
    archiveprefix = {arXiv},
    eventtitle = {Conference on {{Neural Information Processing Systems}}},
    }

    Hint:
    Unlike the paper, this class assumes that input layers, positional encoding
    layers and linear mapping layers are separate from the encoder and decoder,
    i.e. the encoder and decoder only do what is depicted as their sub-layers
    in the paper. For practical purposes, this assumption does not make a
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the
    Encoder() and Decoder() classes.
    """

    def __init__(
        self,
        num_features: int,
        dec_seq_len: int,
        max_seq_len: int,
        out_seq_len: int = 58,
        dim_val: int = 512,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout_encoder: float = 0.2,
        dropout_decoder: float = 0.2,
        dropout_pos_enc: float = 0.2,
        dim_feedforward_encoder: int = 2048,
        dim_feedforward_decoder: int = 2048,
    ):
        """
        Initialize a Transformer.

        Simple Transformer for time series prediction using multiple features.

        Args:
            num_features: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            max_seq_len: int, length of the longest sequence the model will
                         receive. Used in positional encoding.
            out_seq_len: int, the length of the model's output (i.e. the target
                         sequence length)
            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder
        """
        super().__init__()

        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.n_heads = n_heads

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=num_features, out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=num_features, out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=out_seq_len * dim_val, out_features=out_seq_len
        )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=True,
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant,
        # because nn.TransformerEncoderLayer per default normalizes after each
        # sub-layer. See: https://github.com/pytorch/pytorch/issues/24930.
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True,
        )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant,
        # because nn.TransformerEncoderLayer per default normalizes after each
        # sub-layer. See: https://github.com/pytorch/pytorch/issues/24930.
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None
        )

        self.src_mask = None
        self.tgt_mask = None

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Processs input tensor and output tensor in forward pass.

        Args:
            src (torch.Tensor): sequence to encoder. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the feature number
            tgt (torch.Tensor): sequence to the decoder. Shape: (T,E) for unbatched
                input, (T,N,E) if batch_first=False or (N, T, E) if batch_first=True,
                where T is the target sequence length, N is the batch size, E is the
                feature number

        Returns:
            torch.Tensor: output
        """
        # if target is two-dimensional, make it tree dimensional (like input)
        if tgt.dim() == 2:
            tgt = torch.unsqueeze(tgt, 2)

        batch_size, _, _ = tgt.shape

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)

        decoder_output = self.decoder_input_layer(tgt)

        # generate masks, regenerate if no mask exists or batch is different e.g.,
        # last batch.

        if self.src_mask is None or self.src_mask.size(0) != batch_size * self.n_heads:
            self.src_mask = torch.triu(
                torch.ones(
                    batch_size * self.n_heads, self.out_seq_len, self.dec_seq_len
                )
                * float("-inf"),
                diagonal=1,
            ).to(self.device)
        if self.tgt_mask is None or self.tgt_mask.size(0) != batch_size * self.n_heads:
            self.tgt_mask = torch.triu(
                torch.ones(
                    batch_size * self.n_heads, self.out_seq_len, self.out_seq_len
                )
                * float("-inf"),
                diagonal=1,
            ).to(self.device)

        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=self.tgt_mask,
            memory_mask=self.src_mask,
        )

        decoder_output = self.linear_mapping(decoder_output.flatten(start_dim=1))

        return decoder_output
