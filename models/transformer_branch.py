"""
Transformer Branch — Captures complex long-range relationships across the entire input window.

Architecture:
  - Positional encoding (learned)
  - Encoder stack: Multi-Head Attention × N layers
  - Global average pooling over sequence
  - Output: fixed-size embedding for ensemble head
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional encoding added to input embeddings."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable position embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBranch(nn.Module):
    """Transformer encoder branch for time-series feature extraction.

    Input:  (batch, seq_len, n_features)
    Output: (batch, d_model)

    Architecture:
        1. Linear projection from n_features → d_model
        2. Positional encoding
        3. TransformerEncoder (num_layers × [MultiHeadAttention + FFN])
        4. Layer normalization
        5. Global average pooling → (batch, d_model)
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()

        # Project raw features to d_model dimensions
        self.input_projection = nn.Linear(in_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.embedding_dim = d_model

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
            src_key_padding_mask: (batch, seq_len) — True for padded positions
        Returns:
            embedding: (batch, d_model)
        """
        # Project to d_model
        x = self.input_projection(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, T, d_model)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Global average pooling (ignoring padding if mask provided)
        if src_key_padding_mask is not None:
            # Mask out padded positions
            mask = ~src_key_padding_mask  # True = valid
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d_model)
        else:
            x = x.mean(dim=1)  # (B, d_model)

        x = self.output_norm(x)
        return x


class TransformerModel(nn.Module):
    """Wrapper that maintains consistent interface with CNNLSTMModel.

    Input:  (batch, seq_len, n_features)
    Output: (batch, d_model)
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        self.branch = TransformerBranch(
            in_features=in_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.embedding_dim = self.branch.embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
            src_key_padding_mask: optional padding mask
        Returns:
            embedding: (batch, d_model)
        """
        return self.branch(x, src_key_padding_mask=src_key_padding_mask)
