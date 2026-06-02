"""
CNN-LSTM Branch — Captures local patterns (CNN) and temporal dependencies (LSTM+Attention).

Architecture:
  - CNN: Conv1D(64, k=3) → Conv1D(128, k=5) → MaxPool → Dropout
  - BiLSTM(128) → Attention → BiLSTM(64) → Dropout
  - Each branch outputs a fixed-size embedding for the ensemble head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalAttention(nn.Module):
    """Simple temporal attention over LSTM hidden states.

    Computes a weighted sum of all hidden states, learning which
    timesteps are most important for the prediction.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — weighted sum of hidden states
        """
        # lstm_out: (B, T, H)
        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return context


class CNNBranch(nn.Module):
    """1D CNN branch for capturing local price patterns.

    Input:  (batch, seq_len, n_features)
    Output: (batch, embedding_dim)
    """

    def __init__(
        self,
        in_features: int,
        filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        filters = filters or [64, 128]
        kernel_sizes = kernel_sizes or [3, 5]

        assert len(filters) == len(kernel_sizes), (
            f"filters and kernel_sizes must have same length, got {len(filters)} vs {len(kernel_sizes)}"
        )

        layers: list[nn.Module] = []
        in_ch = in_features
        for f, k in zip(filters, kernel_sizes):
            layers.append(nn.Conv1d(in_ch, f, kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm1d(f))
            layers.append(nn.ReLU(inplace=True))
            in_ch = f

        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)  # global max pool → (B, C, 1)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            out: (batch, embedding_dim)
        """
        # Conv1d expects (B, C, T) — C = features, T = seq_len
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv_stack(x)  # (B, last_filter, T)
        x = self.pool(x).squeeze(-1)  # (B, last_filter)
        x = self.dropout(x)
        return x


class LSTMBranch(nn.Module):
    """BiLSTM branch with temporal attention for capturing long-range dependencies.

    Input:  (batch, seq_len, n_features)
    Output: (batch, embedding_dim)
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        attn_input_size = hidden_size * self.num_directions
        if use_attention:
            self.attention = TemporalAttention(attn_input_size)
        else:
            self.attention = None

        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = hidden_size * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            out: (batch, embedding_dim)
        """
        lstm_out, _ = self.lstm(x)  # (B, T, H*directions)

        if self.attention is not None:
            out = self.attention(lstm_out)  # (B, H*directions)
        else:
            # Use last hidden state
            out = lstm_out[:, -1, :]  # (B, H*directions)

        out = self.dropout(out)
        return out


class CNNLSTMModel(nn.Module):
    """Full CNN-LSTM model combining both branches.

    Concatenates CNN and LSTM embeddings and passes through
    a shared dense layer.

    Input:  (batch, seq_len, n_features)
    Output: (batch, embedding_dim)  — to be used by ensemble head
    """

    def __init__(
        self,
        in_features: int,
        cnn_filters: list[int] | None = None,
        cnn_kernel_sizes: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_attention: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = CNNBranch(
            in_features=in_features,
            filters=cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
        )

        self.lstm = LSTMBranch(
            in_features=in_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=lstm_bidirectional,
            use_attention=lstm_attention,
        )

        # Project concatenated embedding to a consistent size
        combined_dim = self.cnn.embedding_dim + self.lstm.embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.embedding_dim = combined_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            embedding: (batch, embedding_dim)
        """
        cnn_emb = self.cnn(x)   # (B, cnn_emb_dim)
        lstm_emb = self.lstm(x)  # (B, lstm_emb_dim)
        combined = torch.cat([cnn_emb, lstm_emb], dim=1)  # (B, combined_dim)
        return self.projection(combined)
