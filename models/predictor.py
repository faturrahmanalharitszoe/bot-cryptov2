"""
Model Predictor — Inference wrapper for the EnsembleModel.

Provides a clean API for:
  - Loading a trained model from checkpoint
  - Converting raw feature DataFrames into model input tensors
  - Running inference with proper preprocessing
  - Returning structured prediction results
  - Sliding window creation for real-time inference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.ensemble import EnsembleModel, DIRECTION_LABELS

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Structured prediction result."""

    # Direction
    direction: str        # "Long" | "Short" | "Neutral"
    direction_idx: int    # 0=Long, 1=Short, 2=Neutral
    direction_probs: dict[str, float]  # {"Long": 0.7, "Short": 0.2, "Neutral": 0.1}

    # Magnitude: expected % price move
    magnitude: float      # e.g. 0.015 = 1.5% expected move

    # Confidence: model's self-assessed confidence
    confidence: float     # 0.0 to 1.0

    # Metadata
    timestamp: pd.Timestamp | None = None
    symbol: str = ""

    @property
    def is_bullish(self) -> bool:
        return self.direction == "Long"

    @property
    def is_bearish(self) -> bool:
        return self.direction == "Short"

    @property
    def is_actionable(self) -> bool:
        """Whether this prediction warrants a trade signal."""
        return self.direction != "Neutral" and self.confidence > 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "direction_idx": self.direction_idx,
            "direction_probs": self.direction_probs,
            "magnitude": self.magnitude,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "symbol": self.symbol,
        }

    def __repr__(self) -> str:
        return (
            f"Prediction({self.symbol} | {self.direction} | "
            f"conf={self.confidence:.3f} | mag={self.magnitude:.4f})"
        )


class Predictor:
    """High-level inference wrapper for the EnsembleModel.

    Handles:
      - Model loading from checkpoint
      - Feature DataFrame → sliding windows → tensor
      - Batch and single-sample inference
      - Device management (CPU/GPU auto-detection)
    """

    def __init__(
        self,
        model: EnsembleModel | None = None,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        input_window: int = 90,
    ):
        """
        Args:
            model: pre-loaded EnsembleModel (mutually exclusive with checkpoint_path)
            checkpoint_path: path to .pt checkpoint file
            device: "auto" | "cpu" | "cuda" | "mps"
            input_window: number of timesteps expected by the model
        """
        self.input_window = input_window

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif checkpoint_path is not None:
            self.model, self._checkpoint = EnsembleModel.load(
                checkpoint_path, device=self.device
            )
            self.model.eval()
        else:
            raise ValueError("Either model or checkpoint_path must be provided")

        logger.info(
            "Predictor initialized: device=%s, input_window=%d, in_features=%d",
            self.device, self.input_window, self.model.in_features,
        )

    # -------------------------------------------------------------------
    # Feature preparation
    # -------------------------------------------------------------------

    @staticmethod
    def create_sliding_windows(
        features: np.ndarray,
        window_size: int = 90,
    ) -> np.ndarray:
        """Create sliding windows from a feature matrix.

        Args:
            features: (n_timesteps, n_features) — full feature matrix
            window_size: number of timesteps per window

        Returns:
            windows: (n_windows, window_size, n_features)
        """
        n = features.shape[0]
        if n < window_size:
            raise ValueError(
                f"Not enough data: need {window_size} timesteps, got {n}"
            )

        n_windows = n - window_size + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            features, (window_size, features.shape[1])
        ).reshape(n_windows, window_size, features.shape[1])

        return windows

    def features_to_tensor(
        self,
        features: np.ndarray | pd.DataFrame,
    ) -> torch.Tensor:
        """Convert features to a model-ready tensor.

        Args:
            features: either:
                - (n_features,) or (1, n_features) for a single timestep (uses last window from history)
                - (window_size, n_features) for a single window
                - (n_windows, window_size, n_features) for batched windows

        Returns:
            tensor: (batch, window_size, n_features) on self.device
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        features = np.asarray(features, dtype=np.float32)

        if features.ndim == 1:
            # Single timestep vector — reshape to (1, 1, n_features)
            # Caller should provide a full window instead
            raise ValueError(
                "1D input not supported. Provide a (window_size, n_features) array."
            )
        elif features.ndim == 2:
            # (window_size, n_features) → (1, window_size, n_features)
            if features.shape[0] == self.model.in_features and features.shape[0] != self.input_window:
                # Looks like (n_features, 1) — unlikely but handle
                features = features.T
            features = features[np.newaxis, ...]
        elif features.ndim == 3:
            # Already (batch, window_size, n_features)
            pass
        else:
            raise ValueError(f"Unsupported feature shape: {features.shape}")

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        tensor = torch.FloatTensor(features).to(self.device)
        return tensor

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, features: np.ndarray | pd.DataFrame) -> Prediction:
        """Predict on a single window.

        Args:
            features: (window_size, n_features) — one sliding window

        Returns:
            Prediction object
        """
        tensor = self.features_to_tensor(features)
        outputs = self.model(tensor)

        # Extract results
        direction_probs = outputs["direction"][0].cpu().numpy()  # (3,)
        direction_idx = int(direction_probs.argmax())
        
        # Bias towards action: if Neutral is max, but Long/Short is > 0.35, take action
        if direction_idx == 2 and (direction_probs[0] > 0.35 or direction_probs[1] > 0.35):
            direction_idx = 0 if direction_probs[0] > direction_probs[1] else 1
            
        direction_label = DIRECTION_LABELS[direction_idx]

        magnitude = float(outputs["magnitude"][0, 0].cpu())
        confidence = float(outputs["confidence"][0, 0].cpu())

        return Prediction(
            direction=direction_label,
            direction_idx=direction_idx,
            direction_probs={
                DIRECTION_LABELS[i]: float(direction_probs[i])
                for i in range(len(DIRECTION_LABELS))
            },
            magnitude=magnitude,
            confidence=confidence,
        )

    @torch.no_grad()
    def predict_batch(self, features: np.ndarray) -> list[Prediction]:
        """Predict on multiple windows.

        Args:
            features: (n_windows, window_size, n_features)

        Returns:
            list of Prediction objects, one per window
        """
        tensor = self.features_to_tensor(features)
        outputs = self.model(tensor)

        batch_size = outputs["direction"].shape[0]
        predictions: list[Prediction] = []

        for i in range(batch_size):
            dir_probs = outputs["direction"][i].cpu().numpy()
            dir_idx = int(dir_probs.argmax())
            
            # Bias towards action
            if dir_idx == 2 and (dir_probs[0] > 0.35 or dir_probs[1] > 0.35):
                dir_idx = 0 if dir_probs[0] > dir_probs[1] else 1
                
            dir_label = DIRECTION_LABELS[dir_idx]

            mag = float(outputs["magnitude"][i, 0].cpu())
            conf = float(outputs["confidence"][i, 0].cpu())

            predictions.append(Prediction(
                direction=dir_label,
                direction_idx=dir_idx,
                direction_probs={
                    DIRECTION_LABELS[j]: float(dir_probs[j])
                    for j in range(len(DIRECTION_LABELS))
                },
                magnitude=mag,
                confidence=conf,
            ))

        return predictions

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> Prediction:
        """Convenience: predict from a feature DataFrame.

        Takes the last `input_window` rows as the sliding window.

        Args:
            df: DataFrame with feature columns, indexed by time
            symbol: trading pair symbol for metadata

        Returns:
            Prediction with timestamp from last row
        """
        if len(df) < self.input_window:
            raise ValueError(
                f"Need {self.input_window} rows, got {len(df)}"
            )

        # Take last N rows as window
        window = df.iloc[-self.input_window:].copy()

        # Drop unscaled raw features so they don't cause exploding gradients or shape mismatches
        # Any feature that has a '_norm' equivalent should be dropped, matching the training pipeline.
        cols_to_drop = [c.replace("_norm", "") for c in window.columns if c.endswith("_norm")]
        window = window.drop(columns=cols_to_drop, errors="ignore")

        # Exclude raw OHLCV + timestamp columns to match training pipeline
        skip_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        num_cols = [
            c for c in window.columns
            if c not in skip_cols and window[c].dtype in [np.float64, np.int64, float, int]
        ]
        features = window[num_cols].values

        if features.shape[1] > self.model.in_features:
            logger.debug(
                "Feature shape mismatch: pipeline generated %d features, but model expects %d. "
                "Truncating the extra features (likely orderbook/sentiment).",
                features.shape[1], self.model.in_features
            )
            features = features[:, :self.model.in_features]
        elif features.shape[1] < self.model.in_features:
            raise ValueError(f"Not enough features: model expects {self.model.in_features}, got {features.shape[1]}")

        prediction = self.predict(features)
        prediction.symbol = symbol

        # Attach timestamp from last row
        if isinstance(df.index, pd.DatetimeIndex):
            prediction.timestamp = df.index[-1]
        elif "timestamp" in df.columns:
            prediction.timestamp = pd.Timestamp(df["timestamp"].iloc[-1])

        return prediction

    def predict_latest_from_history(
        self,
        history_df: pd.DataFrame,
        symbol: str = "",
    ) -> Prediction:
        """Predict the next move using the latest window from historical data.

        Same as predict_from_dataframe but with explicit naming for clarity
        in the trading pipeline.

        Args:
            history_df: full historical feature DataFrame (at least input_window rows)
            symbol: trading pair

        Returns:
            Prediction
        """
        return self.predict_from_dataframe(history_df, symbol=symbol)

    # -------------------------------------------------------------------
    # Model introspection
    # -------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "in_features": self.model.in_features,
            "combined_dim": self.model.combined_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "input_window": self.input_window,
            "cnn_lstm_embedding_dim": self.model.cnn_lstm.embedding_dim,
            "transformer_embedding_dim": self.model.transformer.embedding_dim,
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"Predictor(device={info['device']}, "
            f"in_features={info['in_features']}, "
            f"params={info['total_parameters']:,}, "
            f"window={info['input_window']})"
        )
