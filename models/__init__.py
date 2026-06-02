"""Deep learning models — CNN-LSTM, Transformer, Ensemble, Trainer, Predictor.

Usage:
    from models.ensemble import EnsembleModel
    from models.trainer import Trainer, TrainerConfig
    from models.predictor import Predictor, Prediction
"""

from models.cnn_lstm import CNNLSTMModel, CNNBranch, LSTMBranch
from models.transformer_branch import TransformerModel, TransformerBranch
from models.ensemble import EnsembleModel, EnsembleHead, DIRECTION_LABELS
from models.trainer import Trainer, TrainerConfig, MultiTaskLoss
from models.predictor import Predictor, Prediction

__all__ = [
    # Branches
    "CNNLSTMModel",
    "CNNBranch",
    "LSTMBranch",
    "TransformerModel",
    "TransformerBranch",
    # Ensemble
    "EnsembleModel",
    "EnsembleHead",
    "DIRECTION_LABELS",
    # Training
    "Trainer",
    "TrainerConfig",
    "MultiTaskLoss",
    # Inference
    "Predictor",
    "Prediction",
]
