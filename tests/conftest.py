"""
Conftest — Shared fixtures and path setup for all tests.

Ensures the project root is on sys.path so that flat imports like
`from execution.position_tracker import Position` work during testing.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate a realistic OHLCV DataFrame (100 bars)."""
    np.random.seed(42)
    n = 100
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    base_price = 65000.0
    # Random walk
    closes = [base_price]
    for _ in range(n - 1):
        change = np.random.normal(0, 0.005) * closes[-1]
        closes.append(closes[-1] + change)
    closes = np.array(closes)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.002, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.002, n)))
    opens = closes * (1 + np.random.normal(0, 0.001, n))
    volumes = np.random.uniform(100, 5000, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }).set_index("timestamp")


@pytest.fixture
def sample_features_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a feature DataFrame with close, atr, volume columns."""
    df = sample_ohlcv_df.copy()
    df["atr"] = df["close"].rolling(14).std().fillna(100.0)
    df["rsi"] = 50.0 + np.random.normal(0, 10, len(df))
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    return df


@pytest.fixture
def sample_prediction():
    """Create a mock Prediction object."""
    from models.predictor import Prediction

    return Prediction(
        direction="Long",
        direction_idx=0,
        direction_probs={"Long": 0.75, "Short": 0.15, "Neutral": 0.10},
        magnitude=0.02,
        confidence=0.88,
        timestamp=datetime(2024, 6, 1),
        symbol="BTC/USDT",
    )


@pytest.fixture
def sample_short_prediction():
    """Create a bearish Prediction object."""
    from models.predictor import Prediction

    return Prediction(
        direction="Short",
        direction_idx=1,
        direction_probs={"Long": 0.10, "Short": 0.80, "Neutral": 0.10},
        magnitude=0.025,
        confidence=0.90,
        timestamp=datetime(2024, 6, 1),
        symbol="ETH/USDT",
    )


@pytest.fixture
def sample_neutral_prediction():
    """Create a neutral Prediction object."""
    from models.predictor import Prediction

    return Prediction(
        direction="Neutral",
        direction_idx=2,
        direction_probs={"Long": 0.30, "Short": 0.25, "Neutral": 0.45},
        magnitude=0.001,
        confidence=0.45,
        timestamp=datetime(2024, 6, 1),
        symbol="BTC/USDT",
    )
