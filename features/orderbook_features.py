"""
Orderbook Feature Engineer — Converts raw orderbook snapshots into numeric features.

Features include:
- Bid/Ask imbalance
- Spread metrics
- Depth pressure
- Wall detection
"""

import pandas as pd
import numpy as np
from typing import Optional

from monitoring.logger import get_logger

logger = get_logger("features.orderbook")


class OrderbookFeatures:
    """Converts raw orderbook data into numeric features for model input."""

    def compute(self, orderbook_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute orderbook features from raw orderbook snapshots.
        
        Args:
            orderbook_df: DataFrame from OrderbookScraper with columns like
                         [timestamp, best_bid, best_ask, spread, imbalance, etc.]
                         
        Returns:
            DataFrame with derived orderbook features
        """
        if orderbook_df.empty:
            return pd.DataFrame()

        df = orderbook_df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure numeric
        numeric_cols = [
            "best_bid", "best_ask", "spread", "spread_pct", "mid_price",
            "total_bid_volume", "total_ask_volume", "imbalance",
            "bid_vwap", "ask_vwap", "bid_walls", "ask_walls",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rolling features
        if len(df) >= 5:
            df["imbalance_sma_5"] = df["imbalance"].rolling(5).mean()
            df["spread_pct_sma_5"] = df["spread_pct"].rolling(5).mean()
            df["bid_volume_sma_5"] = df["total_bid_volume"].rolling(5).mean()
            df["ask_volume_sma_5"] = df["total_ask_volume"].rolling(5).mean()
        else:
            df["imbalance_sma_5"] = df["imbalance"]
            df["spread_pct_sma_5"] = df["spread_pct"]
            df["bid_volume_sma_5"] = df["total_bid_volume"]
            df["ask_volume_sma_5"] = df["total_ask_volume"]

        if len(df) >= 10:
            df["imbalance_sma_10"] = df["imbalance"].rolling(10).mean()
            df["spread_pct_sma_10"] = df["spread_pct"].rolling(10).mean()
        else:
            df["imbalance_sma_10"] = df["imbalance"]
            df["spread_pct_sma_10"] = df["spread_pct"]

        # Imbalance momentum
        df["imbalance_change"] = df["imbalance"].diff()
        df["imbalance_momentum"] = df["imbalance_change"].diff()

        # Volume pressure ratio
        total_vol = df["total_bid_volume"] + df["total_ask_volume"]
        df["bid_pressure"] = df["total_bid_volume"] / total_vol.replace(0, np.nan)
        df["ask_pressure"] = df["total_ask_volume"] / total_vol.replace(0, np.nan)

        # Wall intensity
        df["wall_diff"] = df["bid_walls"] - df["ask_walls"]

        # Spread regime
        df["spread_regime"] = pd.cut(
            df["spread_pct"],
            bins=[-np.inf, 0.0005, 0.001, 0.005, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float)

        # Select output columns (features only, not raw data)
        feature_cols = [
            "timestamp",
            "spread_pct",
            "imbalance",
            "imbalance_sma_5",
            "imbalance_sma_10",
            "imbalance_change",
            "imbalance_momentum",
            "bid_pressure",
            "ask_pressure",
            "wall_diff",
            "spread_regime",
            "spread_pct_sma_5",
            "spread_pct_sma_10",
        ]

        output = df[[c for c in feature_cols if c in df.columns]].copy()
        output = output.fillna(0)

        return output
