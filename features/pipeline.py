"""
Feature Pipeline — Orchestrates all feature engineering.

Coordinates:
1. Technical indicators from OHLCV data
2. Sentiment features from sentiment DB
3. Orderbook features from orderbook snapshots
4. Multi-timeframe merging
5. Cross-timeframe features
6. Normalization and scaling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from features.technical import TechnicalFeatures
from features.sentiment_features import SentimentFeatures
from features.orderbook_features import OrderbookFeatures
from features.multi_timeframe import MultiTimeframeFeatures
from storage.parquet_store import ParquetStore
from storage.sqlite_store import SQLiteStore
from monitoring.logger import get_logger

logger = get_logger("features.pipeline")


class FeaturePipeline:
    """
    Full feature engineering pipeline.
    
    Takes raw OHLCV + orderbook + sentiment data and produces
    a normalized feature matrix ready for model input.
    """

    def __init__(
        self,
        parquet_store: Optional[ParquetStore] = None,
        sqlite_store: Optional[SQLiteStore] = None,
        base_timeframe: str = "5m",
        timeframes: Optional[List[str]] = None,
    ):
        self.parquet = parquet_store or ParquetStore()
        self.sqlite = sqlite_store or SQLiteStore()
        self.base_timeframe = base_timeframe
        self.timeframes = timeframes or ["5m", "15m"]

        self.technical = TechnicalFeatures()
        self.sentiment = SentimentFeatures(self.sqlite)
        self.orderbook = OrderbookFeatures()
        self.multi_tf = MultiTimeframeFeatures(base_timeframe)

    def compute(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        include_sentiment: bool = True,
        include_orderbook: bool = True,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full feature pipeline for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframes: List of timeframes to process (default: self.timeframes)
            include_sentiment: Whether to include sentiment features
            include_orderbook: Whether to include orderbook features
            normalize: Whether to normalize features (z-score)
            
        Returns:
            Feature matrix DataFrame aligned to base_timeframe
        """
        timeframes = timeframes or self.timeframes
        logger.info(f"Computing features for {symbol} across {timeframes}")

        # Step 1: Compute technical features for each timeframe
        features_by_tf = {}
        for tf in timeframes:
            ohlcv = self.parquet.load_ohlcv(symbol, tf)
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No OHLCV data for {symbol} {tf}")
                continue

            tech_features = self.technical.compute(ohlcv)
            features_by_tf[tf] = tech_features
            logger.debug(f"  {tf}: {len(tech_features)} rows, {len(tech_features.columns)} columns")

        if not features_by_tf:
            logger.error(f"No data available for any timeframe of {symbol}")
            return pd.DataFrame()

        # Step 2: Merge multi-timeframe features
        merged = self.multi_tf.merge(features_by_tf)
        if merged.empty:
            return merged

        # Step 3: Add cross-timeframe features
        merged = self.multi_tf.compute_cross_tf_features(merged)

        # Step 4: Add sentiment features
        if include_sentiment:
            try:
                base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
                sent_features = self.sentiment.compute(
                    base_symbol,
                    merged["timestamp"],
                    lookback_hours=24,
                )
                # Merge by index (timestamps should align)
                for col in sent_features.columns:
                    merged[col] = sent_features[col].values
                logger.debug(f"  Added {len(sent_features.columns)} sentiment features")
            except Exception as e:
                logger.warning(f"Failed to compute sentiment features: {e}")

        # Step 5: Add orderbook features
        if include_orderbook:
            try:
                ob_data = self.parquet.load_orderbook(symbol, limit=1000)
                if ob_data is not None and not ob_data.empty:
                    ob_features = self.orderbook.compute(ob_data)
                    if not ob_features.empty:
                        # Ensure both timestamps have same precision (ns)
                        merged["timestamp"] = pd.to_datetime(merged["timestamp"]).astype("datetime64[ns]")
                        ob_features["timestamp"] = pd.to_datetime(ob_features["timestamp"]).astype("datetime64[ns]")
                        
                        # Merge with asof for time alignment
                        merged = pd.merge_asof(
                            merged.sort_values("timestamp"),
                            ob_features.sort_values("timestamp"),
                            on="timestamp",
                            direction="backward",
                        )
                        logger.debug(f"  Added orderbook features")
            except Exception as e:
                logger.warning(f"Failed to compute orderbook features: {e}")

        # Step 6: Normalize
        if normalize:
            merged = self._normalize(merged)

        # Step 7: Final cleanup
        # Drop rows with too many NaN (from indicator warmup)
        threshold = 0.5  # Drop rows with >50% NaN
        nan_ratio = merged.isna().sum(axis=1) / len(merged.columns)
        merged = merged[nan_ratio <= threshold].reset_index(drop=True)

        # Fill remaining NaN with 0
        merged = merged.fillna(0)

        # Drop infinite values
        merged = merged.replace([np.inf, -np.inf], 0)

        logger.info(f"Feature pipeline complete: {len(merged)} rows, {len(merged.columns)} columns")
        return merged

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using rolling z-score normalization.
        
        Uses expanding window for training data (avoiding look-ahead bias)
        and rolling window for live data.
        """
        df = df.copy()
        
        # Columns to normalize (exclude timestamp and OHLCV)
        skip_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in df.columns if c not in skip_cols and df[c].dtype in [np.float64, np.int64, float, int]]
        
        if not feature_cols:
            return df
        
        # Build all normalized columns in a dict, then concat once to avoid
        # DataFrame fragmentation (pandas PerformanceWarning on repeated inserts).
        norm_dict: dict[str, pd.Series] = {}
        for col in feature_cols:
            expanding_mean = df[col].expanding(min_periods=20).mean()
            expanding_std = df[col].expanding(min_periods=20).std()
            
            # Avoid division by zero
            expanding_std = expanding_std.replace(0, 1)
            
            norm_dict[f"{col}_norm"] = ((df[col] - expanding_mean) / expanding_std).clip(-5, 5)
        
        # Single concat instead of per-column assignment
        if norm_dict:
            norm_df = pd.DataFrame(norm_dict, index=df.index)
            df = pd.concat([df, norm_df], axis=1)
        
        logger.debug(f"Normalized {len(feature_cols)} features")
        return df

    def compute_for_training(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        prediction_horizon: int = 12,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Compute features and labels for model training.
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            prediction_horizon: Number of candles ahead to predict
            
        Returns:
            Tuple of (features_df, direction_labels, magnitude_labels, confidence_labels)
        """
        features = self.compute(
            symbol, timeframes,
            include_sentiment=True,
            include_orderbook=True,
            normalize=True,
        )

        if features.empty:
            return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series()

        # Create labels from future price movement
        close = features["close"]
        future_return = close.shift(-prediction_horizon) / close - 1

        # Direction labels aligned with ensemble.py constants:
        #   DIRECTION_LONG    = 0  (future_return > +0.8%)
        #   DIRECTION_SHORT   = 1  (future_return < -0.8%)
        #   DIRECTION_NEUTRAL = 2  (|future_return| <= 0.8%)
        # Keep at 0.8% to match the currently-trained model checkpoint.
        # Risk management handles position sizing separately.
        direction = pd.Series(2, index=features.index)           # Neutral default
        direction[future_return > 0.008] = 0               # Long  (>+0.8% move)
        direction[future_return < -0.008] = 1              # Short (<-0.8% move)
        direction = direction.fillna(2).astype(int)

        # Magnitude (regression target): absolute percentage move
        magnitude = future_return.abs().fillna(0.0)
        magnitude = magnitude.replace([np.inf, -np.inf], 0.0)

        # Confidence proxy: based on volatility and trend clarity
        # Higher when trend is clear and volatility is manageable
        if "adx" in features.columns:
            trend_clarity = features["adx"] / 100
        else:
            trend_clarity = pd.Series(0.5, index=features.index)

        if "hist_vol_20" in features.columns:
            vol_factor = 1 - features["hist_vol_20"].clip(0, 1)
        else:
            vol_factor = pd.Series(0.5, index=features.index)

        confidence = (trend_clarity * 0.5 + vol_factor * 0.5).clip(0, 1).fillna(0.5)

        # Remove last prediction_horizon rows (no future data for labels)
        features = features.iloc[:-prediction_horizon]
        direction = direction.iloc[:-prediction_horizon]
        magnitude = magnitude.iloc[:-prediction_horizon]
        confidence = confidence.iloc[:-prediction_horizon]

        # Drop unscaled raw features so they don't cause exploding gradients in the neural network.
        # Any feature that has a '_norm' equivalent should be dropped.
        cols_to_drop = [c.replace("_norm", "") for c in features.columns if c.endswith("_norm")]
        features = features.drop(columns=cols_to_drop, errors="ignore")

        # Log class distribution so imbalance is always visible
        n_total = len(direction)
        n_long    = int((direction == 0).sum())
        n_short   = int((direction == 1).sum())
        n_neutral = int((direction == 2).sum())
        logger.info(
            "Training labels: %d samples | "
            "Long=%d (%.1f%%) | Short=%d (%.1f%%) | Neutral=%d (%.1f%%)",
            n_total,
            n_long,    100.0 * n_long    / max(n_total, 1),
            n_short,   100.0 * n_short   / max(n_total, 1),
            n_neutral, 100.0 * n_neutral / max(n_total, 1),
        )
        if n_neutral / max(n_total, 1) > 0.60:
            logger.warning(
                "High class imbalance detected: Neutral=%.1f%% of labels. "
                "Enable use_class_weights and use_weighted_sampler in config.",
                100.0 * n_neutral / max(n_total, 1),
            )

        return features, direction, magnitude, confidence

    def save_features(self, symbol: str, timeframe: str, features: pd.DataFrame) -> int:
        """Save computed features to storage."""
        return self.parquet.save_features(symbol, timeframe, features)

    def load_features(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load previously computed features."""
        return self.parquet.load_features(symbol, timeframe)
