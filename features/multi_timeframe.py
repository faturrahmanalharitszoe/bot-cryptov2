"""
Multi-Timeframe Feature Merger — Combines features from different timeframes
into a single aligned feature matrix.

For swing trading, combining multiple timeframes gives the model context:
- 5m: Short-term micro-structure
- 15m: Intraday patterns
- 1h: Medium-term trend
- 4h: Swing trend

Higher timeframe features are forward-filled to align with lower timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from monitoring.logger import get_logger

logger = get_logger("features.multi_tf")

# Columns that need suffixing when merged from different timeframes
OHLCV_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


class MultiTimeframeFeatures:
    """Merges features from multiple timeframes into a single aligned matrix."""

    def __init__(self, base_timeframe: str = "5m"):
        """
        Args:
            base_timeframe: The primary timeframe to align to (lowest granularity)
        """
        self.base_timeframe = base_timeframe

    def merge(
        self,
        features_by_tf: Dict[str, pd.DataFrame],
        suffix_strategy: str = "prefix",
    ) -> pd.DataFrame:
        """
        Merge features from multiple timeframes.
        
        Args:
            features_by_tf: Dict mapping timeframe -> feature DataFrame
                          e.g., {"5m": df_5m, "15m": df_15m, "1h": df_1h, "4h": df_4h}
            suffix_strategy: How to handle column name conflicts
                           "prefix" -> add timeframe prefix (e.g., "1h_rsi_14")
                           "suffix" -> add timeframe suffix (e.g., "rsi_14_1h")
                            
        Returns:
            Merged DataFrame aligned to base_timeframe
        """
        # Resolve base timeframe: prefer the configured one, but fall back
        # to the lowest available timeframe when the configured base isn't
        # in the provided data (e.g. backtest only provides "1h").
        _TF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        base_tf = self.base_timeframe
        if base_tf not in features_by_tf and features_by_tf:
            available = list(features_by_tf.keys())
            # Pick the lowest-frequency TF present (finest granularity)
            base_tf = min(
                available,
                key=lambda t: _TF_ORDER.index(t) if t in _TF_ORDER else 999,
            )
            logger.info(
                "Base timeframe %s not in provided data — using %s instead",
                self.base_timeframe, base_tf,
            )

        if base_tf not in features_by_tf:
            logger.error("No timeframe data available for merge")
            return pd.DataFrame()

        base_df = features_by_tf[base_tf].copy()
        base_df = base_df.sort_values("timestamp").reset_index(drop=True)

        if base_df.empty:
            return base_df

        # Start with base timeframe features (no prefix needed)
        merged = base_df.copy()

        # Merge higher timeframes
        for tf, tf_df in features_by_tf.items():
            if tf == base_tf:
                continue

            if tf_df.empty:
                logger.warning(f"Empty data for timeframe {tf}, skipping")
                continue

            tf_df = tf_df.copy()
            tf_df = tf_df.sort_values("timestamp").reset_index(drop=True)

            # Select only feature columns (exclude OHLCV)
            feature_cols = [c for c in tf_df.columns if c not in OHLCV_COLS]
            if not feature_cols:
                continue

            # Rename columns with timeframe prefix/suffix
            rename_map = {}
            for col in feature_cols:
                if suffix_strategy == "prefix":
                    rename_map[col] = f"{tf}_{col}"
                else:
                    rename_map[col] = f"{col}_{tf}"

            tf_features = tf_df[["timestamp"] + feature_cols].rename(columns=rename_map)

            # Forward-fill higher timeframe data to align with base
            # Use merge_asof for time-based alignment
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                tf_features.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )

        # Forward fill any remaining NaN from alignment
        merged = merged.ffill()

        logger.info(
            f"Merged {len(features_by_tf)} timeframes: "
            f"{list(features_by_tf.keys())} -> {len(merged)} rows, "
            f"{len(merged.columns)} columns"
        )

        return merged

    def compute_cross_tf_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features that capture relationships between timeframes.
        
        Examples:
        - RSI alignment (all timeframes bullish/bearish)
        - Trend agreement
        - Divergence detection
        """
        df = merged_df.copy()

        # Find RSI columns across timeframes
        rsi_cols = [c for c in df.columns if "rsi_14" in c]
        if len(rsi_cols) >= 2:
            # Average RSI across timeframes
            df["cross_tf_rsi_avg"] = df[rsi_cols].mean(axis=1)
            # RSI agreement (all > 50 or all < 50)
            rsi_bullish = (df[rsi_cols] > 50).sum(axis=1)
            rsi_bearish = (df[rsi_cols] < 50).sum(axis=1)
            df["cross_tf_rsi_bullish_count"] = rsi_bullish
            df["cross_tf_rsi_bearish_count"] = rsi_bearish
            df["cross_tf_rsi_agreement"] = (rsi_bullish - rsi_bearish) / len(rsi_cols)

        # Find MACD histogram columns
        macd_cols = [c for c in df.columns if "macd_hist" in c]
        if len(macd_cols) >= 2:
            df["cross_tf_macd_avg"] = df[macd_cols].mean(axis=1)
            macd_positive = (df[macd_cols] > 0).sum(axis=1)
            df["cross_tf_macd_agreement"] = (macd_positive - (len(macd_cols) - macd_positive)) / len(macd_cols)

        # Find price change columns
        change_cols = [c for c in df.columns if "price_change_pct" in c]
        if len(change_cols) >= 2:
            df["cross_tf_price_trend"] = df[change_cols].mean(axis=1)

        # Find ADX columns (trend strength)
        adx_cols = [c for c in df.columns if c.endswith("_adx") or c == "adx"]
        if len(adx_cols) >= 2:
            df["cross_tf_adx_avg"] = df[adx_cols].mean(axis=1)
            df["cross_tf_trend_strength"] = (df[adx_cols] > 25).sum(axis=1) / len(adx_cols)

        logger.debug(f"Computed {len(df.columns) - len(merged_df.columns)} cross-TF features")
        return df
