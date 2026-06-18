"""
Technical Feature Engineer — Computes technical analysis indicators from OHLCV data.

Uses pandas_ta for comprehensive indicator computation. Outputs a feature-rich
DataFrame ready for model input.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from monitoring.logger import get_logger

logger = get_logger("features.technical")


class TechnicalFeatures:
    """Computes technical analysis indicators from OHLCV data."""

    # Default indicator set — ~40+ features
    DEFAULT_INDICATORS = [
        # Trend
        "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_10", "ema_20", "ema_50",
        "macd", "macd_signal", "macd_hist",
        "adx", "aroon_up", "aroon_down",
        "ichimoku_a", "ichimoku_b",

        # Momentum
        "rsi_14", "rsi_7",
        "stoch_k", "stoch_d",
        "cci_14",
        "williams_r",
        "mfi_14",
        "roc_10",

        # Volatility
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pband",
        "atr_14",
        "atr_pct",
        "hist_vol_20",

        # Volume
        "obv",
        "vwap",
        "mfi",
        "volume_sma_20",
        "volume_ratio",

        # Derived
        "price_change",
        "price_change_pct",
        "high_low_range",
        "close_to_high",
        "close_to_low",
        "candle_body_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
    ]

    def __init__(self):
        try:
            import ta
            self.ta = ta
        except ImportError:
            logger.warning("ta library not installed, fallback will produce NaNs for advanced features")
            self.ta = None

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Returns:
            DataFrame with all original columns + indicator columns
        """
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data ({len(df)} rows), need at least 50")
            return df

        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Always use the reliable manual implementations + ta library for advanced features
        df = self._compute_all(df)

        # Compute derived features (works either way)
        df = self._compute_derived(df)

        # Drop NaN rows from indicator warmup
        initial_len = len(df)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} NaN rows from warmup")

        return df

    def _compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators using a mix of manual math and 'ta' library."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # SMA/EMA
        for period in [10, 20, 50, 200]:
            df[f"sma_{period}"] = close.rolling(period).mean()
        for period in [10, 20, 50]:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
        for period in [7, 14]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr_14"] / close

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_middle"] = sma20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pband"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # OBV
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv"] = obv

        # Volume SMA
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]

        # Hist vol
        df["hist_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        # Use `ta` package for the complex ones
        if self.ta is not None:
            # Trend
            df["adx"] = self.ta.trend.adx(high, low, close, window=14)
            df["aroon_up"] = self.ta.trend.aroon_up(high, low, window=25)
            df["aroon_down"] = self.ta.trend.aroon_down(high, low, window=25)
            df["ichimoku_a"] = self.ta.trend.ichimoku_a(high, low)
            df["ichimoku_b"] = self.ta.trend.ichimoku_b(high, low)
            
            # Momentum
            df["cci_14"] = self.ta.trend.cci(high, low, close, window=14)
            df["williams_r"] = self.ta.momentum.williams_r(high, low, close, lbp=14)
            df["roc_10"] = self.ta.momentum.roc(close, window=10)
            
            # Volume
            df["mfi_14"] = self.ta.volume.money_flow_index(high, low, close, volume, window=14)
            df["vwap"] = self.ta.volume.volume_weighted_average_price(high, low, close, volume, window=14)
            df["mfi"] = df["mfi_14"]
        else:
            # Fallback (will result in missing features, but won't crash the script immediately)
            for col in ["adx", "aroon_up", "aroon_down", "ichimoku_a", "ichimoku_b",
                         "cci_14", "williams_r", "mfi_14", "roc_10", "vwap", "mfi"]:
                df[col] = np.nan

        return df

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features (price patterns, candle analysis)."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Price changes
        df["price_change"] = close.diff()
        df["price_change_pct"] = close.pct_change()

        # High-low range
        df["high_low_range"] = (high - low) / close

        # Close position relative to high/low
        df["close_to_high"] = (high - close) / close
        df["close_to_low"] = (close - low) / close

        # Candle body analysis
        body = (close - open_).abs()
        total_range = (high - low).replace(0, np.nan)
        df["candle_body_pct"] = body / total_range
        df["upper_shadow_pct"] = (high - close.where(close >= open_, open_)) / total_range
        df["lower_shadow_pct"] = (close.where(close < open_, open_) - low) / total_range

        # Returns at different horizons
        for period in [1, 3, 5, 10, 20]:
            df[f"return_{period}"] = close.pct_change(period)

        # Momentum features
        df["momentum_5"] = close - close.shift(5)
        df["momentum_10"] = close - close.shift(10)

        # Acceleration
        df["acceleration"] = df["price_change"].diff()

        # Volume momentum
        df["volume_change"] = volume = df["volume"].pct_change()

        # Fill NaN with 0 for derived features (they'll be dropped during training anyway)
        derived_cols = [
            "price_change", "price_change_pct", "high_low_range",
            "close_to_high", "close_to_low", "candle_body_pct",
            "upper_shadow_pct", "lower_shadow_pct",
            "momentum_5", "momentum_10", "acceleration", "volume_change",
        ]
        for col in derived_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of all computed feature column names (excluding OHLCV + timestamp)."""
        base_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        # This returns the known feature names; actual columns depend on data
        return [c for c in self.DEFAULT_INDICATORS if c not in base_cols]
