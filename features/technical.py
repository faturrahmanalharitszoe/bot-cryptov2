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
            import pandas_ta as ta
            self.ta = ta
        except ImportError:
            logger.warning("pandas_ta not installed, using manual implementations")
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

        if self.ta is not None:
            df = self._compute_with_pandas_ta(df)
        else:
            df = self._compute_manual(df)

        # Compute derived features (works either way)
        df = self._compute_derived(df)

        # Drop NaN rows from indicator warmup
        initial_len = len(df)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} NaN rows from warmup")

        return df

    def _compute_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators using pandas_ta library."""
        ta = self.ta

        # ===== TREND =====
        df["sma_10"] = ta.sma(df["close"], length=10)
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)
        df["ema_10"] = ta.ema(df["close"], length=10)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 1]
            df["macd_hist"] = macd.iloc[:, 2]

        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and not adx.empty:
            df["adx"] = adx.iloc[:, 0]
            df["aroon_up"] = adx.iloc[:, 1] if adx.shape[1] > 1 else np.nan
            df["aroon_down"] = adx.iloc[:, 2] if adx.shape[1] > 2 else np.nan

        ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
        if ichimoku is not None:
            if isinstance(ichimoku, tuple):
                ichimoku = ichimoku[0]
            if ichimoku is not None and not ichimoku.empty:
                df["ichimoku_a"] = ichimoku.iloc[:, 0] if ichimoku.shape[1] > 0 else np.nan
                df["ichimoku_b"] = ichimoku.iloc[:, 1] if ichimoku.shape[1] > 1 else np.nan

        # ===== MOMENTUM =====
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_7"] = ta.rsi(df["close"], length=7)

        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        df["cci_14"] = ta.cci(df["high"], df["low"], df["close"], length=14)
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
        df["mfi_14"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        df["roc_10"] = ta.roc(df["close"], length=10)

        # ===== VOLATILITY =====
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df["bb_lower"] = bbands.iloc[:, 0]
            df["bb_middle"] = bbands.iloc[:, 1]
            df["bb_upper"] = bbands.iloc[:, 2]
            df["bb_width"] = bbands.iloc[:, 3] if bbands.shape[1] > 3 else (
                (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            )
            df["bb_pband"] = bbands.iloc[:, 4] if bbands.shape[1] > 4 else (
                (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            )

        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_pct"] = df["atr_14"] / df["close"]
        df["hist_vol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        # ===== VOLUME =====
        df["obv"] = ta.obv(df["close"], df["volume"])
        df["mfi"] = df["mfi_14"]  # Already computed above
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # VWAP (approximate from cumulative values)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        df["vwap"] = cum_tp_vol / cum_vol

        return df

    def _compute_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback manual indicator computation (no pandas_ta)."""
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

        # Fill missing columns with NaN
        for col in ["adx", "aroon_up", "aroon_down", "ichimoku_a", "ichimoku_b",
                     "cci_14", "williams_r", "mfi_14", "roc_10", "vwap"]:
            if col not in df.columns:
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
