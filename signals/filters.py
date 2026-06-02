"""
Signal Filters — Additional filters applied before emitting trade signals.

Filters:
  1. Cooldown: minimum time between signals per symbol
  2. Trend alignment: confirm signal aligns with higher-TF trend
  3. Volatility filter: skip signals during extreme volatility
  4. Volume filter: skip if volume is below threshold
  5. Signal strength: combine confidence + magnitude into a single score
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for signal filters."""

    # Cooldown
    cooldown_minutes: int = 30  # min time between signals per symbol

    # Trend alignment
    trend_alignment_enabled: bool = True
    trend_alignment_periods: int = 20  # SMA period for trend check

    # Volatility filter
    volatility_filter_enabled: bool = True
    atr_period: int = 14
    volatility_threshold: float = 3.0  # skip if ATR > N × average ATR

    # Volume filter
    volume_filter_enabled: bool = True
    volume_ma_period: int = 20
    min_volume_ratio: float = 0.5  # skip if volume < 0.5 × average

    # Signal strength
    min_confidence: float = 0.70
    min_magnitude: float = 0.005  # 0.5%

    # Direction agreement (if multiple timeframes agree)
    require_multi_tf_agreement: bool = False
    min_agreeing_timeframes: int = 2


class SignalFilter:
    """Applies filters to raw model predictions before emitting signals."""

    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()
        # Track last signal time per symbol for cooldown
        self._last_signal_time: dict[str, datetime] = {}

    def reset_cooldown(self, symbol: str | None = None) -> None:
        """Reset cooldown timer for a symbol (or all symbols)."""
        if symbol:
            self._last_signal_time.pop(symbol, None)
        else:
            self._last_signal_time.clear()

    def check_cooldown(self, symbol: str, now: datetime | None = None) -> bool:
        """Check if enough time has passed since last signal for this symbol.

        Returns:
            True if signal is allowed (cooldown passed), False if still in cooldown
        """
        now = now or datetime.utcnow()
        last_time = self._last_signal_time.get(symbol)

        if last_time is None:
            return True

        elapsed = (now - last_time).total_seconds() / 60.0
        return elapsed >= self.config.cooldown_minutes

    def record_signal(self, symbol: str, now: datetime | None = None) -> None:
        """Record that a signal was emitted for this symbol."""
        self._last_signal_time[symbol] = now or datetime.utcnow()

    def check_trend_alignment(
        self,
        direction: str,
        features_df: pd.DataFrame,
        price_col: str = "close",
    ) -> bool:
        """Check if signal direction aligns with the higher-timeframe trend.

        Args:
            direction: "Long" or "Short"
            features_df: DataFrame with price data
            price_col: column name for close price

        Returns:
            True if aligned, False otherwise
        """
        if not self.config.trend_alignment_enabled:
            return True

        if price_col not in features_df.columns:
            logger.warning("Price column '%s' not found, skipping trend filter", price_col)
            return True

        if len(features_df) < self.config.trend_alignment_periods:
            return True

        prices = features_df[price_col].iloc[-self.config.trend_alignment_periods:]
        sma = prices.mean()
        current_price = prices.iloc[-1]

        if direction == "Long":
            return bool(current_price > sma)
        elif direction == "Short":
            return bool(current_price < sma)
        return True

    def check_volatility(
        self,
        features_df: pd.DataFrame,
        atr_col: str = "atr",
    ) -> bool:
        """Check if current volatility is within acceptable range.

        Returns:
            True if volatility is acceptable, False if too extreme
        """
        if not self.config.volatility_filter_enabled:
            return True

        if atr_col not in features_df.columns:
            return True

        if len(features_df) < self.config.atr_period * 2:
            return True

        atr = features_df[atr_col]
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-self.config.atr_period * 2:].mean()

        if avg_atr == 0:
            return True

        ratio = current_atr / avg_atr
        return bool(ratio <= self.config.volatility_threshold)

    def check_volume(
        self,
        features_df: pd.DataFrame,
        volume_col: str = "volume",
    ) -> bool:
        """Check if current volume is above minimum threshold.

        Returns:
            True if volume is sufficient
        """
        if not self.config.volume_filter_enabled:
            return True

        if volume_col not in features_df.columns:
            return True

        if len(features_df) < self.config.volume_ma_period:
            return True

        volumes = features_df[volume_col]
        current_vol = volumes.iloc[-1]
        avg_vol = volumes.iloc[-self.config.volume_ma_period:].mean()

        if avg_vol == 0:
            return True

        return bool(current_vol >= avg_vol * self.config.min_volume_ratio)

    def compute_signal_strength(
        self,
        confidence: float,
        magnitude: float,
    ) -> float:
        """Compute a combined signal strength score.

        Args:
            confidence: model confidence (0-1)
            magnitude: expected % move (e.g. 0.015 for 1.5%)

        Returns:
            strength score (0-1)
        """
        # Normalize magnitude to 0-1 range (cap at 5%)
        mag_normalized = min(abs(magnitude) / 0.05, 1.0)

        # Weighted combination: confidence is more important
        strength = 0.7 * confidence + 0.3 * mag_normalized
        return min(strength, 1.0)

    def apply_all(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        magnitude: float,
        features_df: pd.DataFrame | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, list[str], float]:
        """Apply all filters and return whether signal passes.

        Args:
            symbol: trading pair
            direction: "Long" | "Short" | "Neutral"
            confidence: model confidence (0-1)
            magnitude: expected % move
            features_df: optional DataFrame for technical filters
            now: current timestamp

        Returns:
            (passed, reasons, signal_strength)
            - passed: True if signal passes all filters
            - reasons: list of filter reasons if rejected
            - strength: combined signal strength score
        """
        reasons: list[str] = []
        now = now or datetime.utcnow()

        # 1. Neutral direction → always reject
        if direction == "Neutral":
            return False, ["direction_neutral"], 0.0

        # 2. Confidence threshold
        if confidence < self.config.min_confidence:
            reasons.append(f"low_confidence ({confidence:.3f} < {self.config.min_confidence})")

        # 3. Magnitude threshold
        if abs(magnitude) < self.config.min_magnitude:
            reasons.append(f"low_magnitude ({abs(magnitude):.4f} < {self.config.min_magnitude})")

        # 4. Cooldown
        if not self.check_cooldown(symbol, now):
            remaining = self.config.cooldown_minutes - (
                (now - self._last_signal_time.get(symbol, now)).total_seconds() / 60.0
            )
            reasons.append(f"cooldown ({remaining:.0f}m remaining)")

        # 5. Technical filters (if features available)
        if features_df is not None:
            if not self.check_trend_alignment(direction, features_df):
                reasons.append("trend_misalignment")

            if not self.check_volatility(features_df):
                reasons.append("extreme_volatility")

            if not self.check_volume(features_df):
                reasons.append("low_volume")

        # Compute signal strength
        strength = self.compute_signal_strength(confidence, magnitude)

        passed = len(reasons) == 0

        if not passed:
            logger.debug("Signal rejected for %s: %s", symbol, ", ".join(reasons))

        return passed, reasons, strength
