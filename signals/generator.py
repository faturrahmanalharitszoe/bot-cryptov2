"""
Signal Generator — Converts model predictions into actionable trade signals.

Flow:
  1. Receive Prediction from Predictor
  2. Apply filters (cooldown, trend, volatility, volume)
  3. Decide market type (spot vs futures) based on direction + confidence
  4. Compute leverage (if futures)
  5. Emit TradeSignal with all metadata

Spot vs Futures Decision Logic:
  - Long + moderate confidence → Spot (no leverage)
  - Long + high confidence (>0.85) → Futures long (with leverage)
  - Short → Futures short (always futures, need shorting)
  - Neutral → Close positions / no action
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from models.predictor import Prediction
from signals.filters import SignalFilter, FilterConfig

logger = logging.getLogger(__name__)


class MarketType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"


class SignalAction(str, Enum):
    BUY = "buy"           # Long spot
    SELL = "sell"         # Sell spot (close long)
    LONG = "long"         # Long futures
    SHORT = "short"       # Short futures
    CLOSE = "close"       # Close any position
    HOLD = "hold"         # No action


@dataclass
class TradeSignal:
    """A fully qualified trade signal ready for execution."""

    # Identity
    symbol: str
    timestamp: datetime

    # What to do
    action: SignalAction
    market: MarketType

    # Direction from model
    direction: str        # "Long" | "Short" | "Neutral"
    confidence: float     # 0-1
    magnitude: float      # expected % move
    strength: float       # combined signal strength

    # Risk parameters
    leverage: float = 1.0
    position_size_pct: float = 0.05  # % of portfolio

    # Price targets
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_prices: list[float] = field(default_factory=list)

    # Metadata
    direction_probs: dict[str, float] = field(default_factory=dict)
    filters_passed: list[str] = field(default_factory=list)
    filters_rejected: list[str] = field(default_factory=list)

    @property
    def is_entry(self) -> bool:
        """Whether this signal is an entry (opening a position)."""
        return self.action in (SignalAction.BUY, SignalAction.LONG, SignalAction.SHORT)

    @property
    def is_exit(self) -> bool:
        """Whether this signal is an exit (closing a position)."""
        return self.action in (SignalAction.SELL, SignalAction.CLOSE)

    @property
    def side(self) -> str:
        """Return 'buy' or 'sell' for exchange API."""
        if self.action in (SignalAction.BUY, SignalAction.LONG):
            return "buy"
        elif self.action in (SignalAction.SELL, SignalAction.SHORT):
            return "sell"
        return "buy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "market": self.market.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "strength": self.strength,
            "leverage": self.leverage,
            "position_size_pct": self.position_size_pct,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_prices": self.take_profit_prices,
        }

    def __repr__(self) -> str:
        return (
            f"TradeSignal({self.symbol} | {self.action.value} {self.market.value} | "
            f"conf={self.confidence:.3f} | lev={self.leverage}x | str={self.strength:.3f})"
        )


class SignalGenerator:
    """Main signal generation pipeline.

    Usage:
        generator = SignalGenerator(config)
        signal = generator.generate(
            prediction=predictor.predict(features_df),
            current_price=65000.0,
            features_df=features_df,
        )
        if signal and signal.is_entry:
            # Execute trade...
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        filter_config: FilterConfig | None = None,
    ):
        config = config or {}

        # Signal thresholds
        self.confidence_threshold: float = config.get("confidence_threshold", 0.70)
        self.magnitude_threshold: float = config.get("magnitude_threshold", 0.005)

        # Spot/futures switching thresholds
        self.high_confidence_threshold: float = config.get("high_confidence_threshold", 0.85)
        self.max_leverage: float = config.get("max_leverage", 3.0)

        # Leverage scaling
        self.leverage_scale_min: float = config.get("leverage_scale_min", 0.85)
        self.leverage_scale_max: float = config.get("leverage_scale_max", 0.95)

        # Stop-loss / take-profit
        self.stop_loss_pct: float = config.get("stop_loss_pct", 0.02)
        self.take_profit_levels: list[float] = config.get(
            "take_profit_levels", [0.03, 0.05, 0.08]
        )

        # Position sizing
        self.position_size_pct: float = config.get("position_size_pct", 0.05)

        # Initialize filter
        fc = filter_config or FilterConfig(
            cooldown_minutes=config.get("cooldown_minutes", 30),
            min_confidence=self.confidence_threshold,
            min_magnitude=self.magnitude_threshold,
        )
        self.signal_filter = SignalFilter(fc)

        # Stats
        self._signals_emitted: int = 0
        self._signals_rejected: int = 0

        logger.info(
            "SignalGenerator initialized: conf_thresh=%.2f, mag_thresh=%.4f, max_lev=%.1fx",
            self.confidence_threshold, self.magnitude_threshold, self.max_leverage,
        )

    # -------------------------------------------------------------------
    # Market decision
    # -------------------------------------------------------------------

    def decide_market(
        self,
        direction: str,
        confidence: float,
    ) -> tuple[MarketType, float]:
        """Decide whether to use spot or futures, and compute leverage.

        Args:
            direction: "Long" | "Short" | "Neutral"
            confidence: model confidence 0-1

        Returns:
            (market_type, leverage)
        """
        if direction == "Short":
            leverage = self._compute_leverage(confidence)
            return MarketType.FUTURES, leverage

        elif direction == "Long":
            if confidence >= self.high_confidence_threshold:
                leverage = self._compute_leverage(confidence)
                return MarketType.FUTURES, leverage
            else:
                return MarketType.SPOT, 1.0

        else:  # Neutral
            return MarketType.SPOT, 1.0

    def _compute_leverage(self, confidence: float) -> float:
        """Scale leverage based on confidence level.

        Linear interpolation:
          confidence 0.85 → leverage 1.0x
          confidence 0.95 → leverage 3.0x (max)
          confidence > 0.95 → leverage 3.0x (capped)
        """
        if confidence <= self.leverage_scale_min:
            return 1.0

        t = (confidence - self.leverage_scale_min) / (
            self.leverage_scale_max - self.leverage_scale_min
        )
        t = min(t, 1.0)

        leverage = 1.0 + t * (self.max_leverage - 1.0)
        return round(leverage, 1)

    # -------------------------------------------------------------------
    # Price targets
    # -------------------------------------------------------------------

    def compute_price_targets(
        self,
        direction: str,
        entry_price: float,
    ) -> tuple[float, list[float]]:
        """Compute stop-loss and take-profit prices.

        Args:
            direction: "Long" or "Short"
            entry_price: current price

        Returns:
            (stop_loss_price, [take_profit_prices])
        """
        if direction == "Long":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profits = [
                entry_price * (1 + tp) for tp in self.take_profit_levels
            ]
        elif direction == "Short":
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profits = [
                entry_price * (1 - tp) for tp in self.take_profit_levels
            ]
        else:
            stop_loss = entry_price
            take_profits = []

        return stop_loss, take_profits

    # -------------------------------------------------------------------
    # Main generation
    # -------------------------------------------------------------------

    def generate(
        self,
        prediction: Prediction,
        current_price: float,
        features_df: pd.DataFrame | None = None,
        now: datetime | None = None,
    ) -> TradeSignal:
        """Generate a trade signal from a model prediction.

        Always returns a TradeSignal — HOLD if filters reject,
        actionable signal if filters pass.

        Args:
            prediction: Prediction from Predictor
            current_price: current market price for the symbol
            features_df: optional DataFrame for technical filters
            now: current timestamp

        Returns:
            TradeSignal (may be HOLD)
        """
        now = now or datetime.utcnow()

        # Apply filters
        passed, reasons, strength = self.signal_filter.apply_all(
            symbol=prediction.symbol,
            direction=prediction.direction,
            confidence=prediction.confidence,
            magnitude=prediction.magnitude,
            features_df=features_df,
            now=now,
        )

        # Default: HOLD
        if not passed:
            self._signals_rejected += 1
            return TradeSignal(
                symbol=prediction.symbol,
                timestamp=now,
                action=SignalAction.HOLD,
                market=MarketType.SPOT,
                direction=prediction.direction,
                confidence=prediction.confidence,
                magnitude=prediction.magnitude,
                strength=0.0,
                direction_probs=prediction.direction_probs,
                filters_rejected=reasons,
            )

        # Signal passed all filters
        self._signals_emitted += 1
        self.signal_filter.record_signal(prediction.symbol, now)

        # Decide market type and leverage
        market, leverage = self.decide_market(
            prediction.direction, prediction.confidence
        )

        # Determine action
        action = self._direction_to_action(prediction.direction, market)

        # Compute price targets
        stop_loss, take_profits = self.compute_price_targets(
            prediction.direction, current_price
        )

        signal = TradeSignal(
            symbol=prediction.symbol,
            timestamp=now,
            action=action,
            market=market,
            direction=prediction.direction,
            confidence=prediction.confidence,
            magnitude=prediction.magnitude,
            strength=strength,
            leverage=leverage,
            position_size_pct=self.position_size_pct,
            entry_price=current_price,
            stop_loss_price=stop_loss,
            take_profit_prices=take_profits,
            direction_probs=prediction.direction_probs,
            filters_passed=["all"],
        )

        logger.info(
            "Signal generated: %s %s %s | conf=%.3f | mag=%.4f | lev=%.1fx | str=%.3f",
            signal.symbol, signal.action.value, signal.market.value,
            signal.confidence, signal.magnitude, signal.leverage, signal.strength,
        )

        return signal

    @staticmethod
    def _direction_to_action(direction: str, market: MarketType) -> SignalAction:
        """Convert direction + market to a SignalAction."""
        if direction == "Long" and market == MarketType.SPOT:
            return SignalAction.BUY
        elif direction == "Long" and market == MarketType.FUTURES:
            return SignalAction.LONG
        elif direction == "Short":
            return SignalAction.SHORT
        else:
            return SignalAction.HOLD

    # -------------------------------------------------------------------
    # Batch generation
    # -------------------------------------------------------------------

    def generate_batch(
        self,
        predictions: list[Prediction],
        prices: dict[str, float],
        features: dict[str, pd.DataFrame] | None = None,
        now: datetime | None = None,
    ) -> list[TradeSignal]:
        """Generate signals for multiple symbols at once.

        Args:
            predictions: list of Prediction objects (one per symbol)
            prices: dict mapping symbol → current price
            features: optional dict mapping symbol → feature DataFrame
            now: current timestamp

        Returns:
            list of TradeSignal objects
        """
        now = now or datetime.utcnow()
        features = features or {}

        signals: list[TradeSignal] = []
        for pred in predictions:
            price = prices.get(pred.symbol, 0.0)
            feat_df = features.get(pred.symbol)

            signal = self.generate(
                prediction=pred,
                current_price=price,
                features_df=feat_df,
                now=now,
            )
            signals.append(signal)

        # Log summary
        entries = sum(1 for s in signals if s.is_entry)
        holds = sum(1 for s in signals if s.action == SignalAction.HOLD)
        logger.info(
            "Batch signal generation: %d total, %d entries, %d holds",
            len(signals), entries, holds,
        )

        return signals

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return signal generation statistics."""
        total = self._signals_emitted + self._signals_rejected
        return {
            "signals_emitted": self._signals_emitted,
            "signals_rejected": self._signals_rejected,
            "total_processed": total,
            "acceptance_rate": self._signals_emitted / max(total, 1),
        }
