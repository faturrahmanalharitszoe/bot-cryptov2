"""
Risk Manager — Enforces position sizing, stop-loss, drawdown limits, and concurrent caps.

Rules (from config.yaml):
  - Max 5% of portfolio per trade
  - Max 3 concurrent positions
  - Stop-loss: 2% per position (trailing)
  - Take-profit: Scaled exits at 3%, 5%, 8%
  - Max daily drawdown: 5% → halt trading
  - Max weekly drawdown: 10% → halt and alert
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from signals.generator import TradeSignal, SignalAction, MarketType
from execution.position_tracker import PositionTracker, Position

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration from config.yaml."""

    max_position_pct: float = 0.05       # 5% of portfolio per trade
    max_concurrent_positions: int = 3
    stop_loss_pct: float = 0.02          # 2% trailing stop
    take_profit_levels: list[float] = field(default_factory=lambda: [0.03, 0.05, 0.08])
    take_profit_weights: list[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    max_daily_drawdown_pct: float = 0.05 # 5%
    max_weekly_drawdown_pct: float = 0.10 # 10%
    commission_spot: float = 0.001       # 0.1%
    commission_futures: float = 0.0002   # 0.02%

    @classmethod
    def from_config_dict(cls, risk_cfg: dict[str, Any]) -> "RiskConfig":
        return cls(
            max_position_pct=risk_cfg.get("max_position_pct", 0.05),
            max_concurrent_positions=risk_cfg.get("max_concurrent_positions", 3),
            stop_loss_pct=risk_cfg.get("stop_loss_pct", 0.02),
            take_profit_levels=risk_cfg.get("take_profit_levels", [0.03, 0.05, 0.08]),
            take_profit_weights=risk_cfg.get("take_profit_weights", [0.4, 0.35, 0.25]),
            max_daily_drawdown_pct=risk_cfg.get("max_daily_drawdown_pct", 0.05),
            max_weekly_drawdown_pct=risk_cfg.get("max_weekly_drawdown_pct", 0.10),
        )


class RiskManager:
    """Central risk management authority.

    Checks every signal against risk rules before execution.

    Usage:
        rm = RiskManager(config, tracker, portfolio_value=10000)
        approved = rm.check_signal(signal)
        if approved:
            position = rm.create_position(signal, portfolio_value)
    """

    def __init__(
        self,
        config: RiskConfig,
        tracker: PositionTracker,
        initial_capital: float = 10000.0,
    ):
        self.config = config
        self.tracker = tracker
        self.initial_capital = initial_capital
        self.current_portfolio_value = initial_capital

        # Drawdown tracking
        self.peak_portfolio_value = initial_capital
        self.daily_start_value = initial_capital
        self.weekly_start_value = initial_capital
        self._last_daily_reset = datetime.utcnow().date()
        self._last_weekly_reset = datetime.utcnow().isocalendar()[:2]  # (year, week)

        # Trading halt state
        self._trading_halted = False
        self._halt_reason = ""

        logger.info(
            "RiskManager initialized: max_pos=%.1f%%, max_concurrent=%d, stop_loss=%.1f%%, max_daily_dd=%.1f%%",
            config.max_position_pct * 100, config.max_concurrent_positions,
            config.stop_loss_pct * 100, config.max_daily_drawdown_pct * 100,
        )

    @property
    def is_trading_halted(self) -> bool:
        return self._trading_halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    # -------------------------------------------------------------------
    # Portfolio update
    # -------------------------------------------------------------------

    def update_portfolio_value(self, new_value: float) -> None:
        """Update current portfolio value and check drawdown limits."""
        self.current_portfolio_value = new_value

        # Track peak
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value

        # Reset daily/weekly counters if needed
        self._maybe_reset_counters()

        # Check drawdown
        self._check_drawdown()

    def _maybe_reset_counters(self) -> None:
        """Reset daily and weekly drawdown counters on new day/week."""
        now = datetime.utcnow()

        # Daily reset
        if now.date() > self._last_daily_reset:
            self.daily_start_value = self.current_portfolio_value
            self._last_daily_reset = now.date()
            logger.info("Daily drawdown counter reset (start=%.2f)", self.daily_start_value)

        # Weekly reset
        current_week = now.isocalendar()[:2]
        if current_week > self._last_weekly_reset:
            self.weekly_start_value = self.current_portfolio_value
            self._last_weekly_reset = current_week
            logger.info("Weekly drawdown counter reset (start=%.2f)", self.weekly_start_value)

    def _check_drawdown(self) -> None:
        """Check daily and weekly drawdown limits."""
        # Daily drawdown
        if self.daily_start_value > 0:
            daily_dd = (self.daily_start_value - self.current_portfolio_value) / self.daily_start_value
            if daily_dd >= self.config.max_daily_drawdown_pct:
                self._trading_halted = True
                self._halt_reason = f"Daily drawdown limit hit: {daily_dd:.2%} >= {self.config.max_daily_drawdown_pct:.2%}"
                logger.warning("TRADING HALTED: %s", self._halt_reason)
                return

        # Weekly drawdown
        if self.weekly_start_value > 0:
            weekly_dd = (self.weekly_start_value - self.current_portfolio_value) / self.weekly_start_value
            if weekly_dd >= self.config.max_weekly_drawdown_pct:
                self._trading_halted = True
                self._halt_reason = f"Weekly drawdown limit hit: {weekly_dd:.2%} >= {self.config.max_weekly_drawdown_pct:.2%}"
                logger.critical("TRADING HALTED: %s", self._halt_reason)
                return

    def resume_trading(self) -> None:
        """Manually resume trading (e.g., after reset)."""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading manually resumed")

    def reset_daily_for_backtest(self, bar_date) -> None:
        """Reset daily drawdown counter for backtest (called when bar date changes).

        In live trading, _maybe_reset_counters() uses datetime.utcnow() to detect
        new days. In backtest, time doesn't advance, so we must call this manually.

        Args:
            bar_date: a date or datetime object for the current bar
        """
        if hasattr(bar_date, "date"):
            bar_date = bar_date.date()
        if bar_date > self._last_daily_reset:
            self.daily_start_value = self.current_portfolio_value
            self._last_daily_reset = bar_date
            # Also un-halt trading on new day if it was halted by daily drawdown
            if self._trading_halted and "Daily" in self._halt_reason:
                self._trading_halted = False
                self._halt_reason = ""
                logger.info(
                    "Daily drawdown counter reset (backtest): start=%.2f, trading resumed",
                    self.daily_start_value,
                )
            else:
                logger.info("Daily drawdown counter reset (backtest): start=%.2f", self.daily_start_value)

    # -------------------------------------------------------------------
    # Signal checks
    # -------------------------------------------------------------------

    def check_signal(self, signal: TradeSignal) -> tuple[bool, list[str]]:
        """Run all risk checks on a signal.

        Args:
            signal: proposed TradeSignal

        Returns:
            (approved, reasons) where reasons lists any rejections
        """
        if not signal.is_entry:
            # Exits always allowed
            return True, []

        reasons: list[str] = []

        # 1. Trading halted
        if self._trading_halted:
            reasons.append(f"trading_halted: {self._halt_reason}")

        # 2. Concurrent position limit
        if self.tracker.position_count >= self.config.max_concurrent_positions:
            reasons.append(
                f"max_concurrent ({self.tracker.position_count}/{self.config.max_concurrent_positions})"
            )

        # 3. Already have position in this symbol
        if self.tracker.has_position(signal.symbol):
            reasons.append(f"position_exists ({signal.symbol})")

        # 4. Position size check
        position_value = self.current_portfolio_value * signal.position_size_pct
        if signal.position_size_pct > self.config.max_position_pct:
            reasons.append(
                f"position_too_large ({signal.position_size_pct:.1%} > {self.config.max_position_pct:.1%})"
            )

        # 5. Available balance check (rough)
        if position_value > self.current_portfolio_value * self.config.max_position_pct:
            reasons.append("insufficient_portfolio_allocation")

        # 6. Leverage check
        if signal.leverage > 3.0:
            reasons.append(f"leverage_too_high ({signal.leverage}x > 3x)")

        approved = len(reasons) == 0

        if not approved:
            logger.info("Signal rejected by risk manager for %s: %s", signal.symbol, "; ".join(reasons))
        else:
            logger.info("Signal approved for %s (lev=%.1fx, size=%.1f%%)", signal.symbol, signal.leverage, signal.position_size_pct * 100)

        return approved, reasons

    # -------------------------------------------------------------------
    # Position creation
    # -------------------------------------------------------------------

    def create_position(
        self,
        signal: TradeSignal,
        current_price: float,
    ) -> Position | None:
        """Create a Position object from an approved signal.

        Computes actual position size based on portfolio value and risk limits.

        Args:
            signal: approved TradeSignal
            current_price: current market price

        Returns:
            Position object, or None if rejected
        """
        approved, reasons = self.check_signal(signal)
        if not approved:
            return None

        # Compute position value
        position_value = self.current_portfolio_value * signal.position_size_pct

        # Apply commission-adjusted size
        commission = self._get_commission(signal.market)
        position_value_after_fees = position_value * (1 - commission)

        # Compute size in base currency
        if current_price <= 0:
            logger.error("Invalid price for %s: %s", signal.symbol, current_price)
            return None

        size = position_value_after_fees / current_price

        # Set stop-loss price
        stop_loss_price = signal.stop_loss_price or 0.0

        position = Position(
            symbol=signal.symbol,
            market=signal.market.value,
            side="long" if signal.direction == "Long" else "short",
            entry_price=current_price,
            size=size,
            leverage=signal.leverage,
            stop_loss_price=stop_loss_price,
            take_profit_prices=signal.take_profit_prices,
        )

        success = self.tracker.open_position(position)
        if not success:
            return None

        logger.info(
            "Position created: %s %s %s | size=%.8f value=%.2f (portfolio=%.2f)",
            position.market, position.side, signal.symbol,
            size, position_value, self.current_portfolio_value,
        )

        return position

    def _get_commission(self, market: str) -> float:
        """Get commission rate for market type."""
        if market == "futures":
            return self.config.commission_futures
        return self.config.commission_spot

    # -------------------------------------------------------------------
    # Take-profit execution
    # -------------------------------------------------------------------

    def compute_tp_close_size(
        self,
        tp_index: int,
    ) -> float:
        """Compute how much of position to close at a take-profit level.

        Args:
            tp_index: which TP level (0, 1, or 2)

        Returns:
            fraction of initial position to close (0.0 to 1.0)
        """
        if tp_index < len(self.config.take_profit_weights):
            return self.config.take_profit_weights[tp_index]
        # If beyond defined levels, close remaining
        return 1.0

    # -------------------------------------------------------------------
    # Trailing stop update
    # -------------------------------------------------------------------

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
    ) -> bool:
        """Update trailing stop-loss for a position.

        Only tightens the stop, never loosens it.

        Args:
            symbol: trading pair
            current_price: current market price

        Returns:
            True if stop was updated
        """
        position = self.tracker.get_position(symbol)
        if position is None:
            return False

        if position.is_long:
            new_stop = current_price * (1 - self.config.stop_loss_pct)
            # Only tighten (raise the stop)
            if new_stop > position.stop_loss_price:
                old_stop = position.stop_loss_price
                position.stop_loss_price = new_stop
                logger.debug(
                    "Trailing stop updated for %s: %.4f → %.4f",
                    symbol, old_stop, new_stop,
                )
                return True
        else:
            new_stop = current_price * (1 + self.config.stop_loss_pct)
            if new_stop < position.stop_loss_price or position.stop_loss_price == 0:
                position.stop_loss_price = new_stop
                return True

        return False

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return current risk state."""
        daily_dd = 0.0
        weekly_dd = 0.0

        if self.daily_start_value > 0:
            daily_dd = (self.daily_start_value - self.current_portfolio_value) / self.daily_start_value

        if self.weekly_start_value > 0:
            weekly_dd = (self.weekly_start_value - self.current_portfolio_value) / self.weekly_start_value

        total_dd = 0.0
        if self.peak_portfolio_value > 0:
            total_dd = (self.peak_portfolio_value - self.current_portfolio_value) / self.peak_portfolio_value

        return {
            "portfolio_value": self.current_portfolio_value,
            "peak_value": self.peak_portfolio_value,
            "daily_drawdown": daily_dd,
            "weekly_drawdown": weekly_dd,
            "total_drawdown": total_dd,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "open_positions": self.tracker.position_count,
            "max_positions": self.config.max_concurrent_positions,
        }
