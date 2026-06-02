"""
Position Tracker — Tracks all open and historical positions.

Maintains state of:
  - Current positions per symbol (entry price, size, leverage, market type)
  - PnL tracking (realized + unrealized)
  - Position history for analytics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single open position."""

    symbol: str
    market: str            # "spot" | "futures"
    side: str              # "long" | "short"
    entry_price: float
    size: float            # quantity in base currency
    leverage: float = 1.0
    entry_time: float = 0.0  # unix timestamp

    # Trailing stop / take-profit tracking
    stop_loss_price: float = 0.0
    take_profit_prices: list[float] = field(default_factory=list)
    tp_hits: list[float] = field(default_factory=list)  # prices where TP was hit

    # Partial exit tracking
    initial_size: float = 0.0  # original size before partial exits
    closed_size: float = 0.0   # how much was closed via TP

    def __post_init__(self):
        if self.initial_size == 0.0:
            self.initial_size = self.size
        if self.entry_time == 0.0:
            self.entry_time = time.time()

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def notional_value(self) -> float:
        """Current notional value based on entry price."""
        return self.size * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized PnL."""
        if self.is_long:
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Compute unrealized PnL as percentage."""
        if self.is_long:
            return (current_price - self.entry_price) / self.entry_price * self.leverage
        else:
            return (self.entry_price - current_price) / self.entry_price * self.leverage

    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss should trigger."""
        if self.stop_loss_price <= 0:
            return False
        if self.is_long:
            return current_price <= self.stop_loss_price
        else:
            return current_price >= self.stop_loss_price

    def check_take_profit(self, current_price: float) -> int | None:
        """Check if any take-profit level is hit.

        Returns:
            index of TP level hit, or None
        """
        for i, tp_price in enumerate(self.take_profit_prices):
            if i in [int(x) for x in self.tp_hits]:
                continue  # already hit
            if self.is_long and current_price >= tp_price:
                return i
            elif not self.is_long and current_price <= tp_price:
                return i
        return None

    def close(self, size: float) -> dict[str, Any]:
        """Partially or fully close this position.

        Returns:
            closure info dict
        """
        closed = min(size, self.size)
        self.size -= closed
        self.closed_size += closed

        return {
            "symbol": self.symbol,
            "market": self.market,
            "side": self.side,
            "closed_size": closed,
            "remaining_size": self.size,
            "entry_price": self.entry_price,
            "is_full_close": self.size <= 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_prices": self.take_profit_prices,
            "initial_size": self.initial_size,
            "closed_size": self.closed_size,
        }


@dataclass
class ClosedTrade:
    """A completed trade (position fully closed)."""

    symbol: str
    market: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    entry_time: float
    exit_time: float
    pnl: float           # absolute PnL
    pnl_pct: float       # percentage PnL
    fees: float = 0.0
    duration_hours: float = 0.0
    close_reason: str = "signal"  # signal | stop_loss | take_profit | manual

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "leverage": self.leverage,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "fees": self.fees,
            "duration_hours": self.duration_hours,
            "close_reason": self.close_reason,
        }


class PositionTracker:
    """Manages all open positions and closed trade history.

    Usage:
        tracker = PositionTracker()
        tracker.open_position(Position(symbol="BTC/USDT", ...))
        tracker.close_position("BTC/USDT", exit_price=66000, reason="take_profit")
        print(tracker.get_stats())
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._positions: dict[str, Position] = {}  # symbol → Position
        self._trade_history: list[ClosedTrade] = []

        # Running stats
        self._total_realized_pnl: float = 0.0
        self._total_fees: float = 0.0

    # -------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------

    def open_position(self, position: Position) -> bool:
        """Open a new position.

        Returns:
            True if opened, False if rejected (max positions, duplicate)
        """
        if position.symbol in self._positions:
            logger.warning("Position already exists for %s", position.symbol)
            return False

        if len(self._positions) >= self.max_concurrent:
            logger.warning(
                "Max concurrent positions (%d) reached. Cannot open %s",
                self.max_concurrent, position.symbol,
            )
            return False

        self._positions[position.symbol] = position
        logger.info(
            "Position opened: %s %s %s size=%.8f entry=%.4f lev=%.1fx",
            position.market, position.side, position.symbol,
            position.size, position.entry_price, position.leverage,
        )
        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        close_reason: str = "signal",
        fees: float = 0.0,
    ) -> ClosedTrade | None:
        """Close a position by symbol.

        Returns:
            ClosedTrade if position existed, None otherwise
        """
        pos = self._positions.pop(symbol, None)
        if pos is None:
            logger.warning("No position to close for %s", symbol)
            return None

        # Compute PnL
        pnl = pos.unrealized_pnl(exit_price)
        pnl_pct = pos.unrealized_pnl_pct(exit_price)

        exit_time = time.time()
        duration_hours = (exit_time - pos.entry_time) / 3600.0

        trade = ClosedTrade(
            symbol=symbol,
            market=pos.market,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            leverage=pos.leverage,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=fees,
            duration_hours=duration_hours,
            close_reason=close_reason,
        )

        self._trade_history.append(trade)
        self._total_realized_pnl += pnl
        self._total_fees += fees

        logger.info(
            "Position closed: %s %s %s | entry=%.4f exit=%.4f | pnl=%.2f (%.2f%%) | reason=%s",
            pos.market, pos.side, symbol,
            pos.entry_price, exit_price,
            pnl, pnl_pct * 100, close_reason,
        )

        return trade

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists."""
        return symbol in self._positions

    def get_all_positions(self) -> list[Position]:
        """Return all open positions."""
        return list(self._positions.values())

    @property
    def position_count(self) -> int:
        return len(self._positions)

    # -------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------

    def compute_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Compute total unrealized PnL across all positions."""
        total = 0.0
        for symbol, pos in self._positions.items():
            price = prices.get(symbol, pos.entry_price)
            total += pos.unrealized_pnl(price)
        return total

    def get_total_pnl(self, prices: dict[str, float]) -> float:
        """Total PnL = realized + unrealized."""
        return self._total_realized_pnl + self.compute_unrealized_pnl(prices)

    def check_stop_losses(self, prices: dict[str, float]) -> list[str]:
        """Check all positions for stop-loss triggers.

        Returns:
            list of symbols where stop-loss should trigger
        """
        triggered: list[str] = []
        for symbol, pos in self._positions.items():
            price = prices.get(symbol, pos.entry_price)
            if pos.should_stop_loss(price):
                triggered.append(symbol)
        return triggered

    def check_take_profits(self, prices: dict[str, float]) -> list[tuple[str, int]]:
        """Check all positions for take-profit triggers.

        Returns:
            list of (symbol, tp_index) tuples
        """
        triggered: list[tuple[str, int]] = []
        for symbol, pos in self._positions.items():
            price = prices.get(symbol, pos.entry_price)
            tp_idx = pos.check_take_profit(price)
            if tp_idx is not None:
                triggered.append((symbol, tp_idx))
        return triggered

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Compute performance statistics from trade history."""
        trades = self._trade_history
        n = len(trades)

        if n == 0:
            return {
                "total_trades": 0,
                "total_realized_pnl": 0.0,
                "total_fees": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_duration_hours": 0.0,
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
                "profit_factor": 0.0,
                "open_positions": self.position_count,
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))

        return {
            "total_trades": n,
            "total_realized_pnl": self._total_realized_pnl,
            "total_fees": self._total_fees,
            "win_rate": len(wins) / n if n > 0 else 0.0,
            "avg_pnl": sum(t.pnl for t in trades) / n,
            "avg_pnl_pct": sum(t.pnl_pct for t in trades) / n,
            "avg_duration_hours": sum(t.duration_hours for t in trades) / n,
            "best_trade_pnl": max(t.pnl for t in trades),
            "worst_trade_pnl": min(t.pnl for t in trades),
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            "open_positions": self.position_count,
        }

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        """Return recent closed trades."""
        return [t.to_dict() for t in self._trade_history[-limit:]]
