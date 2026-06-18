"""
Backtest Engine — Event-driven historical simulation.

Replays OHLCV bars through the full signal→risk→execution pipeline.
Simulates slippage and commission for realistic results.

Flow per bar:
  1. Update portfolio value (mark-to-market)
  2. Check trailing stops on open positions
  3. Check take-profit levels on open positions
  4. Generate signal from features + model prediction
  5. Risk manager approval gate
  6. Simulate order execution with slippage
  7. Record equity curve point
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from execution.position_tracker import PositionTracker, Position, ClosedTrade
from execution.risk_manager import RiskManager, RiskConfig
from signals.generator import SignalGenerator, TradeSignal, SignalAction, MarketType
from signals.filters import FilterConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration from config.yaml → backtest section."""

    start_date: str = "2024-01-01"
    end_date: str = "2025-05-31"
    initial_capital: float = 10000.0
    commission_spot: float = 0.001       # 0.1%
    commission_futures: float = 0.0002   # 0.02%
    slippage: float = 0.0005             # 0.05%
    max_leverage: float = 3.0

    @classmethod
    def from_config_dict(cls, cfg: dict[str, Any]) -> "BacktestConfig":
        return cls(
            start_date=cfg.get("start_date", "2024-01-01"),
            end_date=cfg.get("end_date", "2025-05-31"),
            initial_capital=cfg.get("initial_capital", 10000.0),
            commission_spot=cfg.get("commission_spot", 0.001),
            commission_futures=cfg.get("commission_futures", 0.0002),
            slippage=cfg.get("slippage", 0.0005),
            max_leverage=cfg.get("max_leverage", 3.0),
        )


@dataclass
class EquityPoint:
    """Single point on the equity curve."""

    timestamp: datetime
    equity: float
    cash: float
    unrealized_pnl: float
    positions: int
    drawdown: float


@dataclass
class BacktestResult:
    """Complete output from a backtest run."""

    config: BacktestConfig
    equity_curve: list[EquityPoint]
    closed_trades: list[dict]
    signals_generated: int
    signals_approved: int
    signals_rejected: int
    total_commission: float
    total_slippage_cost: float
    portfolio_values: list[float]
    timestamps: list[datetime]

    @property
    def final_equity(self) -> float:
        return self.portfolio_values[-1] if self.portfolio_values else self.config.initial_capital

    @property
    def total_return(self) -> float:
        if not self.portfolio_values:
            return 0.0
        return (self.final_equity - self.config.initial_capital) / self.config.initial_capital


class BacktestEngine:
    """Event-driven backtesting engine.

    Replays historical data bar-by-bar through the full trading pipeline.

    Usage:
        engine = BacktestEngine(config)
        result = engine.run(
            ohlcv_data=ohlcv_dict,
            feature_data=feature_dict,
            model_predictor=predictor,
        )
    """

    def __init__(
        self,
        config: BacktestConfig,
        risk_config: RiskConfig | None = None,
        signal_config: dict[str, Any] | None = None,
        filter_config: FilterConfig | None = None,
    ):
        self.config = config

        # Initialize components
        max_concurrent = risk_config.max_concurrent_positions if risk_config else 3
        self.tracker = PositionTracker(max_concurrent=max_concurrent)
        risk_cfg = risk_config or RiskConfig(
            commission_spot=config.commission_spot,
            commission_futures=config.commission_futures,
        )
        self.risk_manager = RiskManager(
            config=risk_cfg,
            tracker=self.tracker,
            initial_capital=config.initial_capital,
        )
        self.signal_generator = SignalGenerator(
            config=signal_config,
            filter_config=filter_config,
        )

        # State
        self.cash = config.initial_capital
        self._slippage_cost_total = 0.0
        self._commission_total = 0.0

    def run(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        feature_data: dict[str, pd.DataFrame],
        model_predictor: Callable[[pd.DataFrame, str], Any] | None = None,
    ) -> BacktestResult:
        """Run backtest across all symbols.

        Args:
            ohlcv_data: {symbol: DataFrame with OHLCV columns indexed by datetime}
            feature_data: {symbol: DataFrame with features indexed by datetime}
            model_predictor: callable(features_df, symbol) → Prediction object

        Returns:
            BacktestResult with full analytics
        """
        logger.info(
            "Starting backtest: %s to %s | Capital=$%.0f | Slippage=%.3f%%",
            self.config.start_date, self.config.end_date,
            self.config.initial_capital, self.config.slippage * 100,
        )

        start_ts = pd.Timestamp(self.config.start_date)
        end_ts = pd.Timestamp(self.config.end_date)

        # Build unified timeline from all symbols
        all_timestamps: set[pd.Timestamp] = set()
        for symbol, df in ohlcv_data.items():
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            all_timestamps.update(df.index[mask])

        timeline = sorted(all_timestamps)

        if not timeline:
            logger.warning("No data in date range %s to %s", self.config.start_date, self.config.end_date)
            return self._empty_result()

        # Equity tracking
        equity_curve: list[EquityPoint] = []
        portfolio_values: list[float] = []
        timestamps: list[datetime] = []
        signals_generated = 0
        signals_approved = 0
        signals_rejected = 0

        peak_equity = self.config.initial_capital

        last_bar_date = None

        for i, bar_time in enumerate(timeline):
            # Reset daily drawdown counter when calendar date changes (backtest)
            bar_date = bar_time.date() if hasattr(bar_time, "date") else bar_time
            if last_bar_date is not None and bar_date != last_bar_date:
                self.risk_manager.reset_daily_for_backtest(bar_date)
            last_bar_date = bar_date

            # Current prices for all symbols at this bar
            current_prices: dict[str, float] = {}
            for symbol, df in ohlcv_data.items():
                if bar_time in df.index:
                    current_prices[symbol] = float(df.loc[bar_time, "close"])

            # --- Step 1: Update portfolio value ---
            # Use positions_value (full market value) NOT unrealized_pnl (just the delta)
            # Cash was debited on entry, so equity = cash + current value of all positions
            positions_value = self.tracker.compute_positions_value(current_prices)
            total_equity = self.cash + positions_value
            self.risk_manager.update_portfolio_value(total_equity)

            if total_equity > peak_equity:
                peak_equity = total_equity

            # Convert bar_time to unix timestamp for duration tracking
            bar_ts = bar_time.timestamp() if hasattr(bar_time, "timestamp") else float(bar_time.value) / 1e9

            # --- Step 2: Check trailing stops ---
            sl_triggers = self.tracker.check_stop_losses(current_prices)
            for symbol in sl_triggers:
                price = current_prices[symbol]
                slippage_price = self._apply_slippage(symbol, price, "sell")
                trade = self.tracker.close_position(
                    symbol, slippage_price,
                    close_reason="stop_loss",
                    bar_timestamp=bar_ts,
                )
                if trade:
                    # Direction-aware close: long sells at market, short buys back
                    if trade.side == "long":
                        close_value = trade.size * slippage_price
                    else:
                        close_value = trade.size * (2 * trade.entry_price - slippage_price)
                    fee = self._compute_commission(trade.market, abs(close_value))
                    self._commission_total += fee
                    self.cash += close_value - fee

            # --- Step 3: Check take-profits ---
            tp_triggers = self.tracker.check_take_profits(current_prices)
            for symbol, tp_idx in tp_triggers:
                price = current_prices[symbol]
                pos = self.tracker.get_position(symbol)
                if pos is None:
                    continue

                close_fraction = self.risk_manager.compute_tp_close_size(tp_idx)
                close_size = pos.initial_size * close_fraction

                slippage_price = self._apply_slippage(symbol, price, "sell" if pos.is_long else "buy")
                info = pos.close(close_size)
                # Direction-aware close value
                if pos.is_long:
                    close_value = close_size * slippage_price
                else:
                    close_value = close_size * (2 * pos.entry_price - slippage_price)
                pnl = 0.0
                if pos.is_long:
                    pnl = (slippage_price - pos.entry_price) * close_size
                else:
                    pnl = (pos.entry_price - slippage_price) * close_size
                fee = self._compute_commission(pos.market, abs(close_value))
                self._commission_total += fee
                self.cash += close_value - fee
                pos.tp_hits.append(float(tp_idx))

                # Remove fully closed positions
                if info["is_full_close"]:
                    self.tracker._positions.pop(symbol, None)
                    duration_hours = (bar_ts - pos.entry_time) / 3600.0
                    trade = ClosedTrade(
                        symbol=symbol,
                        market=pos.market,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=slippage_price,
                        size=pos.size + close_size,
                        leverage=pos.leverage,
                        entry_time=pos.entry_time,
                        exit_time=bar_ts,
                        pnl=pnl,
                        pnl_pct=pnl / (pos.entry_price * pos.initial_size) if pos.entry_price > 0 else 0,
                        fees=fee,
                        duration_hours=duration_hours,
                        close_reason="take_profit",
                    )
                    self.tracker._trade_history.append(trade)

            # --- Step 4: Update trailing stops for remaining positions ---
            for pos in self.tracker.get_all_positions():
                if pos.symbol in current_prices:
                    self.risk_manager.update_trailing_stop(pos.symbol, current_prices[pos.symbol])

            # --- Step 5: Generate signals ---
            for symbol in ohlcv_data:
                if symbol not in current_prices:
                    continue
                if symbol not in feature_data:
                    continue

                # Get features up to current bar
                feat_df = feature_data[symbol]
                if bar_time not in feat_df.index:
                    continue

                # Slice features up to current bar (no look-ahead)
                available = feat_df.loc[:bar_time]
                if len(available) < 30:  # minimum for indicators
                    continue

                # Model prediction
                prediction = None
                if model_predictor is not None:
                    try:
                        prediction = model_predictor(available, symbol)
                    except Exception as e:
                        logger.debug("Predictor error for %s: %s", symbol, e)
                        continue

                # Generate signal
                signal = self.signal_generator.generate(
                    prediction=prediction,
                    current_price=current_prices[symbol],
                    features_df=available,
                )

                if signal.action == SignalAction.HOLD:
                    continue

                signals_generated += 1

                # --- Step 6: Risk check ---
                if signal.is_entry:
                    approved, reasons = self.risk_manager.check_signal(signal)
                    if not approved:
                        signals_rejected += 1
                        continue

                    signals_approved += 1

                    # --- Step 7: Simulate execution ---
                    self._execute_entry(signal, current_prices[symbol], bar_ts=bar_ts)

                elif signal.is_exit:
                    # Close existing position
                    pos = self.tracker.get_position(signal.symbol)
                    if pos:
                        exec_price = self._apply_slippage(
                            signal.symbol, current_prices[signal.symbol],
                            "sell" if pos.is_long else "buy",
                        )
                        trade = self.tracker.close_position(
                            signal.symbol, exec_price, close_reason="signal",
                            bar_timestamp=bar_ts,
                        )
                        if trade:
                            # Direction-aware close
                            if trade.side == "long":
                                close_value = trade.size * exec_price
                            else:
                                close_value = trade.size * (2 * trade.entry_price - exec_price)
                            fee = self._compute_commission(trade.market, abs(close_value))
                            self._commission_total += fee
                            self.cash += close_value - fee

            # --- Record equity ---
            positions_value = self.tracker.compute_positions_value(current_prices)
            total_equity = self.cash + positions_value
            drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0

            equity_curve.append(EquityPoint(
                timestamp=bar_time.to_pydatetime(),
                equity=total_equity,
                cash=self.cash,
                unrealized_pnl=self.tracker.compute_unrealized_pnl(current_prices),
                positions=self.tracker.position_count,
                drawdown=drawdown,
            ))
            portfolio_values.append(total_equity)
            timestamps.append(bar_time.to_pydatetime())

        # Close any remaining positions at last price
        if timeline:
            last_prices: dict[str, float] = {}
            for symbol, df in ohlcv_data.items():
                if timeline[-1] in df.index:
                    last_prices[symbol] = float(df.loc[timeline[-1], "close"])

            last_bar_ts = timeline[-1].timestamp() if hasattr(timeline[-1], "timestamp") else float(timeline[-1].value) / 1e9
            for symbol in list(self.tracker._positions.keys()):
                price = last_prices.get(symbol)
                if price:
                    trade = self.tracker.close_position(
                        symbol, price, close_reason="backtest_end",
                        bar_timestamp=last_bar_ts,
                    )
                    if trade:
                        # Direction-aware close
                        if trade.side == "long":
                            close_value = trade.size * price
                        else:
                            close_value = trade.size * (2 * trade.entry_price - price)
                        fee = self._compute_commission(trade.market, abs(close_value))
                        self._commission_total += fee
                        self.cash += close_value - fee

        logger.info(
            "Backtest complete: %d signals (%d approved) | $%.2f → $%.2f (%.2f%%)",
            signals_generated, signals_approved,
            self.config.initial_capital, self.cash,
            (self.cash - self.config.initial_capital) / self.config.initial_capital * 100,
        )

        return BacktestResult(
            config=self.config,
            equity_curve=equity_curve,
            closed_trades=[t.to_dict() for t in self.tracker._trade_history],
            signals_generated=signals_generated,
            signals_approved=signals_approved,
            signals_rejected=signals_rejected,
            total_commission=self._commission_total,
            total_slippage_cost=self._slippage_cost_total,
            portfolio_values=portfolio_values,
            timestamps=timestamps,
        )

    def _execute_entry(self, signal: TradeSignal, current_price: float, bar_ts: float = 0.0) -> None:
        """Simulate order entry with slippage."""
        exec_price = self._apply_slippage(
            signal.symbol, current_price,
            "buy" if signal.is_entry and signal.direction == "Long" else "sell",
        )

        # Recompute position with slippage-adjusted price
        position_value = self.cash * signal.position_size_pct
        commission = self._get_commission(signal.market)
        position_value_after_fees = position_value * (1 - commission)
        size = position_value_after_fees / exec_price if exec_price > 0 else 0

        if size <= 0:
            return

        # Deduct from cash
        self.cash -= position_value

        stop_loss_price = signal.stop_loss_price or 0.0

        position = Position(
            symbol=signal.symbol,
            market=signal.market.value,
            side="long" if signal.direction == "Long" else "short",
            entry_price=exec_price,
            size=size,
            leverage=signal.leverage,
            entry_time=bar_ts if bar_ts > 0 else time.time(),
            stop_loss_price=stop_loss_price,
            take_profit_prices=signal.take_profit_prices or [],
        )

        self.tracker.open_position(position)

    def _apply_slippage(self, symbol: str, price: float, side: str) -> float:
        """Apply slippage to execution price.

        Buy:  price goes up (unfavorable)
        Sell: price goes down (unfavorable)
        """
        slippage = self.config.slippage
        if side == "buy":
            exec_price = price * (1 + slippage)
        else:
            exec_price = price * (1 - slippage)

        cost = abs(exec_price - price) * 0.001  # rough slippage cost tracking
        self._slippage_cost_total += cost

        return exec_price

    def _get_commission(self, market: str) -> float:
        """Get commission rate for market."""
        if market == "futures":
            return self.config.commission_futures
        return self.config.commission_spot

    def _compute_commission(self, market: str, notional: float) -> float:
        """Compute commission fee."""
        rate = self._get_commission(market)
        return notional * rate

    def _empty_result(self) -> BacktestResult:
        """Return empty result for no-data case."""
        return BacktestResult(
            config=self.config,
            equity_curve=[],
            closed_trades=[],
            signals_generated=0,
            signals_approved=0,
            signals_rejected=0,
            total_commission=0.0,
            total_slippage_cost=0.0,
            portfolio_values=[self.config.initial_capital],
            timestamps=[],
        )
