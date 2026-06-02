"""
Backtest Analytics — Performance metrics and reporting.

Computes:
  - Sharpe Ratio (annualized, rf=0)
  - Sortino Ratio
  - Max Drawdown + Duration
  - Win Rate, Profit Factor
  - Calmar Ratio
  - Average Trade Duration
  - Per-symbol breakdown
  - Monthly returns heatmap data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

# 365 trading days for crypto (24/7)
TRADING_DAYS_PER_YEAR = 365


@dataclass
class PerformanceMetrics:
    """All computed performance metrics."""

    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    cagr_pct: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    volatility_annual: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0
    avg_trade_duration_hours: float = 0.0
    expectancy: float = 0.0

    # Cost metrics
    total_commission: float = 0.0
    total_slippage_cost: float = 0.0
    total_costs: float = 0.0

    # Signal metrics
    signals_generated: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    signal_approval_rate_pct: float = 0.0

    # Monthly returns (for heatmap)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    # Per-symbol breakdown
    symbol_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "volatility_annual": self.volatility_annual,
            "total_trades": self.total_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "expectancy": self.expectancy,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "total_commission": self.total_commission,
            "total_slippage_cost": self.total_slippage_cost,
            "signals_generated": self.signals_generated,
            "signals_approved": self.signals_approved,
            "signal_approval_rate_pct": self.signal_approval_rate_pct,
            "monthly_returns": self.monthly_returns,
            "symbol_stats": self.symbol_stats,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═══════════════════════════════════════════",
            "         BACKTEST PERFORMANCE SUMMARY      ",
            "═══════════════════════════════════════════",
            f"  Total Return:         {self.total_return_pct:>+8.2f}%",
            f"  Annualized Return:    {self.annualized_return_pct:>+8.2f}%",
            f"  Sharpe Ratio:         {self.sharpe_ratio:>8.3f}",
            f"  Sortino Ratio:        {self.sortino_ratio:>8.3f}",
            f"  Calmar Ratio:         {self.calmar_ratio:>8.3f}",
            f"  Max Drawdown:         {self.max_drawdown_pct:>8.2f}%",
            f"  Max DD Duration:      {self.max_drawdown_duration_days:>8.1f} days",
            f"  Volatility (ann.):    {self.volatility_annual:>8.2f}%",
            "───────────────────────────────────────────",
            f"  Total Trades:         {self.total_trades:>8d}",
            f"  Win Rate:             {self.win_rate_pct:>8.1f}%",
            f"  Profit Factor:        {self.profit_factor:>8.3f}",
            f"  Avg Win:              {self.avg_win_pct:>+8.2f}%",
            f"  Avg Loss:             {self.avg_loss_pct:>+8.2f}%",
            f"  Expectancy:           {self.expectancy:>+8.4f}",
            f"  Avg Duration:         {self.avg_trade_duration_hours:>8.1f} hrs",
            "───────────────────────────────────────────",
            f"  Total Commission:     ${self.total_commission:>8.2f}",
            f"  Total Slippage:       ${self.total_slippage_cost:>8.2f}",
            f"  Signal Approval Rate: {self.signal_approval_rate_pct:>8.1f}%",
            "═══════════════════════════════════════════",
        ]
        return "\n".join(lines)


class Analytics:
    """Compute performance analytics from backtest results."""

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: annual risk-free rate (default 0 for crypto)
        """
        self.rf = risk_free_rate

    def compute(self, result: BacktestResult) -> PerformanceMetrics:
        """Compute all metrics from backtest result."""
        metrics = PerformanceMetrics()

        if not result.portfolio_values or len(result.portfolio_values) < 2:
            return metrics

        values = np.array(result.portfolio_values, dtype=np.float64)
        initial = result.config.initial_capital

        # --- Return metrics ---
        metrics.total_return_pct = (values[-1] - initial) / initial * 100

        n_bars = len(values)
        # Estimate bars per year from timestamps
        if result.timestamps and len(result.timestamps) >= 2:
            total_seconds = (result.timestamps[-1] - result.timestamps[0]).total_seconds()
            bars_per_year = n_bars / (total_seconds / (365.25 * 86400)) if total_seconds > 0 else n_bars
        else:
            bars_per_year = n_bars

        years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
        metrics.cagr_pct = ((values[-1] / initial) ** (1 / max(years, 0.01)) - 1) * 100
        metrics.annualized_return_pct = metrics.cagr_pct

        # --- Daily returns for Sharpe/Sortino ---
        returns = np.diff(values) / values[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) > 0:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-10

            # Annualize
            annual_factor = np.sqrt(bars_per_year)
            metrics.volatility_annual = std_ret * annual_factor * 100

            # Sharpe
            excess = mean_ret - self.rf / bars_per_year
            metrics.sharpe_ratio = (excess / std_ret * annual_factor) if std_ret > 0 else 0.0

            # Sortino (downside deviation)
            downside = returns[returns < 0]
            if len(downside) > 0:
                downside_std = np.std(downside, ddof=1)
                metrics.sortino_ratio = (excess / downside_std * annual_factor) if downside_std > 0 else 0.0
            else:
                metrics.sortino_ratio = float("inf") if excess > 0 else 0.0

        # --- Max Drawdown ---
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        metrics.max_drawdown_pct = float(np.max(drawdowns)) * 100

        # Max drawdown duration
        in_dd = drawdowns > 0
        if np.any(in_dd):
            dd_groups = np.diff(np.concatenate(([0], in_dd.astype(int), [0])))
            dd_starts = np.where(dd_groups == 1)[0]
            dd_ends = np.where(dd_groups == -1)[0]
            if len(dd_starts) > 0 and len(dd_ends) > 0:
                durations = dd_ends[:len(dd_starts)] - dd_starts[:len(dd_ends)]
                max_dd_bars = int(np.max(durations))
                # Convert bars to days
                metrics.max_drawdown_duration_days = max_dd_bars / (bars_per_year / 365.25)

        # Calmar
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct

        # --- Trade metrics ---
        trades = result.closed_trades
        metrics.total_trades = len(trades)
        metrics.signals_generated = result.signals_generated
        metrics.signals_approved = result.signals_approved
        metrics.signals_rejected = result.signals_rejected

        if result.signals_generated > 0:
            metrics.signal_approval_rate_pct = result.signals_approved / result.signals_generated * 100

        if trades:
            pnls = [t.get("pnl", 0) for t in trades]
            pnl_pcts = [t.get("pnl_pct", 0) for t in trades]
            durations = [t.get("duration_hours", 0) for t in trades]

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_pcts = [p for p in pnl_pcts if p > 0]
            loss_pcts = [p for p in pnl_pcts if p <= 0]

            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            metrics.win_rate_pct = len(wins) / len(trades) * 100 if trades else 0

            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

            metrics.avg_win_pct = np.mean(win_pcts) * 100 if win_pcts else 0
            metrics.avg_loss_pct = np.mean(loss_pcts) * 100 if loss_pcts else 0
            metrics.largest_win_pct = max(win_pcts) * 100 if win_pcts else 0
            metrics.largest_loss_pct = min(loss_pcts) * 100 if loss_pcts else 0
            metrics.avg_trade_duration_hours = np.mean(durations) if durations else 0

            # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            wr = metrics.win_rate_pct / 100
            avg_w = np.mean(win_pcts) if win_pcts else 0
            avg_l = abs(np.mean(loss_pcts)) if loss_pcts else 0
            metrics.expectancy = (wr * avg_w) - ((1 - wr) * avg_l)

        # --- Cost metrics ---
        metrics.total_commission = result.total_commission
        metrics.total_slippage_cost = result.total_slippage_cost
        metrics.total_costs = result.total_commission + result.total_slippage_cost

        # --- Monthly returns ---
        metrics.monthly_returns = self._compute_monthly_returns(result)

        # --- Per-symbol breakdown ---
        metrics.symbol_stats = self._compute_symbol_stats(trades)

        return metrics

    def _compute_monthly_returns(self, result: BacktestResult) -> dict[str, float]:
        """Compute monthly return percentages."""
        if not result.portfolio_values or not result.timestamps:
            return {}

        monthly: dict[str, list[float]] = {}
        for val, ts in zip(result.portfolio_values, result.timestamps):
            key = ts.strftime("%Y-%m")
            monthly.setdefault(key, []).append(val)

        monthly_returns: dict[str, float] = {}
        prev_end = result.config.initial_capital

        for month_key in sorted(monthly.keys()):
            month_vals = monthly[month_key]
            month_end = month_vals[-1]
            monthly_returns[month_key] = (month_end - prev_end) / prev_end * 100
            prev_end = month_end

        return monthly_returns

    def _compute_symbol_stats(self, trades: list[dict]) -> dict[str, dict[str, Any]]:
        """Per-symbol trade statistics."""
        symbol_trades: dict[str, list[dict]] = {}
        for t in trades:
            sym = t.get("symbol", "unknown")
            symbol_trades.setdefault(sym, []).append(t)

        stats: dict[str, dict[str, Any]] = {}
        for sym, sym_trades in symbol_trades.items():
            pnls = [t.get("pnl", 0) for t in sym_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            total_wins = sum(wins)
            total_losses = abs(sum(losses))

            stats[sym] = {
                "total_trades": len(sym_trades),
                "win_rate": len(wins) / len(sym_trades) * 100 if sym_trades else 0,
                "total_pnl": sum(pnls),
                "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "best_trade": max(pnls) if pnls else 0,
                "worst_trade": min(pnls) if pnls else 0,
            }

        return stats

    def compare_results(self, results: list[BacktestResult]) -> dict[str, Any]:
        """Compare multiple backtest results (e.g., walk-forward windows)."""
        metrics_list = [self.compute(r) for r in results]

        if not metrics_list:
            return {}

        return {
            "num_windows": len(metrics_list),
            "avg_return_pct": np.mean([m.total_return_pct for m in metrics_list]),
            "std_return_pct": np.std([m.total_return_pct for m in metrics_list]),
            "avg_sharpe": np.mean([m.sharpe_ratio for m in metrics_list]),
            "avg_win_rate": np.mean([m.win_rate_pct for m in metrics_list]),
            "avg_profit_factor": np.mean([m.profit_factor for m in metrics_list]),
            "avg_max_drawdown": np.mean([m.max_drawdown_pct for m in metrics_list]),
            "worst_drawdown": max(m.max_drawdown_pct for m in metrics_list),
            "total_trades": sum(m.total_trades for m in metrics_list),
            "per_window": [m.to_dict() for m in metrics_list],
        }
