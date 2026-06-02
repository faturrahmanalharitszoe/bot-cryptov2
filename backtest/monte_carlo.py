"""
Monte Carlo Simulation — Robustness testing via trade resampling.

Shuffles trade outcomes to generate confidence intervals for:
  - Final equity distribution
  - Max drawdown distribution
  - Sharpe ratio distribution
  - Probability of ruin

Helps answer: "Was our backtest result luck or skill?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtest.engine import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration."""

    num_simulations: int = 1000
    confidence_levels: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])
    random_seed: int | None = None


@dataclass
class MonteCarloResult:
    """Output from Monte Carlo simulation."""

    num_simulations: int
    final_equities: np.ndarray
    max_drawdowns: np.ndarray
    sharpe_ratios: np.ndarray
    annual_returns: np.ndarray

    # Percentile distributions
    equity_percentiles: dict[float, float] = field(default_factory=dict)
    drawdown_percentiles: dict[float, float] = field(default_factory=dict)
    sharpe_percentiles: dict[float, float] = field(default_factory=dict)

    # Risk metrics
    probability_of_ruin: float = 0.0  # P(equity < 50% of initial)
    median_final_equity: float = 0.0
    worst_case_equity: float = 0.0
    best_case_equity: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═══════════════════════════════════════════",
            "       MONTE CARLO SIMULATION RESULTS      ",
            "═══════════════════════════════════════════",
            f"  Simulations:          {self.num_simulations:>8d}",
            f"  Prob. of Ruin:        {self.probability_of_ruin:>8.2%}",
            f"  Median Final Equity:  ${self.median_final_equity:>10.2f}",
            f"  Worst Case Equity:    ${self.worst_case_equity:>10.2f}",
            f"  Best Case Equity:     ${self.best_case_equity:>10.2f}",
            "───────────────────────────────────────────",
            "  Equity Distribution:",
        ]

        for level, value in sorted(self.equity_percentiles.items()):
            lines.append(f"    {level:.0%} percentile:     ${value:>10.2f}")

        lines.extend([
            "───────────────────────────────────────────",
            "  Max Drawdown Distribution:",
        ])
        for level, value in sorted(self.drawdown_percentiles.items()):
            lines.append(f"    {level:.0%} percentile:     {value:>8.2f}%")

        lines.extend([
            "───────────────────────────────────────────",
            "  Sharpe Ratio Distribution:",
        ])
        for level, value in sorted(self.sharpe_percentiles.items()):
            lines.append(f"    {level:.0%} percentile:     {value:>8.3f}")

        lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_simulations": self.num_simulations,
            "probability_of_ruin": self.probability_of_ruin,
            "median_final_equity": self.median_final_equity,
            "worst_case_equity": self.worst_case_equity,
            "best_case_equity": self.best_case_equity,
            "equity_percentiles": self.equity_percentiles,
            "drawdown_percentiles": self.drawdown_percentiles,
            "sharpe_percentiles": self.sharpe_percentiles,
        }


class MonteCarloSimulator:
    """Monte Carlo simulation via trade resampling.

    Usage:
        simulator = MonteCarloSimulator(config)
        mc_result = simulator.run(backtest_result)
    """

    def __init__(self, config: MonteCarloConfig | None = None):
        self.config = config or MonteCarloConfig()

    def run(self, result: BacktestResult) -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Method: Shuffle trade PnLs and replay them through the portfolio
        to generate distribution of possible outcomes.

        Args:
            result: BacktestResult with equity curve and trades

        Returns:
            MonteCarloResult with distributions and risk metrics
        """
        trades = result.closed_trades
        initial_capital = result.config.initial_capital

        if not trades or len(trades) < 5:
            logger.warning("Not enough trades (%d) for meaningful Monte Carlo", len(trades))
            return self._empty_result(initial_capital)

        rng = np.random.default_rng(self.config.random_seed)

        # Extract trade PnLs as percentage of portfolio at entry
        trade_returns = np.array([t.get("pnl_pct", 0) for t in trades], dtype=np.float64)
        trade_pnls = np.array([t.get("pnl", 0) for t in trades], dtype=np.float64)

        n_sims = self.config.num_simulations
        n_trades = len(trades)

        # Run simulations
        final_equities = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        sharpe_ratios = np.zeros(n_sims)
        annual_returns = np.zeros(n_sims)

        for i in range(n_sims):
            # Shuffle trade order
            perm = rng.permutation(n_trades)
            sim_returns = trade_returns[perm]

            # Replay through portfolio
            equity = initial_capital
            peak = equity
            max_dd = 0.0
            equity_path = [equity]

            for ret in sim_returns:
                pnl = equity * ret
                equity += pnl
                equity = max(equity, 0)  # floor at 0

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                equity_path.append(equity)

            final_equities[i] = equity
            max_drawdowns[i] = max_dd * 100

            # Compute Sharpe from simulated path
            path_returns = np.diff(equity_path) / np.array(equity_path[:-1])
            path_returns = path_returns[np.isfinite(path_returns)]

            if len(path_returns) > 1:
                mean_r = np.mean(path_returns)
                std_r = np.std(path_returns, ddof=1)
                # Assume ~365 trading days, scale to annual
                sharpe_ratios[i] = (mean_r / std_r * np.sqrt(365)) if std_r > 0 else 0
            else:
                sharpe_ratios[i] = 0

            # Annualized return (approximate from total return)
            total_ret = (equity - initial_capital) / initial_capital
            # Assume backtest covers some period; use simple annualization
            annual_returns[i] = total_ret * (365 / max(len(trades), 1))

        # Compute percentiles
        mc = MonteCarloResult(
            num_simulations=n_sims,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            sharpe_ratios=sharpe_ratios,
            annual_returns=annual_returns,
        )

        for level in self.config.confidence_levels:
            mc.equity_percentiles[level] = float(np.percentile(final_equities, level * 100))
            mc.drawdown_percentiles[level] = float(np.percentile(max_drawdowns, level * 100))
            mc.sharpe_percentiles[level] = float(np.percentile(sharpe_ratios, level * 100))

        # Risk of ruin: P(final equity < 50% of initial)
        ruin_threshold = initial_capital * 0.5
        mc.probability_of_ruin = float(np.mean(final_equities < ruin_threshold))

        mc.median_final_equity = float(np.median(final_equities))
        mc.worst_case_equity = float(np.min(final_equities))
        mc.best_case_equity = float(np.max(final_equities))

        logger.info(
            "Monte Carlo complete: median=$%.0f, ruin=%.1f%%, worst=$%.0f, best=$%.0f",
            mc.median_final_equity, mc.probability_of_ruin * 100,
            mc.worst_case_equity, mc.best_case_equity,
        )

        return mc

    def run_with_bootstrap(
        self,
        result: BacktestResult,
        bootstrap_samples: int = 100,
    ) -> MonteCarloResult:
        """Bootstrap variant: sample trades with replacement.

        Better for small trade samples — generates more diverse scenarios.
        """
        trades = result.closed_trades
        initial_capital = result.config.initial_capital

        if not trades or len(trades) < 5:
            return self._empty_result(initial_capital)

        rng = np.random.default_rng(self.config.random_seed)
        trade_returns = np.array([t.get("pnl_pct", 0) for t in trades], dtype=np.float64)

        n_sims = self.config.num_simulations
        n_trades = len(trades)

        final_equities = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        sharpe_ratios = np.zeros(n_sims)
        annual_returns = np.zeros(n_sims)

        for i in range(n_sims):
            # Bootstrap sample with replacement
            sample = rng.choice(trade_returns, size=n_trades, replace=True)

            equity = initial_capital
            peak = equity
            max_dd = 0.0
            equity_path = [equity]

            for ret in sample:
                pnl = equity * ret
                equity += pnl
                equity = max(equity, 0)

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                equity_path.append(equity)

            final_equities[i] = equity
            max_drawdowns[i] = max_dd * 100

            path_returns = np.diff(equity_path) / np.array(equity_path[:-1])
            path_returns = path_returns[np.isfinite(path_returns)]
            if len(path_returns) > 1:
                mean_r = np.mean(path_returns)
                std_r = np.std(path_returns, ddof=1)
                sharpe_ratios[i] = (mean_r / std_r * np.sqrt(365)) if std_r > 0 else 0

            total_ret = (equity - initial_capital) / initial_capital
            annual_returns[i] = total_ret * (365 / max(n_trades, 1))

        mc = MonteCarloResult(
            num_simulations=n_sims,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            sharpe_ratios=sharpe_ratios,
            annual_returns=annual_returns,
        )

        for level in self.config.confidence_levels:
            mc.equity_percentiles[level] = float(np.percentile(final_equities, level * 100))
            mc.drawdown_percentiles[level] = float(np.percentile(max_drawdowns, level * 100))
            mc.sharpe_percentiles[level] = float(np.percentile(sharpe_ratios, level * 100))

        mc.probability_of_ruin = float(np.mean(final_equities < initial_capital * 0.5))
        mc.median_final_equity = float(np.median(final_equities))
        mc.worst_case_equity = float(np.min(final_equities))
        mc.best_case_equity = float(np.max(final_equities))

        logger.info(
            "Bootstrap Monte Carlo complete: median=$%.0f, ruin=%.1f%%",
            mc.median_final_equity, mc.probability_of_ruin * 100,
        )

        return mc

    def _empty_result(self, initial_capital: float) -> MonteCarloResult:
        return MonteCarloResult(
            num_simulations=0,
            final_equities=np.array([initial_capital]),
            max_drawdowns=np.array([0.0]),
            sharpe_ratios=np.array([0.0]),
            annual_returns=np.array([0.0]),
        )
