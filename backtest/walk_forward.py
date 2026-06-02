"""
Walk-Forward Validation — Rolling train/test window validation.

Prevents overfitting by training on historical data, testing on unseen future data,
then stepping forward and repeating.

Walk-forward process:
  Window 1: Train [2024-01 → 2024-06], Test [2024-07]
  Window 2: Train [2024-02 → 2024-07], Test [2024-08]
  ...
  Window N: Train [...], Test [...]

Each window retrains the model and runs a fresh backtest on the test period.
Aggregated metrics show out-of-sample performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from backtest.analytics import Analytics, PerformanceMetrics
from execution.risk_manager import RiskConfig
from signals.filters import FilterConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""

    train_months: int = 6
    test_months: int = 1
    step_months: int = 1

    @classmethod
    def from_config_dict(cls, cfg: dict[str, Any]) -> "WalkForwardConfig":
        wf = cfg.get("walk_forward", cfg)
        return cls(
            train_months=wf.get("train_months", 6),
            test_months=wf.get("test_months", 1),
            step_months=wf.get("step_months", 1),
        )


@dataclass
class WalkForwardWindow:
    """Result from a single walk-forward window."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None
    train_metrics: PerformanceMetrics | None = None
    test_metrics: PerformanceMetrics | None = None


@dataclass
class WalkForwardResult:
    """Aggregated result from all walk-forward windows."""

    windows: list[WalkForwardWindow]
    config: WalkForwardConfig
    overall_test_metrics: dict[str, Any] = field(default_factory=dict)
    stability_score: float = 0.0  # 0-1, how consistent are window results

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═══════════════════════════════════════════",
            "       WALK-FORWARD VALIDATION SUMMARY     ",
            "═══════════════════════════════════════════",
            f"  Windows:              {len(self.windows):>8d}",
            f"  Train Period:         {self.config.train_months:>8d} months",
            f"  Test Period:          {self.config.test_months:>8d} months",
            f"  Step:                 {self.config.step_months:>8d} months",
            f"  Stability Score:      {self.stability_score:>8.3f}",
            "───────────────────────────────────────────",
        ]

        for w in self.windows:
            test_ret = w.test_metrics.total_return_pct if w.test_metrics else 0
            test_sharpe = w.test_metrics.sharpe_ratio if w.test_metrics else 0
            test_dd = w.test_metrics.max_drawdown_pct if w.test_metrics else 0
            lines.append(
                f"  Window {w.window_id:>2d}: {w.test_start} → {w.test_end} | "
                f"Return={test_ret:>+6.2f}% | Sharpe={test_sharpe:>6.3f} | DD={test_dd:>5.2f}%"
            )

        if self.overall_test_metrics:
            lines.extend([
                "───────────────────────────────────────────",
                f"  Avg OOS Return:       {self.overall_test_metrics.get('avg_return_pct', 0):>+8.2f}%",
                f"  Avg OOS Sharpe:       {self.overall_test_metrics.get('avg_sharpe', 0):>8.3f}",
                f"  Avg OOS Win Rate:     {self.overall_test_metrics.get('avg_win_rate', 0):>8.1f}%",
                f"  Worst Drawdown:       {self.overall_test_metrics.get('worst_drawdown', 0):>8.2f}%",
            ])

        lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)


class WalkForwardValidator:
    """Walk-forward validation engine.

    Usage:
        validator = WalkForwardValidator(config, wf_config)
        result = validator.run(
            ohlcv_data=ohlcv_data,
            feature_data=feature_data,
            train_fn=train_model_fn,
            predict_fn=predict_fn,
        )
    """

    def __init__(
        self,
        config: BacktestConfig,
        wf_config: WalkForwardConfig,
        risk_config: RiskConfig | None = None,
        signal_config: dict[str, Any] | None = None,
        filter_config: FilterConfig | None = None,
    ):
        self.config = config
        self.wf_config = wf_config
        self.risk_config = risk_config
        self.signal_config = signal_config
        self.filter_config = filter_config

    def generate_windows(
        self,
        data_start: str,
        data_end: str,
    ) -> list[WalkForwardWindow]:
        """Generate train/test window pairs.

        Returns:
            list of WalkForwardWindow with train/test date ranges
        """
        start = pd.Timestamp(data_start)
        end = pd.Timestamp(data_end)
        windows: list[WalkForwardWindow] = []

        window_id = 0
        current_train_start = start

        while True:
            train_end = current_train_start + pd.DateOffset(months=self.wf_config.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.wf_config.test_months)

            # Stop if test window extends beyond data
            if test_end > end:
                break

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=current_train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            ))

            window_id += 1
            current_train_start += pd.DateOffset(months=self.wf_config.step_months)

        logger.info(
            "Generated %d walk-forward windows (train=%dm, test=%dm, step=%dm)",
            len(windows), self.wf_config.train_months,
            self.wf_config.test_months, self.wf_config.step_months,
        )

        return windows

    def run(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        feature_data: dict[str, pd.DataFrame],
        train_fn: Callable[[dict[str, pd.DataFrame], dict[str, pd.DataFrame]], Any],
        predict_fn: Callable[[Any, pd.DataFrame, str], Any],
    ) -> WalkForwardResult:
        """Run full walk-forward validation.

        Args:
            ohlcv_data: {symbol: OHLCV DataFrame}
            feature_data: {symbol: features DataFrame}
            train_fn: callable(ohlcv_train, features_train) → trained_model
            predict_fn: callable(model, features_df, symbol) → Prediction

        Returns:
            WalkForwardResult with per-window and aggregate metrics
        """
        # Determine data range from available data
        all_starts = [df.index.min() for df in ohlcv_data.values() if len(df) > 0]
        all_ends = [df.index.max() for df in ohlcv_data.values() if len(df) > 0]

        if not all_starts or not all_ends:
            logger.error("No OHLCV data available for walk-forward")
            return WalkForwardResult(windows=[], config=self.wf_config)

        data_start = min(all_starts).strftime("%Y-%m-%d")
        data_end = max(all_ends).strftime("%Y-%m-%d")

        windows = self.generate_windows(data_start, data_end)

        if not windows:
            logger.warning("No valid walk-forward windows generated")
            return WalkForwardResult(windows=[], config=self.wf_config)

        analytics = Analytics()
        all_test_metrics: list[PerformanceMetrics] = []

        for w in windows:
            logger.info(
                "Window %d: Train [%s → %s] Test [%s → %s]",
                w.window_id, w.train_start, w.train_end,
                w.test_start, w.test_end,
            )

            # --- Train phase ---
            train_ohlcv = self._slice_data(ohlcv_data, w.train_start, w.train_end)
            train_features = self._slice_data(feature_data, w.train_start, w.train_end)

            try:
                model = train_fn(train_ohlcv, train_features)
            except Exception as e:
                logger.error("Training failed for window %d: %s", w.window_id, e)
                continue

            # --- Test phase (backtest on unseen data) ---
            test_ohlcv = self._slice_data(ohlcv_data, w.test_start, w.test_end)
            test_features = self._slice_data(feature_data, w.test_start, w.test_end)

            # Create predict function bound to this model
            def make_predictor(m):
                def predict(features_df, symbol):
                    return predict_fn(m, features_df, symbol)
                return predict

            predictor = make_predictor(model)

            engine = BacktestEngine(
                config=BacktestConfig(
                    start_date=w.test_start,
                    end_date=w.test_end,
                    initial_capital=self.config.initial_capital,
                    commission_spot=self.config.commission_spot,
                    commission_futures=self.config.commission_futures,
                    slippage=self.config.slippage,
                ),
                risk_config=self.risk_config,
                signal_config=self.signal_config,
                filter_config=self.filter_config,
            )

            try:
                w.test_result = engine.run(
                    ohlcv_data=test_ohlcv,
                    feature_data=test_features,
                    model_predictor=predictor,
                )
                w.test_metrics = analytics.compute(w.test_result)
                all_test_metrics.append(w.test_metrics)

                logger.info(
                    "Window %d test: return=%.2f%% sharpe=%.3f win_rate=%.1f%%",
                    w.window_id,
                    w.test_metrics.total_return_pct,
                    w.test_metrics.sharpe_ratio,
                    w.test_metrics.win_rate_pct,
                )
            except Exception as e:
                logger.error("Backtest failed for window %d: %s", w.window_id, e)
                continue

        # Aggregate results
        overall = self._aggregate_results(windows, all_test_metrics, analytics)
        stability = self._compute_stability(all_test_metrics)

        return WalkForwardResult(
            windows=windows,
            config=self.wf_config,
            overall_test_metrics=overall,
            stability_score=stability,
        )

    def _slice_data(
        self,
        data: dict[str, pd.DataFrame],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """Slice all DataFrames to date range."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        sliced: dict[str, pd.DataFrame] = {}

        for symbol, df in data.items():
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            sliced[symbol] = df[mask].copy()

        return sliced

    def _aggregate_results(
        self,
        windows: list[WalkForwardWindow],
        test_metrics: list[PerformanceMetrics],
        analytics: Analytics,
    ) -> dict[str, Any]:
        """Aggregate test metrics across all windows."""
        if not test_metrics:
            return {}

        import numpy as np

        return {
            "num_windows": len(test_metrics),
            "avg_return_pct": float(np.mean([m.total_return_pct for m in test_metrics])),
            "std_return_pct": float(np.std([m.total_return_pct for m in test_metrics])),
            "avg_sharpe": float(np.mean([m.sharpe_ratio for m in test_metrics])),
            "avg_win_rate": float(np.mean([m.win_rate_pct for m in test_metrics])),
            "avg_profit_factor": float(np.mean([m.profit_factor for m in test_metrics])),
            "avg_max_drawdown": float(np.mean([m.max_drawdown_pct for m in test_metrics])),
            "worst_drawdown": float(max(m.max_drawdown_pct for m in test_metrics)),
            "total_trades": sum(m.total_trades for m in test_metrics),
            "positive_windows": sum(1 for m in test_metrics if m.total_return_pct > 0),
            "window_details": [
                {
                    "window_id": w.window_id,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "return_pct": w.test_metrics.total_return_pct if w.test_metrics else 0,
                    "sharpe": w.test_metrics.sharpe_ratio if w.test_metrics else 0,
                }
                for w in windows if w.test_metrics
            ],
        }

    def _compute_stability(self, metrics: list[PerformanceMetrics]) -> float:
        """Compute stability score (0-1).

        Higher = more consistent returns across windows.
        Based on coefficient of variation of returns.
        """
        import numpy as np

        if len(metrics) < 2:
            return 0.0

        returns = [m.total_return_pct for m in metrics]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if abs(mean_ret) < 1e-10:
            return 0.0

        # Coefficient of variation (inverted and normalized)
        cv = abs(std_ret / mean_ret)
        # Stability = 1 / (1 + cv), bounded [0, 1]
        stability = 1.0 / (1.0 + cv)

        return float(stability)
