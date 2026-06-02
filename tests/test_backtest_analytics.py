"""
Tests for BacktestAnalytics and PerformanceMetrics.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from backtest.analytics import Analytics, PerformanceMetrics
from backtest.engine import BacktestConfig, BacktestResult, EquityPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backtest_result(
    initial_capital=10000.0,
    n_bars=100,
    final_value=11000.0,
    trades=None,
    signals_generated=50,
    signals_approved=20,
    signals_rejected=30,
) -> BacktestResult:
    """Create a synthetic BacktestResult for analytics testing."""
    config = BacktestConfig(initial_capital=initial_capital)

    # Linear equity curve from initial to final
    values = np.linspace(initial_capital, final_value, n_bars).tolist()
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]

    equity_curve = [
        EquityPoint(
            timestamp=ts,
            equity=val,
            cash=val,
            unrealized_pnl=0.0,
            positions=0,
            drawdown=0.0,
        )
        for ts, val in zip(timestamps, values)
    ]

    if trades is None:
        trades = [
            {"symbol": "BTC/USDT", "pnl": 200.0, "pnl_pct": 0.02, "duration_hours": 5.0, "side": "long"},
            {"symbol": "ETH/USDT", "pnl": -50.0, "pnl_pct": -0.01, "duration_hours": 3.0, "side": "short"},
            {"symbol": "BTC/USDT", "pnl": 150.0, "pnl_pct": 0.015, "duration_hours": 8.0, "side": "long"},
        ]

    return BacktestResult(
        config=config,
        equity_curve=equity_curve,
        closed_trades=trades,
        signals_generated=signals_generated,
        signals_approved=signals_approved,
        signals_rejected=signals_rejected,
        total_commission=15.0,
        total_slippage_cost=5.0,
        portfolio_values=values,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# PerformanceMetrics tests
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_defaults(self):
        m = PerformanceMetrics()
        assert m.total_return_pct == 0.0
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0

    def test_to_dict(self):
        m = PerformanceMetrics(total_return_pct=10.0, sharpe_ratio=1.5)
        d = m.to_dict()
        assert d["total_return_pct"] == 10.0
        assert d["sharpe_ratio"] == 1.5

    def test_summary_contains_key_metrics(self):
        m = PerformanceMetrics(total_return_pct=10.0, sharpe_ratio=1.5, max_drawdown_pct=5.0)
        s = m.summary()
        assert "Total Return" in s
        assert "Sharpe Ratio" in s
        assert "Max Drawdown" in s


# ---------------------------------------------------------------------------
# Analytics tests
# ---------------------------------------------------------------------------


class TestAnalytics:
    def test_compute_basic(self):
        result = _make_backtest_result(initial_capital=10000.0, final_value=11000.0)
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.total_return_pct == pytest.approx(10.0, abs=0.1)
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate_pct == pytest.approx(66.67, abs=0.1)

    def test_compute_profit_factor(self):
        result = _make_backtest_result()
        analytics = Analytics()
        metrics = analytics.compute(result)

        # Wins: 200 + 150 = 350, Losses: 50 → PF = 7.0
        assert metrics.profit_factor == pytest.approx(7.0, abs=0.1)

    def test_compute_with_drawdown(self):
        # Create result with a dip in the middle
        config = BacktestConfig(initial_capital=10000.0)
        n = 50
        values = [10000.0] * 10 + [9000.0] * 10 + [10500.0] * 10 + [11000.0] * 20
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]

        result = BacktestResult(
            config=config,
            equity_curve=[],
            closed_trades=[],
            signals_generated=0,
            signals_approved=0,
            signals_rejected=0,
            total_commission=0.0,
            total_slippage_cost=0.0,
            portfolio_values=values,
            timestamps=timestamps,
        )

        analytics = Analytics()
        metrics = analytics.compute(result)

        # Max DD = (10000 - 9000) / 10000 = 10%
        assert metrics.max_drawdown_pct == pytest.approx(10.0, abs=0.1)

    def test_compute_no_trades(self):
        result = _make_backtest_result(trades=[], signals_generated=0, signals_approved=0, signals_rejected=0)
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.total_trades == 0
        assert metrics.win_rate_pct == 0.0
        assert metrics.profit_factor == 0.0

    def test_compute_all_wins(self):
        trades = [
            {"symbol": "BTC/USDT", "pnl": 100.0, "pnl_pct": 0.01, "duration_hours": 2.0},
            {"symbol": "ETH/USDT", "pnl": 50.0, "pnl_pct": 0.005, "duration_hours": 1.0},
        ]
        result = _make_backtest_result(trades=trades)
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 0
        assert metrics.win_rate_pct == pytest.approx(100.0)
        assert metrics.profit_factor == float("inf")

    def test_compute_all_losses(self):
        trades = [
            {"symbol": "BTC/USDT", "pnl": -100.0, "pnl_pct": -0.01, "duration_hours": 2.0},
            {"symbol": "ETH/USDT", "pnl": -50.0, "pnl_pct": -0.005, "duration_hours": 1.0},
        ]
        result = _make_backtest_result(trades=trades)
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 2
        assert metrics.win_rate_pct == pytest.approx(0.0)

    def test_compute_signal_metrics(self):
        result = _make_backtest_result(signals_generated=100, signals_approved=60, signals_rejected=40)
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.signals_generated == 100
        assert metrics.signals_approved == 60
        assert metrics.signals_rejected == 40
        assert metrics.signal_approval_rate_pct == pytest.approx(60.0)

    def test_compute_costs(self):
        result = _make_backtest_result()
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.total_commission == 15.0
        assert metrics.total_slippage_cost == 5.0
        assert metrics.total_costs == 20.0

    def test_compute_monthly_returns(self):
        result = _make_backtest_result()
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert isinstance(metrics.monthly_returns, dict)
        # Should have at least one month
        assert len(metrics.monthly_returns) >= 1

    def test_compute_symbol_stats(self):
        trades = [
            {"symbol": "BTC/USDT", "pnl": 200.0, "pnl_pct": 0.02, "duration_hours": 5.0},
            {"symbol": "BTC/USDT", "pnl": -50.0, "pnl_pct": -0.005, "duration_hours": 3.0},
            {"symbol": "ETH/USDT", "pnl": 100.0, "pnl_pct": 0.01, "duration_hours": 2.0},
        ]
        result = _make_backtest_result(trades=trades)
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert "BTC/USDT" in metrics.symbol_stats
        assert "ETH/USDT" in metrics.symbol_stats
        assert metrics.symbol_stats["BTC/USDT"]["total_trades"] == 2
        assert metrics.symbol_stats["ETH/USDT"]["total_trades"] == 1

    def test_compute_empty_result(self):
        config = BacktestConfig(initial_capital=10000.0)
        result = BacktestResult(
            config=config,
            equity_curve=[],
            closed_trades=[],
            signals_generated=0,
            signals_approved=0,
            signals_rejected=0,
            total_commission=0.0,
            total_slippage_cost=0.0,
            portfolio_values=[],
            timestamps=[],
        )
        analytics = Analytics()
        metrics = analytics.compute(result)

        assert metrics.total_return_pct == 0.0
        assert metrics.total_trades == 0

    def test_sharpe_ratio_positive_for_profitable(self):
        result = _make_backtest_result(initial_capital=10000.0, final_value=12000.0, n_bars=200)
        analytics = Analytics()
        metrics = analytics.compute(result)
        assert metrics.sharpe_ratio > 0

    def test_compute_expectancy(self):
        result = _make_backtest_result()
        analytics = Analytics()
        metrics = analytics.compute(result)

        # Expectancy should be positive since we have more wins than losses
        assert metrics.expectancy > 0

    def test_annualized_return(self):
        result = _make_backtest_result(initial_capital=10000.0, final_value=11000.0, n_bars=100)
        analytics = Analytics()
        metrics = analytics.compute(result)

        # With ~100 hourly bars, annualized return should be much higher than 10%
        assert metrics.annualized_return_pct != 0.0
