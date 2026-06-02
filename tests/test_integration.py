"""
Integration test — Simulates a mini trading pipeline without real exchanges.

Tests the full flow: Prediction → SignalGenerator → RiskManager → PositionTracker → Analytics
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from models.predictor import Prediction
from signals.generator import SignalGenerator, SignalAction, MarketType
from signals.filters import FilterConfig
from execution.position_tracker import PositionTracker, Position
from execution.risk_manager import RiskManager, RiskConfig
from backtest.analytics import Analytics
from backtest.engine import BacktestConfig, BacktestResult, EquityPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prediction(direction="Long", confidence=0.90, magnitude=0.02, symbol="BTC/USDT"):
    return Prediction(
        direction=direction,
        direction_idx=0 if direction == "Long" else (1 if direction == "Short" else 2),
        direction_probs={"Long": 0.7, "Short": 0.2, "Neutral": 0.1},
        magnitude=magnitude,
        confidence=confidence,
        timestamp=datetime(2024, 6, 1),
        symbol=symbol,
    )


def _make_uptrend_features(n=50, start=100.0):
    """Create an uptrending features DataFrame."""
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    closes = start + np.arange(n) * 0.5
    return pd.DataFrame({
        "close": closes,
        "volume": np.full(n, 1000.0),
        "atr": np.full(n, 50.0),
    }, index=dates)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Test the full prediction → signal → risk → position → analytics pipeline."""

    def _setup_pipeline(self):
        """Set up all pipeline components."""
        # Signal generator (no cooldown for testing)
        sig_gen = SignalGenerator(config={"cooldown_minutes": 0})

        # Risk manager + tracker
        risk_cfg = RiskConfig(max_concurrent_positions=3, max_position_pct=0.05)
        tracker = PositionTracker(max_concurrent=3)
        risk_mgr = RiskManager(config=risk_cfg, tracker=tracker, initial_capital=10000.0)

        return sig_gen, risk_mgr, tracker

    def test_long_spot_pipeline(self):
        """Long signal with moderate confidence → spot buy."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Long", confidence=0.82, magnitude=0.02)
        features = _make_uptrend_features()

        # Generate signal
        signal = sig_gen.generate(pred, current_price=65000.0, features_df=features)
        assert signal.action == SignalAction.BUY
        assert signal.market == MarketType.SPOT

        # Risk check
        approved, reasons = risk_mgr.check_signal(signal)
        assert approved is True

        # Create position
        position = risk_mgr.create_position(signal, current_price=65000.0)
        assert position is not None
        assert position.market == "spot"
        assert position.side == "long"
        assert tracker.has_position("BTC/USDT")

    def test_short_futures_pipeline(self):
        """Short signal → futures short."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Short", confidence=0.90, magnitude=0.025, symbol="ETH/USDT")

        signal = sig_gen.generate(pred, current_price=3500.0)
        assert signal.action == SignalAction.SHORT
        assert signal.market == MarketType.FUTURES

        approved, _ = risk_mgr.check_signal(signal)
        assert approved is True

        position = risk_mgr.create_position(signal, current_price=3500.0)
        assert position is not None
        assert position.market == "futures"
        assert position.side == "short"

    def test_high_confidence_long_futures_pipeline(self):
        """High confidence long → futures long with leverage."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Long", confidence=0.92, magnitude=0.03)
        features = _make_uptrend_features()

        signal = sig_gen.generate(pred, current_price=65000.0, features_df=features)
        assert signal.action == SignalAction.LONG
        assert signal.market == MarketType.FUTURES
        assert signal.leverage > 1.0

        approved, _ = risk_mgr.check_signal(signal)
        assert approved is True

        position = risk_mgr.create_position(signal, current_price=65000.0)
        assert position is not None
        assert position.leverage > 1.0

    def test_neutral_signal_no_position(self):
        """Neutral prediction → HOLD → no position created."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Neutral", confidence=0.45, magnitude=0.001)

        signal = sig_gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.HOLD
        assert signal.is_entry is False

    def test_low_confidence_rejected_by_filter(self):
        """Low confidence → filtered out → HOLD."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Long", confidence=0.50, magnitude=0.02)

        signal = sig_gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.HOLD

    def test_position_close_and_pnl(self):
        """Open a position, then close it and verify PnL."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Long", confidence=0.82, magnitude=0.02)
        features = _make_uptrend_features()

        signal = sig_gen.generate(pred, current_price=100.0, features_df=features)
        position = risk_mgr.create_position(signal, current_price=100.0)
        assert position is not None

        # Close at profit
        trade = tracker.close_position("BTC/USDT", exit_price=105.0, close_reason="take_profit")
        assert trade is not None
        assert trade.pnl > 0
        assert trade.close_reason == "take_profit"

    def test_trailing_stop_update(self):
        """Test trailing stop tightening after price increase."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        pred = _make_prediction("Long", confidence=0.82, magnitude=0.02)
        features = _make_uptrend_features()

        signal = sig_gen.generate(pred, current_price=100.0, features_df=features)
        position = risk_mgr.create_position(signal, current_price=100.0)
        assert position is not None

        initial_sl = position.stop_loss_price

        # Price goes up → trailing stop should tighten
        updated = risk_mgr.update_trailing_stop("BTC/USDT", 110.0)
        assert updated is True
        assert position.stop_loss_price > initial_sl

    def test_max_concurrent_positions(self):
        """Verify max concurrent position limit works."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        # Fill all 3 slots
        for sym, price in [("BTC/USDT", 65000.0), ("ETH/USDT", 3500.0), ("SOL/USDT", 150.0)]:
            pred = _make_prediction("Long", confidence=0.82, magnitude=0.02, symbol=sym)
            features = _make_uptrend_features()
            signal = sig_gen.generate(pred, current_price=price, features_df=features)
            pos = risk_mgr.create_position(signal, current_price=price)
            assert pos is not None

        # 4th should be rejected
        pred = _make_prediction("Long", confidence=0.82, magnitude=0.02, symbol="ADA/USDT")
        features = _make_uptrend_features()
        signal = sig_gen.generate(pred, current_price=0.5, features_df=features)
        approved, reasons = risk_mgr.check_signal(signal)
        assert approved is False
        assert any("max_concurrent" in r for r in reasons)

    def test_drawdown_halts_trading(self):
        """Portfolio drawdown exceeding limit should halt trading."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        # Drop portfolio by 6% (exceeds 5% daily limit)
        risk_mgr.update_portfolio_value(9400.0)
        assert risk_mgr.is_trading_halted is True

        # New entry signals should be rejected
        pred = _make_prediction("Long", confidence=0.90, magnitude=0.03)
        features = _make_uptrend_features()
        signal = sig_gen.generate(pred, current_price=65000.0, features_df=features)
        approved, reasons = risk_mgr.check_signal(signal)
        assert approved is False
        assert any("trading_halted" in r for r in reasons)

    def test_full_backtest_analytics(self):
        """End-to-end: generate synthetic backtest result → compute analytics."""
        config = BacktestConfig(initial_capital=10000.0)

        # Synthetic profitable equity curve
        n = 200
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.01, n)
        values = [10000.0]
        for r in daily_returns:
            values.append(values[-1] * (1 + r))

        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n + 1)]

        # Synthetic trades
        trades = []
        for i in range(20):
            pnl = np.random.normal(50, 200)
            trades.append({
                "symbol": "BTC/USDT" if i % 2 == 0 else "ETH/USDT",
                "pnl": pnl,
                "pnl_pct": pnl / 10000,
                "duration_hours": np.random.uniform(1, 24),
            })

        result = BacktestResult(
            config=config,
            equity_curve=[],
            closed_trades=trades,
            signals_generated=100,
            signals_approved=50,
            signals_rejected=50,
            total_commission=25.0,
            total_slippage_cost=10.0,
            portfolio_values=values,
            timestamps=timestamps,
        )

        analytics = Analytics()
        metrics = analytics.compute(result)

        # Verify all metric categories are populated
        assert metrics.total_trades == 20
        assert metrics.sharpe_ratio != 0.0
        assert metrics.max_drawdown_pct >= 0.0
        assert metrics.win_rate_pct >= 0.0
        assert metrics.profit_factor >= 0.0
        assert metrics.total_commission == 25.0
        assert metrics.signal_approval_rate_pct == pytest.approx(50.0)
        assert "BTC/USDT" in metrics.symbol_stats
        assert "ETH/USDT" in metrics.symbol_stats

    def test_multiple_symbols_batch_signals(self):
        """Generate signals for multiple symbols in batch."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        predictions = [
            _make_prediction("Long", 0.82, 0.02, "BTC/USDT"),
            _make_prediction("Short", 0.90, 0.025, "ETH/USDT"),
            _make_prediction("Neutral", 0.45, 0.001, "SOL/USDT"),
        ]
        prices = {"BTC/USDT": 65000.0, "ETH/USDT": 3500.0, "SOL/USDT": 150.0}

        signals = sig_gen.generate_batch(predictions, prices)
        assert len(signals) == 3

        entries = [s for s in signals if s.is_entry]
        holds = [s for s in signals if s.action == SignalAction.HOLD]

        assert len(entries) == 2  # BTC buy + ETH short
        assert len(holds) == 1    # SOL neutral

    def test_risk_manager_stats(self):
        """Verify risk manager stats are accurate after operations."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        # Open a position
        pred = _make_prediction("Long", confidence=0.82, magnitude=0.02)
        features = _make_uptrend_features()
        signal = sig_gen.generate(pred, current_price=65000.0, features_df=features)
        risk_mgr.create_position(signal, current_price=65000.0)

        stats = risk_mgr.get_stats()
        assert stats["portfolio_value"] == 10000.0
        assert stats["open_positions"] == 1
        assert stats["trading_halted"] is False

    def test_position_tracker_stats_after_trades(self):
        """Verify position tracker stats after multiple trades."""
        sig_gen, risk_mgr, tracker = self._setup_pipeline()

        # Trade 1: Win
        pred = _make_prediction("Long", 0.82, 0.02, "BTC/USDT")
        features = _make_uptrend_features()
        signal = sig_gen.generate(pred, 100.0, features_df=features)
        risk_mgr.create_position(signal, 100.0)
        tracker.close_position("BTC/USDT", 110.0, "take_profit")

        # Trade 2: Loss
        pred = _make_prediction("Long", 0.82, 0.02, "BTC/USDT")
        signal = sig_gen.generate(pred, 100.0, features_df=features)
        risk_mgr.create_position(signal, 100.0)
        tracker.close_position("BTC/USDT", 95.0, "stop_loss")

        stats = tracker.get_stats()
        assert stats["total_trades"] == 2
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["best_trade_pnl"] > 0
        assert stats["worst_trade_pnl"] < 0
