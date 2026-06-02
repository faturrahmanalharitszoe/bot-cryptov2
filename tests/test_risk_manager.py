"""
Tests for RiskConfig and RiskManager.
"""

import pytest
from datetime import datetime, timedelta
from execution.risk_manager import RiskConfig, RiskManager
from execution.position_tracker import PositionTracker, Position
from signals.generator import TradeSignal, SignalAction, MarketType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    symbol="BTC/USDT",
    action=SignalAction.BUY,
    market=MarketType.SPOT,
    direction="Long",
    confidence=0.90,
    magnitude=0.02,
    leverage=1.0,
    position_size_pct=0.05,
    stop_loss_price=64000.0,
    take_profit_prices=[66950.0, 68250.0, 70200.0],
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        action=action,
        market=market,
        direction=direction,
        confidence=confidence,
        magnitude=magnitude,
        strength=0.8,
        leverage=leverage,
        position_size_pct=position_size_pct,
        entry_price=65000.0,
        stop_loss_price=stop_loss_price,
        take_profit_prices=take_profit_prices,
    )


# ---------------------------------------------------------------------------
# RiskConfig tests
# ---------------------------------------------------------------------------


class TestRiskConfig:
    def test_defaults(self):
        cfg = RiskConfig()
        assert cfg.max_position_pct == 0.05
        assert cfg.max_concurrent_positions == 3
        assert cfg.stop_loss_pct == 0.02
        assert cfg.take_profit_levels == [0.03, 0.05, 0.08]

    def test_from_config_dict(self):
        d = {
            "max_position_pct": 0.10,
            "max_concurrent_positions": 5,
            "stop_loss_pct": 0.03,
        }
        cfg = RiskConfig.from_config_dict(d)
        assert cfg.max_position_pct == 0.10
        assert cfg.max_concurrent_positions == 5
        assert cfg.stop_loss_pct == 0.03

    def test_from_config_dict_defaults(self):
        cfg = RiskConfig.from_config_dict({})
        assert cfg.max_position_pct == 0.05


# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------


class TestRiskManager:
    def _make_manager(self, capital=10000.0, max_concurrent=3):
        cfg = RiskConfig(max_concurrent_positions=max_concurrent)
        tracker = PositionTracker(max_concurrent=max_concurrent)
        return RiskManager(config=cfg, tracker=tracker, initial_capital=capital)

    def test_init(self):
        rm = self._make_manager()
        assert rm.current_portfolio_value == 10000.0
        assert rm.is_trading_halted is False

    def test_check_signal_approved(self):
        rm = self._make_manager()
        signal = _make_signal(position_size_pct=0.05)
        approved, reasons = rm.check_signal(signal)
        assert approved is True
        assert reasons == []

    def test_check_signal_rejected_position_too_large(self):
        rm = self._make_manager()
        signal = _make_signal(position_size_pct=0.10)  # > 5% default limit
        approved, reasons = rm.check_signal(signal)
        assert approved is False
        assert any("position_too_large" in r for r in reasons)

    def test_check_signal_rejected_leverage_too_high(self):
        rm = self._make_manager()
        signal = _make_signal(leverage=5.0)
        approved, reasons = rm.check_signal(signal)
        assert approved is False
        assert any("leverage_too_high" in r for r in reasons)

    def test_check_signal_rejected_max_concurrent(self):
        rm = self._make_manager(max_concurrent=1)
        # Fill the slot
        pos = Position(symbol="ETH/USDT", market="spot", side="long", entry_price=3500.0, size=1.0)
        rm.tracker.open_position(pos)

        signal = _make_signal(symbol="BTC/USDT")
        approved, reasons = rm.check_signal(signal)
        assert approved is False
        assert any("max_concurrent" in r for r in reasons)

    def test_check_signal_rejected_duplicate_symbol(self):
        rm = self._make_manager()
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        rm.tracker.open_position(pos)

        signal = _make_signal(symbol="BTC/USDT")
        approved, reasons = rm.check_signal(signal)
        assert approved is False
        assert any("position_exists" in r for r in reasons)

    def test_check_signal_exit_always_approved(self):
        rm = self._make_manager()
        rm._trading_halted = True
        signal = _make_signal(action=SignalAction.SELL, market=MarketType.SPOT, direction="Long")
        approved, reasons = rm.check_signal(signal)
        assert approved is True

    def test_create_position(self):
        rm = self._make_manager()
        signal = _make_signal(position_size_pct=0.05)
        pos = rm.create_position(signal, current_price=65000.0)
        assert pos is not None
        assert pos.symbol == "BTC/USDT"
        assert pos.market == "spot"
        assert pos.side == "long"
        assert rm.tracker.has_position("BTC/USDT")

    def test_create_position_rejected(self):
        rm = self._make_manager()
        rm._trading_halted = True
        signal = _make_signal()
        pos = rm.create_position(signal, current_price=65000.0)
        assert pos is None

    def test_create_position_invalid_price(self):
        rm = self._make_manager()
        signal = _make_signal()
        pos = rm.create_position(signal, current_price=0.0)
        assert pos is None

    def test_create_position_futures(self):
        rm = self._make_manager()
        signal = _make_signal(
            action=SignalAction.LONG,
            market=MarketType.FUTURES,
            leverage=2.0,
        )
        pos = rm.create_position(signal, current_price=65000.0)
        assert pos is not None
        assert pos.market == "futures"
        assert pos.leverage == 2.0

    def test_update_portfolio_value_drawdown_halt(self):
        rm = self._make_manager(capital=10000.0)
        # 6% drop should trigger 5% daily DD limit
        rm.update_portfolio_value(9400.0)
        assert rm.is_trading_halted is True
        assert "Daily drawdown" in rm.halt_reason

    def test_update_portfolio_value_within_limits(self):
        rm = self._make_manager(capital=10000.0)
        rm.update_portfolio_value(9700.0)  # 3% drop, within 5% limit
        assert rm.is_trading_halted is False

    def test_resume_trading(self):
        rm = self._make_manager(capital=10000.0)
        rm.update_portfolio_value(9000.0)
        assert rm.is_trading_halted is True
        rm.resume_trading()
        assert rm.is_trading_halted is False
        assert rm.halt_reason == ""

    def test_compute_tp_close_size(self):
        rm = self._make_manager()
        assert rm.compute_tp_close_size(0) == pytest.approx(0.4)
        assert rm.compute_tp_close_size(1) == pytest.approx(0.35)
        assert rm.compute_tp_close_size(2) == pytest.approx(0.25)
        assert rm.compute_tp_close_size(99) == pytest.approx(1.0)

    def test_update_trailing_stop_long(self):
        rm = self._make_manager()
        pos = Position(
            symbol="BTC/USDT", market="spot", side="long",
            entry_price=100.0, size=1.0, stop_loss_price=95.0,
        )
        rm.tracker.open_position(pos)

        # Price goes up, trailing stop should move up
        updated = rm.update_trailing_stop("BTC/USDT", 110.0)
        assert updated is True
        assert pos.stop_loss_price > 95.0
        expected = 110.0 * (1 - rm.config.stop_loss_pct)
        assert pos.stop_loss_price == pytest.approx(expected)

    def test_update_trailing_stop_long_no_tighten(self):
        rm = self._make_manager()
        pos = Position(
            symbol="BTC/USDT", market="spot", side="long",
            entry_price=100.0, size=1.0, stop_loss_price=99.0,
        )
        rm.tracker.open_position(pos)
        # Price goes down, trailing stop should NOT move down
        updated = rm.update_trailing_stop("BTC/USDT", 98.0)
        assert updated is False
        assert pos.stop_loss_price == pytest.approx(99.0)

    def test_update_trailing_stop_no_position(self):
        rm = self._make_manager()
        assert rm.update_trailing_stop("BTC/USDT", 100.0) is False

    def test_get_stats(self):
        rm = self._make_manager()
        stats = rm.get_stats()
        assert stats["portfolio_value"] == 10000.0
        assert stats["trading_halted"] is False
        assert stats["open_positions"] == 0

    def test_get_stats_with_drawdown(self):
        rm = self._make_manager(capital=10000.0)
        rm.update_portfolio_value(9700.0)
        stats = rm.get_stats()
        assert stats["daily_drawdown"] == pytest.approx(0.03, abs=0.001)
