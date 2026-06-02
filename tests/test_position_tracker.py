"""
Tests for Position, ClosedTrade, and PositionTracker.
"""

import time
import pytest
from execution.position_tracker import Position, ClosedTrade, PositionTracker


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_create_long_position(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        assert pos.symbol == "BTC/USDT"
        assert pos.is_long is True
        assert pos.initial_size == 0.1
        assert pos.entry_time > 0

    def test_create_short_position(self):
        pos = Position(symbol="ETH/USDT", market="futures", side="short", entry_price=3500.0, size=1.0)
        assert pos.is_long is False
        assert pos.market == "futures"

    def test_notional_value(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        assert pos.notional_value == pytest.approx(6500.0)

    def test_unrealized_pnl_long_profit(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        pnl = pos.unrealized_pnl(66000.0)
        assert pnl == pytest.approx(100.0)

    def test_unrealized_pnl_long_loss(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        pnl = pos.unrealized_pnl(64000.0)
        assert pnl == pytest.approx(-100.0)

    def test_unrealized_pnl_short_profit(self):
        pos = Position(symbol="ETH/USDT", market="futures", side="short", entry_price=3500.0, size=1.0)
        pnl = pos.unrealized_pnl(3400.0)
        assert pnl == pytest.approx(100.0)

    def test_unrealized_pnl_short_loss(self):
        pos = Position(symbol="ETH/USDT", market="futures", side="short", entry_price=3500.0, size=1.0)
        pnl = pos.unrealized_pnl(3600.0)
        assert pnl == pytest.approx(-100.0)

    def test_unrealized_pnl_pct_long(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=100.0, size=1.0, leverage=2.0)
        pct = pos.unrealized_pnl_pct(105.0)
        assert pct == pytest.approx(0.1)  # 5% * 2x = 10%

    def test_unrealized_pnl_pct_short(self):
        pos = Position(symbol="BTC/USDT", market="futures", side="short", entry_price=100.0, size=1.0, leverage=2.0)
        pct = pos.unrealized_pnl_pct(95.0)
        assert pct == pytest.approx(0.1)  # 5% * 2x = 10%

    def test_stop_loss_long_triggered(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1,
                       stop_loss_price=64000.0)
        assert pos.should_stop_loss(63500.0) is True
        assert pos.should_stop_loss(64500.0) is False

    def test_stop_loss_short_triggered(self):
        pos = Position(symbol="ETH/USDT", market="futures", side="short", entry_price=3500.0, size=1.0,
                       stop_loss_price=3600.0)
        assert pos.should_stop_loss(3650.0) is True
        assert pos.should_stop_loss(3550.0) is False

    def test_stop_loss_no_sl(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        assert pos.should_stop_loss(1.0) is False

    def test_check_take_profit_long(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=100.0, size=1.0,
                       take_profit_prices=[103.0, 105.0, 108.0])
        assert pos.check_take_profit(104.0) == 0
        assert pos.check_take_profit(106.0) == 0
        assert pos.check_take_profit(101.0) is None

    def test_check_take_profit_short(self):
        pos = Position(symbol="ETH/USDT", market="futures", side="short", entry_price=100.0, size=1.0,
                       take_profit_prices=[97.0, 95.0, 92.0])
        assert pos.check_take_profit(96.0) == 0
        assert pos.check_take_profit(98.0) is None

    def test_check_take_profit_already_hit(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=100.0, size=1.0,
                       take_profit_prices=[103.0, 105.0], tp_hits=[0])
        # TP 0 already hit, TP 1 not yet
        assert pos.check_take_profit(104.0) is None  # 0 is already hit
        assert pos.check_take_profit(106.0) == 1

    def test_close_partial(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=1.0)
        result = pos.close(0.4)
        assert result["closed_size"] == pytest.approx(0.4)
        assert result["remaining_size"] == pytest.approx(0.6)
        assert result["is_full_close"] is False
        assert pos.closed_size == pytest.approx(0.4)

    def test_close_full(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=1.0)
        result = pos.close(1.0)
        assert result["is_full_close"] is True
        assert pos.size == pytest.approx(0.0)

    def test_close_over_size(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.5)
        result = pos.close(2.0)
        assert result["closed_size"] == pytest.approx(0.5)
        assert result["remaining_size"] == pytest.approx(0.0)

    def test_to_dict(self):
        pos = Position(symbol="BTC/USDT", market="spot", side="long", entry_price=65000.0, size=0.1)
        d = pos.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["entry_price"] == 65000.0
        assert "size" in d


# ---------------------------------------------------------------------------
# ClosedTrade tests
# ---------------------------------------------------------------------------


class TestClosedTrade:
    def test_to_dict(self):
        trade = ClosedTrade(
            symbol="BTC/USDT", market="spot", side="long",
            entry_price=65000.0, exit_price=66000.0, size=0.1,
            leverage=1.0, entry_time=1000.0, exit_time=13600.0,
            pnl=100.0, pnl_pct=0.015, fees=1.0,
            duration_hours=1.0, close_reason="take_profit",
        )
        d = trade.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["pnl"] == 100.0
        assert d["close_reason"] == "take_profit"


# ---------------------------------------------------------------------------
# PositionTracker tests
# ---------------------------------------------------------------------------


class TestPositionTracker:
    def _make_position(self, symbol="BTC/USDT", side="long", entry_price=65000.0, size=0.1):
        return Position(symbol=symbol, market="spot", side=side, entry_price=entry_price, size=size)

    def test_open_position(self):
        tracker = PositionTracker(max_concurrent=3)
        pos = self._make_position()
        assert tracker.open_position(pos) is True
        assert tracker.position_count == 1
        assert tracker.has_position("BTC/USDT")

    def test_open_duplicate_rejected(self):
        tracker = PositionTracker(max_concurrent=3)
        pos = self._make_position()
        tracker.open_position(pos)
        assert tracker.open_position(self._make_position()) is False

    def test_max_concurrent_rejected(self):
        tracker = PositionTracker(max_concurrent=2)
        tracker.open_position(self._make_position("BTC/USDT"))
        tracker.open_position(self._make_position("ETH/USDT", entry_price=3500.0))
        assert tracker.open_position(self._make_position("SOL/USDT", entry_price=150.0)) is False

    def test_close_position(self):
        tracker = PositionTracker()
        tracker.open_position(self._make_position())
        trade = tracker.close_position("BTC/USDT", exit_price=66000.0, close_reason="signal")
        assert trade is not None
        assert trade.pnl == pytest.approx(100.0)
        assert tracker.position_count == 0

    def test_close_nonexistent(self):
        tracker = PositionTracker()
        assert tracker.close_position("BTC/USDT", exit_price=66000.0) is None

    def test_get_position(self):
        tracker = PositionTracker()
        pos = self._make_position()
        tracker.open_position(pos)
        assert tracker.get_position("BTC/USDT") is pos
        assert tracker.get_position("ETH/USDT") is None

    def test_get_all_positions(self):
        tracker = PositionTracker(max_concurrent=3)
        tracker.open_position(self._make_position("BTC/USDT"))
        tracker.open_position(self._make_position("ETH/USDT", entry_price=3500.0))
        all_pos = tracker.get_all_positions()
        assert len(all_pos) == 2

    def test_compute_unrealized_pnl(self):
        tracker = PositionTracker()
        tracker.open_position(self._make_position("BTC/USDT", side="long", entry_price=100.0, size=1.0))
        pnl = tracker.compute_unrealized_pnl({"BTC/USDT": 110.0})
        assert pnl == pytest.approx(10.0)

    def test_check_stop_losses(self):
        tracker = PositionTracker()
        pos = self._make_position(entry_price=100.0, size=1.0)
        pos.stop_loss_price = 95.0
        tracker.open_position(pos)
        triggered = tracker.check_stop_losses({"BTC/USDT": 94.0})
        assert "BTC/USDT" in triggered
        assert tracker.check_stop_losses({"BTC/USDT": 96.0}) == []

    def test_check_take_profits(self):
        tracker = PositionTracker()
        pos = self._make_position(entry_price=100.0, size=1.0)
        pos.take_profit_prices = [103.0, 105.0, 108.0]
        tracker.open_position(pos)
        triggered = tracker.check_take_profits({"BTC/USDT": 104.0})
        assert ("BTC/USDT", 0) in triggered

    def test_stats_empty(self):
        tracker = PositionTracker()
        stats = tracker.get_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_stats_with_trades(self):
        tracker = PositionTracker()
        # Win
        tracker.open_position(self._make_position("BTC/USDT", entry_price=100.0, size=1.0))
        tracker.close_position("BTC/USDT", exit_price=110.0)
        # Loss
        tracker.open_position(self._make_position("ETH/USDT", entry_price=200.0, size=1.0))
        tracker.close_position("ETH/USDT", exit_price=190.0)

        stats = tracker.get_stats()
        assert stats["total_trades"] == 2
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["best_trade_pnl"] == pytest.approx(10.0)
        assert stats["worst_trade_pnl"] == pytest.approx(-10.0)

    def test_trade_history(self):
        tracker = PositionTracker()
        tracker.open_position(self._make_position("BTC/USDT"))
        tracker.close_position("BTC/USDT", exit_price=66000.0)
        history = tracker.get_trade_history(limit=10)
        assert len(history) == 1
        assert history[0]["symbol"] == "BTC/USDT"
