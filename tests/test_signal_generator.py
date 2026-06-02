"""
Tests for SignalGenerator, TradeSignal, and related classes.
"""

import pytest
import pandas as pd
from datetime import datetime
from models.predictor import Prediction
from signals.generator import SignalGenerator, TradeSignal, SignalAction, MarketType
from signals.filters import FilterConfig


# ---------------------------------------------------------------------------
# TradeSignal tests
# ---------------------------------------------------------------------------


class TestTradeSignal:
    def _make_signal(self, action=SignalAction.BUY, market=MarketType.SPOT, direction="Long"):
        return TradeSignal(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 6, 1),
            action=action,
            market=market,
            direction=direction,
            confidence=0.90,
            magnitude=0.02,
            strength=0.8,
        )

    def test_is_entry_buy(self):
        s = self._make_signal(action=SignalAction.BUY)
        assert s.is_entry is True
        assert s.is_exit is False

    def test_is_entry_long(self):
        s = self._make_signal(action=SignalAction.LONG, market=MarketType.FUTURES)
        assert s.is_entry is True

    def test_is_entry_short(self):
        s = self._make_signal(action=SignalAction.SHORT, market=MarketType.FUTURES, direction="Short")
        assert s.is_entry is True

    def test_is_exit_sell(self):
        s = self._make_signal(action=SignalAction.SELL)
        assert s.is_exit is True
        assert s.is_entry is False

    def test_is_exit_close(self):
        s = self._make_signal(action=SignalAction.CLOSE)
        assert s.is_exit is True

    def test_hold_neither_entry_nor_exit(self):
        s = self._make_signal(action=SignalAction.HOLD)
        assert s.is_entry is False
        assert s.is_exit is False

    def test_side_buy(self):
        s = self._make_signal(action=SignalAction.BUY)
        assert s.side == "buy"

    def test_side_long(self):
        s = self._make_signal(action=SignalAction.LONG, market=MarketType.FUTURES)
        assert s.side == "buy"

    def test_side_sell(self):
        s = self._make_signal(action=SignalAction.SELL)
        assert s.side == "sell"

    def test_side_short(self):
        s = self._make_signal(action=SignalAction.SHORT, market=MarketType.FUTURES, direction="Short")
        assert s.side == "sell"

    def test_side_close(self):
        s = self._make_signal(action=SignalAction.CLOSE)
        assert s.side == "buy"  # default fallback

    def test_to_dict(self):
        s = self._make_signal()
        d = s.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["action"] == "buy"
        assert d["market"] == "spot"

    def test_repr(self):
        s = self._make_signal()
        r = repr(s)
        assert "BTC/USDT" in r
        assert "buy" in r


# ---------------------------------------------------------------------------
# SignalGenerator tests
# ---------------------------------------------------------------------------


class TestSignalGenerator:
    def _make_generator(self, **kwargs):
        # Disable cooldown for most tests
        cfg = {"cooldown_minutes": 0, **kwargs}
        return SignalGenerator(config=cfg)

    def _make_prediction(self, direction="Long", confidence=0.90, magnitude=0.02, symbol="BTC/USDT"):
        return Prediction(
            direction=direction,
            direction_idx=0 if direction == "Long" else (1 if direction == "Short" else 2),
            direction_probs={"Long": 0.7, "Short": 0.2, "Neutral": 0.1},
            magnitude=magnitude,
            confidence=confidence,
            timestamp=datetime(2024, 6, 1),
            symbol=symbol,
        )

    def test_generate_long_spot(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Long", confidence=0.80)
        signal = gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.BUY
        assert signal.market == MarketType.SPOT
        assert signal.leverage == 1.0

    def test_generate_long_futures_high_confidence(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Long", confidence=0.92)
        signal = gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.LONG
        assert signal.market == MarketType.FUTURES
        assert signal.leverage > 1.0

    def test_generate_short_always_futures(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Short", confidence=0.88, magnitude=0.025)
        signal = gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.SHORT
        assert signal.market == MarketType.FUTURES

    def test_generate_neutral_hold(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Neutral", confidence=0.45, magnitude=0.001)
        signal = gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.HOLD

    def test_generate_low_confidence_hold(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Long", confidence=0.50, magnitude=0.02)
        signal = gen.generate(pred, current_price=65000.0)
        assert signal.action == SignalAction.HOLD

    def test_generate_has_price_targets(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Long", confidence=0.80)
        signal = gen.generate(pred, current_price=100.0)
        assert signal.stop_loss_price is not None
        assert signal.stop_loss_price < 100.0
        assert len(signal.take_profit_prices) == 3
        assert all(tp > 100.0 for tp in signal.take_profit_prices)

    def test_generate_short_price_targets(self):
        gen = self._make_generator()
        pred = self._make_prediction(direction="Short", confidence=0.88, magnitude=0.025)
        signal = gen.generate(pred, current_price=100.0)
        assert signal.stop_loss_price > 100.0
        assert all(tp < 100.0 for tp in signal.take_profit_prices)

    def test_decide_market_long_moderate(self):
        gen = self._make_generator()
        market, lev = gen.decide_market("Long", 0.80)
        assert market == MarketType.SPOT
        assert lev == 1.0

    def test_decide_market_long_high(self):
        gen = self._make_generator()
        market, lev = gen.decide_market("Long", 0.92)
        assert market == MarketType.FUTURES
        assert lev > 1.0

    def test_decide_market_short(self):
        gen = self._make_generator()
        market, lev = gen.decide_market("Short", 0.80)
        assert market == MarketType.FUTURES

    def test_decide_market_neutral(self):
        gen = self._make_generator()
        market, lev = gen.decide_market("Neutral", 0.50)
        assert market == MarketType.SPOT
        assert lev == 1.0

    def test_compute_leverage_scaling(self):
        gen = self._make_generator()
        # At min threshold → 1.0
        assert gen._compute_leverage(0.85) == pytest.approx(1.0)
        # At max threshold → 3.0
        assert gen._compute_leverage(0.95) == pytest.approx(3.0)
        # Above max → capped at 3.0
        assert gen._compute_leverage(0.99) == pytest.approx(3.0)
        # Below min → 1.0
        assert gen._compute_leverage(0.70) == pytest.approx(1.0)

    def test_compute_price_targets_long(self):
        gen = self._make_generator()
        sl, tps = gen.compute_price_targets("Long", 100.0)
        assert sl == pytest.approx(98.0)  # 2% stop loss
        assert len(tps) == 3
        assert tps[0] == pytest.approx(103.0)
        assert tps[1] == pytest.approx(105.0)
        assert tps[2] == pytest.approx(108.0)

    def test_compute_price_targets_short(self):
        gen = self._make_generator()
        sl, tps = gen.compute_price_targets("Short", 100.0)
        assert sl == pytest.approx(102.0)
        assert tps[0] == pytest.approx(97.0)

    def test_generate_batch(self):
        gen = self._make_generator()
        preds = [
            self._make_prediction("Long", 0.80, 0.02, "BTC/USDT"),
            self._make_prediction("Short", 0.88, 0.025, "ETH/USDT"),
            self._make_prediction("Neutral", 0.45, 0.001, "SOL/USDT"),
        ]
        prices = {"BTC/USDT": 65000.0, "ETH/USDT": 3500.0, "SOL/USDT": 150.0}
        signals = gen.generate_batch(preds, prices)
        assert len(signals) == 3
        assert signals[0].action == SignalAction.BUY
        assert signals[1].action == SignalAction.SHORT
        assert signals[2].action == SignalAction.HOLD

    def test_stats(self):
        gen = self._make_generator()
        pred = self._make_prediction("Long", 0.80)
        gen.generate(pred, 65000.0)
        stats = gen.get_stats()
        assert stats["signals_emitted"] >= 1
        assert stats["total_processed"] >= 1

    def test_direction_to_action_buy(self):
        assert SignalGenerator._direction_to_action("Long", MarketType.SPOT) == SignalAction.BUY

    def test_direction_to_action_long(self):
        assert SignalGenerator._direction_to_action("Long", MarketType.FUTURES) == SignalAction.LONG

    def test_direction_to_action_short(self):
        assert SignalGenerator._direction_to_action("Short", MarketType.FUTURES) == SignalAction.SHORT

    def test_direction_to_action_neutral(self):
        assert SignalGenerator._direction_to_action("Neutral", MarketType.SPOT) == SignalAction.HOLD
