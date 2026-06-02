"""
Tests for FilterConfig and SignalFilter.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from signals.filters import FilterConfig, SignalFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features_df(n=50, close_start=100.0, trend="up"):
    """Create a simple features DataFrame for filter testing."""
    np.random.seed(123)
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    if trend == "up":
        closes = close_start + np.arange(n) * 0.5 + np.random.normal(0, 0.1, n)
    elif trend == "down":
        closes = close_start - np.arange(n) * 0.5 + np.random.normal(0, 0.1, n)
    else:
        closes = close_start + np.random.normal(0, 0.5, n)

    volumes = np.full(n, 1000.0)
    atr = np.full(n, 50.0)
    return pd.DataFrame({
        "close": closes,
        "volume": volumes,
        "atr": atr,
    }, index=dates)


# ---------------------------------------------------------------------------
# FilterConfig tests
# ---------------------------------------------------------------------------


class TestFilterConfig:
    def test_defaults(self):
        cfg = FilterConfig()
        assert cfg.cooldown_minutes == 30
        assert cfg.min_confidence == 0.70
        assert cfg.min_magnitude == 0.005
        assert cfg.volatility_threshold == 3.0

    def test_custom(self):
        cfg = FilterConfig(cooldown_minutes=15, min_confidence=0.80)
        assert cfg.cooldown_minutes == 15
        assert cfg.min_confidence == 0.80


# ---------------------------------------------------------------------------
# SignalFilter tests
# ---------------------------------------------------------------------------


class TestSignalFilter:
    def test_cooldown_no_prior_signal(self):
        sf = SignalFilter()
        assert sf.check_cooldown("BTC/USDT") is True

    def test_cooldown_active(self):
        sf = SignalFilter(FilterConfig(cooldown_minutes=30))
        now = datetime(2024, 6, 1, 12, 0)
        sf.record_signal("BTC/USDT", now=now)
        # 10 minutes later — still in cooldown
        assert sf.check_cooldown("BTC/USDT", now=now + timedelta(minutes=10)) is False

    def test_cooldown_expired(self):
        sf = SignalFilter(FilterConfig(cooldown_minutes=30))
        now = datetime(2024, 6, 1, 12, 0)
        sf.record_signal("BTC/USDT", now=now)
        # 31 minutes later — cooldown expired
        assert sf.check_cooldown("BTC/USDT", now=now + timedelta(minutes=31)) is True

    def test_cooldown_exact_boundary(self):
        sf = SignalFilter(FilterConfig(cooldown_minutes=30))
        now = datetime(2024, 6, 1, 12, 0)
        sf.record_signal("BTC/USDT", now=now)
        assert sf.check_cooldown("BTC/USDT", now=now + timedelta(minutes=30)) is True

    def test_reset_cooldown_single(self):
        sf = SignalFilter()
        sf.record_signal("BTC/USDT")
        sf.reset_cooldown("BTC/USDT")
        assert sf.check_cooldown("BTC/USDT") is True

    def test_reset_cooldown_all(self):
        sf = SignalFilter()
        sf.record_signal("BTC/USDT")
        sf.record_signal("ETH/USDT")
        sf.reset_cooldown()
        assert sf.check_cooldown("BTC/USDT") is True
        assert sf.check_cooldown("ETH/USDT") is True

    def test_trend_alignment_long_uptrend(self):
        sf = SignalFilter()
        df = _make_features_df(trend="up")
        assert sf.check_trend_alignment("Long", df) is True

    def test_trend_alignment_long_downtrend(self):
        sf = SignalFilter()
        df = _make_features_df(trend="down")
        assert sf.check_trend_alignment("Long", df) is False

    def test_trend_alignment_short_downtrend(self):
        sf = SignalFilter()
        df = _make_features_df(trend="down")
        assert sf.check_trend_alignment("Short", df) is True

    def test_trend_alignment_short_uptrend(self):
        sf = SignalFilter()
        df = _make_features_df(trend="up")
        assert sf.check_trend_alignment("Short", df) is False

    def test_trend_alignment_disabled(self):
        cfg = FilterConfig(trend_alignment_enabled=False)
        sf = SignalFilter(cfg)
        df = _make_features_df(trend="down")
        assert sf.check_trend_alignment("Long", df) is True

    def test_trend_alignment_missing_column(self):
        sf = SignalFilter()
        df = pd.DataFrame({"price": [1, 2, 3]})
        assert sf.check_trend_alignment("Long", df) is True  # lenient

    def test_volatility_normal(self):
        sf = SignalFilter()
        df = _make_features_df()
        assert sf.check_volatility(df) is True

    def test_volatility_extreme(self):
        cfg = FilterConfig(volatility_threshold=0.5)
        sf = SignalFilter(cfg)
        df = _make_features_df()
        # Make last ATR much higher than average
        df.iloc[-1, df.columns.get_loc("atr")] = 999.0
        assert sf.check_volatility(df) is False

    def test_volatility_disabled(self):
        cfg = FilterConfig(volatility_filter_enabled=False)
        sf = SignalFilter(cfg)
        df = _make_features_df()
        df.iloc[-1, df.columns.get_loc("atr")] = 999.0
        assert sf.check_volatility(df) is True

    def test_volume_sufficient(self):
        sf = SignalFilter()
        df = _make_features_df()
        assert sf.check_volume(df) is True

    def test_volume_low(self):
        sf = SignalFilter()
        df = _make_features_df()
        df.iloc[-1, df.columns.get_loc("volume")] = 1.0  # very low
        assert sf.check_volume(df) is False

    def test_volume_disabled(self):
        cfg = FilterConfig(volume_filter_enabled=False)
        sf = SignalFilter(cfg)
        df = _make_features_df()
        df.iloc[-1, df.columns.get_loc("volume")] = 0.0
        assert sf.check_volume(df) is True

    def test_compute_signal_strength(self):
        sf = SignalFilter()
        # confidence=0.9, magnitude=0.025 → mag_norm=0.5
        # strength = 0.7*0.9 + 0.3*0.5 = 0.63 + 0.15 = 0.78
        s = sf.compute_signal_strength(0.9, 0.025)
        assert s == pytest.approx(0.78)

    def test_compute_signal_strength_capped(self):
        sf = SignalFilter()
        s = sf.compute_signal_strength(1.0, 1.0)
        assert s <= 1.0

    def test_apply_all_passes(self):
        sf = SignalFilter()
        df = _make_features_df(trend="up")
        passed, reasons, strength = sf.apply_all(
            symbol="BTC/USDT",
            direction="Long",
            confidence=0.85,
            magnitude=0.02,
            features_df=df,
        )
        assert passed is True
        assert reasons == []
        assert strength > 0

    def test_apply_all_rejected_neutral(self):
        sf = SignalFilter()
        passed, reasons, strength = sf.apply_all(
            symbol="BTC/USDT",
            direction="Neutral",
            confidence=0.85,
            magnitude=0.02,
        )
        assert passed is False
        assert "direction_neutral" in reasons

    def test_apply_all_rejected_low_confidence(self):
        sf = SignalFilter()
        passed, reasons, _ = sf.apply_all(
            symbol="BTC/USDT",
            direction="Long",
            confidence=0.50,
            magnitude=0.02,
        )
        assert passed is False
        assert any("low_confidence" in r for r in reasons)

    def test_apply_all_rejected_low_magnitude(self):
        sf = SignalFilter()
        passed, reasons, _ = sf.apply_all(
            symbol="BTC/USDT",
            direction="Long",
            confidence=0.85,
            magnitude=0.001,
        )
        assert passed is False
        assert any("low_magnitude" in r for r in reasons)

    def test_apply_all_rejected_cooldown(self):
        sf = SignalFilter(FilterConfig(cooldown_minutes=30))
        now = datetime(2024, 6, 1, 12, 0)
        sf.record_signal("BTC/USDT", now=now)
        passed, reasons, _ = sf.apply_all(
            symbol="BTC/USDT",
            direction="Long",
            confidence=0.85,
            magnitude=0.02,
            now=now + timedelta(minutes=5),
        )
        assert passed is False
        assert any("cooldown" in r for r in reasons)

    def test_apply_all_rejected_trend_misalignment(self):
        sf = SignalFilter()
        df = _make_features_df(trend="down")
        passed, reasons, _ = sf.apply_all(
            symbol="BTC/USDT",
            direction="Long",
            confidence=0.85,
            magnitude=0.02,
            features_df=df,
        )
        assert passed is False
        assert any("trend_misalignment" in r for r in reasons)

    def test_apply_all_multiple_rejections(self):
        sf = SignalFilter()
        df = _make_features_df(trend="down")
        passed, reasons, _ = sf.apply_all(
            symbol="BTC/USDT",
            direction="Long",
            confidence=0.50,   # low confidence
            magnitude=0.001,   # low magnitude
            features_df=df,    # trend misalignment
        )
        assert passed is False
        assert len(reasons) >= 3
