"""Signals — Converts model predictions into trade signals.

Usage:
    from signals.generator import SignalGenerator, TradeSignal, SignalAction, MarketType
    from signals.filters import SignalFilter, FilterConfig
"""

from signals.generator import (
    SignalGenerator,
    TradeSignal,
    SignalAction,
    MarketType,
)
from signals.filters import SignalFilter, FilterConfig

__all__ = [
    "SignalGenerator",
    "TradeSignal",
    "SignalAction",
    "MarketType",
    "SignalFilter",
    "FilterConfig",
]
