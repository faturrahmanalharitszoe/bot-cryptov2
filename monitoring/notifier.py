"""
Telegram Notifier — Send trade alerts and status updates via Telegram.

Usage:
    notifier = TelegramNotifier(bot_token, chat_id)
    notifier.send_trade_alert(trade_signal)
    notifier.send_daily_summary(stats)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications via Telegram Bot API.

    Usage:
        notifier = TelegramNotifier(
            bot_token="123456:ABC-DEF...",
            chat_id="-100123456789",
        )
        notifier.send_message("Trade executed: BTC/USDT LONG")
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

        if not bot_token or not chat_id:
            self.enabled = False
            logger.warning("Telegram notifier disabled: missing bot_token or chat_id")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a plain text message.

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            url = f"{self._base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Telegram message sent: %s", text[:50])
            return True
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_trade_alert(self, signal: Any) -> bool:
        """Send a formatted trade signal alert.

        Args:
            signal: TradeSignal object
        """
        emoji_map = {
            "buy": "🟢",
            "sell": "🔴",
            "long": "🟢",
            "short": "🔴",
            "close": "🟡",
            "hold": "⚪",
        }

        action = getattr(signal, "action", None)
        action_val = action.value if action else "unknown"
        emoji = emoji_map.get(action_val, "📊")

        text = (
            f"{emoji} <b>Trade Signal</b>\n\n"
            f"<b>Symbol:</b> {getattr(signal, 'symbol', 'N/A')}\n"
            f"<b>Action:</b> {action_val.upper()}\n"
            f"<b>Market:</b> {getattr(signal, 'market', 'N/A')}\n"
            f"<b>Direction:</b> {getattr(signal, 'direction', 'N/A')}\n"
            f"<b>Confidence:</b> {getattr(signal, 'confidence', 0):.1%}\n"
            f"<b>Magnitude:</b> {getattr(signal, 'magnitude', 0):.2%}\n"
            f"<b>Leverage:</b> {getattr(signal, 'leverage', 1.0)}x\n"
            f"<b>Strength:</b> {getattr(signal, 'strength', 0):.3f}\n"
        )

        entry = getattr(signal, "entry_price", None)
        if entry:
            text += f"<b>Entry:</b> ${entry:,.2f}\n"

        sl = getattr(signal, "stop_loss_price", None)
        if sl:
            text += f"<b>Stop Loss:</b> ${sl:,.2f}\n"

        tps = getattr(signal, "take_profit_prices", [])
        if tps:
            tp_str = " / ".join(f"${tp:,.2f}" for tp in tps)
            text += f"<b>Take Profit:</b> {tp_str}\n"

        return self.send_message(text)

    def send_trade_executed(
        self,
        symbol: str,
        side: str,
        market: str,
        amount: float,
        price: float,
        leverage: float = 1.0,
    ) -> bool:
        """Send trade execution confirmation."""
        emoji = "✅" if side in ("buy", "long") else "❌"

        text = (
            f"{emoji} <b>Trade Executed</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Market:</b> {market}\n"
            f"<b>Amount:</b> {amount:.8f}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>Leverage:</b> {leverage}x\n"
            f"<b>Value:</b> ${amount * price:,.2f}\n"
        )

        return self.send_message(text)

    def send_stop_loss_alert(
        self, symbol: str, price: float, pnl: float,
    ) -> bool:
        """Send stop-loss triggered alert."""
        text = (
            f"🛑 <b>Stop-Loss Triggered</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>PnL:</b> ${pnl:,.2f}\n"
        )
        return self.send_message(text)

    def send_daily_summary(self, stats: dict[str, Any]) -> bool:
        """Send daily performance summary."""
        portfolio = stats.get("portfolio_value", 0)
        total_pnl = stats.get("total_realized_pnl", 0)
        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0) * 100
        drawdown = stats.get("total_drawdown", 0) * 100

        text = (
            f"📊 <b>Daily Summary</b>\n\n"
            f"<b>Portfolio:</b> ${portfolio:,.2f}\n"
            f"<b>Total PnL:</b> ${total_pnl:,.2f}\n"
            f"<b>Trades Today:</b> {total_trades}\n"
            f"<b>Win Rate:</b> {win_rate:.1f}%\n"
            f"<b>Drawdown:</b> {drawdown:.2f}%\n"
            f"<b>Open Positions:</b> {stats.get('open_positions', 0)}\n"
        )

        if stats.get("trading_halted"):
            text += f"\n🚨 <b>TRADING HALTED:</b> {stats.get('halt_reason', 'unknown')}\n"

        return self.send_message(text)

    def send_drawdown_warning(self, daily_dd: float, weekly_dd: float) -> bool:
        """Send drawdown warning."""
        text = (
            f"⚠️ <b>Drawdown Warning</b>\n\n"
            f"<b>Daily DD:</b> {daily_dd:.2%}\n"
            f"<b>Weekly DD:</b> {weekly_dd:.2%}\n"
        )

        if daily_dd >= 0.04:
            text += "\n🔴 Approaching daily limit (5%)"
        if weekly_dd >= 0.08:
            text += "\n🔴 Approaching weekly limit (10%)"

        return self.send_message(text)

    def send_error(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        text = (
            f"🚨 <b>Error</b>\n\n"
            f"<b>Context:</b> {context}\n"
            f"<b>Error:</b> <code>{error}</code>\n"
        )
        return self.send_message(text)
