"""
Exchange Execution — ccxt wrapper for Binance spot + futures.

Handles:
  - Connecting to Binance (testnet or live)
  - Spot trading: market/limit orders
  - Futures trading: market/limit orders with leverage
  - Account balance queries
  - Order status tracking
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import ccxt

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order execution."""

    order_id: str
    symbol: str
    side: str           # buy | sell
    order_type: str     # market | limit
    market: str         # spot | futures
    status: str         # open | closed | canceled
    price: float        # fill price or limit price
    amount: float       # order quantity
    cost: float         # total cost (price × amount)
    fee: float          # trading fee
    leverage: float     # 1.0 for spot
    timestamp: float    # unix timestamp
    raw: dict = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.status == "closed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "market": self.market,
            "status": self.status,
            "price": self.price,
            "amount": self.amount,
            "cost": self.cost,
            "fee": self.fee,
            "leverage": self.leverage,
            "timestamp": self.timestamp,
        }


class ExchangeClient:
    """Unified ccxt client for Binance spot and futures.

    Usage:
        client = ExchangeClient(testnet=True)
        client.connect()

        # Spot
        result = client.create_spot_order("BTC/USDT", "buy", "market", 0.001)

        # Futures
        result = client.create_futures_order("BTC/USDT", "buy", "market", 0.001, leverage=2)
    """

    def __init__(
        self,
        testnet: bool = True,
        api_key: str = "",
        secret: str = "",
        rate_limit_per_minute: int = 1200,
    ):
        self.testnet = testnet
        self.api_key = api_key
        self.secret = secret
        self.rate_limit = rate_limit_per_minute

        self.spot_exchange: ccxt.binance | None = None
        self.futures_exchange: ccxt.binance | None = None

        self._connected = False

    def connect(self) -> None:
        """Initialize ccxt exchange instances."""
        common_config = {
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
            "rateLimit": 60_000 / self.rate_limit,  # ms between requests
            "options": {"defaultType": "spot"},
        }

        # Spot
        self.spot_exchange = ccxt.binance(common_config)
        if self.testnet:
            self.spot_exchange.set_sandbox_mode(True)

        # Futures
        futures_config = {**common_config, "options": {"defaultType": "future"}}
        self.futures_exchange = ccxt.binance(futures_config)
        if self.testnet:
            self.futures_exchange.set_sandbox_mode(True)

        self._connected = True
        logger.info(
            "Exchange connected: testnet=%s, rate_limit=%d/min",
            self.testnet, self.rate_limit,
        )

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Exchange not connected. Call connect() first.")

    # -------------------------------------------------------------------
    # Spot orders
    # -------------------------------------------------------------------

    def create_spot_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        params: dict | None = None,
    ) -> OrderResult:
        """Create a spot order.

        Args:
            symbol: e.g. "BTC/USDT"
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: quantity in base currency
            price: limit price (required for limit orders)
            params: additional ccxt params

        Returns:
            OrderResult
        """
        self._ensure_connected()
        assert self.spot_exchange is not None

        params = params or {}
        logger.info("Spot %s %s %s amount=%.8f", side, order_type, symbol, amount)

        if order_type == "market":
            order = self.spot_exchange.create_order(
                symbol, "market", side, amount, params=params
            )
        elif order_type == "limit":
            if price is None:
                raise ValueError("Price required for limit orders")
            order = self.spot_exchange.create_order(
                symbol, "limit", side, amount, price, params=params
            )
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        return self._parse_order(order, market="spot", leverage=1.0)

    # -------------------------------------------------------------------
    # Futures orders
    # -------------------------------------------------------------------

    def create_futures_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        leverage: float = 1.0,
        params: dict | None = None,
    ) -> OrderResult:
        """Create a futures order with leverage.

        Args:
            symbol: e.g. "BTC/USDT"
            side: "buy" or "sell"
            order_type: "market" or "limit"
            amount: quantity in base currency
            price: limit price (required for limit orders)
            leverage: leverage multiplier (1-3)
            params: additional ccxt params

        Returns:
            OrderResult
        """
        self._ensure_connected()
        assert self.futures_exchange is not None

        params = params or {}

        # Set leverage before order
        if leverage > 1.0:
            try:
                self.futures_exchange.set_leverage(int(leverage), symbol)
                logger.info("Set leverage %dx for %s", leverage, symbol)
            except Exception as e:
                logger.warning("Could not set leverage for %s: %s", symbol, e)

        # Set margin mode to isolated
        try:
            self.futures_exchange.set_margin_mode("isolated", symbol)
        except Exception:
            pass  # may already be isolated

        logger.info(
            "Futures %s %s %s amount=%.8f leverage=%.1fx",
            side, order_type, symbol, amount, leverage,
        )

        if order_type == "market":
            order = self.futures_exchange.create_order(
                symbol, "market", side, amount, params=params
            )
        elif order_type == "limit":
            if price is None:
                raise ValueError("Price required for limit orders")
            order = self.futures_exchange.create_order(
                symbol, "limit", side, amount, price, params=params
            )
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        return self._parse_order(order, market="futures", leverage=leverage)

    # -------------------------------------------------------------------
    # Order management
    # -------------------------------------------------------------------

    def cancel_order(self, order_id: str, symbol: str, market: str = "spot") -> bool:
        """Cancel an open order."""
        self._ensure_connected()
        exchange = self._get_exchange(market)

        try:
            exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s for %s on %s", order_id, symbol, market)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def get_order_status(
        self, order_id: str, symbol: str, market: str = "spot"
    ) -> OrderResult | None:
        """Fetch order status."""
        self._ensure_connected()
        exchange = self._get_exchange(market)

        try:
            order = exchange.fetch_order(order_id, symbol)
            return self._parse_order(order, market=market, leverage=1.0)
        except Exception as e:
            logger.error("Failed to fetch order %s: %s", order_id, e)
            return None

    def get_open_orders(
        self, symbol: str | None = None, market: str = "spot"
    ) -> list[OrderResult]:
        """Fetch open orders."""
        self._ensure_connected()
        exchange = self._get_exchange(market)

        try:
            orders = exchange.fetch_open_orders(symbol)
            return [self._parse_order(o, market=market) for o in orders]
        except Exception as e:
            logger.error("Failed to fetch open orders: %s", e)
            return []

    # -------------------------------------------------------------------
    # Account info
    # -------------------------------------------------------------------

    def get_balance(self, market: str = "spot") -> dict[str, Any]:
        """Fetch account balance."""
        self._ensure_connected()
        exchange = self._get_exchange(market)

        try:
            balance = exchange.fetch_balance()
            # Filter to non-zero balances
            result: dict[str, Any] = {}
            for currency, data in balance.get("total", {}).items():
                if data and float(data) > 0:
                    result[currency] = {
                        "total": float(data),
                        "free": float(balance.get("free", {}).get(currency, 0)),
                        "used": float(balance.get("used", {}).get(currency, 0)),
                    }
            return result
        except Exception as e:
            logger.error("Failed to fetch balance: %s", e)
            return {}

    def get_ticker(self, symbol: str, market: str = "spot") -> dict[str, Any] | None:
        """Fetch current ticker for a symbol."""
        self._ensure_connected()
        exchange = self._get_exchange(market)

        try:
            ticker = exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "volume": ticker.get("baseVolume"),
                "change_pct": ticker.get("percentage"),
            }
        except Exception as e:
            logger.error("Failed to fetch ticker for %s: %s", symbol, e)
            return None

    def close_position(self, symbol: str, market: str = "futures") -> OrderResult | None:
        """Close an open futures position."""
        self._ensure_connected()
        assert self.futures_exchange is not None

        try:
            # Fetch current position
            positions = self.futures_exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get("contracts", 0))
                if contracts > 0:
                    side = "sell"
                    amount = contracts
                elif contracts < 0:
                    side = "buy"
                    amount = abs(contracts)
                else:
                    continue

                order = self.futures_exchange.create_order(
                    symbol, "market", side, amount,
                    params={"reduceOnly": True},
                )
                return self._parse_order(order, market="futures")

            logger.info("No open position for %s", symbol)
            return None
        except Exception as e:
            logger.error("Failed to close position for %s: %s", symbol, e)
            return None

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_exchange(self, market: str) -> ccxt.binance:
        if market == "futures":
            assert self.futures_exchange is not None
            return self.futures_exchange
        assert self.spot_exchange is not None
        return self.spot_exchange

    @staticmethod
    def _parse_order(
        order: dict,
        market: str = "spot",
        leverage: float = 1.0,
    ) -> OrderResult:
        """Parse ccxt order dict into OrderResult."""
        fee = order.get("fee", {})
        fee_cost = float(fee.get("cost", 0)) if fee else 0.0

        return OrderResult(
            order_id=str(order.get("id", "")),
            symbol=order.get("symbol", ""),
            side=order.get("side", ""),
            order_type=order.get("type", ""),
            market=market,
            status=order.get("status", "open"),
            price=float(order.get("average") or order.get("price") or 0),
            amount=float(order.get("amount") or 0),
            cost=float(order.get("cost") or 0),
            fee=fee_cost,
            leverage=leverage,
            timestamp=float(order.get("timestamp") or time.time() * 1000) / 1000,
            raw=order,
        )
