"""
Orderbook Scraper — Fetches L2 order book depth snapshots from Binance.

Captures bid/ask levels for order imbalance analysis and spread monitoring.
"""

import ccxt
import pandas as pd
import time as time_module
from datetime import datetime
from typing import Optional, List, Dict

from scrapers.base_scraper import BaseScraper
from storage.parquet_store import ParquetStore
from monitoring.logger import get_logger

logger = get_logger("scraper.orderbook")


class OrderbookScraper(BaseScraper):
    """Scrapes L2 order book depth from Binance via ccxt."""

    def __init__(
        self,
        testnet: bool = True,
        store: Optional[ParquetStore] = None,
        depth: int = 20,
    ):
        super().__init__(
            name="Orderbook",
            max_calls_per_minute=600,
            max_retries=3,
        )
        
        exchange_class = ccxt.binance
        exchange_config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        }
        
        if testnet:
            exchange_config["sandbox"] = True
        
        self.exchange = exchange_class(exchange_config)
        self.store = store or ParquetStore()
        self.depth = depth

    def scrape(self, symbol: str, depth: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch current order book snapshot.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            depth: Number of price levels (default: self.depth)
            
        Returns:
            DataFrame with bid/ask levels and derived metrics
        """
        depth = depth or self.depth
        
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=depth)
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limited, waiting 60s")
            time_module.sleep(60)
            orderbook = self.exchange.fetch_order_book(symbol, limit=depth)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            raise
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            logger.warning(f"Empty orderbook for {symbol}")
            return pd.DataFrame()
        
        # Build summary row
        timestamp = datetime.utcnow()
        
        # Price and volume aggregates
        bid_prices = [b[0] for b in bids]
        bid_volumes = [b[1] for b in bids]
        ask_prices = [a[0] for a in asks]
        ask_volumes = [a[1] for a in asks]
        
        total_bid_volume = sum(bid_volumes)
        total_ask_volume = sum(ask_volumes)
        
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        spread = best_ask - best_bid
        spread_pct = spread / best_bid if best_bid > 0 else 0
        
        # Order book imbalance (bid vs ask pressure)
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) \
            if (total_bid_volume + total_ask_volume) > 0 else 0
        
        # VWAP for bids and asks
        bid_vwap = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / total_bid_volume \
            if total_bid_volume > 0 else 0
        ask_vwap = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / total_ask_volume \
            if total_ask_volume > 0 else 0
        
        # Wall detection (large orders)
        avg_bid_vol = total_bid_volume / len(bids) if bids else 0
        avg_ask_vol = total_ask_volume / len(asks) if asks else 0
        bid_walls = sum(1 for v in bid_volumes if v > avg_bid_vol * 3)
        ask_walls = sum(1 for v in ask_volumes if v > avg_ask_vol * 3)
        
        row = {
            "timestamp": timestamp,
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "mid_price": (best_bid + best_ask) / 2,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "imbalance": imbalance,
            "bid_vwap": bid_vwap,
            "ask_vwap": ask_vwap,
            "bid_walls": bid_walls,
            "ask_walls": ask_walls,
        }
        
        df = pd.DataFrame([row])
        
        # Save to store
        self.store.save_orderbook(symbol, df, append=True)
        logger.debug(f"Saved orderbook snapshot for {symbol}: spread={spread_pct:.4%}")
        
        return df

    def scrape_all(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch orderbook for multiple symbols and return combined DataFrame."""
        all_rows = []
        for symbol in symbols:
            try:
                df = self.scrape(symbol)
                if not df.empty:
                    all_rows.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
        
        if all_rows:
            return pd.concat(all_rows, ignore_index=True)
        return pd.DataFrame()

    def close(self):
        """Close exchange connection."""
        self.exchange.close()
        super().close()
