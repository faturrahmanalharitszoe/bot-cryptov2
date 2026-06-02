"""
OHLCV Scraper — Fetches candlestick (klines) data from Binance.

Uses ccxt for unified exchange access, supporting both spot and futures.
Handles incremental updates (only fetch new candles since last scrape).
"""

import ccxt
import pandas as pd
import time as time_module
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from scrapers.base_scraper import BaseScraper
from storage.parquet_store import ParquetStore
from monitoring.logger import get_logger

logger = get_logger("scraper.ohlcv")

# Binance timeframe mapping
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}

# Binance max klines per request
MAX_KLINES_PER_REQUEST = 1000


class OHLCVScraper(BaseScraper):
    """Scrapes OHLCV candlestick data from Binance via ccxt."""

    def __init__(
        self,
        testnet: bool = True,
        store: Optional[ParquetStore] = None,
    ):
        super().__init__(
            name="OHLCV",
            max_calls_per_minute=1200,
            max_retries=3,
        )
        
        # Initialize ccxt exchange
        exchange_class = ccxt.binance
        exchange_config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        }
        
        if testnet:
            exchange_config["sandbox"] = True
            logger.info("Using Binance TESTNET")
        
        self.exchange = exchange_class(exchange_config)
        self.store = store or ParquetStore()

    def scrape(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol/timeframe.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle interval (e.g., '5m', '1h')
            since: Start datetime (None = auto from last saved or 90 days ago)
            limit: Max number of candles (None = all available)
            
        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        # Determine start time
        if since is None:
            last_ts = self.store.get_latest_timestamp(symbol, timeframe)
            if last_ts:
                # Start from last saved timestamp + 1 candle
                since = last_ts + timedelta(milliseconds=TIMEFRAME_MS.get(timeframe, 3_600_000))
            else:
                # Default: 90 days of history
                since = datetime.utcnow() - timedelta(days=90)
        
        since_ms = int(since.timestamp() * 1000)
        all_candles = []
        fetched = 0
        
        logger.info(f"Fetching {symbol} {timeframe} from {since.isoformat()}")
        
        while True:
            try:
                # Fetch candles
                remaining = (limit - fetched) if limit else MAX_KLINES_PER_REQUEST
                batch_size = min(remaining, MAX_KLINES_PER_REQUEST)
                
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since_ms,
                    limit=batch_size,
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                fetched += len(candles)
                
                # Move to next batch
                last_candle_ts = candles[-1][0]
                since_ms = last_candle_ts + TIMEFRAME_MS.get(timeframe, 3_600_000)
                
                # Check if we've reached present time
                now_ms = int(datetime.utcnow().timestamp() * 1000)
                if since_ms >= now_ms:
                    break
                
                # Check limit
                if limit and fetched >= limit:
                    break
                
                # Small delay between batches
                time_module.sleep(0.1)
                
            except ccxt.RateLimitExceeded:
                logger.warning(f"Rate limited by Binance, waiting 60s")
                time_module.sleep(60)
            except ccxt.NetworkError as e:
                logger.error(f"Network error: {e}")
                time_module.sleep(5)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise
        
        if not all_candles:
            logger.warning(f"No candles returned for {symbol} {timeframe}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        # Save to store
        rows_saved = self.store.save_ohlcv(symbol, timeframe, df, append=True)
        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}, saved {rows_saved} rows")
        
        return df

    def scrape_all(
        self,
        symbols: List[str],
        timeframes: List[str],
        since: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch OHLCV data for all symbol/timeframe combinations.
        
        Returns:
            Nested dict: results[symbol][timeframe] = DataFrame
        """
        results = {}
        total = len(symbols) * len(timeframes)
        current = 0
        
        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                current += 1
                logger.info(f"[{current}/{total}] Fetching {symbol} {timeframe}")
                try:
                    df = self.scrape(symbol, timeframe, since=since)
                    results[symbol][timeframe] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
                    results[symbol][timeframe] = pd.DataFrame()
        
        return results

    def get_available_symbols(self, quote: str = "USDT", min_volume: float = 1_000_000) -> List[str]:
        """
        Get available trading pairs filtered by quote currency and volume.
        
        Args:
            quote: Quote currency (e.g., 'USDT')
            min_volume: Minimum 24h volume in quote currency
            
        Returns:
            List of symbol strings
        """
        try:
            self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            symbols = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith(f"/{quote}"):
                    continue
                volume = ticker.get("quoteVolume", 0) or 0
                if volume >= min_volume:
                    symbols.append(symbol)
            
            # Sort by volume descending
            symbols.sort(
                key=lambda s: tickers.get(s, {}).get("quoteVolume", 0) or 0,
                reverse=True,
            )
            
            logger.info(f"Found {len(symbols)} {quote} pairs with volume >= {min_volume:,.0f}")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    def close(self):
        """Close exchange connection."""
        self.exchange.close()
        super().close()
