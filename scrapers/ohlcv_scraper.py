"""
OHLCV Scraper — Fetches candlestick (klines) data from Binance or Yahoo Finance.

Primary source : Binance mainnet public API via ccxt (auto-detects working mirror).
Fallback source: Yahoo Finance via yfinance (used automatically when Binance is
                 geo-blocked or otherwise unreachable — no API key required).

Handles incremental updates (only fetch new candles since last scrape).
"""

import ccxt
import pandas as pd
import time as time_module
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

from scrapers.base_scraper import BaseScraper
from storage.parquet_store import ParquetStore
from monitoring.logger import get_logger

logger = get_logger("scraper.ohlcv")

# Binance timeframe → milliseconds per candle
TIMEFRAME_MS = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
    "6h":  21_600_000,
    "8h":  28_800_000,
    "12h": 43_200_000,
    "1d":  86_400_000,
    "1w":  604_800_000,
}

# Binance max klines per request
MAX_KLINES_PER_REQUEST = 1000

# yfinance interval strings  (Binance tf → yfinance interval)
# yfinance history limits: 1m=7d, 2/5/15/30/90m=60d, 1h=730d, 1d+=full
_YF_INTERVAL_MAP = {
    "1m":  "1m",
    "3m":  "2m",   # closest available
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "1h",   # closest available
    "4h":  "1h",   # closest available
    "1d":  "1d",
    "1w":  "1wk",
}

# Maximum history yfinance will return per interval
_YF_MAX_DAYS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "1d":  3650,
    "1wk": 3650,
}

# Try the alias if the primary returns empty.
_YF_TICKER_ALIASES: dict[str, list[str]] = {
    "SOL-USD":  ["SOL-USD"],
    "DOT-USD":  ["DOT-USD"],
    "BTC-USD":  ["BTC-USD"],
    "ETH-USD":  ["ETH-USD"],
}

# Max retry attempts for transient yfinance DNS/network errors
_YF_MAX_RETRIES = 3
_YF_RETRY_DELAY = 2.0  # seconds between retries


def _symbol_to_yf(symbol: str) -> str:
    """Convert ccxt-style symbol to Yahoo Finance ticker.

    Examples:
        BTC/USDT → BTC-USD
        ETH/USDT → ETH-USD
        SOL/USDT → SOL-USD
    """
    base = symbol.split("/")[0] if "/" in symbol else symbol
    return f"{base}-USD"


class OHLCVScraper(BaseScraper):
    """Scrapes OHLCV candlestick data.

    Primary:  Binance mainnet public API (auto-selects working mirror).
    Fallback: Yahoo Finance (yfinance) when Binance is unreachable.
    """

    # Binance provides several geographically distributed API clusters.
    # We probe them in order and use the first one that responds.
    _BINANCE_BASE_URLS = [
        "https://api-gcp.binance.com",   # GCP mirror (often unblocked in SEA)
        "https://api1.binance.com",       # AWS mirror #1
        "https://api2.binance.com",       # AWS mirror #2
        "https://api3.binance.com",       # AWS mirror #3
        "https://api.binance.com",        # default (may be geo-blocked)
    ]

    def __init__(
        self,
        testnet: bool = False,
        store: Optional[ParquetStore] = None,
    ):
        super().__init__(
            name="OHLCV",
            max_calls_per_minute=1200,
            max_retries=3,
        )

        # ----------------------------------------------------------------
        # NOTE: use mainnet (testnet=False) for historical OHLCV data.
        # Binance testnet only exposes ~100 bars of history which is
        # nowhere near enough for model training (need 8 000+ bars).
        # The public klines endpoint requires NO API key, so mainnet
        # is safe to use here even when trading via testnet.
        # ----------------------------------------------------------------
        self.store = store or ParquetStore()
        self._use_yfinance = False  # set to True if Binance unreachable

        if testnet:
            logger.warning(
                "OHLCVScraper: testnet=True — Binance testnet only has ~100 bars. "
                "Set testnet=False to fetch full history from the public mainnet API."
            )
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "sandbox": True,
                "options": {"defaultType": "spot"},
            })
            return

        # Mainnet: auto-detect a working endpoint
        exchange, reachable = self._connect_mainnet()
        self.exchange = exchange
        if not reachable:
            self._use_yfinance = True
            logger.warning(
                "OHLCVScraper: Binance is unreachable — switching to Yahoo Finance "
                "(yfinance). Data will be in USD instead of USDT, which is fine for "
                "training. Max history: 60 days for 15m/30m, 730 days for 1h."
            )

    # ----------------------------------------------------------------
    # Connection helpers
    # ----------------------------------------------------------------

    def _connect_mainnet(self) -> tuple:
        """Probe Binance mirror endpoints.

        Returns:
            (exchange, reachable: bool)
            reachable=False means all mirrors failed → caller should use yfinance.
        """
        import requests

        base_config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            "timeout": 8000,
        }

        for url in self._BINANCE_BASE_URLS:
            try:
                resp = requests.get(
                    f"{url}/api/v3/ping",
                    timeout=5,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    logger.info("OHLCVScraper: connected to Binance via %s", url)
                    exchange = ccxt.binance(base_config)
                    # Override every URL entry to use the working mirror
                    exchange.urls["api"] = {
                        k: v.replace("https://api.binance.com", url)
                        for k, v in exchange.urls["api"].items()
                    }
                    return exchange, True
            except Exception as e:
                logger.debug("OHLCVScraper: %s unreachable (%s)", url, e)

        logger.warning(
            "OHLCVScraper: all Binance mirrors unreachable — will use Yahoo Finance."
        )
        return ccxt.binance(base_config), False

    # ----------------------------------------------------------------
    # yfinance backend
    # ----------------------------------------------------------------

    def _scrape_yfinance(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance with retry and ticker aliasing.

        Args:
            symbol:    ccxt-style pair e.g. 'BTC/USDT'
            timeframe: Binance-style interval e.g. '15m', '1h'
            since:     start datetime (UTC, timezone-naive or aware)

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        import yfinance as yf

        yf_interval = _YF_INTERVAL_MAP.get(timeframe)
        if yf_interval is None:
            logger.warning(
                "yfinance does not support timeframe '%s'. Skipping %s.",
                timeframe, symbol,
            )
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # yfinance max lookback per interval
        max_days = _YF_MAX_DAYS.get(yf_interval, 60)

        # Clamp since to yfinance limit
        now_utc = datetime.now(timezone.utc)
        earliest = now_utc - timedelta(days=max_days - 1)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if since < earliest:
            logger.warning(
                "yfinance only supports %d days of %s history. "
                "Clamping start from %s to %s.",
                max_days, yf_interval,
                since.strftime("%Y-%m-%d"), earliest.strftime("%Y-%m-%d"),
            )
            since = earliest

        start_str = since.strftime("%Y-%m-%d")
        end_str   = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

        primary_ticker = _symbol_to_yf(symbol)
        # Build list of tickers to try: aliases first if defined, otherwise just primary
        tickers_to_try = _YF_TICKER_ALIASES.get(primary_ticker, [primary_ticker])
        # Always include primary as last resort if not already present
        if primary_ticker not in tickers_to_try:
            tickers_to_try = list(tickers_to_try) + [primary_ticker]

        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        for ticker in tickers_to_try:
            for attempt in range(1, _YF_MAX_RETRIES + 1):
                try:
                    logger.info(
                        "yfinance: fetching %s (%s) interval=%s from %s (attempt %d/%d)",
                        ticker, symbol, yf_interval, start_str, attempt, _YF_MAX_RETRIES,
                    )
                    raw = yf.download(
                        ticker,
                        start=start_str,
                        end=end_str,
                        interval=yf_interval,
                        progress=False,
                        auto_adjust=True,
                    )
                    if raw is not None and not raw.empty:
                        # Flatten MultiIndex columns if present (yfinance >=0.2.x)
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = [col[0].lower() for col in raw.columns]
                        else:
                            raw.columns = [col.lower() for col in raw.columns]

                        df = raw.reset_index()
                        time_col = next(
                            (c for c in df.columns if c.lower() in ("datetime", "date", "index")),
                            df.columns[0],
                        )
                        df = df.rename(columns={time_col: "timestamp"})
                        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
                        df = df.dropna(subset=["close"])
                        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                        logger.info(
                            "yfinance: fetched %d bars for %s %s (ticker=%s)",
                            len(df), symbol, timeframe, ticker,
                        )
                        return df
                    else:
                        logger.warning(
                            "yfinance: empty response for %s %s (ticker=%s, attempt %d)",
                            symbol, timeframe, ticker, attempt,
                        )
                        break  # empty → try next alias, not retry same ticker

                except Exception as e:
                    logger.warning(
                        "yfinance error for %s %s (ticker=%s, attempt %d/%d): %s",
                        symbol, timeframe, ticker, attempt, _YF_MAX_RETRIES, e,
                    )
                    if attempt < _YF_MAX_RETRIES:
                        time_module.sleep(_YF_RETRY_DELAY * attempt)
                    # If last attempt fails, fall through to next ticker alias

        logger.warning("yfinance returned no data for %s %s after all attempts", symbol, timeframe)
        return empty

    # ----------------------------------------------------------------
    # Public scrape interface
    # ----------------------------------------------------------------

    def scrape(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol/timeframe.

        Automatically uses Yahoo Finance when Binance is unreachable.

        Args:
            symbol:    Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle interval (e.g., '15m', '1h')
            since:     Start datetime (None = auto from last saved or 90 days ago)
            limit:     Max number of candles (None = all available)

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        # Determine start time
        if since is None:
            last_ts = self.store.get_latest_timestamp(symbol, timeframe)
            if last_ts:
                since = last_ts + timedelta(milliseconds=TIMEFRAME_MS.get(timeframe, 3_600_000))
            else:
                since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)

        # ---- Yahoo Finance path ----
        if self._use_yfinance:
            df = self._scrape_yfinance(symbol, timeframe, since)
            if not df.empty:
                rows_saved = self.store.save_ohlcv(symbol, timeframe, df, append=True)
                logger.info(
                    "Saved %d rows for %s %s (source=yfinance)", rows_saved, symbol, timeframe
                )
            return df

        # ---- Binance/ccxt path ----
        since_ms = int(since.timestamp() * 1000)
        all_candles: list = []
        fetched = 0

        logger.info("Fetching %s %s from %s (Binance)", symbol, timeframe, since.isoformat())

        while True:
            try:
                remaining = (limit - fetched) if limit else MAX_KLINES_PER_REQUEST
                batch_size = min(remaining, MAX_KLINES_PER_REQUEST)

                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=batch_size,
                )

                if not candles:
                    break

                all_candles.extend(candles)
                fetched += len(candles)

                last_candle_ts = candles[-1][0]
                since_ms = last_candle_ts + TIMEFRAME_MS.get(timeframe, 3_600_000)

                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                if since_ms >= now_ms:
                    break
                if limit and fetched >= limit:
                    break

                time_module.sleep(0.1)

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limited by Binance, waiting 60s")
                time_module.sleep(60)
            except ccxt.NetworkError as e:
                logger.error("Network error: %s — retrying with yfinance...", e)
                # Degrade gracefully to yfinance for the rest of the session
                self._use_yfinance = True
                return self.scrape(symbol, timeframe, since, limit)
            except ccxt.ExchangeError as e:
                logger.error("Exchange error: %s", e)
                raise

        if not all_candles:
            logger.warning("No candles returned for %s %s", symbol, timeframe)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        rows_saved = self.store.save_ohlcv(symbol, timeframe, df, append=True)
        logger.info(
            "Fetched %d candles for %s %s, saved %d rows (source=Binance)",
            len(df), symbol, timeframe, rows_saved,
        )
        return df

    def scrape_all(
        self,
        symbols: List[str],
        timeframes: List[str],
        since: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch OHLCV data for all symbol/timeframe combinations."""
        results: Dict[str, Dict[str, pd.DataFrame]] = {}
        total = len(symbols) * len(timeframes)
        current = 0

        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                current += 1
                logger.info("[%d/%d] Fetching %s %s", current, total, symbol, timeframe)
                try:
                    df = self.scrape(symbol, timeframe, since=since)
                    results[symbol][timeframe] = df
                except Exception as e:
                    logger.error("Failed to fetch %s %s: %s", symbol, timeframe, e)
                    results[symbol][timeframe] = pd.DataFrame()

        return results

    def get_available_symbols(self, quote: str = "USDT", min_volume: float = 1_000_000) -> List[str]:
        """Get available trading pairs filtered by quote currency and volume."""
        if self._use_yfinance:
            logger.warning(
                "get_available_symbols() is not supported in yfinance mode. "
                "Returning configured pairs only."
            )
            return []

        try:
            self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()

            symbols = [
                s for s, t in tickers.items()
                if s.endswith(f"/{quote}") and (t.get("quoteVolume") or 0) >= min_volume
            ]
            symbols.sort(
                key=lambda s: tickers.get(s, {}).get("quoteVolume", 0) or 0,
                reverse=True,
            )
            logger.info("Found %d %s pairs with volume >= %,.0f", len(symbols), quote, min_volume)
            return symbols

        except Exception as e:
            logger.error("Failed to get symbols: %s", e)
            return []

    def close(self):
        """Close exchange connection."""
        if hasattr(self, "exchange"):
            self.exchange.close()
        super().close()
