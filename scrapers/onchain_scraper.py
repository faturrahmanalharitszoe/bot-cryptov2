"""
On-Chain Scraper — Fetches on-chain data for crypto analysis.

Sources:
- Blockchain.com API (Bitcoin stats)
- Etherscan API (Ethereum gas, transactions)
- Public whale alert data (large transactions)

Provides signals like:
- Large exchange inflows (potential selling pressure)
- Large exchange outflows (potential accumulation)
- Network activity changes
- Gas price spikes (Ethereum congestion)
"""

import requests
from datetime import datetime
from typing import Optional, List, Dict, Any

from scrapers.base_scraper import BaseScraper
from storage.sqlite_store import SQLiteStore
from monitoring.logger import get_logger

logger = get_logger("scraper.onchain")


class OnChainScraper(BaseScraper):
    """Scrapes on-chain data from public APIs."""

    def __init__(
        self,
        db_store: Optional[SQLiteStore] = None,
    ):
        super().__init__(
            name="OnChain",
            max_calls_per_minute=60,
            max_retries=3,
        )
        self.db = db_store or SQLiteStore()

    def scrape(self, symbols: Optional[List[str]] = None, **kwargs) -> int:
        """
        Scrape on-chain data from all configured sources.
        
        Returns:
            Number of entries saved
        """
        total = 0
        
        # Bitcoin stats
        count = self._scrape_bitcoin_stats()
        total += count
        
        # Ethereum stats
        count = self._scrape_ethereum_stats()
        total += count
        
        # Blockchain.com market data
        count = self._scrape_blockchain_market()
        total += count
        
        return total

    def _scrape_bitcoin_stats(self) -> int:
        """Scrape Bitcoin network statistics from Blockchain.com API."""
        url = "https://api.blockchain.info/stats"
        
        try:
            data = self.request_sync("GET", url)
        except Exception as e:
            logger.error(f"Blockchain.info stats error: {e}")
            return 0
        
        # Extract useful metrics
        entries = []
        timestamp = datetime.utcnow()
        
        metrics = {
            "hash_rate": data.get("hash_rate", 0),
            "difficulty": data.get("difficulty", 0),
            "total_btc_sent": data.get("total_btc_sent", 0),
            "total_btc_in_blocks": data.get("total_btc_in_blocks", 0),
            "n_btc_mined": data.get("n_btc_mined", 0),
            "n_tx": data.get("n_tx", 0),
            "trade_volume_usd": data.get("trade_volume_usd", 0),
            "miners_revenue_usd": data.get("miners_revenue_usd", 0),
            "market_price_usd": data.get("market_price_usd", 0),
            "market_cap": data.get("market_cap", 0),
        }
        
        for metric_name, value in metrics.items():
            self.db.save_sentiment(
                source="blockchain_info",
                symbol="BTC",
                title=f"BTC {metric_name}: {value}",
                content="",
                sentiment_score=0.0,
                sentiment_label="neutral",
                published_at=timestamp,
                metadata={"metric": metric_name, "value": value},
            )
            entries.append(1)
        
        logger.debug(f"Scraped {len(entries)} Bitcoin metrics")
        return len(entries)

    def _scrape_ethereum_stats(self) -> int:
        """Scrape Ethereum gas price from public API."""
        url = "https://api.etherscan.io/api"
        params = {
            "module": "gastracker",
            "action": "gasoracle",
        }
        
        try:
            data = self.request_sync("GET", url, params=params)
        except Exception as e:
            logger.error(f"Etherscan gas error: {e}")
            return 0
        
        result = data.get("result", {})
        if isinstance(result, dict):
            safe = result.get("SafeGasPrice", 0)
            propose = result.get("ProposeGasPrice", 0)
            fast = result.get("FastGasPrice", 0)
            
            timestamp = datetime.utcnow()
            
            # Gas price is a sentiment indicator (high gas = high activity/congestion)
            try:
                gas_price = float(propose)
            except (ValueError, TypeError):
                gas_price = 0
            
            # Score: very high gas (>100) = negative (expensive), moderate = positive (active)
            if gas_price > 100:
                score = -0.3
                label = "negative"
            elif gas_price > 50:
                score = -0.1
                label = "neutral"
            elif gas_price > 10:
                score = 0.2
                label = "positive"
            else:
                score = 0.0
                label = "neutral"
            
            self.db.save_sentiment(
                source="etherscan",
                symbol="ETH",
                title=f"ETH Gas: Safe={safe}, Propose={propose}, Fast={fast} Gwei",
                content="",
                sentiment_score=score,
                sentiment_label=label,
                published_at=timestamp,
                metadata={
                    "safe_gas": safe,
                    "propose_gas": propose,
                    "fast_gas": fast,
                },
            )
            return 1
        
        return 0

    def _scrape_blockchain_market(self) -> int:
        """Scrape BTC price from Blockchain.com ticker."""
        url = "https://api.blockchain.info/ticker"
        
        try:
            data = self.request_sync("GET", url)
        except Exception as e:
            logger.error(f"Blockchain.info ticker error: {e}")
            return 0
        
        usd = data.get("USD", {})
        if usd:
            price = usd.get("last", 0)
            volume = usd.get("volume", 0)
            
            self.db.save_sentiment(
                source="blockchain_info_ticker",
                symbol="BTC",
                title=f"BTC Price: ${price:,.2f}, Volume: {volume:,.0f}",
                content="",
                sentiment_score=0.0,
                sentiment_label="neutral",
                published_at=datetime.utcnow(),
                metadata={"price_usd": price, "volume": volume},
            )
            return 1
        
        return 0
