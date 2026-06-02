"""
Sentiment Scraper — Fetches news and social media sentiment for crypto.

Supports:
- CryptoPanic API (crypto-specific news aggregator)
- RSS feeds (CoinTelegraph, CoinDesk, etc.)
- Reddit (via public JSON API)

Each entry is scored for sentiment using a simple keyword-based approach
(faster than loading a full NLP model for scraping).
"""

import re
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from bs4 import BeautifulSoup

from scrapers.base_scraper import BaseScraper
from storage.sqlite_store import SQLiteStore
from monitoring.logger import get_logger

logger = get_logger("scraper.sentiment")

# Simple keyword-based sentiment scoring (fast, no ML needed during scraping)
POSITIVE_WORDS = {
    "bullish", "surge", "rally", "pump", "breakout", "moon", "gain", "profit",
    "adoption", "partnership", "upgrade", "launch", "growth", "record", "high",
    "buy", "accumulate", "support", "optimistic", "recovery", "boost", "milestone",
    "approval", "etf", "institutional", "mainstream", "innovation",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "dump", "plunge", "sell-off", "hack", "scam", "fraud",
    "regulation", "ban", "lawsuit", "sec", "crackdown", "fear", "panic", "loss",
    "decline", "drop", "correction", "fud", "risk", "warning", "collapse",
    "bankruptcy", "insolvency", "liquidation", "exploit", "vulnerability",
}


def simple_sentiment_score(text: str) -> tuple:
    """
    Compute a simple keyword-based sentiment score.
    
    Returns:
        (score: float [-1, 1], label: str)
    """
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    total = pos_count + neg_count
    
    if total == 0:
        return 0.0, "neutral"
    
    score = (pos_count - neg_count) / total
    
    if score > 0.2:
        return score, "positive"
    elif score < -0.2:
        return score, "negative"
    return score, "neutral"


class SentimentScraper(BaseScraper):
    """Scrapes crypto news and social sentiment from multiple sources."""

    def __init__(
        self,
        db_store: Optional[SQLiteStore] = None,
        cryptopanic_api_key: Optional[str] = None,
    ):
        super().__init__(
            name="Sentiment",
            max_calls_per_minute=60,
            max_retries=3,
        )
        self.db = db_store or SQLiteStore()
        self.cryptopanic_key = cryptopanic_api_key

    def scrape(self, symbols: Optional[List[str]] = None, **kwargs) -> int:
        """
        Scrape sentiment from all configured sources.
        
        Args:
            symbols: List of symbols to filter for (e.g., ['BTC', 'ETH'])
            
        Returns:
            Total number of entries saved
        """
        total = 0
        
        # CryptoPanic
        if self.cryptopanic_key:
            count = self._scrape_cryptopanic(symbols)
            total += count
            logger.info(f"CryptoPanic: scraped {count} entries")
        
        # RSS Feeds
        count = self._scrape_rss_feeds(symbols)
        total += count
        logger.info(f"RSS feeds: scraped {count} entries")
        
        # Reddit
        count = self._scrape_reddit(symbols)
        total += count
        logger.info(f"Reddit: scraped {count} entries")
        
        return total

    def _scrape_cryptopanic(self, symbols: Optional[List[str]] = None) -> int:
        """Scrape news from CryptoPanic API."""
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.cryptopanic_key,
            "kind": "news",
            "filter": "important",
            "regions": "en",
        }
        
        if symbols:
            params["currencies"] = ",".join(s.upper() for s in symbols)
        
        try:
            data = self.request_sync("GET", url, params=params)
        except Exception as e:
            logger.error(f"CryptoPanic API error: {e}")
            return 0
        
        entries = data.get("results", [])
        saved = 0
        
        for entry in entries:
            title = entry.get("title", "")
            url = entry.get("url", "")
            published = entry.get("published_at", "")
            
            # Extract currencies
            currencies = [c.get("code", "") for c in entry.get("currencies", [])]
            
            # Score sentiment
            score, label = simple_sentiment_score(title)
            
            # If specific symbols requested, only save matching
            if symbols:
                symbol_set = set(s.upper() for s in symbols)
                if not any(c.upper() in symbol_set for c in currencies):
                    continue
            
            for currency in (currencies or ["UNKNOWN"]):
                self.db.save_sentiment(
                    source="cryptopanic",
                    symbol=currency,
                    title=title,
                    url=url,
                    sentiment_score=score,
                    sentiment_label=label,
                    published_at=datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None,
                )
                saved += 1
        
        return saved

    def _scrape_rss_feeds(self, symbols: Optional[List[str]] = None) -> int:
        """Scrape crypto news RSS feeds."""
        feeds = [
            ("CoinTelegraph", "https://cointelegraph.com/rss"),
            ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("Decrypt", "https://decrypt.co/feed"),
        ]
        
        saved = 0
        
        for source_name, feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
            except Exception as e:
                logger.warning(f"Failed to parse {source_name} RSS: {e}")
                continue
            
            for entry in feed.entries[:20]:  # Limit to recent 20
                title = entry.get("title", "")
                link = entry.get("link", "")
                summary = entry.get("summary", "")
                
                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6])
                    except Exception:
                        pass
                
                # Score sentiment from title + summary
                text = f"{title} {summary}"
                score, label = simple_sentiment_score(text)
                
                # Filter by symbols if specified
                if symbols:
                    text_lower = text.lower()
                    matched_symbols = [
                        s for s in symbols
                        if s.lower() in text_lower
                        or f"${s.lower()}" in text_lower
                    ]
                    if not matched_symbols:
                        continue
                    
                    for sym in matched_symbols:
                        self.db.save_sentiment(
                            source=source_name.lower(),
                            symbol=sym.upper(),
                            title=title,
                            content=BeautifulSoup(summary, "html.parser").get_text()[:500],
                            url=link,
                            sentiment_score=score,
                            sentiment_label=label,
                            published_at=published,
                        )
                        saved += 1
                else:
                    # Save with generic crypto symbol
                    self.db.save_sentiment(
                        source=source_name.lower(),
                        symbol="CRYPTO",
                        title=title,
                        content=BeautifulSoup(summary, "html.parser").get_text()[:500],
                        url=link,
                        sentiment_score=score,
                        sentiment_label=label,
                        published_at=published,
                    )
                    saved += 1
        
        return saved

    def _scrape_reddit(self, symbols: Optional[List[str]] = None) -> int:
        """Scrape Reddit crypto subreddits."""
        subreddits = ["cryptocurrency", "cryptomarkets", "bitcoin", "ethereum"]
        saved = 0
        
        for subreddit in subreddits:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "application/json",
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.warning(f"Failed to fetch r/{subreddit}: {e}")
                continue
            
            posts = data.get("data", {}).get("children", [])
            
            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")[:500]
                permalink = post_data.get("permalink", "")
                created = post_data.get("created_utc", 0)
                upvotes = post_data.get("ups", 0)
                
                text = f"{title} {selftext}"
                score, label = simple_sentiment_score(text)
                
                # Weight score by upvotes (more upvotes = more impactful)
                upvote_weight = min(upvotes / 1000, 2.0)  # Cap at 2x
                weighted_score = score * (1 + upvote_weight * 0.5)
                weighted_score = max(-1.0, min(1.0, weighted_score))
                
                published = datetime.utcfromtimestamp(created) if created else None
                
                if symbols:
                    text_lower = text.lower()
                    matched = [s for s in symbols if s.lower() in text_lower]
                    if not matched:
                        continue
                    
                    for sym in matched:
                        self.db.save_sentiment(
                            source="reddit",
                            symbol=sym.upper(),
                            title=title,
                            content=selftext,
                            url=f"https://reddit.com{permalink}",
                            sentiment_score=weighted_score,
                            sentiment_label=label,
                            published_at=published,
                            metadata={"upvotes": upvotes, "subreddit": subreddit},
                        )
                        saved += 1
                else:
                    self.db.save_sentiment(
                        source="reddit",
                        symbol="CRYPTO",
                        title=title,
                        content=selftext,
                        url=f"https://reddit.com{permalink}",
                        sentiment_score=weighted_score,
                        sentiment_label=label,
                        published_at=published,
                        metadata={"upvotes": upvotes, "subreddit": subreddit},
                    )
                    saved += 1
        
        return saved
