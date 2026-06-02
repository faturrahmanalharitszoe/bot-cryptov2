"""
Base scraper with rate limiting, retry logic, and shared utilities.

All scrapers inherit from this base class to ensure consistent
behavior for API calls, error handling, and backoff strategies.
"""

import time
import asyncio
import aiohttp
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from monitoring.logger import get_logger

logger = get_logger("scraper.base")


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_calls: int = 100, period: float = 60.0):
        """
        Args:
            max_calls: Maximum number of calls in the period
            period: Time window in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a rate limit slot is available."""
        async with self._lock:
            now = time.monotonic()
            # Remove expired timestamps
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest = self.calls[0]
                wait_time = self.period - (now - oldest)
                if wait_time > 0:
                    logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            self.calls.append(time.monotonic())

    def acquire_sync(self):
        """Synchronous version of acquire."""
        now = time.monotonic()
        self.calls = [t for t in self.calls if now - t < self.period]
        
        if len(self.calls) >= self.max_calls:
            oldest = self.calls[0]
            wait_time = self.period - (now - oldest)
            if wait_time > 0:
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
        
        self.calls.append(time.monotonic())


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.
    
    Provides:
    - Rate limiting
    - Retry with exponential backoff
    - HTTP session management
    - Logging
    """

    def __init__(
        self,
        name: str = "base",
        max_calls_per_minute: int = 1200,
        max_retries: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
    ):
        self.name = name
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.rate_limiter = RateLimiter(max_calls=max_calls_per_minute, period=60.0)
        self._session: Optional[requests.Session] = None
        self._aio_session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Initialized {self.name} scraper")

    @property
    def session(self) -> requests.Session:
        """Lazy-init synchronous HTTP session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Bot-CryptoV2/2.0",
                "Accept": "application/json",
            })
        return self._session

    async def get_aio_session(self) -> aiohttp.ClientSession:
        """Lazy-init async HTTP session."""
        if self._aio_session is None or self._aio_session.closed:
            self._aio_session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Bot-CryptoV2/2.0",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._aio_session

    def request_sync(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a synchronous HTTP request with rate limiting and retry.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response JSON as dict
            
        Raises:
            requests.RequestException: After all retries exhausted
        """
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire_sync()
            try:
                response = self.session.request(
                    method, url, params=params, headers=headers, timeout=30, **kwargs
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 0
                if status == 429:  # Rate limited
                    backoff = min(
                        self.base_backoff * (2 ** attempt),
                        self.max_backoff,
                    )
                    logger.warning(f"[{self.name}] Rate limited (429), backing off {backoff:.1f}s")
                    time.sleep(backoff)
                elif status >= 500:  # Server error
                    backoff = min(
                        self.base_backoff * (2 ** attempt),
                        self.max_backoff,
                    )
                    logger.warning(f"[{self.name}] Server error {status}, retry {attempt+1}")
                    time.sleep(backoff)
                else:
                    logger.error(f"[{self.name}] HTTP error {status}: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                    logger.warning(f"[{self.name}] Request failed, retry {attempt+1}: {e}")
                    time.sleep(backoff)
                else:
                    logger.error(f"[{self.name}] All retries exhausted: {e}")
                    raise

        raise RuntimeError(f"[{self.name}] Failed after {self.max_retries} retries")

    async def request_async(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request with rate limiting and retry.
        
        Returns:
            Response JSON as dict
        """
        session = await self.get_aio_session()

        for attempt in range(self.max_retries + 1):
            await self.rate_limiter.acquire()
            try:
                async with session.request(method, url, params=params, headers=headers) as response:
                    if response.status == 429:
                        backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                        logger.warning(f"[{self.name}] Rate limited (429), backing off {backoff:.1f}s")
                        await asyncio.sleep(backoff)
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                if e.status >= 500 and attempt < self.max_retries:
                    backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                    logger.warning(f"[{self.name}] Server error {e.status}, retry {attempt+1}")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"[{self.name}] HTTP error: {e}")
                    raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    backoff = min(self.base_backoff * (2 ** attempt), self.max_backoff)
                    logger.warning(f"[{self.name}] Request failed, retry {attempt+1}: {e}")
                    await asyncio.sleep(backoff)
                else:
                    raise

        raise RuntimeError(f"[{self.name}] Failed after {self.max_retries} retries")

    @abstractmethod
    def scrape(self, **kwargs) -> Any:
        """Main scraping method to be implemented by subclasses."""
        ...

    def close(self):
        """Close HTTP sessions."""
        if self._session:
            self._session.close()
            self._session = None
        logger.info(f"Closed {self.name} scraper")

    async def close_async(self):
        """Close async HTTP session."""
        if self._aio_session and not self._aio_session.closed:
            await self._aio_session.close()
        logger.info(f"Closed {self.name} async scraper")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
