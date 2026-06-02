"""
SQLiteStore — Flexible schema storage for sentiment data, trade logs, and metadata.

Handles reading/writing text-heavy data that doesn't fit the columnar Parquet model.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from monitoring.logger import get_logger

logger = get_logger("storage.sqlite")


class SQLiteStore:
    """Manages SQLite database for sentiment, trade logs, and metadata."""

    def __init__(self, db_path: str = "data/raw/sentiment/bot_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """Initialize database tables if they don't exist."""
        with self._connect() as conn:
            # Sentiment data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    url TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    published_at TIMESTAMP,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            # Trade log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    market TEXT NOT NULL,
                    order_type TEXT,
                    price REAL,
                    quantity REAL,
                    total_value REAL,
                    fee REAL,
                    signal_confidence REAL,
                    signal_direction TEXT,
                    signal_magnitude REAL,
                    order_id TEXT,
                    status TEXT DEFAULT 'pending',
                    opened_at TIMESTAMP,
                    closed_at TIMESTAMP,
                    pnl REAL,
                    close_reason TEXT,
                    metadata TEXT
                )
            """)

            # Signal history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    magnitude REAL,
                    market_type TEXT,
                    executed INTEGER DEFAULT 0,
                    trade_id INTEGER,
                    features_snapshot TEXT,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            """)

            # Scraping metadata (last scrape timestamps, etc.)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scrape_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    symbol TEXT,
                    last_scraped_at TIMESTAMP,
                    rows_added INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'success',
                    error_message TEXT
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_published ON sentiment(published_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")

    # ==================== Sentiment Methods ====================

    def save_sentiment(
        self,
        source: str,
        symbol: str,
        title: str,
        content: str = "",
        url: str = "",
        sentiment_score: float = 0.0,
        sentiment_label: str = "neutral",
        published_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Save a sentiment entry."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sentiment 
                (source, symbol, title, content, url, sentiment_score, sentiment_label, published_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source, symbol, title, content, url,
                    sentiment_score, sentiment_label,
                    published_at.isoformat() if published_at else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def save_sentiment_batch(self, entries: List[Dict[str, Any]]) -> int:
        """Save multiple sentiment entries in a batch."""
        with self._connect() as conn:
            count = 0
            for entry in entries:
                conn.execute(
                    """
                    INSERT INTO sentiment 
                    (source, symbol, title, content, url, sentiment_score, sentiment_label, published_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.get("source", ""),
                        entry.get("symbol", ""),
                        entry.get("title", ""),
                        entry.get("content", ""),
                        entry.get("url", ""),
                        entry.get("sentiment_score", 0.0),
                        entry.get("sentiment_label", "neutral"),
                        entry.get("published_at").isoformat() if entry.get("published_at") else None,
                        json.dumps(entry.get("metadata")) if entry.get("metadata") else None,
                    ),
                )
                count += 1
            logger.debug(f"Saved {count} sentiment entries")
            return count

    def get_sentiment(
        self,
        symbol: str,
        source: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query sentiment data."""
        query = "SELECT * FROM sentiment WHERE symbol = ?"
        params = [symbol]

        if source:
            query += " AND source = ?"
            params.append(source)
        if start:
            query += " AND published_at >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND published_at <= ?"
            params.append(end.isoformat())

        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_avg_sentiment(self, symbol: str, hours: int = 24) -> float:
        """Get average sentiment score for a symbol over the last N hours."""
        cutoff = datetime.utcnow() - pd.Timedelta(hours=hours)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT AVG(sentiment_score) FROM sentiment 
                WHERE symbol = ? AND published_at >= ?
                """,
                (symbol, cutoff.isoformat()),
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else 0.0

    # ==================== Trade Methods ====================

    def save_trade(
        self,
        symbol: str,
        side: str,
        market: str,
        order_type: str = "limit",
        price: float = 0.0,
        quantity: float = 0.0,
        total_value: float = 0.0,
        fee: float = 0.0,
        signal_confidence: float = 0.0,
        signal_direction: str = "",
        signal_magnitude: float = 0.0,
        order_id: str = "",
        status: str = "open",
        opened_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Save a trade entry."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades 
                (symbol, side, market, order_type, price, quantity, total_value, fee,
                 signal_confidence, signal_direction, signal_magnitude, order_id, status,
                 opened_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol, side, market, order_type, price, quantity, total_value, fee,
                    signal_confidence, signal_direction, signal_magnitude, order_id, status,
                    (opened_at or datetime.utcnow()).isoformat(),
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def close_trade(
        self,
        trade_id: int,
        close_price: float,
        pnl: float,
        close_reason: str = "signal",
        fee: float = 0.0,
    ) -> None:
        """Close a trade."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades SET
                    status = 'closed',
                    closed_at = ?,
                    pnl = ?,
                    close_reason = ?,
                    fee = fee + ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), pnl, close_reason, fee, trade_id),
            )

    def get_open_trades(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get all open trades."""
        query = "SELECT * FROM trades WHERE status = 'open'"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY opened_at DESC"

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Get trade history."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND opened_at >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND opened_at <= ?"
            params.append(end.isoformat())

        query += " ORDER BY opened_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ==================== Signal Methods ====================

    def save_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        magnitude: float = 0.0,
        market_type: str = "spot",
        executed: bool = False,
        trade_id: Optional[int] = None,
        features_snapshot: Optional[str] = None,
    ) -> int:
        """Save a signal entry."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals 
                (symbol, direction, confidence, magnitude, market_type, executed, trade_id, features_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol, direction, confidence, magnitude,
                    market_type, int(executed), trade_id, features_snapshot,
                ),
            )
            return cursor.lastrowid

    def get_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get signal history."""
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ==================== Metadata Methods ====================

    def update_scrape_metadata(
        self,
        source: str,
        symbol: str = "",
        rows_added: int = 0,
        status: str = "success",
        error_message: str = "",
    ) -> None:
        """Update scraping metadata."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scrape_metadata (source, symbol, last_scraped_at, rows_added, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, symbol, datetime.utcnow().isoformat(), rows_added, status, error_message),
            )

    def get_last_scrape_time(self, source: str, symbol: str = "") -> Optional[datetime]:
        """Get the last successful scrape time for a source/symbol."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT last_scraped_at FROM scrape_metadata 
                WHERE source = ? AND symbol = ? AND status = 'success'
                ORDER BY last_scraped_at DESC LIMIT 1
                """,
                (source, symbol),
            )
            row = cursor.fetchone()
            if row:
                return datetime.fromisoformat(row[0])
            return None

    # ==================== Utility Methods ====================

    def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        tables = ["sentiment", "trades", "signals", "scrape_metadata"]
        counts = {}
        with self._connect() as conn:
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
        return counts

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        with self._connect() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")
