"""
ParquetStore — Efficient columnar storage for time-series data.

Handles reading/writing OHLCV klines and orderbook snapshots
using Apache Parquet format via pyarrow.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from monitoring.logger import get_logger

logger = get_logger("storage.parquet")


class ParquetStore:
    """Manages Parquet file storage for time-series market data."""

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, category: str, symbol: str, timeframe: str = "") -> Path:
        """
        Get the file path for a given category/symbol/timeframe.
        
        Directory structure:
            data/raw/ohlcv/BTC_USDT/5m.parquet
            data/raw/orderbook/BTC_USDT.parquet
        """
        # Sanitize symbol for filesystem (BTC/USDT -> BTC_USDT)
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        
        if timeframe:
            path = self.base_dir / category / safe_symbol / f"{timeframe}.parquet"
        else:
            path = self.base_dir / category / f"{safe_symbol}.parquet"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        append: bool = True,
    ) -> int:
        """
        Save OHLCV data to Parquet.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle interval (e.g., '5m', '1h')
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            append: If True, merge with existing data (dedup by timestamp)
            
        Returns:
            Number of rows saved
        """
        path = self._get_path("ohlcv", symbol, timeframe)
        
        # Ensure timestamp column exists and is datetime
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        if append and path.exists():
            existing = pd.read_parquet(path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"])
            
            # Merge and deduplicate
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            df = combined
        
        # Save
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.debug(f"Saved {len(df)} rows for {symbol} {timeframe} to {path}")
        return len(df)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data from Parquet.
        
        Args:
            symbol: Trading pair
            timeframe: Candle interval
            start: Start datetime filter
            end: End datetime filter
            limit: Max number of rows (most recent)
            
        Returns:
            DataFrame or None if no data exists
        """
        path = self._get_path("ohlcv", symbol, timeframe)
        
        if not path.exists():
            logger.warning(f"No OHLCV data found for {symbol} {timeframe}")
            return None
        
        df = pd.read_parquet(path, engine="pyarrow")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Apply filters
        if start:
            df = df[df["timestamp"] >= pd.to_datetime(start)]
        if end:
            df = df[df["timestamp"] <= pd.to_datetime(end)]
        if limit:
            df = df.tail(limit)
        
        return df.reset_index(drop=True)

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the timestamp of the most recent candle for a symbol/timeframe."""
        path = self._get_path("ohlcv", symbol, timeframe)
        
        if not path.exists():
            return None
        
        # Read only the metadata to get row count, then read last row
        df = pd.read_parquet(path, engine="pyarrow", columns=["timestamp"])
        if df.empty:
            return None
        
        return pd.to_datetime(df["timestamp"]).max()

    def save_orderbook(
        self,
        symbol: str,
        df: pd.DataFrame,
        append: bool = True,
    ) -> int:
        """Save orderbook snapshot data."""
        path = self._get_path("orderbook", symbol)
        
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        if append and path.exists():
            existing = pd.read_parquet(path)
            if "timestamp" in existing.columns:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"])
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)
                df = combined
        
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.debug(f"Saved {len(df)} orderbook rows for {symbol}")
        return len(df)

    def load_orderbook(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Load orderbook data."""
        path = self._get_path("orderbook", symbol)
        
        if not path.exists():
            return None
        
        df = pd.read_parquet(path, engine="pyarrow")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if start:
                df = df[df["timestamp"] >= pd.to_datetime(start)]
            if end:
                df = df[df["timestamp"] <= pd.to_datetime(end)]
        
        if limit:
            df = df.tail(limit)
        
        return df.reset_index(drop=True)

    def save_features(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> int:
        """Save computed feature data to the features directory."""
        features_dir = Path(self.base_dir).parent / "features" / "technical"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        path = features_dir / safe_symbol / f"{timeframe}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.debug(f"Saved {len(df)} feature rows for {symbol} {timeframe}")
        return len(df)

    def load_features(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Load computed feature data."""
        features_dir = Path(self.base_dir).parent / "features" / "technical"
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        path = features_dir / safe_symbol / f"{timeframe}.parquet"
        
        if not path.exists():
            return None
        
        return pd.read_parquet(path, engine="pyarrow")

    def list_available(self, category: str = "ohlcv") -> Dict[str, List[str]]:
        """
        List all available data for a category.
        
        Returns:
            Dict mapping symbol -> list of available timeframes/files
        """
        category_dir = self.base_dir / category
        if not category_dir.exists():
            return {}
        
        available = {}
        for symbol_dir in sorted(category_dir.iterdir()):
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                files = [f.stem for f in symbol_dir.glob("*.parquet")]
                if files:
                    available[symbol] = sorted(files)
        
        return available

    def get_data_stats(self, category: str = "ohlcv") -> pd.DataFrame:
        """Get summary statistics for all stored data."""
        stats = []
        available = self.list_available(category)
        
        for symbol, timeframes in available.items():
            for tf in timeframes:
                if category == "ohlcv":
                    df = self.load_ohlcv(
                        symbol.replace("_", "/"),
                        tf,
                        limit=1,  # Just check existence
                    )
                    if df is not None:
                        path = self._get_path(category, symbol.replace("_", "/"), tf)
                        full_df = pd.read_parquet(path, engine="pyarrow")
                        stats.append({
                            "symbol": symbol,
                            "timeframe": tf,
                            "rows": len(full_df),
                            "start": pd.to_datetime(full_df["timestamp"]).min(),
                            "end": pd.to_datetime(full_df["timestamp"]).max(),
                            "size_mb": path.stat().st_size / (1024 * 1024),
                        })
        
        return pd.DataFrame(stats) if stats else pd.DataFrame()

    def delete_data(self, category: str, symbol: str, timeframe: str = "") -> bool:
        """Delete stored data for a specific symbol/timeframe."""
        path = self._get_path(category, symbol, timeframe)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted {path}")
            return True
        return False
