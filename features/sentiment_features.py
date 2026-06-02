"""
Sentiment Feature Engineer — Converts raw sentiment data into numeric features
suitable for model input.

Aggregates sentiment scores over time windows and produces features like:
- Average sentiment score (1h, 4h, 24h)
- Sentiment momentum (change in sentiment)
- News volume (number of articles)
- Sentiment volatility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from storage.sqlite_store import SQLiteStore
from monitoring.logger import get_logger

logger = get_logger("features.sentiment")


class SentimentFeatures:
    """Converts raw sentiment data into numeric features for model input."""

    def __init__(self, db_store: Optional[SQLiteStore] = None):
        self.db = db_store or SQLiteStore()

    def compute(
        self,
        symbol: str,
        timestamps: pd.DatetimeIndex,
        lookback_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Compute sentiment features aligned to given timestamps.
        
        Args:
            symbol: Trading pair base symbol (e.g., 'BTC')
            timestamps: DatetimeIndex to align features to
            lookback_hours: How far back to look for sentiment data
            
        Returns:
            DataFrame with sentiment feature columns, indexed by timestamp
        """
        # Fetch recent sentiment data
        start = timestamps.min() - timedelta(hours=lookback_hours)
        end = timestamps.max() + timedelta(hours=1)
        
        # Try both the pair symbol and base symbol
        base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
        
        sentiment_df = self.db.get_sentiment(
            symbol=base_symbol,
            start=start,
            end=end,
            limit=10000,
        )
        
        if sentiment_df.empty:
            logger.debug(f"No sentiment data for {symbol}, using defaults")
            return self._default_features(timestamps)
        
        # Ensure datetime
        sentiment_df["published_at"] = pd.to_datetime(sentiment_df["published_at"], errors="coerce")
        sentiment_df = sentiment_df.dropna(subset=["published_at"])
        sentiment_df = sentiment_df.sort_values("published_at")
        
        # Compute features for each timestamp window
        features = []
        for ts in timestamps:
            window_start = ts - timedelta(hours=lookback_hours)
            window_data = sentiment_df[
                (sentiment_df["published_at"] >= window_start) &
                (sentiment_df["published_at"] <= ts)
            ]
            
            if window_data.empty:
                feat = self._empty_feature_row()
            else:
                feat = self._compute_window_features(window_data, ts)
            
            features.append(feat)
        
        result = pd.DataFrame(features, index=timestamps)
        return result

    def _compute_window_features(self, window_data: pd.DataFrame, current_ts: datetime) -> dict:
        """Compute features from sentiment data within a time window."""
        scores = window_data["sentiment_score"].astype(float)
        
        # Basic stats
        avg_score = scores.mean()
        std_score = scores.std() if len(scores) > 1 else 0.0
        max_score = scores.max()
        min_score = scores.min()
        
        # Count by label
        total_count = len(window_data)
        pos_count = (window_data["sentiment_label"] == "positive").sum()
        neg_count = (window_data["sentiment_label"] == "negative").sum()
        neu_count = (window_data["sentiment_label"] == "neutral").sum()
        
        # Ratios
        pos_ratio = pos_count / total_count if total_count > 0 else 0.0
        neg_ratio = neg_count / total_count if total_count > 0 else 0.0
        
        # Sentiment momentum (recent vs older)
        half_point = window_data["published_at"].min() + (
            current_ts - window_data["published_at"].min()
        ) / 2
        
        older = window_data[window_data["published_at"] < half_point]["sentiment_score"]
        recent = window_data[window_data["published_at"] >= half_point]["sentiment_score"]
        
        sentiment_momentum = (recent.mean() - older.mean()) if len(older) > 0 and len(recent) > 0 else 0.0
        
        # Source breakdown
        sources = window_data["source"].value_counts()
        cryptopanic_count = sources.get("cryptopanic", 0)
        reddit_count = sources.get("reddit", 0)
        rss_count = sources.get("cointelegraph", 0) + sources.get("coindesk", 0)
        
        # Weighted score (by recency — more recent = higher weight)
        if len(window_data) > 0:
            time_weights = (window_data["published_at"] - window_data["published_at"].min()).dt.total_seconds()
            time_weights = time_weights / time_weights.max() if time_weights.max() > 0 else pd.Series([1.0] * len(time_weights))
            weighted_score = (scores * time_weights).sum() / time_weights.sum()
        else:
            weighted_score = 0.0
        
        return {
            "sentiment_avg": avg_score,
            "sentiment_std": std_score,
            "sentiment_max": max_score,
            "sentiment_min": min_score,
            "sentiment_pos_ratio": pos_ratio,
            "sentiment_neg_ratio": neg_ratio,
            "sentiment_momentum": sentiment_momentum,
            "sentiment_weighted": weighted_score,
            "sentiment_count": total_count,
            "sentiment_cryptopanic_count": cryptopanic_count,
            "sentiment_reddit_count": reddit_count,
            "sentiment_rss_count": rss_count,
        }

    def _empty_feature_row(self) -> dict:
        """Return a feature row with neutral/default values."""
        return {
            "sentiment_avg": 0.0,
            "sentiment_std": 0.0,
            "sentiment_max": 0.0,
            "sentiment_min": 0.0,
            "sentiment_pos_ratio": 0.0,
            "sentiment_neg_ratio": 0.0,
            "sentiment_momentum": 0.0,
            "sentiment_weighted": 0.0,
            "sentiment_count": 0.0,
            "sentiment_cryptopanic_count": 0.0,
            "sentiment_reddit_count": 0.0,
            "sentiment_rss_count": 0.0,
        }

    def _default_features(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Return default neutral features when no sentiment data exists."""
        default = self._empty_feature_row()
        return pd.DataFrame([default] * len(timestamps), index=timestamps)
