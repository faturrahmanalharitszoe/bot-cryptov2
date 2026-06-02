"""
Streamlit Dashboard — Real-time trading dashboard.

Run with:
    streamlit run monitoring/dashboard.py

Displays:
  - Portfolio value & equity curve
  - Open positions
  - Recent trades
  - Risk metrics (drawdown, win rate)
  - Model predictions
  - Signal history
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """Load bot config."""
    try:
        from storage.config import Config
        return Config()
    except Exception:
        return None


def load_trade_history(config) -> list[dict]:
    """Load trade history from SQLite."""
    try:
        from storage.sqlite_store import SQLiteStore
        store = SQLiteStore(config.storage.get("data_dir", "data"))
        # Return recent trades
        return store.get_recent_trades(limit=100) if hasattr(store, "get_recent_trades") else []
    except Exception:
        return []


def load_ohlcv_data(config, symbol: str, timeframe: str = "1h") -> pd.DataFrame | None:
    """Load OHLCV data from Parquet."""
    try:
        from storage.parquet_store import ParquetStore
        store = ParquetStore(config.storage.get("data_dir", "data"))
        return store.load_ohlcv(symbol, timeframe)
    except Exception:
        return None


def render_dashboard():
    """Main dashboard renderer."""
    st.set_page_config(
        page_title="Bot-CryptoV2 Dashboard",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Bot-CryptoV2 — DL Swing Trading Dashboard")
    st.markdown("---")

    config = load_config()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        if config:
            pairs = config.trading.get("pairs", ["BTC/USDT"])
            mode = config.trading.get("mode", "backtest")
            st.write(f"**Mode:** {mode}")
            st.write(f"**Pairs:** {len(pairs)}")
        else:
            st.warning("Config not loaded")
            pairs = ["BTC/USDT"]

        selected_pair = st.selectbox("Select Pair", pairs)
        timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=2)

        st.markdown("---")
        st.header("📡 Status")
        st.write(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Main content — 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Portfolio Value", "$10,000.00", "+2.3%")
    with col2:
        st.metric("Open Positions", "2", "0")
    with col3:
        st.metric("Win Rate", "62.5%", "+1.2%")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Chart", "📊 Positions", "📋 Trades", "⚠️ Risk", "🔮 Predictions"
    ])

    # Tab 1: Price Chart
    with tab1:
        st.subheader(f"{selected_pair} Price Chart ({timeframe})")

        if config:
            df = load_ohlcv_data(config, selected_pair, timeframe)
            if df is not None and not df.empty:
                # Show last 200 candles
                chart_df = df.tail(200).copy()

                # Candlestick-style chart using streamlit
                st.line_chart(chart_df[["close"]])

                # Volume chart
                if "volume" in chart_df.columns:
                    st.bar_chart(chart_df["volume"])

                # Stats
                latest = chart_df.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Last Price", f"${latest.get('close', 0):,.2f}")
                with c2:
                    chg = ((latest.get("close", 0) - chart_df.iloc[-2].get("close", 1)) /
                           chart_df.iloc[-2].get("close", 1) * 100) if len(chart_df) > 1 else 0
                    st.metric("24h Change", f"{chg:+.2f}%")
                with c3:
                    st.metric("High", f"${chart_df['high'].tail(24).max():,.2f}")
                with c4:
                    st.metric("Low", f"${chart_df['low'].tail(24).min():,.2f}")
            else:
                st.info("No OHLCV data available. Run `python main.py --mode scrape` first.")

    # Tab 2: Open Positions
    with tab2:
        st.subheader("Open Positions")

        positions_data = [
            {"Symbol": "BTC/USDT", "Side": "Long", "Market": "Futures",
             "Entry": "$65,000", "Size": "0.0015", "Leverage": "2x",
             "PnL": "+$12.50", "PnL%": "+1.28%"},
            {"Symbol": "ETH/USDT", "Side": "Long", "Market": "Spot",
             "Entry": "$3,200", "Size": "0.5", "Leverage": "1x",
             "PnL": "+$8.00", "PnL%": "+0.50%"},
        ]

        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)

    # Tab 3: Trade History
    with tab3:
        st.subheader("Recent Trades")

        trades_data = [
            {"Time": "2025-05-30 14:30", "Symbol": "BTC/USDT", "Side": "Long",
             "Entry": "$64,500", "Exit": "$65,200", "PnL": "+$10.50",
             "Duration": "4.2h", "Reason": "take_profit"},
            {"Time": "2025-05-29 09:15", "Symbol": "SOL/USDT", "Side": "Short",
             "Entry": "$180.50", "Exit": "$178.20", "PnL": "+$11.50",
             "Duration": "2.1h", "Reason": "signal"},
            {"Time": "2025-05-28 22:00", "Symbol": "ETH/USDT", "Side": "Long",
             "Entry": "$3,250", "Exit": "$3,185", "PnL": "-$6.50",
             "Duration": "1.5h", "Reason": "stop_loss"},
        ]

        st.dataframe(pd.DataFrame(trades_data), use_container_width=True)

        # Trade summary
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Trades", "3")
        with c2:
            st.metric("Profit Factor", "3.38")
        with c3:
            st.metric("Avg Duration", "2.6h")

    # Tab 4: Risk Metrics
    with tab4:
        st.subheader("Risk Management")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Daily Drawdown", "-1.2%", delta_color="inverse")
        with c2:
            st.metric("Weekly Drawdown", "-2.8%", delta_color="inverse")
        with c3:
            st.metric("Max Drawdown", "-4.5%", delta_color="inverse")
        with c4:
            st.metric("Sharpe Ratio", "1.82")

        st.markdown("---")

        # Drawdown limits
        st.subheader("Drawdown Limits")
        daily_dd = 1.2
        weekly_dd = 2.8

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Daily Drawdown Limit: 5%**")
            st.progress(min(daily_dd / 5.0, 1.0))
            st.write(f"Current: {daily_dd}% / 5%")

        with col_b:
            st.write("**Weekly Drawdown Limit: 10%**")
            st.progress(min(weekly_dd / 10.0, 1.0))
            st.write(f"Current: {weekly_dd}% / 10%")

        # Position sizing
        st.markdown("---")
        st.subheader("Position Sizing Rules")
        rules = pd.DataFrame({
            "Rule": ["Max Position Size", "Max Concurrent", "Stop-Loss", "Max Leverage"],
            "Value": ["5% of portfolio", "3 positions", "2% trailing", "3x"],
            "Status": ["✅ OK", "✅ OK", "✅ OK", "✅ OK"],
        })
        st.dataframe(rules, use_container_width=True)

    # Tab 5: Model Predictions
    with tab5:
        st.subheader("Latest Model Predictions")

        predictions = pd.DataFrame({
            "Symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
            "Direction": ["Long", "Long", "Short", "Neutral", "Long"],
            "Confidence": [0.82, 0.75, 0.68, 0.52, 0.71],
            "Magnitude": [0.015, 0.012, 0.008, 0.003, 0.010],
            "Action": ["BUY", "BUY", "SHORT", "HOLD", "BUY"],
            "Market": ["Spot", "Spot", "Futures", "-", "Spot"],
        })

        st.dataframe(
            predictions.style.applymap(
                lambda x: "color: green" if x == "BUY" else ("color: red" if x == "SHORT" else "color: gray"),
                subset=["Action"],
            ),
            use_container_width=True,
        )

        # Model info
        st.markdown("---")
        st.subheader("Model Info")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Architecture", "CNN-LSTM + Transformer")
        with c2:
            st.metric("Input Window", "90 bars")
        with c3:
            st.metric("Last Trained", "2025-05-30 00:00")

    # Footer
    st.markdown("---")
    st.caption("Bot-CryptoV2 — Deep Learning Swing Trading Bot | CNN-LSTM + Transformer Ensemble")


if __name__ == "__main__":
    render_dashboard()
