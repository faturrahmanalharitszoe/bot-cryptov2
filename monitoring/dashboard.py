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


def load_trade_history(config) -> pd.DataFrame:
    """Load trade history from SQLite."""
    try:
        from storage.sqlite_store import SQLiteStore
        store = SQLiteStore()
        return store.get_trade_history(limit=100)
    except Exception:
        return pd.DataFrame()

def get_model_last_trained(config) -> str:
    """Get the last modification time of the best model."""
    try:
        model_path = config.get_model_path("ensemble_best.pt")
        if model_path.exists():
            mtime = model_path.stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    return "N/A"

def load_open_trades(config) -> pd.DataFrame:
    """Load open trades from SQLite."""
    try:
        from storage.sqlite_store import SQLiteStore
        store = SQLiteStore()
        return store.get_open_trades()
    except Exception:
        return pd.DataFrame()

def load_signals(config) -> pd.DataFrame:
    """Load signals from SQLite."""
    try:
        from storage.sqlite_store import SQLiteStore
        store = SQLiteStore()
        return store.get_signals(limit=20)
    except Exception:
        return pd.DataFrame()

def load_state(config) -> dict:
    """Load orchestrator state."""
    try:
        state_file = Path(config.storage.get("data_dir", "data")) / "state.json"
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


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
        page_title="Bloomberg Crypto Terminal",
        page_icon="📟",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # -------------------------------------------------------------------
    # Custom Bloomberg-style CSS
    # -------------------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Base background and text */
        .stApp {
            background-color: #050505 !important;
            color: #ffb000 !important;
            font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #00e5ff !important;
            font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace !important;
            text-transform: uppercase;
            border-bottom: 1px solid #333;
            padding-bottom: 5px;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            color: #00e5ff !important;
            font-size: 1.8rem !important;
            font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        }
        [data-testid="stMetricLabel"] {
            color: #ffb000 !important;
            text-transform: uppercase;
            font-size: 0.9rem !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        }
        
        /* Containers / Borders */
        [data-testid="stVerticalBlock"] > div > div {
            border: 1px solid #1a1a1a;
            padding: 5px;
            background-color: #0a0a0a;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            font-family: 'JetBrains Mono', 'Consolas', monospace !important;
            font-size: 0.85rem !important;
        }
        
        /* Hide sidebar completely if possible, or style it */
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #333 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    config = load_config()
    state = load_state(config) if config else {}
    risk_stats = state.get("risk", {})
    tracker_stats = state.get("tracker", {})

    st.markdown("### 📟 BOT-CRYPTOV2 TERMINAL (DL SWING TRADING)")

    # -------------------------------------------------------------------
    # Ticker Tape Row (Top Metrics)
    # -------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("PORTFOLIO", f"${risk_stats.get('portfolio_value', 10000.0):,.2f}")
    with m2:
        st.metric("OPEN RISK", f"${risk_stats.get('open_risk', 0.0):.2f}")
    with m3:
        dd = risk_stats.get("total_drawdown", 0.0) * 100
        st.metric("DRAWDOWN", f"{dd:.2f}%", delta=f"{dd:.2f}%", delta_color="inverse")
    with m4:
        st.metric("OPEN POS", str(risk_stats.get('open_positions', 0)))
    with m5:
        st.metric("WIN RATE", f"{tracker_stats.get('win_rate', 0.0) * 100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # Main Layout: 70% Chart / 30% Panel
    # -------------------------------------------------------------------
    col_chart, col_panel = st.columns([7, 3])

    with col_chart:
        st.markdown("#### PRICE ACTION")
        pairs = config.trading.get("pairs", ["BTC/USDT"]) if config else ["BTC/USDT"]
        
        # Mini selector
        c1, c2 = st.columns([1, 1])
        with c1:
            selected_pair = st.selectbox("SYMBOL", pairs, label_visibility="collapsed")
        with c2:
            timeframe = st.selectbox("TIMEFRAME", ["5m", "15m", "1h", "4h"], index=2, label_visibility="collapsed")

        if config:
            df = load_ohlcv_data(config, selected_pair, timeframe)
            if df is not None and not df.empty:
                chart_df = df.tail(150).copy()
                st.line_chart(chart_df[["close"]], height=400, use_container_width=True)
                
                # Volume
                if "volume" in chart_df.columns:
                    st.bar_chart(chart_df["volume"], height=100)
            else:
                st.info(f"NO DATA FOR {selected_pair}")

    with col_panel:
        st.markdown("#### LATEST SIGNALS")
        signals_df = load_signals(config) if config else pd.DataFrame()
        if not signals_df.empty:
            # Format dataframe to look dense
            display_sig = signals_df[["symbol", "direction", "confidence"]].copy()
            st.dataframe(
                display_sig.style.map(
                    lambda x: "color: #00ff00" if x == "BUY" else ("color: #ff0044" if x == "SHORT" else "color: #555"),
                    subset=["direction"],
                ),
                use_container_width=True,
                height=250,
                hide_index=True
            )
        else:
            st.caption("WAITING FOR SIGNALS...")

        st.markdown("#### OPEN POSITIONS")
        open_df = load_open_trades(config) if config else pd.DataFrame()
        if not open_df.empty:
            display_open = open_df[["symbol", "side", "pnl"]].copy()
            st.dataframe(
                display_open.style.map(
                    lambda x: "color: #00ff00" if float(x) > 0 else ("color: #ff0044" if float(x) < 0 else "color: #555"),
                    subset=["pnl"],
                ),
                use_container_width=True,
                height=200,
                hide_index=True
            )
        else:
            st.caption("NO OPEN POSITIONS.")

    # -------------------------------------------------------------------
    # Trade History (Bottom)
    # -------------------------------------------------------------------
    st.markdown("#### TRADE LOG")
    trades_df = load_trade_history(config) if config else pd.DataFrame()
    if not trades_df.empty:
        st.dataframe(
            trades_df[["symbol", "side", "market", "price", "pnl", "close_reason", "opened_at", "closed_at"]], 
            use_container_width=True,
            height=250,
            hide_index=True
        )
    else:
        st.caption("NO RECENT TRADES.")

    st.caption(f"LAST TRAINED: {get_model_last_trained(config) if config else 'N/A'}")

if __name__ == "__main__":
    render_dashboard()
