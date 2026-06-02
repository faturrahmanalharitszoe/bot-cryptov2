# Bot-CryptoV2 — Deep Learning Swing Trading Bot

A modular, production-grade crypto swing trading bot powered by a **CNN-LSTM + Transformer ensemble** deep learning model. It scrapes multi-source data, generates trading signals, and executes trades across both **spot and futures** markets on Binance.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
  - [Environment Variables](#environment-variables)
  - [config.yaml Reference](#configyaml-reference)
- [Usage](#usage)
  - [Backtest Mode](#backtest-mode)
  - [Train Mode](#train-mode)
  - [Scrape Mode](#scrape-mode)
  - [Testnet Mode](#testnet-mode)
  - [Live Mode](#live-mode)
- [Module Documentation](#module-documentation)
  - [Storage Layer](#storage-layer)
  - [Scrapers](#scrapers)
  - [Feature Engineering](#feature-engineering)
  - [Deep Learning Models](#deep-learning-models)
  - [Signal Generation](#signal-generation)
  - [Execution & Risk Management](#execution--risk-management)
  - [Backtesting](#backtesting)
  - [Orchestrator](#orchestrator)
  - [Monitoring](#monitoring)
- [Testing](#testing)
- [Workflow](#workflow)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

| Category | Description |
|----------|-------------|
| **Multi-Source Scraping** | OHLCV klines (5m/15m/1h/4h), order book depth, news sentiment, on-chain whale data |
| **Deep Learning Ensemble** | CNN-LSTM (local patterns + temporal deps) + Transformer (complex relationships) → multi-task head |
| **Spot & Futures Switching** | Spot long for moderate confidence; Futures long for high confidence (>0.85); Futures short always |
| **Risk Management** | Position sizing (5% max), trailing stop-loss (2%), scaled take-profit (3 levels), drawdown halts |
| **Backtesting** | Event-driven simulation with slippage, commissions, walk-forward validation, Monte Carlo simulation |
| **Monitoring** | Real-time Streamlit dashboard + Telegram notifications for trades and alerts |
| **Extensible** | Modular architecture — swap models, add scrapers, customize filters independently |

---

## Architecture Overview

The bot follows an **event-driven pipeline**:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Scrapers    │───▶│  Feature     │───▶│  ML Models   │───▶│  Signal  │
│  (OHLCV,     │    │  Engineering │    │  (CNN-LSTM + │    │  Gen +   │
│  Orderbook,  │    │  Pipeline    │    │  Transformer)│    │  Filters │
│  Sentiment)  │    │              │    │              │    │          │
└─────────────┘    └──────────────┘    └──────────────┘    └────┬─────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────────────────┐
│  Monitoring  │◀──│  Orchestrator│◀──▶│  Risk Manager → Execution    │
│  (Dashboard, │    │  (APScheduler│    │  (Position Tracker → ccxt)   │
│  Telegram)   │    │  event loop) │    │                              │
└─────────────┘    └──────────────┘    └──────────────────────────────┘
```

**Flow**: Scrapers collect data → Feature pipeline computes indicators → Model predicts direction/magnitude/confidence → Signal generator creates trade signals with filters → Risk manager approves/sizes positions → Execution layer places orders via ccxt → Orchestrator schedules everything → Monitoring displays dashboard and sends alerts.

For detailed architecture diagrams and module descriptions, see [`plans/architecture.md`](plans/architecture.md).

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Exchange API | [ccxt](https://github.com/ccxt/ccxt) | Unified Binance spot + futures interface |
| Deep Learning | [PyTorch](https://pytorch.org/) | CNN-LSTM + Transformer ensemble model |
| Technical Analysis | [pandas_ta](https://github.com/twopirllc/pandas_ta) | 20+ indicators (RSI, MACD, ATR, Bollinger, etc.) |
| Data Storage | [PyArrow](https://arrow.apache.org/) + [SQLite](https://docs.python.org/3/library/sqlite3.html) | Parquet for time-series, SQLite for sentiment/trades |
| Scheduling | [APScheduler](https://apscheduler.readthedocs.io/) | Orchestrating live scraping and signal generation |
| Dashboard | [Streamlit](https://streamlit.io/) | Real-time monitoring dashboard |
| Notifications | [python-telegram-bot](https://python-telegram-bot.org/) | Trade alerts and daily summaries |
| Configuration | [PyYAML](https://pyyaml.org/) + [python-dotenv](https://github.com/theskumar/python-dotenv) | YAML config + .env secrets |
| Logging | [Rich](https://github.com/Textualize/rich) | Beautiful terminal logging with rotation |
| Testing | [pytest](https://docs.pytest.org/) | Unit and integration tests |

---

## Project Structure

```
bot-cryptov2/
├── main.py                    # CLI entry point (backtest/testnet/live/train/scrape)
├── orchestrator.py            # LiveOrchestrator — APScheduler event loop
├── config.yaml                # All trading, model, risk, and scraping parameters
├── .env.example               # Template for API keys and secrets
├── requirements.txt           # Python dependencies
│
├── storage/                   # Data access layer
│   ├── config.py              # Config singleton (loads config.yaml + .env)
│   ├── parquet_store.py       # OHLCV, orderbook, features in Parquet files
│   └── sqlite_store.py        # Sentiment, trade logs, signals in SQLite
│
├── scrapers/                  # Data collection modules
│   ├── base_scraper.py        # Abstract base with rate limiting
│   ├── ohlcv_scraper.py       # Binance kline scraper (5m/15m/1h/4h)
│   ├── orderbook_scraper.py   # Order book depth scraper
│   ├── sentiment_scraper.py   # News/reddit sentiment scraper
│   └── onchain_scraper.py     # Whale alert / on-chain data scraper
│
├── features/                  # Feature engineering
│   ├── technical.py           # Technical indicators (RSI, MACD, ATR, BB, etc.)
│   ├── sentiment_features.py  # Sentiment score aggregation
│   ├── orderbook_features.py  # Order book imbalance, spread features
│   ├── multi_timeframe.py     # Multi-TF merge with forward-fill alignment
│   └── pipeline.py            # FeaturePipeline — orchestrates all feature computation
│
├── models/                    # Deep learning models (PyTorch)
│   ├── cnn_lstm.py            # CNN branch + BiLSTM+Attention branch
│   ├── transformer_branch.py  # Transformer encoder branch
│   ├── ensemble.py            # EnsembleModel — combines all branches + multi-task head
│   ├── trainer.py             # Training loop with early stopping, LR scheduling
│   └── predictor.py           # Predictor — high-level inference wrapper
│
├── signals/                   # Signal generation & filtering
│   ├── filters.py             # SignalFilter — cooldown, trend, volatility, volume checks
│   └── generator.py           # SignalGenerator — prediction → TradeSignal pipeline
│
├── execution/                 # Trade execution & risk management
│   ├── exchange.py            # ExchangeClient — ccxt wrapper for Binance
│   ├── position_tracker.py    # PositionTracker — open/close positions, PnL tracking
│   └── risk_manager.py        # RiskManager — sizing, stop-loss, drawdown, TP management
│
├── backtest/                  # Backtesting engine
│   ├── engine.py              # BacktestEngine — event-driven bar-by-bar simulation
│   ├── analytics.py           # Analytics — Sharpe, Sortino, drawdown, monthly returns
│   ├── walk_forward.py        # WalkForwardValidator — rolling train/test windows
│   └── monte_carlo.py         # MonteCarloSimulator — trade resampling for confidence intervals
│
├── monitoring/                # Observability
│   ├── logger.py              # setup_logger() with Rich console + rotating file handler
│   ├── dashboard.py           # Streamlit dashboard (positions, trades, equity, predictions)
│   └── notifier.py            # TelegramNotifier — trade alerts, daily summaries
│
├── tests/                     # Test suite
│   ├── conftest.py            # Shared fixtures (sample DataFrames, predictions)
│   ├── test_position_tracker.py
│   ├── test_risk_manager.py
│   ├── test_signal_filters.py
│   ├── test_signal_generator.py
│   ├── test_backtest_analytics.py
│   └── test_integration.py    # Full pipeline integration tests
│
├── data/                      # Data directories (git-ignored contents)
│   ├── raw/                   # Raw OHLCV, orderbook Parquet files
│   └── features/              # Computed feature Parquet files
│
├── models/saved/              # Saved model checkpoints (.pt files)
├── logs/                      # Rotating log files
└── plans/                     # Architecture documentation
    └── architecture.md
```

---

## Quick Start

### Prerequisites

- **Python 3.11+** (recommended 3.12)
- **Git** (for cloning)
- **CUDA GPU** (optional, for faster training — CPU works but is slower)

### 1. Clone & Setup

```bash
cd bot-cryptov2
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
```

Edit `.env` with your API keys (see [Environment Variables](#environment-variables)).

### 3. Configure Trading Parameters

Edit `config.yaml` to set your trading pairs, risk parameters, and model settings (see [config.yaml Reference](#configyaml-reference)).

### 4. Scrape Historical Data

```bash
python main.py --mode scrape
```

This fetches 90 days of OHLCV data for all configured pairs across all timeframes.

### 5. Train the Model

```bash
python main.py --mode train
```

### 6. Run Backtest

```bash
python main.py --mode backtest
```

### 7. Run on Testnet (Optional)

```bash
python main.py --mode testnet
```

---

## Configuration Guide

### Environment Variables

All secrets are stored in `.env` (never committed to git). See [`.env.example`](.env.example) for the template.

| Variable | Required | Description |
|----------|----------|-------------|
| `BINANCE_TESTNET_API_KEY` | Yes (for testnet/live) | Binance testnet API key |
| `BINANCE_TESTNET_API_SECRET` | Yes (for testnet/live) | Binance testnet API secret |
| `BINANCE_LIVE_API_KEY` | Yes (for live only) | Binance production API key |
| `BINANCE_LIVE_API_SECRET` | Yes (for live only) | Binance production API secret |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat ID for notifications |
| `CRYPTOPANIC_API_KEY` | Optional | CryptoPanic API key for news sentiment |
| `CUDA_VISIBLE_DEVICES` | Optional | GPU device ID (default: `0`) |

**Getting Binance Testnet Keys:**
1. Go to [testnet.binancefuture.com](https://testnet.binancefuture.com)
2. Log in with GitHub
3. Generate API key and secret
4. Copy them to your `.env` file

### config.yaml Reference

#### `trading` — Trading Parameters

```yaml
trading:
  mode: backtest          # backtest | testnet | live
  pairs:                  # Trading pairs to monitor
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
  timeframes:             # Candle timeframes to scrape
    - 5m
    - 15m
    - 1h
    - 4h
  default_market: spot    # spot | futures — default market
  max_leverage: 3         # Maximum leverage for futures (1-3x)
```

#### `model` — Deep Learning Model

```yaml
model:
  input_window: 90              # Number of timesteps to look back
  retrain_interval_hours: 24    # Retrain model every N hours
  ensemble_weights:             # Weight of each branch in ensemble
    cnn_lstm: 0.35
    transformer: 0.35
    mlp_head: 0.30
  cnn:                          # CNN branch configuration
    filters: [64, 128]
    kernel_sizes: [3, 5]
    dropout: 0.3
  lstm:                         # LSTM branch configuration
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    bidirectional: true
    attention: true
  transformer:                  # Transformer branch configuration
    d_model: 128
    nhead: 8
    num_layers: 4
    dim_feedforward: 256
    dropout: 0.1
  training:                     # Training hyperparameters
    batch_size: 64
    epochs: 100
    learning_rate: 0.001
    weight_decay: 0.0001
    scheduler: cosine           # cosine | step | plateau
    early_stopping_patience: 15
    validation_split: 0.15
    test_split: 0.1
```

#### `signal` — Signal Generation

```yaml
signal:
  confidence_threshold: 0.70    # Minimum model confidence to generate signal
  magnitude_threshold: 0.005    # 0.5% minimum expected price move
  cooldown_minutes: 30          # Minimum time between signals per pair
```

#### `risk` — Risk Management

```yaml
risk:
  max_position_pct: 0.05        # Max 5% of portfolio per trade
  max_concurrent_positions: 3   # Maximum simultaneous open positions
  stop_loss_pct: 0.02           # 2% trailing stop-loss
  take_profit_levels: [0.03, 0.05, 0.08]   # TP levels (3%, 5%, 8%)
  take_profit_weights: [0.4, 0.35, 0.25]   # % of position to close at each TP
  max_daily_drawdown_pct: 0.05  # 5% daily drawdown halts trading
  max_weekly_drawdown_pct: 0.10 # 10% weekly drawdown halts trading
```

#### `exchange` — Exchange Configuration

```yaml
exchange:
  testnet: true                 # Use Binance testnet
  rate_limit_per_minute: 1200   # API rate limit
  order_type: limit             # limit | market
  slippage_pct: 0.001           # Expected slippage (0.1%)
```

#### `scraping` — Data Collection

```yaml
scraping:
  ohlcv:
    history_days: 90            # Initial history to fetch
    update_interval_sec: 300    # Scrape every 5 minutes
  orderbook:
    depth: 20                   # Order book levels
    update_interval_sec: 60     # Scrape every minute
  sentiment:
    sources: [cryptopanic, reddit]
    update_interval_sec: 900    # Scrape every 15 minutes
  onchain:
    sources: [whale_alert]
    update_interval_sec: 1800   # Scrape every 30 minutes
```

#### `backtest` — Backtesting Configuration

```yaml
backtest:
  start_date: "2024-01-01"      # Backtest start date
  end_date: "2025-05-31"        # Backtest end date
  initial_capital: 10000.0      # Starting capital in USDT
  commission_spot: 0.001        # Spot commission (0.1%)
  commission_futures: 0.0002    # Futures commission (0.02%)
  slippage: 0.0005              # Simulated slippage (0.05%)
  walk_forward:                 # Walk-forward validation
    train_months: 6
    test_months: 1
    step_months: 1
```

#### `monitoring` — Dashboard & Notifications

```yaml
monitoring:
  dashboard:
    enabled: true
    port: 8501
  telegram:
    enabled: false
    bot_token: ""
    chat_id: ""
  logging:
    level: INFO                 # DEBUG | INFO | WARNING | ERROR
    file: logs/bot.log
    max_bytes: 10485760         # 10MB per log file
    backup_count: 5             # Keep 5 rotated log files
```

---

## Usage

### CLI Commands

```bash
python main.py --mode <MODE> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--mode` | Operating mode: `backtest`, `testnet`, `live`, `train`, `scrape` |
| `--config` | Path to custom config.yaml (default: `./config.yaml`) |
| `--pairs` | Override trading pairs (e.g., `BTC/USDT ETH/USDT`) |
| `--symbols` | Override symbols for training/scraping |
| `--log-level` | Override log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Backtest Mode

Runs an event-driven simulation over historical data:

```bash
python main.py --mode backtest
```

**What it does:**
1. Loads historical OHLCV data from Parquet files
2. Computes features for each bar
3. Generates predictions using the trained model
4. Simulates signal → risk → execution pipeline
5. Outputs performance metrics (Sharpe, Sortino, drawdown, win rate, etc.)

**Output includes:**
- Total return, annualized return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown, daily/weekly drawdown
- Win rate, profit factor, average win/loss
- Monthly returns breakdown
- Per-symbol performance stats

### Train Mode

Trains or retrains the deep learning model:

```bash
python main.py --mode train
```

**What it does:**
1. Loads and prepares features from historical data
2. Creates sliding windows for time-series input
3. Trains the CNN-LSTM + Transformer ensemble
4. Uses cosine annealing LR scheduler with early stopping
5. Saves the best model checkpoint to `models/saved/`

**Training tips:**
- Ensure you have at least 90 days of scraped data
- Training on GPU is ~10x faster than CPU
- Adjust `model.training.epochs` and `early_stopping_patience` in config

### Scrape Mode

Fetches historical and real-time data:

```bash
python main.py --mode scrape
```

**What it does:**
1. Scrapes OHLCV klines for all pairs and timeframes
2. Fetches order book snapshots
3. Collects news sentiment from CryptoPanic/Reddit
4. Retrieves on-chain whale alert data
5. Stores everything in Parquet/SQLite

### Testnet Mode

Runs live trading on Binance testnet:

```bash
python main.py --mode testnet
```

**What it does:**
1. Connects to `testnet.binancefuture.com` via ccxt
2. Starts the LiveOrchestrator with APScheduler
3. Periodically scrapes data, computes features, generates signals
4. Executes trades on testnet with simulated money
5. Sends Telegram notifications (if configured)
6. Runs the Streamlit dashboard on port 8501

### Live Mode

⚠️ **Use with extreme caution — trades real money!**

```bash
python main.py --mode live
```

Same as testnet mode but connects to production Binance API. Requires `BINANCE_LIVE_API_KEY` and `BINANCE_LIVE_API_SECRET` in `.env`.

**Safety checklist before going live:**
- [ ] Backtested extensively with positive results
- [ ] Walk-forward validation shows consistent performance
- [ ] Monte Carlo simulation shows acceptable risk
- [ ] Risk parameters are conservative
- [ ] Tested on testnet for at least 1 week
- [ ] Telegram notifications are working
- [ ] You understand you can lose money

---

## Module Documentation

### Storage Layer

**[`storage/config.py`](storage/config.py)** — Singleton configuration manager that loads `config.yaml` and `.env`. Access via `get_config()`.

```python
from storage.config import get_config
config = get_config()
api_key, api_secret = config.get_api_keys("testnet")
```

**[`storage/parquet_store.py`](storage/parquet_store.py)** — Manages Parquet file storage for OHLCV, orderbook, and feature data.

```python
from storage.parquet_store import ParquetStore
store = ParquetStore(base_dir="data")
store.save_ohlcv(df, "BTC/USDT", "1h")
df = store.load_ohlcv("BTC/USDT", "1h", start=start_dt, end=end_dt)
```

**[`storage/sqlite_store.py`](storage/sqlite_store.py)** — Manages SQLite database for sentiment, trade logs, and signals.

```python
from storage.sqlite_store import SQLiteStore
store = SQLiteStore(db_path="data/raw/sentiment/bot_data.db")
store.save_sentiment("BTC/USDT", "positive", 0.85, "cryptopanic", "Title", "url")
store.save_trade("BTC/USDT", "spot", "buy", 50000.0, 0.001)
```

### Scrapers

All scrapers inherit from [`scrapers/base_scraper.py`](scrapers/base_scraper.py) which provides rate limiting and retry logic.

| Scraper | File | Data |
|---------|------|------|
| OHLCV | [`scrapers/ohlcv_scraper.py`](scrapers/ohlcv_scraper.py) | Binance klines (5m/15m/1h/4h) |
| Order Book | [`scrapers/orderbook_scraper.py`](scrapers/orderbook_scraper.py) | Bid/ask depth snapshots |
| Sentiment | [`scrapers/sentiment_scraper.py`](scrapers/sentiment_scraper.py) | CryptoPanic + Reddit news |
| On-Chain | [`scrapers/onchain_scraper.py`](scrapers/onchain_scraper.py) | Whale alerts and large transfers |

### Feature Engineering

**[`features/pipeline.py`](features/pipeline.py)** — [`FeaturePipeline`](features/pipeline.py:29) orchestrates all feature computation:

1. **Technical indicators** ([`features/technical.py`](features/technical.py)): RSI, MACD, ATR, Bollinger Bands, EMA/SMA, Stochastic, OBV, MFI, ADX, CCI, Williams %R, Ichimoku
2. **Sentiment features** ([`features/sentiment_features.py`](features/sentiment_features.py)): Aggregated sentiment scores, news volume
3. **Order book features** ([`features/orderbook_features.py`](features/orderbook_features.py)): Bid/ask imbalance, spread, depth pressure
4. **Multi-timeframe merge** ([`features/multi_timeframe.py`](features/multi_timeframe.py)): Merges 5m/15m/1h/4h features with forward-fill alignment

### Deep Learning Models

The ensemble model ([`models/ensemble.py`](models/ensemble.py)) combines three branches:

```
Input Features (90 timesteps × N features)
        │
        ├─── CNN Branch ──────────────▶ [Local patterns]
        │    (Conv1D layers)                │
        │                                  │
        ├─── BiLSTM + Attention ──────▶ [Temporal dependencies]
        │    (Bidirectional LSTM)           │
        │                                  ├─── Concatenate ──▶ Multi-Task Head
        └─── Transformer ─────────────▶ [Complex relationships]     │
             (Transformer Encoder)            │                     ├── Direction (3-class softmax)
                                              │                     ├── Magnitude (linear)
                                              │                     └── Confidence (sigmoid)
```

**Multi-task output:**
- **Direction**: Long / Short / Neutral (3-class softmax)
- **Magnitude**: Expected % price move (linear regression)
- **Confidence**: Prediction confidence 0-1 (sigmoid)

**Key files:**
- [`models/cnn_lstm.py`](models/cnn_lstm.py) — CNN and BiLSTM+Attention branches
- [`models/transformer_branch.py`](models/transformer_branch.py) — Transformer encoder branch
- [`models/ensemble.py`](models/ensemble.py) — Combines branches with multi-task head
- [`models/trainer.py`](models/trainer.py) — Training loop with early stopping
- [`models/predictor.py`](models/predictor.py) — [`Prediction`](models/predictor.py:29) dataclass and [`Predictor`](models/predictor.py:78) inference wrapper

### Signal Generation

**[`signals/generator.py`](signals/generator.py)** — [`SignalGenerator`](signals/generator.py:123) converts model predictions into [`TradeSignal`](signals/generator.py:49) objects:

| Model Output | Signal Decision |
|-------------|-----------------|
| Long + High confidence (>0.85) | LONG via **futures** with leverage |
| Long + Moderate confidence | LONG via **spot** |
| Short | SHORT via **futures** (always) |
| Neutral or Low confidence | HOLD (no trade) |

**[`signals/filters.py`](signals/filters.py)** — [`SignalFilter`](signals/filters.py:56) applies pre-trade filters:

- **Cooldown**: Minimum time between signals per pair
- **Trend alignment**: Signal must align with EMA trend
- **Volatility**: Reject signals during extreme volatility
- **Volume**: Reject signals with insufficient volume
- **Confidence/Magnitude**: Minimum thresholds

### Execution & Risk Management

**[`execution/position_tracker.py`](execution/position_tracker.py)** — [`PositionTracker`](execution/position_tracker.py:167) manages all open positions and trade history. Tracks stop-loss and take-profit levels with partial close support.

**[`execution/risk_manager.py`](execution/risk_manager.py)** — [`RiskManager`](execution/risk_manager.py:54) is the central authority:

- **Position sizing**: Max 5% of portfolio per trade
- **Concurrent positions**: Max 3 simultaneous
- **Stop-loss**: 2% trailing stop-loss
- **Take-profit**: Scaled TP at 3%, 5%, 8% (closes 40%, 35%, 25%)
- **Drawdown limits**: 5% daily / 10% weekly halts all trading
- **Commission tracking**: Spot (0.1%) and futures (0.02%)

**[`execution/exchange.py`](execution/exchange.py)** — [`ExchangeClient`](execution/exchange.py:63) wraps ccxt for Binance spot and futures with order management.

### Backtesting

**[`backtest/engine.py`](backtest/engine.py)** — [`BacktestEngine`](backtest/engine.py:99) simulates the full pipeline bar-by-bar with:
- Slippage simulation (0.05% default)
- Commission simulation (spot/futures rates)
- Signal → Risk → Execution pipeline replay
- Equity curve tracking with [`EquityPoint`](backtest/engine.py:62)

**[`backtest/analytics.py`](backtest/analytics.py)** — [`Analytics`](backtest/analytics.py:135) computes comprehensive [`PerformanceMetrics`](backtest/analytics.py:32):
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown, daily/weekly drawdown
- Win rate, profit factor, average win/loss
- Monthly returns, per-symbol stats
- Total commission and slippage costs

**[`backtest/walk_forward.py`](backtest/walk_forward.py)** — Rolling train/test window validation (6-month train, 1-month test).

**[`backtest/monte_carlo.py`](backtest/monte_carlo.py)** — Trade resampling for confidence intervals on equity, drawdown, and Sharpe.

### Orchestrator

**[`orchestrator.py`](orchestrator.py)** — [`LiveOrchestrator`](orchestrator.py) uses APScheduler to:
1. Scrape data at configured intervals
2. Compute features and generate signals
3. Execute approved trades
4. Update portfolio tracking
5. Send notifications

### Monitoring

**[`monitoring/dashboard.py`](monitoring/dashboard.py)** — Streamlit dashboard showing:
- Real-time portfolio value and equity curve
- Open positions with PnL
- Recent trade history
- Signal log with model predictions
- Risk metrics and drawdown status

```bash
streamlit run monitoring/dashboard.py --server.port 8501
```

**[`monitoring/notifier.py`](monitoring/notifier.py)** — [`TelegramNotifier`](monitoring/notifier.py) sends:
- Trade entry/exit alerts
- Daily P&L summaries
- Drawdown warnings
- Model retrain notifications

---

## Testing

### Run All Tests

```bash
cd bot-cryptov2
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Run Specific Test File

```bash
pytest tests/test_position_tracker.py -v
pytest tests/test_integration.py -v
```

### Test Suite Overview

| Test File | Coverage | Tests |
|-----------|----------|-------|
| [`test_position_tracker.py`](tests/test_position_tracker.py) | Position, ClosedTrade, PositionTracker | 30+ tests |
| [`test_risk_manager.py`](tests/test_risk_manager.py) | RiskConfig, RiskManager | 20+ tests |
| [`test_signal_filters.py`](tests/test_signal_filters.py) | FilterConfig, SignalFilter | 25+ tests |
| [`test_signal_generator.py`](tests/test_signal_generator.py) | TradeSignal, SignalGenerator | 30+ tests |
| [`test_backtest_analytics.py`](tests/test_backtest_analytics.py) | PerformanceMetrics, Analytics | 15+ tests |
| [`test_integration.py`](tests/test_integration.py) | Full pipeline (prediction → signal → risk → position → analytics) | 12+ tests |

### Shared Fixtures

[`tests/conftest.py`](tests/conftest.py) provides:
- `sample_ohlcv_df` — 100-bar random walk OHLCV DataFrame
- `sample_features_df` — Features with atr, rsi, ema_20, ema_50
- `sample_prediction` — Long prediction (conf=0.88)
- `sample_short_prediction` — Short prediction (conf=0.90)
- `sample_neutral_prediction` — Neutral prediction (conf=0.45)

---

## Workflow

### Typical Development Workflow

```
1. Scrape data          →  python main.py --mode scrape
2. Train model          →  python main.py --mode train
3. Backtest strategy    →  python main.py --mode backtest
4. Analyze results      →  Review analytics output
5. Adjust config        →  Tune risk/signal/model parameters
6. Re-backtest          →  Validate improvements
7. Test on testnet      →  python main.py --mode testnet
8. Monitor dashboard    →  streamlit run monitoring/dashboard.py
9. Go live (carefully)  →  python main.py --mode live
```

### Spot vs Futures Decision Logic

```
Model Prediction → Direction + Confidence + Magnitude
                          │
                    ┌─────┴─────┐
                    │  Neutral   │──▶ HOLD (no trade)
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │   Long    │──▶ conf > 0.85? ──▶ YES ──▶ Futures LONG (leverage 1-3x)
                    │           │                   └─ NO  ──▶ Spot LONG
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │   Short   │──▶ Futures SHORT (always, leverage 1-3x)
                    └───────────┘
```

---

## Troubleshooting

### Common Issues

**`ModuleNotFoundError` when running tests:**
```bash
# Ensure you're in the bot-cryptov2 directory
cd bot-cryptov2
pytest tests/ -v
```

**`ccxt` connection errors:**
- Check your API keys in `.env`
- Ensure testnet keys are for `testnet.binancefuture.com`
- Check if Binance testnet is available (sometimes down for maintenance)

**Model training is very slow:**
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Reduce `model.training.epochs` or `model.input_window` in config
- Use fewer trading pairs during development

**Backtest shows no trades:**
- Ensure you have scraped data: `python main.py --mode scrape`
- Check `signal.confidence_threshold` — try lowering it (e.g., 0.60)
- Check `signal.magnitude_threshold` — try lowering it (e.g., 0.003)
- Ensure model is trained: `python main.py --mode train`

**Telegram notifications not working:**
- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
- Set `monitoring.telegram.enabled: true` in `config.yaml`
- Start a conversation with your bot on Telegram first

---

## License

MIT
