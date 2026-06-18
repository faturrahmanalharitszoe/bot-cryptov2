"""
Bot-CryptoV2 — Main Entry Point / Orchestrator

Usage:
    python main.py --mode backtest          # Run backtesting
    python main.py --mode testnet           # Run on Binance testnet
    python main.py --mode live              # Run live trading
    python main.py --mode train             # Train/retrain model only
    python main.py --mode scrape            # Scrape data only
"""

from __future__ import annotations

import argparse
import signal as sig_module
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from storage.config import Config, get_config
from monitoring.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bot-CryptoV2 — Deep Learning Scalping Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "testnet", "live", "train", "scrape"],
        default="backtest",
        help="Operating mode (default: backtest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="Override trading pairs (e.g., BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Override symbols for training/scraping",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Mode handlers
# -----------------------------------------------------------------------

def run_backtest(config: Config, logger) -> None:
    """Run the backtesting engine with analytics."""
    logger.info("=" * 60)
    logger.info("MODE: BACKTESTING")
    logger.info("=" * 60)

    import numpy as np
    from storage.parquet_store import ParquetStore
    from features.pipeline import FeaturePipeline
    from models.predictor import Predictor
    from models.ensemble import EnsembleModel
    from models.trainer import Trainer, TrainerConfig
    from backtest.engine import BacktestEngine, BacktestConfig
    from backtest.analytics import Analytics
    from backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig
    from execution.risk_manager import RiskConfig
    from signals.filters import FilterConfig
    import pandas as pd

    bt_cfg = BacktestConfig.from_config_dict(config.backtest)
    pairs = config.trading.get("pairs", ["BTC/USDT"])
    timeframes = config.trading.get("timeframes", ["1h"])
    # Prefer 5m for scalping (288 bars/day); fall back through 15m → 1h
    primary_tf = "5m" if "5m" in timeframes else ("15m" if "15m" in timeframes else ("1h" if "1h" in timeframes else (timeframes[-1] if timeframes else "1h")))
    input_window = config.model.get("input_window", 90)

    store = ParquetStore(config.storage.get("data_dir", "data"))
    pipeline = FeaturePipeline(parquet_store=store)

    # Load data for all pairs
    ohlcv_data: dict = {}
    feature_data: dict = {}

    for symbol in pairs:
        # Load OHLCV
        df = store.load_ohlcv(symbol, primary_tf)
        if df is None or df.empty:
            logger.warning("No OHLCV data for %s %s. Run 'scrape' mode first.", symbol, primary_tf)
            continue

        # Filter to date range (timestamp is a column, not the index)
        start = pd.Timestamp(bt_cfg.start_date)
        end = pd.Timestamp(bt_cfg.end_date)
        if "timestamp" in df.columns:
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            filtered = df[mask]
            if filtered.empty:
                # Config date range doesn't match data — use all available data
                data_start = df["timestamp"].min()
                data_end = df["timestamp"].max()
                logger.warning(
                    "Config date range %s–%s has no overlap with data %s–%s. Using all available data.",
                    start.date(), end.date(), data_start.date(), data_end.date(),
                )
                # Update bt_cfg to match actual data range so the engine doesn't re-filter
                bt_cfg.start_date = str(data_start.date())
                bt_cfg.end_date = str(data_end.date())
                df = df  # keep all rows
            else:
                df = filtered
        if df.empty:
            continue

        ohlcv_data[symbol] = df

        # Compute features using ALL timeframes (must match training)
        logger.info("Computing features for %s (%d bars)...", symbol, len(df))
        features = pipeline.compute(symbol, timeframes)
        if features is not None and not features.empty:
            feature_data[symbol] = features

    if not ohlcv_data:
        logger.error("No data available. Run 'python main.py --mode scrape' first.")
        return

    # Set timestamp as DatetimeIndex for the backtest engine (expects datetime-indexed DataFrames)
    for sym in ohlcv_data:
        if "timestamp" in ohlcv_data[sym].columns:
            ohlcv_data[sym] = ohlcv_data[sym].set_index("timestamp").sort_index()
    for sym in feature_data:
        if "timestamp" in feature_data[sym].columns:
            feature_data[sym] = feature_data[sym].set_index("timestamp").sort_index()

    # Try to load model for predictions
    model_path = config.get_model_path("ensemble_best.pt")
    model_loaded = False
    predictor = None

    # Prefer find_latest_checkpoint() to avoid Windows Defender blocking ensemble_best.pt
    load_path = EnsembleModel.find_latest_checkpoint(model_path.parent)
    if load_path is None and model_path.exists():
        load_path = model_path

    if load_path is not None:
        try:
            model_obj, _ = EnsembleModel.load(str(load_path))
            predictor = Predictor(model=model_obj, input_window=input_window)
            model_loaded = True
            logger.info("Loaded model from %s", load_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)

    if not model_loaded:
        logger.warning("No trained model found. Running quick training...")
        # Build training data from features
        skip_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        all_feat_list, all_dir_list, all_mag_list, all_conf_list = [], [], [], []

        for symbol, features_df in feature_data.items():
            feat_cols = [
                c for c in features_df.columns
                if c not in skip_cols and features_df[c].dtype in [np.float64, np.int64, float, int]
            ]
            numeric = features_df[feat_cols].values.astype(np.float32)
            numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)

            # Generate SCALPING labels — aligned with ensemble.py:
            #   DIRECTION_LONG=0, DIRECTION_SHORT=1, DIRECTION_NEUTRAL=2
            # 6 bars × 5m = 30 min forward horizon; 0.3% threshold for scalp moves
            if "close" in features_df.columns:
                close = features_df["close"]
                future_return = close.shift(-6) / close - 1
                direction = pd.Series(2, index=close.index)      # Neutral default
                direction[future_return > 0.003] = 0              # Long (>0.3% in 30min)
                direction[future_return < -0.003] = 1             # Short (<-0.3% in 30min)
                magnitude = future_return.abs().fillna(0)
                confidence = pd.Series(0.5, index=close.index)
            else:
                continue

            try:
                windows = Predictor.create_sliding_windows(numeric, window_size=input_window)
            except ValueError:
                continue

            n_win = windows.shape[0]
            all_feat_list.append(windows)
            all_dir_list.append(direction.values[-n_win:].astype(np.int64))
            all_mag_list.append(magnitude.values[-n_win:].astype(np.float32))
            all_conf_list.append(confidence.values[-n_win:].astype(np.float32))

        if all_feat_list:
            feats_np = np.concatenate(all_feat_list, axis=0)
            dir_np = np.concatenate(all_dir_list, axis=0)
            mag_np = np.concatenate(all_mag_list, axis=0)
            conf_np = np.concatenate(all_conf_list, axis=0)

            in_features = feats_np.shape[2]
            model_cfg = config.model
            ensemble_model = EnsembleModel(
                in_features=in_features,
                cnn_filters=model_cfg.get("cnn", {}).get("filters"),
                cnn_kernel_sizes=model_cfg.get("cnn", {}).get("kernel_sizes"),
                lstm_hidden=model_cfg.get("lstm", {}).get("hidden_size", 128),
                lstm_layers=model_cfg.get("lstm", {}).get("num_layers", 2),
                lstm_bidirectional=model_cfg.get("lstm", {}).get("bidirectional", True),
                lstm_attention=model_cfg.get("lstm", {}).get("attention", True),
                transformer_d_model=model_cfg.get("transformer", {}).get("d_model", 128),
                transformer_nhead=model_cfg.get("transformer", {}).get("nhead", 8),
                transformer_layers=model_cfg.get("transformer", {}).get("num_layers", 4),
                transformer_ff_dim=model_cfg.get("transformer", {}).get("dim_feedforward", 256),
                dropout=model_cfg.get("cnn", {}).get("dropout", 0.3),
                transformer_dropout=model_cfg.get("transformer", {}).get("dropout", 0.1),
            )
            trainer_config = TrainerConfig.from_config_dict(model_cfg)
            trainer = Trainer(model=ensemble_model, config=trainer_config)

            train_loader, val_loader, _ = Trainer.prepare_datasets(
                features=feats_np,
                direction_labels=dir_np,
                magnitude_labels=mag_np,
                confidence_labels=conf_np,
                val_split=trainer_config.validation_split,
                test_split=trainer_config.test_split,
                batch_size=trainer_config.batch_size,
            )

            trainer.fit(train_loader=train_loader, val_loader=val_loader, save_path=str(model_path))
            predictor = Predictor(model=ensemble_model, input_window=input_window)
            model_loaded = True
            logger.info("Quick training complete.")

    # Build predictor function
    def model_predictor(features_df: pd.DataFrame, symbol: str):
        if not model_loaded:
            return None
        pred = predictor.predict_from_dataframe(features_df, symbol)
        # Override the learned confidence head with max direction probability.
        # The confidence head requires many more training epochs to calibrate;
        # max(direction_probs) is immediately meaningful (0.33 = random, 1.0 = certain).
        pred.confidence = max(pred.direction_probs.values())
        return pred

    # Run backtest
    engine = BacktestEngine(
        config=bt_cfg,
        risk_config=RiskConfig.from_config_dict(config.risk),
        signal_config=config.signal,
        filter_config=FilterConfig(
            cooldown_minutes=config.signal.get("cooldown_minutes", 15),
            min_confidence=config.signal.get("confidence_threshold", 0.50),
            min_magnitude=config.signal.get("magnitude_threshold", 0.002),
            trend_alignment_enabled=False,   # disabled for backtest — too aggressive with limited data
            volatility_filter_enabled=False,  # disabled for backtest — ATR unreliable in short windows
            volume_filter_enabled=False,      # disabled for backtest — volume data may be incomplete
        ),
    )

    result = engine.run(
        ohlcv_data=ohlcv_data,
        feature_data=feature_data,
        model_predictor=model_predictor if model_loaded else None,
    )

    # Analytics
    analytics = Analytics()
    metrics = analytics.compute(result)
    try:
        print(metrics.summary())
    except UnicodeEncodeError:
        print(metrics.summary().encode("ascii", errors="replace").decode("ascii"))

    # Monte Carlo
    if result.closed_trades and len(result.closed_trades) >= 10:
        logger.info("Running Monte Carlo simulation...")
        mc_sim = MonteCarloSimulator(MonteCarloConfig(
            num_simulations=1000,
            random_seed=42,
        ))
        mc_result = mc_sim.run(result)
        try:
            print(mc_result.summary())
        except UnicodeEncodeError:
            print(mc_result.summary().encode("ascii", errors="replace").decode("ascii"))
    else:
        logger.info("Skipping Monte Carlo (need >= 10 trades, got %d)", len(result.closed_trades))

    logger.info("Backtest complete!")


def run_train(config: Config, logger) -> None:
    """Train or retrain the deep learning model."""
    logger.info("=" * 60)
    logger.info("MODE: MODEL TRAINING")
    logger.info("=" * 60)

    import numpy as np
    from storage.parquet_store import ParquetStore
    from features.pipeline import FeaturePipeline
    from models.trainer import Trainer, TrainerConfig
    from models.ensemble import EnsembleModel
    from models.predictor import Predictor

    pairs = config.trading.get("pairs", ["BTC/USDT"])
    timeframes = config.trading.get("timeframes", ["1h"])
    input_window = config.model.get("input_window", 90)

    store = ParquetStore(config.storage.get("data_dir", "data"))
    pipeline = FeaturePipeline(parquet_store=store)

    # Columns to exclude from model input (raw OHLCV + timestamp)
    skip_cols = {"timestamp", "open", "high", "low", "close", "volume"}

    # Collect features and labels for all pairs
    all_features_list = []
    all_direction_list = []
    all_magnitude_list = []
    all_confidence_list = []

    for symbol in pairs:
        logger.info("Computing training data for %s...", symbol)
        features_df, direction, magnitude, confidence = pipeline.compute_for_training(
            symbol, timeframes
        )
        if features_df.empty:
            logger.warning("No training data for %s. Skipping.", symbol)
            continue

        # Extract numeric feature columns (exclude OHLCV + timestamp)
        feature_cols = [
            c for c in features_df.columns
            if c not in skip_cols and features_df[c].dtype in [np.float64, np.int64, float, int]
        ]
        numeric_features = features_df[feature_cols].values.astype(np.float32)
        numeric_features = np.nan_to_num(numeric_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sliding windows
        try:
            windows = Predictor.create_sliding_windows(numeric_features, window_size=input_window)
        except ValueError as e:
            logger.warning("Cannot create windows for %s: %s", symbol, e)
            continue

        # Align labels: last N windows correspond to the end of the series
        n_windows = windows.shape[0]
        dir_vals = direction.values[-n_windows:].astype(np.int64)
        mag_vals = magnitude.values[-n_windows:].astype(np.float32)
        conf_vals = confidence.values[-n_windows:].astype(np.float32)

        all_features_list.append(windows)
        all_direction_list.append(dir_vals)
        all_magnitude_list.append(mag_vals)
        all_confidence_list.append(conf_vals)

        logger.info(
            "  %s: %d windows, %d features per timestep",
            symbol, n_windows, numeric_features.shape[1],
        )

    if not all_features_list:
        logger.error("No training data available. Run 'python main.py --mode scrape' first.")
        return

    # Concatenate all symbols
    all_features = np.concatenate(all_features_list, axis=0)
    all_direction = np.concatenate(all_direction_list, axis=0)
    all_magnitude = np.concatenate(all_magnitude_list, axis=0)
    all_confidence = np.concatenate(all_confidence_list, axis=0)

    logger.info(
        "Total training samples: %d, features: %d, window: %d",
        all_features.shape[0], all_features.shape[2], input_window,
    )

    # Determine in_features from data
    in_features = all_features.shape[2]

    # Create EnsembleModel from config
    model_cfg = config.model
    ensemble_model = EnsembleModel(
        in_features=in_features,
        cnn_filters=model_cfg.get("cnn", {}).get("filters"),
        cnn_kernel_sizes=model_cfg.get("cnn", {}).get("kernel_sizes"),
        lstm_hidden=model_cfg.get("lstm", {}).get("hidden_size", 128),
        lstm_layers=model_cfg.get("lstm", {}).get("num_layers", 2),
        lstm_bidirectional=model_cfg.get("lstm", {}).get("bidirectional", True),
        lstm_attention=model_cfg.get("lstm", {}).get("attention", True),
        transformer_d_model=model_cfg.get("transformer", {}).get("d_model", 128),
        transformer_nhead=model_cfg.get("transformer", {}).get("nhead", 8),
        transformer_layers=model_cfg.get("transformer", {}).get("num_layers", 4),
        transformer_ff_dim=model_cfg.get("transformer", {}).get("dim_feedforward", 256),
        dropout=model_cfg.get("cnn", {}).get("dropout", 0.3),
        transformer_dropout=model_cfg.get("transformer", {}).get("dropout", 0.1),
    )

    # Create TrainerConfig from config.yaml
    trainer_config = TrainerConfig.from_config_dict(model_cfg)

    # Compute class weights (inverse-frequency) to handle Long/Short/Neutral imbalance
    class_weights = None
    if trainer_config.use_class_weights:
        class_weights = Trainer.compute_class_weights(all_direction)
        logger.info("Class weights computed: %s", class_weights.tolist())

    # Create Trainer with model, config, and optional class weights
    trainer = Trainer(model=ensemble_model, config=trainer_config, class_weights=class_weights)

    # Prepare DataLoaders (static method — takes numpy arrays)
    train_loader, val_loader, test_loader = Trainer.prepare_datasets(
        features=all_features,
        direction_labels=all_direction,
        magnitude_labels=all_magnitude,
        confidence_labels=all_confidence,
        val_split=trainer_config.validation_split,
        test_split=trainer_config.test_split,
        batch_size=trainer_config.batch_size,
        use_weighted_sampler=trainer_config.use_weighted_sampler,
    )

    # Train  (use .pt extension — Windows Defender may block .pth writes)
    save_path = str(config.get_model_path("ensemble_best.pt"))
    try:
        result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            save_path=save_path,
        )
        logger.info("Training complete: %s", result)

        # Evaluate on test set
        eval_result = trainer.evaluate(test_loader)
        logger.info("Test evaluation: %s", eval_result)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user — checkpoint saved by trainer.")


def run_scrape(config: Config, logger) -> None:
    """Run data scraping for all configured pairs."""
    logger.info("=" * 60)
    logger.info("MODE: DATA SCRAPING")
    logger.info("=" * 60)

    from datetime import datetime, timedelta
    from storage.parquet_store import ParquetStore
    from storage.sqlite_store import SQLiteStore
    from scrapers.ohlcv_scraper import OHLCVScraper
    from scrapers.sentiment_scraper import SentimentScraper

    pairs = config.trading.get("pairs", ["BTC/USDT"])
    timeframes = config.trading.get("timeframes", ["1h"])
    history_days = config.scraping.get("ohlcv", {}).get("history_days", 90)

    store = ParquetStore(config.storage.get("data_dir", "data"))
    db = SQLiteStore()  # uses default path: data/raw/sentiment/bot_data.db

    # Expected bar counts (sanity check)
    bars_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
                    "1h": 24, "2h": 12, "4h": 6, "6h": 4, "8h": 3, "12h": 2, "1d": 1}

    # OHLCV scraping — use mainnet public API (no key required)
    # The since= parameter ensures a full history_days window is fetched
    # on first run, even if the parquet file already has some data.
    since_dt = datetime.utcnow() - timedelta(days=history_days)

    ohlcv_scraper = OHLCVScraper(testnet=False, store=store)  # mainnet for historical data
    for symbol in pairs:
        for tf in timeframes:
            expected_bars = bars_per_day.get(tf, 24) * history_days
            logger.info(
                "Scraping %s %s (history=%d days, ~%d bars expected)...",
                symbol, tf, history_days, expected_bars,
            )
            try:
                # Pass since= so the scraper starts from history_days ago
                # (OHLCVScraper.scrape already saves to parquet internally)
                df = ohlcv_scraper.scrape(symbol, tf, since=since_dt, limit=None)
                if df is not None and not df.empty:
                    pct = 100.0 * len(df) / max(expected_bars, 1)
                    logger.info(
                        "  %s %s: %d bars fetched (%.0f%% of expected %d)",
                        symbol, tf, len(df), pct, expected_bars,
                    )
                    if len(df) < expected_bars * 0.5:
                        logger.warning(
                            "  WARNING: Only %.0f%% of expected bars. "
                            "Possible testnet/API issue or exchange outage.",
                            pct,
                        )
                else:
                    logger.warning("  No data returned for %s %s", symbol, tf)
            except Exception as e:
                logger.error("Failed to scrape %s %s: %s", symbol, tf, e)

    # Sentiment scraping
    logger.info("Scraping sentiment data...")
    try:
        sentiment_scraper = SentimentScraper()
        count = sentiment_scraper.scrape()
        logger.info("Saved %d sentiment items", count)
    except Exception as e:
        logger.error("Sentiment scraping failed: %s", e)

    logger.info("Scraping complete!")


def run_live(config: Config, logger, mode: str = "testnet") -> None:
    """Run live/testnet trading with the orchestrator."""
    logger.info("=" * 60)
    logger.info("MODE: %s TRADING", mode.upper())
    logger.info("=" * 60)

    from orchestrator import LiveOrchestrator

    orch = LiveOrchestrator(config, mode=mode)

    # Graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received. Stopping orchestrator...")
        orch.stop()
        sys.exit(0)

    sig_module.signal(sig_module.SIGINT, shutdown_handler)
    sig_module.signal(sig_module.SIGTERM, shutdown_handler)

    orch.start()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = get_config(config_path=args.config)

    # Setup logger
    log_level = args.log_level or config.monitoring.get("logging", {}).get("level", "INFO")
    log_file = config.monitoring.get("logging", {}).get("file", "logs/bot.log")
    max_bytes = config.monitoring.get("logging", {}).get("max_bytes", 10_485_760)
    backup_count = config.monitoring.get("logging", {}).get("backup_count", 5)

    logger = setup_logger(
        level=log_level,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )

    # Override pairs if specified
    if args.pairs:
        config._config["trading"]["pairs"] = args.pairs

    # Banner
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║       Bot-CryptoV2 — DL Swing Trading        ║")
    logger.info("║   CNN-LSTM + Transformer Ensemble Model       ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info(f"Config: {config}")
    logger.info(f"Mode: {args.mode}")

    # Route to appropriate mode
    mode_handlers = {
        "backtest": run_backtest,
        "testnet": lambda c, l: run_live(c, l, mode="testnet"),
        "live": lambda c, l: run_live(c, l, mode="live"),
        "train": run_train,
        "scrape": run_scrape,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler(config, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
