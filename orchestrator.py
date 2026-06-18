"""
Orchestrator — Live/Testnet trading loop with APScheduler.

Manages:
  - Periodic OHLCV scraping (5m intervals)
  - Periodic orderbook scraping (1m intervals)
  - Periodic sentiment scraping (15m intervals)
  - Periodic signal generation + execution
  - Periodic trailing stop updates
  - Model retraining (24h intervals)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from storage.config import Config
from storage.parquet_store import ParquetStore
from storage.sqlite_store import SQLiteStore
from scrapers.ohlcv_scraper import OHLCVScraper
from scrapers.orderbook_scraper import OrderbookScraper
from scrapers.sentiment_scraper import SentimentScraper
from scrapers.onchain_scraper import OnChainScraper
from features.pipeline import FeaturePipeline
from models.predictor import Predictor
from models.ensemble import EnsembleModel
from signals.generator import SignalGenerator, SignalAction
from signals.filters import FilterConfig
from execution.exchange import ExchangeClient
from execution.position_tracker import PositionTracker
from execution.risk_manager import RiskManager, RiskConfig

logger = logging.getLogger(__name__)


class LiveOrchestrator:
    """Orchestrates live/testnet trading with scheduled jobs.

    Usage:
        orch = LiveOrchestrator(config, mode="testnet")
        orch.start()  # blocks until shutdown
    """

    def __init__(self, config: Config, mode: str = "testnet"):
        self.config = config
        self.mode = mode  # "testnet" or "live"
        self._running = False

        # Initialize storage
        self.parquet_store = ParquetStore(config.storage.get("data_dir", "data"))
        self.sqlite_store = SQLiteStore()  # uses default path

        # Initialize scrapers
        self.ohlcv_scraper = OHLCVScraper()
        self.orderbook_scraper = OrderbookScraper()
        self.sentiment_scraper = SentimentScraper(db_store=self.sqlite_store)

        # Initialize feature pipeline
        self.feature_pipeline = FeaturePipeline(parquet_store=self.parquet_store)

        # Initialize model (deferred until checkpoint is loaded)
        self.predictor = None
        self._try_load_model()

        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            config=config.signal,
            filter_config=FilterConfig(
                cooldown_minutes=config.signal.get("cooldown_minutes", 30),
                min_confidence=config.signal.get("confidence_threshold", 0.70),
                min_magnitude=config.signal.get("magnitude_threshold", 0.005),
            ),
        )

        # Initialize execution
        api_key, secret = config.get_api_keys("testnet" if mode == "testnet" else "live")
        self.exchange = ExchangeClient(
            testnet=(mode == "testnet"),
            api_key=api_key,
            secret=secret,
            rate_limit_per_minute=config.exchange.get("rate_limit_per_minute", 1200),
        )
        self.tracker = PositionTracker(
            max_concurrent=config.risk.get("max_concurrent_positions", 3),
        )
        self.risk_manager = RiskManager(
            config=RiskConfig.from_config_dict(config.risk),
            tracker=self.tracker,
            initial_capital=10000.0,  # TODO: get from exchange balance
        )

        # Scheduler
        self.scheduler = BackgroundScheduler()

        # Trading pairs
        self.pairs: list[str] = config.trading.get("pairs", ["BTC/USDT"])

        logger.info("LiveOrchestrator initialized: mode=%s, pairs=%d", mode, len(self.pairs))

    def _try_load_model(self) -> None:
        """Attempt to load a saved model checkpoint."""
        try:
            model_path = self.config.get_model_path("ensemble_best.pt")
            # EnsembleModel.load auto-discovers latest checkpoint if exact path missing
            model_obj, _ = EnsembleModel.load(str(model_path))
            input_window = self.config.model.get("input_window", 90)
            self.predictor = Predictor(model=model_obj, input_window=input_window)
            logger.info("Loaded model from %s", model_path)
        except FileNotFoundError:
            logger.warning("No saved model found. Run 'train' mode first.")
        except Exception as e:
            logger.error("Failed to load model: %s", e)

    # -------------------------------------------------------------------
    # Scheduled jobs
    # -------------------------------------------------------------------

    def _job_scrape_ohlcv(self) -> None:
        """Scrape OHLCV data for all pairs."""
        logger.info("Scraping OHLCV for %d pairs...", len(self.pairs))
        for symbol in self.pairs:
            try:
                for tf in self.config.trading.get("timeframes", ["1h"]):
                    df = self.ohlcv_scraper.scrape(symbol, tf, limit=200)
                    if df is not None and not df.empty:
                        self.parquet_store.save_ohlcv(symbol, tf, df)
                        logger.debug("Saved %d OHLCV bars for %s %s", len(df), symbol, tf)
            except Exception as e:
                logger.error("OHLCV scrape failed for %s: %s", symbol, e)

    def _job_scrape_orderbook(self) -> None:
        """Scrape orderbook data for all pairs."""
        for symbol in self.pairs:
            try:
                data = self.orderbook_scraper.scrape(symbol)
                if data is not None and not data.empty:
                    self.parquet_store.save_orderbook(symbol, data)
            except Exception as e:
                logger.error("Orderbook scrape failed for %s: %s", symbol, e)

    def _job_scrape_sentiment(self) -> None:
        """Scrape sentiment data."""
        try:
            count = self.sentiment_scraper.scrape()
            logger.info("Scraped %d sentiment items", count)
        except Exception as e:
            logger.error("Sentiment scrape failed: %s", e)

    def _job_generate_signals(self) -> None:
        """Generate signals and execute trades."""
        logger.info("Running signal generation cycle...")
        current_prices: dict[str, float] = {}

        for symbol in self.pairs:
            try:
                # Get current price
                ticker = self.exchange.get_ticker(symbol)
                if ticker is None:
                    continue
                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    continue
                current_prices[symbol] = current_price

                # Load features
                primary_tf = self.config.trading.get("timeframes", ["1h"])[-1]
                features_df = self.parquet_store.load_features(symbol, primary_tf)
                if features_df is None or features_df.empty:
                    # Build features from OHLCV
                    ohlcv = self.parquet_store.load_ohlcv(symbol, primary_tf)
                    if ohlcv is None or ohlcv.empty:
                        continue
                    features_df = self.feature_pipeline.compute(symbol)

                if features_df is None or len(features_df) < 30:
                    continue

                if self.predictor is None:
                    logger.debug("No model loaded, skipping prediction for %s", symbol)
                    continue

                # Model prediction
                prediction = self.predictor.predict_from_dataframe(features_df, symbol)
                if prediction is None:
                    continue

                # Generate signal
                signal = self.signal_generator.generate(
                    prediction=prediction,
                    current_price=current_price,
                    features_df=features_df,
                )

                try:
                    self.sqlite_store.save_signal(
                        symbol=signal.symbol,
                        direction=signal.action.value,
                        confidence=prediction.confidence,
                        magnitude=prediction.magnitude,
                        market_type=signal.market.value,
                    )
                except Exception as e:
                    logger.warning("Failed to save signal to DB: %s", e)

                if signal.action == SignalAction.HOLD:
                    continue

                logger.info("Signal: %s", signal)

                # Risk check
                approved, reasons = self.risk_manager.check_signal(signal)
                if not approved:
                    logger.info("Signal rejected: %s", "; ".join(reasons))
                    continue

                # Execute
                if signal.is_entry:
                    position = self.risk_manager.create_position(signal, current_price)
                    if position:
                        try:
                            if signal.market.value == "spot":
                                result = self.exchange.create_spot_order(
                                    symbol=signal.symbol,
                                    side=signal.side,
                                    order_type="market",
                                    amount=position.size,
                                )
                            else:
                                result = self.exchange.create_futures_order(
                                    symbol=signal.symbol,
                                    side=signal.side,
                                    order_type="market",
                                    amount=position.size,
                                    leverage=signal.leverage,
                                )
                            logger.info("Order result: %s", result)
                            
                            try:
                                db_id = self.sqlite_store.save_trade(
                                    symbol=position.symbol,
                                    side=position.side,
                                    market=position.market,
                                    order_type="market",
                                    price=position.entry_price,
                                    quantity=position.size,
                                    total_value=position.notional_value,
                                    signal_confidence=prediction.confidence,
                                    signal_direction=signal.action.value,
                                    signal_magnitude=prediction.magnitude,
                                    status="open"
                                )
                                position.db_id = db_id
                            except Exception as e:
                                logger.warning("Failed to save trade to DB: %s", e)
                                
                        except Exception as e:
                            logger.error("Failed to execute order on exchange for %s: %s", signal.symbol, e)
                            # Rollback the in-memory tracker since order failed
                            self.tracker.close_position(signal.symbol, current_price, close_reason="execution_failed")
                            continue

                elif signal.is_exit:
                    pos = self.tracker.get_position(symbol)
                    if pos:
                        db_id = pos.db_id
                        result = self.exchange.close_position(
                            symbol=symbol,
                            market=pos.market,
                        )
                        trade = self.tracker.close_position(
                            symbol, current_price, close_reason="signal",
                        )
                        if trade:
                            self.risk_manager.update_portfolio_value(
                                self.risk_manager.current_portfolio_value + trade.pnl,
                            )
                            if db_id != -1:
                                try:
                                    self.sqlite_store.close_trade(
                                        trade_id=db_id,
                                        close_price=trade.exit_price,
                                        pnl=trade.pnl,
                                        close_reason="signal",
                                    )
                                except Exception as e:
                                    logger.warning("Failed to close trade in DB: %s", e)
                        logger.info("Position closed: %s", result)

            except Exception as e:
                logger.error("Signal generation failed for %s: %s", symbol, e)

        # Update portfolio value
        if current_prices:
            unrealized = self.tracker.compute_unrealized_pnl(current_prices)
            total = self.risk_manager.current_portfolio_value + unrealized
            self.risk_manager.update_portfolio_value(total)

    def _job_update_trailing_stops(self) -> None:
        """Update trailing stop-losses for all open positions."""
        for pos in self.tracker.get_all_positions():
            try:
                ticker = self.exchange.get_ticker(pos.symbol)
                if ticker:
                    price = ticker.get("last", 0)
                    if price > 0:
                        self.risk_manager.update_trailing_stop(pos.symbol, price)

                        # Check stop-loss
                        if pos.should_stop_loss(price):
                            logger.warning("STOP-LOSS triggered for %s at %.4f", pos.symbol, price)
                            db_id = pos.db_id
                            result = self.exchange.close_position(
                                symbol=pos.symbol,
                                market=pos.market,
                            )
                            trade = self.tracker.close_position(
                                pos.symbol, price, close_reason="stop_loss",
                            )
                            if trade:
                                self.risk_manager.update_portfolio_value(
                                    self.risk_manager.current_portfolio_value + trade.pnl,
                                )
                                if db_id != -1:
                                    try:
                                        self.sqlite_store.close_trade(
                                            trade_id=db_id,
                                            close_price=trade.exit_price,
                                            pnl=trade.pnl,
                                            close_reason="stop_loss",
                                        )
                                    except Exception as e:
                                        logger.warning("Failed to close trade in DB: %s", e)
                            logger.info("Stop-loss executed: %s", result)

                        # Check take-profits
                        tp_idx = pos.check_take_profit(price)
                        if tp_idx is not None:
                            close_fraction = self.risk_manager.compute_tp_close_size(tp_idx)
                            close_size = pos.initial_size * close_fraction
                            logger.info("TAKE-PROFIT %d for %s: closing %.2f%%",
                                        tp_idx, pos.symbol, close_fraction * 100)
                            db_id = pos.db_id
                            result = self.exchange.close_position(
                                symbol=pos.symbol,
                                market=pos.market,
                            )
                            trade = self.tracker.close_position(
                                pos.symbol, price, close_reason="take_profit",
                            )
                            if trade:
                                self.risk_manager.update_portfolio_value(
                                    self.risk_manager.current_portfolio_value + trade.pnl,
                                )
                                if db_id != -1:
                                    try:
                                        self.sqlite_store.close_trade(
                                            trade_id=db_id,
                                            close_price=trade.exit_price,
                                            pnl=trade.pnl,
                                            close_reason="take_profit",
                                        )
                                    except Exception as e:
                                        logger.warning("Failed to close trade in DB: %s", e)
                            logger.info("TP execution: %s", result)

            except Exception as e:
                logger.error("Trailing stop update failed for %s: %s", pos.symbol, e)

    def _job_save_state(self) -> None:
        """Periodically save trading state."""
        import json
        from pathlib import Path
        stats = self.tracker.get_stats()
        risk_stats = self.risk_manager.get_stats()
        logger.info(
            "State: portfolio=$%.2f | positions=%d | trades=%d | win_rate=%.1f%% | drawdown=%.2f%%",
            risk_stats["portfolio_value"],
            risk_stats["open_positions"],
            stats.get("total_trades", 0),
            stats.get("win_rate", 0) * 100,
            risk_stats["total_drawdown"] * 100,
        )
        
        try:
            state_data = {
                "tracker": stats,
                "risk": risk_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            state_file = Path(self.config.storage.get("data_dir", "data")) / "state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save state.json: %s", e)

    # -------------------------------------------------------------------
    # Start / Stop
    # -------------------------------------------------------------------

    def start(self) -> None:
        """Start the orchestrator with scheduled jobs."""
        self._running = True

        try:
            self.exchange.connect()
        except Exception as e:
            logger.error("Failed to connect to exchange: %s", e)
            return

        # Update intervals from config
        ohlcv_interval = self.config.scraping.get("ohlcv", {}).get("update_interval_sec", 300)
        orderbook_interval = self.config.scraping.get("orderbook", {}).get("update_interval_sec", 60)
        sentiment_interval = self.config.scraping.get("sentiment", {}).get("update_interval_sec", 900)

        # Schedule jobs
        self.scheduler.add_job(
            self._job_scrape_ohlcv,
            IntervalTrigger(seconds=ohlcv_interval),
            id="scrape_ohlcv",
            name="Scrape OHLCV",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._job_scrape_orderbook,
            IntervalTrigger(seconds=orderbook_interval),
            id="scrape_orderbook",
            name="Scrape Orderbook",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._job_scrape_sentiment,
            IntervalTrigger(seconds=sentiment_interval),
            id="scrape_sentiment",
            name="Scrape Sentiment",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._job_generate_signals,
            IntervalTrigger(seconds=ohlcv_interval),  # Same as OHLCV refresh
            id="generate_signals",
            name="Signal Generation",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._job_update_trailing_stops,
            IntervalTrigger(seconds=30),
            id="trailing_stops",
            name="Trailing Stops",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._job_save_state,
            IntervalTrigger(minutes=5),
            id="save_state",
            name="Save State",
            max_instances=1,
        )

        self.scheduler.start()
        logger.info("Orchestrator started in %s mode with %d scheduled jobs", self.mode, len(self.scheduler.get_jobs()))

        # Run initial scrape and state save immediately
        self._job_scrape_ohlcv()
        self._job_save_state()

        # Block until interrupted
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        self._running = False
        self.scheduler.shutdown(wait=False)

        # Close all open positions on shutdown
        for pos in self.tracker.get_all_positions():
            try:
                ticker = self.exchange.get_ticker(pos.symbol)
                if ticker:
                    price = ticker.get("last", 0)
                    if price > 0:
                        db_id = pos.db_id
                        self.exchange.close_position(
                            symbol=pos.symbol,
                            market=pos.market,
                        )
                        trade = self.tracker.close_position(pos.symbol, price, close_reason="shutdown")
                        if trade and db_id != -1:
                            try:
                                self.sqlite_store.close_trade(
                                    trade_id=db_id,
                                    close_price=trade.exit_price,
                                    pnl=trade.pnl,
                                    close_reason="shutdown",
                                )
                            except Exception as e:
                                logger.warning("Failed to close trade in DB on shutdown: %s", e)
                        logger.info("Closed position on shutdown: %s", pos.symbol)
            except Exception as e:
                logger.error("Failed to close position %s on shutdown: %s", pos.symbol, e)

        self._job_save_state()

        # Final stats
        stats = self.tracker.get_stats()
        logger.info("Final stats: %s", stats)
        logger.info("Orchestrator stopped.")
