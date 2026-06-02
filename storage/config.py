"""
Configuration loader for Bot-CryptoV2.

Loads config.yaml and .env, provides a unified Config object
that all modules can access.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Optional


# Project root is two levels up: storage/config.py → bot-cryptov2/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class Config:
    """Centralized configuration manager."""

    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        """Load configuration from YAML and .env files."""
        # Load .env
        env_file = Path(env_path) if env_path else ENV_PATH
        if env_file.exists():
            load_dotenv(env_file)

        # Load YAML config
        cfg_file = Path(config_path) if config_path else CONFIG_PATH
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

        with open(cfg_file, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.
        Example: config.get('trading.mode') -> 'backtest'
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    @property
    def trading(self) -> Dict:
        return self._config.get("trading", {})

    @property
    def model(self) -> Dict:
        return self._config.get("model", {})

    @property
    def signal(self) -> Dict:
        return self._config.get("signal", {})

    @property
    def risk(self) -> Dict:
        return self._config.get("risk", {})

    @property
    def exchange(self) -> Dict:
        return self._config.get("exchange", {})

    @property
    def scraping(self) -> Dict:
        return self._config.get("scraping", {})

    @property
    def storage(self) -> Dict:
        return self._config.get("storage", {})

    @property
    def backtest(self) -> Dict:
        return self._config.get("backtest", {})

    @property
    def monitoring(self) -> Dict:
        return self._config.get("monitoring", {})

    def get_api_keys(self, market: str = "testnet") -> tuple:
        """Get API key and secret for the specified market."""
        if market == "testnet":
            key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        elif market == "live":
            key = os.getenv("BINANCE_LIVE_API_KEY", "")
            secret = os.getenv("BINANCE_LIVE_API_SECRET", "")
        else:
            raise ValueError(f"Unknown market: {market}")
        return key, secret

    def get_data_path(self, *subdirs: str) -> Path:
        """Get a path under the data directory, creating it if needed."""
        base = PROJECT_ROOT / self.storage.get("data_dir", "data")
        path = base.joinpath(*subdirs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_path(self, *subdirs: str) -> Path:
        """Get a path under the models directory, creating it if needed."""
        base = PROJECT_ROOT / self.storage.get("models_dir", "models/saved")
        path = base.joinpath(*subdirs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:
        mode = self.trading.get("mode", "unknown")
        pairs = len(self.trading.get("pairs", []))
        return f"Config(mode={mode}, pairs={pairs})"


# Singleton instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> Config:
    """Get or create the global Config singleton."""
    global _config
    if _config is None:
        _config = Config(config_path, env_path)
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None
