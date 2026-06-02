"""
Model Trainer — Multi-task training loop for the EnsembleModel.

Features:
  - Multi-task loss: Cross-Entropy (direction) + MSE (magnitude) + BCE (confidence)
  - AdamW optimizer with cosine annealing LR scheduler
  - Early stopping with patience
  - Train/validation/test split
  - Gradient clipping
  - Best model checkpointing
  - Training metrics logging
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from models.ensemble import EnsembleModel, DIRECTION_LONG, DIRECTION_SHORT, DIRECTION_NEUTRAL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Training hyperparameters and settings."""

    # Optimizer
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # Scheduler
    scheduler: str = "cosine"  # cosine | step | plateau
    step_size: int = 30        # for step scheduler
    gamma: float = 0.1         # for step scheduler
    T_0: int = 10              # for cosine warm restarts

    # Training
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0

    # Loss weights (multi-task)
    direction_loss_weight: float = 1.0
    magnitude_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.3

    # Data splits
    validation_split: float = 0.15
    test_split: float = 0.1

    # Device
    device: str = "auto"  # auto | cpu | cuda | mps

    # Checkpointing
    checkpoint_dir: str = "models/saved"
    save_best_only: bool = True

    # Memory management
    max_ram_mb: int = 0        # 0 = unlimited; set to e.g. 4096 to cap at 4 GB
    gc_every_n_epochs: int = 1 # run gc.collect() every N epochs (1 = every epoch)

    @classmethod
    def from_config_dict(cls, model_cfg: dict[str, Any]) -> "TrainerConfig":
        """Create from config.yaml model.training section."""
        training = model_cfg.get("training", {})
        return cls(
            learning_rate=training.get("learning_rate", 0.001),
            weight_decay=training.get("weight_decay", 0.0001),
            scheduler=training.get("scheduler", "cosine"),
            batch_size=training.get("batch_size", 32),
            epochs=training.get("epochs", 100),
            early_stopping_patience=training.get("early_stopping_patience", 15),
            validation_split=training.get("validation_split", 0.15),
            test_split=training.get("test_split", 0.1),
            max_ram_mb=training.get("max_ram_mb", 0),
            gc_every_n_epochs=training.get("gc_every_n_epochs", 1),
        )


# ---------------------------------------------------------------------------
# Multi-task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """Combined loss for direction (CE) + magnitude (MSE) + confidence (BCE).

    The confidence target is derived from whether the direction prediction
    was correct — this teaches the model to be confident when it's right
    and uncertain when it's wrong.
    """

    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        confidence_weight: float = 0.3,
        # Optional: class weights for imbalanced direction classes
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.confidence_weight = confidence_weight

        self.direction_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.magnitude_loss = nn.MSELoss()
        self.confidence_loss = nn.BCELoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: model output dict with direction_logits, magnitude, confidence
            targets: dict with:
                - direction: (batch,) long tensor of class indices (0=Long, 1=Short, 2=Neutral)
                - magnitude: (batch, 1) float tensor of actual % price change
                - confidence: (batch, 1) float tensor 0-1 (1 if direction correct)
        Returns:
            dict with total_loss, direction_loss, magnitude_loss, confidence_loss
        """
        # Direction loss (uses logits, not softmax)
        d_loss = self.direction_loss(
            predictions["direction_logits"],
            targets["direction"],
        )

        # Magnitude loss
        m_loss = self.magnitude_loss(
            predictions["magnitude"],
            targets["magnitude"],
        )

        # Confidence loss
        c_loss = self.confidence_loss(
            predictions["confidence"],
            targets["confidence"],
        )

        total = (
            self.direction_weight * d_loss
            + self.magnitude_weight * m_loss
            + self.confidence_weight * c_loss
        )

        return {
            "total_loss": total,
            "direction_loss": d_loss,
            "magnitude_loss": m_loss,
            "confidence_loss": c_loss,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Tracks training progress."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_direction_acc: list[float] = field(default_factory=list)
    val_direction_acc: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_trained: int = 0


class Trainer:
    """Handles the full training lifecycle of an EnsembleModel."""

    def __init__(
        self,
        model: EnsembleModel,
        config: TrainerConfig,
        class_weights: torch.Tensor | None = None,
    ):
        self.config = config
        self.model = model
        self.metrics = TrainingMetrics()

        # Resolve device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model.to(self.device)
        logger.info("Trainer using device: %s", self.device)

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = MultiTaskLoss(
            direction_weight=config.direction_loss_weight,
            magnitude_weight=config.magnitude_loss_weight,
            confidence_weight=config.confidence_loss_weight,
            class_weights=class_weights,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR Scheduler
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler based on config."""
        cfg = self.config
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cfg.T_0, T_mult=2
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.step_size, gamma=cfg.gamma
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        else:
            logger.warning("Unknown scheduler '%s', defaulting to cosine", cfg.scheduler)
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cfg.T_0, T_mult=2
            )

    # -----------------------------------------------------------------------
    # Memory helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_process_memory_mb() -> float:
        """Get current process RSS memory in MB (cross-platform, no deps)."""
        try:
            import ctypes
            import sys
            if sys.platform == "win32":
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                ctypes.windll.kernel32.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                    ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                    ctypes.byref(counters),
                    counters.cb,
                )
                return counters.WorkingSetSize / (1024 * 1024)
            else:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                ru = usage.ru_maxrss
                return ru / 1024 if sys.platform == "linux" else ru / (1024 * 1024)
        except Exception:
            return 0.0

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def prepare_datasets(
        features: np.ndarray,
        direction_labels: np.ndarray,
        magnitude_labels: np.ndarray,
        confidence_labels: np.ndarray,
        val_split: float = 0.15,
        test_split: float = 0.1,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Convert numpy arrays into train/val/test DataLoaders.

        Args:
            features: (N, seq_len, n_features) — sliding window inputs
            direction_labels: (N,) — class indices 0/1/2
            magnitude_labels: (N, 1) — % price change
            confidence_labels: (N, 1) — 0 or 1
            val_split: fraction for validation
            test_split: fraction for test

        Returns:
            (train_loader, val_loader, test_loader)
        """
        # Convert to tensors
        X = torch.FloatTensor(features)
        y_dir = torch.LongTensor(direction_labels)
        y_mag = torch.FloatTensor(magnitude_labels).unsqueeze(-1) if magnitude_labels.ndim == 1 else torch.FloatTensor(magnitude_labels)
        y_conf = torch.FloatTensor(confidence_labels).unsqueeze(-1) if confidence_labels.ndim == 1 else torch.FloatTensor(confidence_labels)

        dataset = TensorDataset(X, y_dir, y_mag, y_conf)

        # Split
        n = len(dataset)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_val - n_test

        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        logger.info("Dataset split: train=%d, val=%d, test=%d", n_train, n_val, n_test)

        # num_workers=0 avoids duplicating data in subprocess memory
        # pin_memory=False reduces host memory overhead
        dl_kwargs = dict(num_workers=0, pin_memory=False, persistent_workers=False)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, **dl_kwargs)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **dl_kwargs)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, **dl_kwargs)

        return train_loader, val_loader, test_loader

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_d_correct = 0
        total_samples = 0

        for batch_idx, (X, y_dir, y_mag, y_conf) in enumerate(loader):
            X = X.to(self.device)
            y_dir = y_dir.to(self.device)
            y_mag = y_mag.to(self.device)
            y_conf = y_conf.to(self.device)

            # Forward
            predictions = self.model(X)

            targets = {
                "direction": y_dir,
                "magnitude": y_mag,
                "confidence": y_conf,
            }
            losses = self.criterion(predictions, targets)

            # Backward
            self.optimizer.zero_grad()
            losses["total_loss"].backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()

            # Metrics
            total_loss += losses["total_loss"].item() * X.size(0)
            pred_dir = predictions["direction"].argmax(dim=1)
            total_d_correct += (pred_dir == y_dir).sum().item()
            total_samples += X.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_d_correct / max(total_samples, 1)

        return {"loss": avg_loss, "direction_accuracy": accuracy}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate on a data loader."""
        self.model.eval()
        total_loss = 0.0
        total_d_correct = 0
        total_samples = 0

        for X, y_dir, y_mag, y_conf in loader:
            X = X.to(self.device)
            y_dir = y_dir.to(self.device)
            y_mag = y_mag.to(self.device)
            y_conf = y_conf.to(self.device)

            predictions = self.model(X)

            targets = {
                "direction": y_dir,
                "magnitude": y_mag,
                "confidence": y_conf,
            }
            losses = self.criterion(predictions, targets)

            total_loss += losses["total_loss"].item() * X.size(0)
            pred_dir = predictions["direction"].argmax(dim=1)
            total_d_correct += (pred_dir == y_dir).sum().item()
            total_samples += X.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_d_correct / max(total_samples, 1)

        return {"loss": avg_loss, "direction_accuracy": accuracy}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str | Path | None = None,
    ) -> TrainingMetrics:
        """Full training loop with early stopping.

        Args:
            train_loader: training data
            val_loader: validation data
            save_path: path to save best model checkpoint

        Returns:
            TrainingMetrics with history
        """
        cfg = self.config
        save_path = Path(save_path) if save_path else Path(cfg.checkpoint_dir) / "ensemble_best.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        patience_counter = 0
        metrics = self.metrics

        logger.info(
            "Starting training: epochs=%d, batch_size=%d, lr=%s",
            cfg.epochs, cfg.batch_size, cfg.learning_rate,
        )

        interrupted = False
        try:
            for epoch in range(1, cfg.epochs + 1):
                t0 = time.time()

                # Train
                train_metrics = self.train_epoch(train_loader)
                # Validate
                val_metrics = self.validate(val_loader)

                # Update scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

                elapsed = time.time() - t0
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Record metrics
                metrics.train_losses.append(train_metrics["loss"])
                metrics.val_losses.append(val_metrics["loss"])
                metrics.train_direction_acc.append(train_metrics["direction_accuracy"])
                metrics.val_direction_acc.append(val_metrics["direction_accuracy"])
                metrics.epochs_trained = epoch

                # Logging
                if epoch % 5 == 0 or epoch == 1:
                    ram_mb = self._get_process_memory_mb()
                    logger.info(
                        "Epoch %d/%d | train_loss=%.6f val_loss=%.6f | "
                        "train_acc=%.4f val_acc=%.4f | lr=%.2e | %.1fs | RAM=%.0fMB",
                        epoch, cfg.epochs,
                        train_metrics["loss"], val_metrics["loss"],
                        train_metrics["direction_accuracy"],
                        val_metrics["direction_accuracy"],
                        current_lr, elapsed, ram_mb,
                    )

                # Garbage collection every N epochs to reduce memory pressure
                if cfg.gc_every_n_epochs > 0 and epoch % cfg.gc_every_n_epochs == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # RAM limit enforcement
                if cfg.max_ram_mb > 0:
                    current_ram = self._get_process_memory_mb()
                    if current_ram > cfg.max_ram_mb:
                        logger.warning(
                            "RAM limit exceeded: %.0f MB > %d MB (epoch %d). "
                            "Running aggressive GC...",
                            current_ram, cfg.max_ram_mb, epoch,
                        )
                        gc.collect()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        # Re-check after GC
                        current_ram = self._get_process_memory_mb()
                        if current_ram > cfg.max_ram_mb:
                            logger.warning(
                                "RAM still %.0f MB after GC (limit %d MB). "
                                "Training continues but consider reducing batch_size or sequence_length.",
                                current_ram, cfg.max_ram_mb,
                            )

                # Checkpoint / Early stopping
                if val_metrics["loss"] < metrics.best_val_loss:
                    metrics.best_val_loss = val_metrics["loss"]
                    metrics.best_epoch = epoch
                    patience_counter = 0

                    if cfg.save_best_only:
                        self.model.save(
                            save_path,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            loss=val_metrics["loss"],
                        )
                        logger.info("  ✓ Best model saved (val_loss=%.6f)", val_metrics["loss"])
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d, best_epoch=%d, best_loss=%.6f)",
                            epoch, cfg.early_stopping_patience,
                            metrics.best_epoch, metrics.best_val_loss,
                        )
                        break
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Training interrupted by user at epoch %d", metrics.epochs_trained)

        # Save checkpoint on interrupt (if we have at least one epoch of data)
        if interrupted and metrics.epochs_trained > 0:
            try:
                saved = self.model.save(
                    save_path,
                    optimizer=self.optimizer,
                    epoch=metrics.epochs_trained,
                    loss=metrics.train_losses[-1] if metrics.train_losses else 0.0,
                )
                logger.info("Interrupt checkpoint saved to %s", saved)
            except Exception as exc:
                logger.warning("Could not save interrupt checkpoint: %s", exc)

        logger.info(
            "Training complete: %d epochs, best_val_loss=%.6f at epoch %d",
            metrics.epochs_trained, metrics.best_val_loss, metrics.best_epoch,
        )

        return metrics

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate on test set with detailed metrics."""
        self.model.eval()

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_confidences: list[torch.Tensor] = []
        all_magnitudes_pred: list[torch.Tensor] = []
        all_magnitudes_true: list[torch.Tensor] = []

        total_loss = 0.0
        total_samples = 0

        for X, y_dir, y_mag, y_conf in test_loader:
            X = X.to(self.device)
            y_dir = y_dir.to(self.device)
            y_mag = y_mag.to(self.device)
            y_conf = y_conf.to(self.device)

            predictions = self.model(X)

            targets = {
                "direction": y_dir,
                "magnitude": y_mag,
                "confidence": y_conf,
            }
            losses = self.criterion(predictions, targets)

            total_loss += losses["total_loss"].item() * X.size(0)
            total_samples += X.size(0)

            all_preds.append(predictions["direction"].argmax(dim=1).cpu())
            all_targets.append(y_dir.cpu())
            all_confidences.append(predictions["confidence"].cpu())
            all_magnitudes_pred.append(predictions["magnitude"].cpu())
            all_magnitudes_true.append(y_mag.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        confidences = torch.cat(all_confidences)

        # Overall accuracy
        accuracy = (preds == targets).float().mean().item()

        # Per-class accuracy
        per_class: dict[str, float] = {}
        for cls_idx, cls_name in enumerate(["long", "short", "neutral"]):
            mask = targets == cls_idx
            if mask.sum() > 0:
                per_class[f"{cls_name}_accuracy"] = (preds[mask] == targets[mask]).float().mean().item()
                per_class[f"{cls_name}_count"] = mask.sum().item()

        # Confidence calibration: avg confidence for correct vs incorrect
        correct_mask = preds == targets
        avg_conf_correct = confidences[correct_mask].mean().item() if correct_mask.sum() > 0 else 0.0
        avg_conf_incorrect = confidences[~correct_mask].mean().item() if (~correct_mask).sum() > 0 else 0.0

        # Magnitude MAE
        mag_pred = torch.cat(all_magnitudes_pred)
        mag_true = torch.cat(all_magnitudes_true)
        magnitude_mae = (mag_pred - mag_true).abs().mean().item()

        results = {
            "test_loss": total_loss / max(total_samples, 1),
            "test_accuracy": accuracy,
            "magnitude_mae": magnitude_mae,
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_incorrect": avg_conf_incorrect,
            **per_class,
        }

        logger.info(
            "Test results: loss=%.6f, acc=%.4f, mag_mae=%.6f",
            results["test_loss"], results["test_accuracy"], results["magnitude_mae"],
        )

        return results
