"""
Model Trainer — Multi-task training loop for the EnsembleModel.

Features:
  - Multi-task loss: Cross-Entropy (direction) + MSE (magnitude) + BCE (confidence)
  - Class imbalance handling:
      * Inverse-frequency class weights in CrossEntropyLoss
      * WeightedRandomSampler for balanced mini-batches
      * Optional label smoothing (regularisation + confidence calibration)
      * Optional Focal Loss (down-weights easy/frequent Neutral examples)
      * Optional Mixup augmentation (interpolates between samples)
  - AdamW optimizer with cosine annealing LR scheduler
  - Early stopping with patience (on val_loss or macro_f1)
  - Train/validation/test split
  - Gradient clipping
  - Best model checkpointing
  - Per-class accuracy + macro-F1 logged every epoch
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
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, DataLoader, TensorDataset, WeightedRandomSampler, random_split,
)

from models.ensemble import EnsembleModel, DIRECTION_LONG, DIRECTION_SHORT, DIRECTION_NEUTRAL

# Use the main bot logger so training output appears in terminal with Rich formatting
from monitoring.logger import get_logger
logger = get_logger("bot_crypto")

# Number of direction classes (Long / Short / Neutral)
_N_CLASSES = 3
_CLASS_NAMES = ["long", "short", "neutral"]


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

    # -----------------------------------------------------------------------
    # Class-imbalance handling
    # -----------------------------------------------------------------------
    use_class_weights: bool = True
    """Compute inverse-frequency class weights and pass to CrossEntropyLoss."""

    use_weighted_sampler: bool = True
    """Use WeightedRandomSampler so every mini-batch is approximately balanced."""

    label_smoothing: float = 0.1
    """Label smoothing epsilon for CrossEntropyLoss. 0 = disabled."""

    focal_loss_gamma: float = 0.0
    """Focal Loss focusing parameter γ. 0 = standard CE; 2 is a good starting value."""

    mixup_alpha: float = 0.0
    """Mixup Beta distribution parameter α. 0 = disabled; 0.4 is a good value."""

    # -----------------------------------------------------------------------
    # Early-stopping metric
    # -----------------------------------------------------------------------
    early_stopping_metric: str = "val_loss"
    """Metric for early stopping / best-model selection. Options: val_loss | macro_f1."""

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
            # Imbalance handling
            use_class_weights=training.get("use_class_weights", True),
            use_weighted_sampler=training.get("use_weighted_sampler", True),
            label_smoothing=training.get("label_smoothing", 0.1),
            focal_loss_gamma=training.get("focal_loss_gamma", 0.0),
            mixup_alpha=training.get("mixup_alpha", 0.0),
            early_stopping_metric=training.get("early_stopping_metric", "val_loss"),
        )


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy / frequent examples (Neutral) so the model focuses
    on hard / rare examples (Long, Short).

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch, n_classes) — raw logits (pre-softmax)
            targets: (batch,) — class indices
        """
        # Standard CE loss (per-sample, unreduced)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # p_t = exp(-CE)  (probability of the true class)
        p_t = torch.exp(-ce)

        # Focal factor
        focal_loss = (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Multi-task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """Combined loss for direction (CE/Focal) + magnitude (MSE) + confidence (BCE).

    The confidence target is derived from whether the direction prediction
    was correct — this teaches the model to be confident when it's right
    and uncertain when it's wrong.
    """

    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        confidence_weight: float = 0.3,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        focal_loss_gamma: float = 0.0,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.confidence_weight = confidence_weight

        # Direction loss: Focal (if gamma > 0) else CrossEntropy with label smoothing
        if focal_loss_gamma > 0:
            self.direction_loss = FocalLoss(
                gamma=focal_loss_gamma,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            logger.info(
                "Direction loss: FocalLoss(gamma=%.1f, label_smoothing=%.2f)",
                focal_loss_gamma, label_smoothing,
            )
        else:
            self.direction_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            logger.info(
                "Direction loss: CrossEntropyLoss(label_smoothing=%.2f, class_weights=%s)",
                label_smoothing,
                "enabled" if class_weights is not None else "disabled",
            )

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
    val_macro_f1: list[float] = field(default_factory=list)
    train_direction_acc: list[float] = field(default_factory=list)
    val_direction_acc: list[float] = field(default_factory=list)
    # Per-class accuracy tracked per epoch (train)
    train_class_acc: list[dict[str, float]] = field(default_factory=list)
    # Per-class accuracy tracked per epoch (val)
    val_class_acc: list[dict[str, float]] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_macro_f1: float = 0.0
    best_epoch: int = 0
    epochs_trained: int = 0


def _compute_macro_f1(preds: torch.Tensor, targets: torch.Tensor, n_classes: int = _N_CLASSES) -> float:
    """Compute macro-averaged F1 score from prediction and target tensors."""
    f1_scores: list[float] = []
    for cls in range(n_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _compute_per_class_acc(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = _N_CLASSES,
) -> dict[str, float]:
    """Return per-class accuracy dict keyed by class name."""
    acc: dict[str, float] = {}
    for cls_idx, cls_name in enumerate(_CLASS_NAMES[:n_classes]):
        mask = targets == cls_idx
        count = mask.sum().item()
        if count > 0:
            acc[f"{cls_name}_acc"] = (preds[mask] == targets[mask]).float().mean().item()
            acc[f"{cls_name}_count"] = count
        else:
            acc[f"{cls_name}_acc"] = 0.0
            acc[f"{cls_name}_count"] = 0
    return acc


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
            label_smoothing=config.label_smoothing,
            focal_loss_gamma=config.focal_loss_gamma,
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
    # Class-weight helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_class_weights(
        direction_labels: np.ndarray,
        n_classes: int = _N_CLASSES,
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights.

        Uses the sklearn convention:
            weight_c = n_samples / (n_classes * count_c)

        Rare classes get a higher weight, pushing the loss to penalise
        mistakes on Long/Short more than on Neutral.

        Args:
            direction_labels: (N,) array of class indices
            n_classes: total number of classes

        Returns:
            FloatTensor of shape (n_classes,)
        """
        n_samples = len(direction_labels)
        weights: list[float] = []
        for cls in range(n_classes):
            count = int((direction_labels == cls).sum())
            if count == 0:
                # Class not present — assign a large weight so it's used if it
                # ever appears (avoids division by zero)
                w = float(n_samples)
            else:
                w = n_samples / (n_classes * count)
            weights.append(w)

        weight_tensor = torch.FloatTensor(weights)

        # Log for transparency
        for cls_idx, (cls_name, w) in enumerate(zip(_CLASS_NAMES[:n_classes], weights)):
            count = int((direction_labels == cls_idx).sum())
            logger.info(
                "Class weight — %s (idx=%d): count=%d (%.1f%%), weight=%.4f",
                cls_name, cls_idx,
                count, 100.0 * count / max(n_samples, 1), w,
            )

        return weight_tensor

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
        batch_size: int = 256,
        use_weighted_sampler: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Convert numpy arrays into train/val/test DataLoaders.

        Args:
            features: (N, seq_len, n_features) — sliding window inputs
            direction_labels: (N,) — class indices 0/1/2
            magnitude_labels: (N, 1) — % price change
            confidence_labels: (N, 1) — 0 or 1
            val_split: fraction for validation
            test_split: fraction for test
            batch_size: samples per batch (larger = faster on CPU/GPU)
            use_weighted_sampler: if True use WeightedRandomSampler for balanced batches

        Returns:
            (train_loader, val_loader, test_loader)
        """
        # Convert to tensors
        X = torch.FloatTensor(features)
        y_dir = torch.LongTensor(direction_labels)
        y_mag = torch.FloatTensor(magnitude_labels).unsqueeze(-1) if magnitude_labels.ndim == 1 else torch.FloatTensor(magnitude_labels)
        y_conf = torch.FloatTensor(confidence_labels).unsqueeze(-1) if confidence_labels.ndim == 1 else torch.FloatTensor(confidence_labels)

        dataset = TensorDataset(X, y_dir, y_mag, y_conf)

        # Split (time-ordered: train first, then val, then test — avoid leakage)
        n = len(dataset)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_val - n_test

        # Use sequential indices so the split is time-ordered, not random
        from torch.utils.data import Subset
        train_ds = Subset(dataset, list(range(n_train)))
        val_ds   = Subset(dataset, list(range(n_train, n_train + n_val)))
        test_ds  = Subset(dataset, list(range(n_train + n_val, n)))

        logger.info(
            "Dataset split: train=%d, val=%d, test=%d (batch_size=%d)",
            n_train, n_val, n_test, batch_size,
        )

        # num_workers=0 avoids subprocess memory duplication on Windows
        # pin_memory speeds up CPU→GPU transfer when CUDA is available
        use_pin = torch.cuda.is_available()
        dl_kwargs: dict[str, Any] = dict(num_workers=0, pin_memory=use_pin, persistent_workers=False)

        # Build sampler for training set to balance class distribution per batch
        if use_weighted_sampler:
            train_labels = direction_labels[:n_train]
            n_samples = len(train_labels)
            # Per-sample weight = inverse of its class frequency
            class_counts = np.bincount(train_labels, minlength=_N_CLASSES).astype(float)
            class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid /0
            sample_weights = torch.FloatTensor(
                [n_samples / (_N_CLASSES * class_counts[lbl]) for lbl in train_labels]
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=n_train,
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,  # mutually exclusive with shuffle=
                drop_last=True,
                **dl_kwargs,
            )
            logger.info(
                "WeightedRandomSampler enabled: class counts train=%s",
                dict(zip(_CLASS_NAMES, class_counts.astype(int).tolist())),
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs
            )

        val_loader  = DataLoader(val_ds,  batch_size=batch_size * 2, shuffle=False, **dl_kwargs)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)

        return train_loader, val_loader, test_loader

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def _mixup_batch(
        self,
        X: torch.Tensor,
        y_dir: torch.Tensor,
        y_mag: torch.Tensor,
        y_conf: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Mixup augmentation to a batch.

        Returns mixed (X, y_a, y_b, lam, y_mag_mixed, y_conf_mixed) where
        the loss should be lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b).
        For magnitude and confidence we simply interpolate.
        """
        lam = float(np.random.beta(alpha, alpha))
        batch_size = X.size(0)
        idx = torch.randperm(batch_size, device=X.device)

        X_mixed = lam * X + (1 - lam) * X[idx]
        y_mag_mixed  = lam * y_mag  + (1 - lam) * y_mag[idx]
        y_conf_mixed = lam * y_conf + (1 - lam) * y_conf[idx]

        return X_mixed, y_dir, y_dir[idx], lam, y_mag_mixed, y_conf_mixed

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        cfg = self.config
        total_loss = 0.0
        total_samples = 0

        # Accumulators for per-class tracking
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        use_mixup = cfg.mixup_alpha > 0

        for batch_idx, (X, y_dir, y_mag, y_conf) in enumerate(loader):
            X = X.to(self.device)
            y_dir  = y_dir.to(self.device)
            y_mag  = y_mag.to(self.device)
            y_conf = y_conf.to(self.device)

            if use_mixup:
                X, y_a, y_b, lam, y_mag, y_conf = self._mixup_batch(
                    X, y_dir, y_mag, y_conf, cfg.mixup_alpha
                )
                predictions = self.model(X)

                # Mixup: loss = λ*L(y_a) + (1−λ)*L(y_b)
                losses_a = self.criterion(predictions, {"direction": y_a, "magnitude": y_mag, "confidence": y_conf})
                losses_b = self.criterion(predictions, {"direction": y_b, "magnitude": y_mag, "confidence": y_conf})
                total_batch_loss = lam * losses_a["total_loss"] + (1 - lam) * losses_b["total_loss"]

                # For accuracy tracking use the primary label (y_a)
                pred_dir = predictions["direction"].argmax(dim=1)
                all_preds.append(pred_dir.detach().cpu())
                all_targets.append(y_a.detach().cpu())
            else:
                predictions = self.model(X)
                targets = {
                    "direction": y_dir,
                    "magnitude": y_mag,
                    "confidence": y_conf,
                }
                losses = self.criterion(predictions, targets)
                total_batch_loss = losses["total_loss"]

                pred_dir = predictions["direction"].argmax(dim=1)
                all_preds.append(pred_dir.detach().cpu())
                all_targets.append(y_dir.detach().cpu())

            # Backward
            self.optimizer.zero_grad()
            total_batch_loss.backward()

            # Gradient clipping
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += total_batch_loss.item() * X.size(0)
            total_samples += X.size(0)

        avg_loss = total_loss / max(total_samples, 1)

        preds_all   = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)
        accuracy = (preds_all == targets_all).float().mean().item()
        per_class = _compute_per_class_acc(preds_all, targets_all)

        result = {"loss": avg_loss, "direction_accuracy": accuracy}
        result.update(per_class)
        return result

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate on a data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for X, y_dir, y_mag, y_conf in loader:
            X = X.to(self.device)
            y_dir  = y_dir.to(self.device)
            y_mag  = y_mag.to(self.device)
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

            pred_dir = predictions["direction"].argmax(dim=1)
            all_preds.append(pred_dir.cpu())
            all_targets.append(y_dir.cpu())

        avg_loss = total_loss / max(total_samples, 1)

        preds_all   = torch.cat(all_preds)
        targets_all = torch.cat(all_targets)
        accuracy  = (preds_all == targets_all).float().mean().item()
        macro_f1  = _compute_macro_f1(preds_all, targets_all)
        per_class = _compute_per_class_acc(preds_all, targets_all)

        result = {
            "loss": avg_loss,
            "direction_accuracy": accuracy,
            "macro_f1": macro_f1,
        }
        result.update(per_class)
        return result

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

        # Decide which metric to minimise/maximise for early stopping
        use_f1_stopping = cfg.early_stopping_metric == "macro_f1"

        logger.info(
            "Starting training: epochs=%d, batch_size=%d, lr=%s, "
            "early_stopping_metric=%s, label_smoothing=%.2f, focal_gamma=%.1f",
            cfg.epochs, cfg.batch_size, cfg.learning_rate,
            cfg.early_stopping_metric, cfg.label_smoothing, cfg.focal_loss_gamma,
        )
        print(
            f"Starting training: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
            f"lr={cfg.learning_rate}, stop_metric={cfg.early_stopping_metric}",
            flush=True,
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
                metrics.val_macro_f1.append(val_metrics["macro_f1"])
                metrics.train_direction_acc.append(train_metrics["direction_accuracy"])
                metrics.val_direction_acc.append(val_metrics["direction_accuracy"])
                metrics.train_class_acc.append({k: train_metrics[k] for k in train_metrics if k.endswith("_acc")})
                metrics.val_class_acc.append({k: val_metrics[k] for k in val_metrics if k.endswith("_acc")})
                metrics.epochs_trained = epoch

                # Per-class accuracy strings
                tr_cls = " ".join(
                    f"{cls}={train_metrics.get(f'{cls}_acc', 0.0):.3f}"
                    for cls in _CLASS_NAMES
                )
                vl_cls = " ".join(
                    f"{cls}={val_metrics.get(f'{cls}_acc', 0.0):.3f}"
                    for cls in _CLASS_NAMES
                )

                # Logging (every epoch)
                ram_mb = self._get_process_memory_mb()
                epoch_msg = (
                    f"Epoch {epoch}/{cfg.epochs} | "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"val_loss={val_metrics['loss']:.6f} | "
                    f"train_acc={train_metrics['direction_accuracy']:.4f} "
                    f"val_acc={val_metrics['direction_accuracy']:.4f} | "
                    f"val_f1={val_metrics['macro_f1']:.4f} | "
                    f"train[{tr_cls}] | val[{vl_cls}] | "
                    f"lr={current_lr:.2e} | {elapsed:.1f}s | RAM={ram_mb:.0f}MB"
                )
                logger.info(epoch_msg)
                print(epoch_msg, flush=True)

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
                        current_ram = self._get_process_memory_mb()
                        if current_ram > cfg.max_ram_mb:
                            logger.warning(
                                "RAM still %.0f MB after GC (limit %d MB). "
                                "Training continues but consider reducing batch_size or sequence_length.",
                                current_ram, cfg.max_ram_mb,
                            )

                # -----------------------------------------------------------------
                # Best-model checkpoint / early stopping
                # -----------------------------------------------------------------
                if use_f1_stopping:
                    # Higher macro_f1 = better
                    current_metric = val_metrics["macro_f1"]
                    is_best = current_metric > metrics.best_macro_f1
                    if is_best:
                        metrics.best_macro_f1 = current_metric
                        # Also track corresponding val loss for reference
                        metrics.best_val_loss = val_metrics["loss"]
                else:
                    # Lower val_loss = better
                    current_metric = val_metrics["loss"]
                    is_best = current_metric < metrics.best_val_loss
                    if is_best:
                        metrics.best_val_loss = current_metric

                if is_best:
                    metrics.best_epoch = epoch
                    patience_counter = 0

                    if cfg.save_best_only:
                        self.model.save(
                            save_path,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            loss=val_metrics["loss"],
                        )
                        logger.info(
                            "  ✓ Best model saved (val_loss=%.6f, macro_f1=%.4f)",
                            val_metrics["loss"], val_metrics["macro_f1"],
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d, best_epoch=%d, "
                            "best_val_loss=%.6f, best_macro_f1=%.4f)",
                            epoch, cfg.early_stopping_patience,
                            metrics.best_epoch, metrics.best_val_loss, metrics.best_macro_f1,
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
            "Training complete: %d epochs, best_val_loss=%.6f, best_macro_f1=%.4f at epoch %d",
            metrics.epochs_trained, metrics.best_val_loss, metrics.best_macro_f1, metrics.best_epoch,
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
            y_dir  = y_dir.to(self.device)
            y_mag  = y_mag.to(self.device)
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

        preds   = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        confidences = torch.cat(all_confidences)

        # Overall accuracy
        accuracy = (preds == targets_t).float().mean().item()

        # Macro-F1
        macro_f1 = _compute_macro_f1(preds, targets_t)

        # Per-class accuracy + counts
        per_class = _compute_per_class_acc(preds, targets_t)

        # Confidence calibration: avg confidence for correct vs incorrect
        correct_mask = preds == targets_t
        avg_conf_correct   = confidences[correct_mask].mean().item()  if correct_mask.sum() > 0 else 0.0
        avg_conf_incorrect = confidences[~correct_mask].mean().item() if (~correct_mask).sum() > 0 else 0.0

        # Magnitude MAE
        mag_pred = torch.cat(all_magnitudes_pred)
        mag_true = torch.cat(all_magnitudes_true)
        magnitude_mae = (mag_pred - mag_true).abs().mean().item()

        # Log class distribution of test set
        for cls_idx, cls_name in enumerate(_CLASS_NAMES):
            count = int((targets_t == cls_idx).sum().item())
            logger.info(
                "  Test class dist — %s: %d (%.1f%%)",
                cls_name, count, 100.0 * count / max(total_samples, 1),
            )

        results = {
            "test_loss": total_loss / max(total_samples, 1),
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
            "magnitude_mae": magnitude_mae,
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_incorrect": avg_conf_incorrect,
            **per_class,
        }

        logger.info(
            "Test results: loss=%.6f, acc=%.4f, macro_f1=%.4f, mag_mae=%.6f",
            results["test_loss"], results["test_accuracy"],
            results["test_macro_f1"], results["magnitude_mae"],
        )

        return results
