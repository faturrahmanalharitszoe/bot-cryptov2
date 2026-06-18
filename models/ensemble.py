"""
Ensemble Model — Combines CNN-LSTM and Transformer branches with a multi-task head.

Architecture:
  1. CNN-LSTM branch → embedding (batch, cnn_lstm_dim)
  2. Transformer branch → embedding (batch, d_model)
  3. Concatenate → (batch, combined_dim)
  4. Dense 128 + ReLU → Dropout → Dense 64 + ReLU → Dropout
  5. Three output heads:
     - Direction:    Linear(64, 3) → Softmax  (Long / Short / Neutral)
     - Magnitude:    Linear(64, 1)            (expected % price move)
     - Confidence:   Linear(64, 1) → Sigmoid  (model confidence 0-1)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_lstm import CNNLSTMModel
from models.transformer_branch import TransformerModel

logger = logging.getLogger(__name__)

# Direction class indices
DIRECTION_LONG = 0
DIRECTION_SHORT = 1
DIRECTION_NEUTRAL = 2
DIRECTION_LABELS = ["Long", "Short", "Neutral"]


class EnsembleHead(nn.Module):
    """Multi-task prediction head that sits on top of concatenated branch embeddings."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.shared = nn.Sequential(*layers)

        # Three output heads
        self.direction_head = nn.Linear(in_dim, 3)   # 3-class softmax
        self.magnitude_head = nn.Linear(in_dim, 1)    # regression
        self.confidence_head = nn.Linear(in_dim, 1)   # sigmoid

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) — concatenated branch embeddings
        Returns:
            dict with keys:
              - direction:   (batch, 3) — softmax probabilities
              - magnitude:   (batch, 1) — predicted % move
              - confidence:  (batch, 1) — confidence 0-1
        """
        shared = self.shared(x)

        direction_logits = self.direction_head(shared)
        direction = F.softmax(direction_logits, dim=-1)

        magnitude = self.magnitude_head(shared)

        confidence = torch.sigmoid(self.confidence_head(shared))

        return {
            "direction": direction,
            "direction_logits": direction_logits,
            "magnitude": magnitude,
            "confidence": confidence,
        }


class EnsembleModel(nn.Module):
    """Full ensemble model: CNN-LSTM + Transformer → Multi-task Head.

    This is the main model used for training and inference.
    """

    def __init__(
        self,
        in_features: int,
        # CNN-LSTM params
        cnn_filters: list[int] | None = None,
        cnn_kernel_sizes: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_attention: bool = True,
        # Transformer params
        transformer_d_model: int = 128,
        transformer_nhead: int = 8,
        transformer_layers: int = 4,
        transformer_ff_dim: int = 256,
        # Shared params
        dropout: float = 0.3,
        transformer_dropout: float = 0.1,
        max_len: int = 500,
    ):
        # Store all architecture params for checkpointing
        self._arch_config = {
            "in_features": in_features,
            "cnn_filters": cnn_filters,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "lstm_bidirectional": lstm_bidirectional,
            "lstm_attention": lstm_attention,
            "transformer_d_model": transformer_d_model,
            "transformer_nhead": transformer_nhead,
            "transformer_layers": transformer_layers,
            "transformer_ff_dim": transformer_ff_dim,
            "dropout": dropout,
            "transformer_dropout": transformer_dropout,
            "max_len": max_len,
        }
        super().__init__()

        # --- Branch 1: CNN-LSTM ---
        self.cnn_lstm = CNNLSTMModel(
            in_features=in_features,
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_bidirectional=lstm_bidirectional,
            lstm_attention=lstm_attention,
            dropout=dropout,
        )

        # --- Branch 2: Transformer ---
        self.transformer = TransformerModel(
            in_features=in_features,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            max_len=max_len,
        )

        # --- Ensemble Head ---
        combined_dim = self.cnn_lstm.embedding_dim + self.transformer.embedding_dim
        self.head = EnsembleHead(input_dim=combined_dim, dropout=dropout)

        # Store dimensions for reference
        self.in_features = in_features
        self.combined_dim = combined_dim

        logger.info(
            "EnsembleModel initialized: in_features=%d, cnn_lstm_emb=%d, transformer_emb=%d, combined=%d",
            in_features,
            self.cnn_lstm.embedding_dim,
            self.transformer.embedding_dim,
            combined_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, in_features) — input window
            src_key_padding_mask: optional padding mask for transformer
        Returns:
            dict with keys: direction, direction_logits, magnitude, confidence
        """
        # Get embeddings from both branches
        cnn_lstm_emb = self.cnn_lstm(x)                    # (B, cnn_lstm_dim)
        transformer_emb = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, d_model)

        # Concatenate
        combined = torch.cat([cnn_lstm_emb, transformer_emb], dim=1)  # (B, combined_dim)

        # Multi-task head
        return self.head(combined)

    def predict_direction(self, x: torch.Tensor) -> tuple[int, float]:
        """Convenience: returns (direction_index, confidence) for a single sample.

        Args:
            x: (1, seq_len, in_features) or (seq_len, in_features)
        Returns:
            (direction_idx, confidence_scalar)
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(x)

        direction_probs = out["direction"][0]  # (3,)
        confidence = out["confidence"][0, 0].item()
        direction_idx = direction_probs.argmax().item()

        return direction_idx, confidence

    def save(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        loss: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        keep_last: int = 5,
    ) -> Path:
        """Save model checkpoint.

        Strategy:
        1. Save to a timestamped file (``ckpt_YYYYMMDD_HHMMSS.pth``) which
           avoids conflicts with locked files.
        2. Attempt to also save/copy as the requested ``path`` (e.g.
           ``ensemble_best.pth``).  On Windows, antivirus or file-locking
           can make this fail, so we catch the error gracefully.
        3. Always return the path to the successfully saved checkpoint.

        Returns:
            Path to the saved checkpoint file (may differ from *path* if the
            alias copy failed).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": self.state_dict(),
            "in_features": self.in_features,
            "combined_dim": self.combined_dim,
            "arch_config": self._arch_config,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Always save a uniquely-named checkpoint first (avoids locked-file issues).
        # Use .pt extension (not .pth) because Windows Defender / Controlled
        # Folder Access may block .pth writes while allowing .pt through.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"ckpt_{ts}_e{epoch}.pt"
        unique_path = path.parent / unique_name

        saved_path = None

        # Save via BytesIO buffer first, then write atomically to disk.
        # This prevents Windows Defender real-time scanning from corrupting
        # the zip file mid-write (the root cause of inline_container errors).
        import io

        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        buf_bytes = buf.getvalue()
        buf.close()

        # Attempt 1: write to the unique path.
        for attempt in range(1, max_retries + 1):
            try:
                with open(unique_path, "wb") as f:
                    f.write(buf_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                saved_path = unique_path
                logger.info(
                    "Checkpoint saved to %s (epoch=%d, loss=%.6f)",
                    unique_path, epoch, loss,
                )
                break
            except (OSError, RuntimeError) as exc:
                logger.warning(
                    "Save attempt %d/%d to %s failed: %s",
                    attempt, max_retries, unique_path, exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        # Attempt 1b: fallback to a temp directory if the primary path is
        # blocked (e.g. Windows Defender Controlled Folder Access).
        if saved_path is None:
            import tempfile
            fallback_dir = Path(tempfile.gettempdir()) / "bot-cryptov2-checkpoints"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / unique_name
            try:
                with open(fallback_path, "wb") as f:
                    f.write(buf_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                saved_path = fallback_path
                logger.warning(
                    "Primary save blocked — checkpoint saved to fallback: %s",
                    fallback_path,
                )
            except (OSError, RuntimeError) as exc:
                logger.error("Fallback save to %s also failed: %s", fallback_path, exc)

        if saved_path is None:
            # All retries exhausted — raise so the caller knows training
            # cannot continue.
            raise OSError(
                f"Failed to save checkpoint after {max_retries} attempts "
                f"(including fallback). Check directory permissions: {path.parent}"
            )

        # Attempt 2: create / overwrite the "canonical" alias (e.g.
        # ensemble_best.pt).  This is best-effort; training already
        # succeeded and the unique checkpoint is safe.
        if saved_path != path:
            try:
                # If the target is a phantom directory created by Windows
                # Defender, try to remove it.  If that also fails (locked),
                # skip the alias entirely — the timestamped checkpoint is
                # the authoritative copy.
                if path.exists():
                    if path.is_dir():
                        try:
                            shutil.rmtree(path)
                            logger.info("Removed phantom directory (Windows Defender): %s", path)
                        except OSError:
                            logger.info(
                                "Skipping canonical alias — Windows Defender "
                                "protected directory: %s", path,
                            )
                            # Do NOT fall through to copy; the directory is locked.
                            path = None  # signal to skip
                    else:
                        path.unlink()

                if path is not None:
                    shutil.copy2(saved_path, path)
                    logger.info("Canonical alias updated: %s", path)
            except Exception as exc:
                logger.warning(
                    "Could not update canonical alias %s (non-fatal): %s",
                    path, exc,
                )

        # Attempt 3: rotate old checkpoints to prevent disk fill-up.
        # Keep only the ``keep_last`` most recent ckpt_*.pt files in
        # both the primary save directory and the temp fallback directory.
        self._rotate_checkpoints(path.parent, keep_last)
        import tempfile as _tf
        fallback_dir = Path(_tf.gettempdir()) / "bot-cryptov2-checkpoints"
        if fallback_dir.is_dir() and fallback_dir.resolve() != path.parent.resolve():
            self._rotate_checkpoints(fallback_dir, keep_last)

        return saved_path

    @staticmethod
    def _rotate_checkpoints(directory: Path, keep_last: int = 5) -> None:
        """Delete old ``ckpt_*.pt`` / ``ckpt_*.pth`` files, keeping the *keep_last* newest.

        Never deletes the canonical alias (``ensemble_best.*``).
        """
        if keep_last <= 0:
            return
        candidates: list[Path] = []
        for pat in ("ckpt_*.pt", "ckpt_*.pth"):
            candidates.extend(directory.glob(pat))
        if len(candidates) <= keep_last:
            return
        # Sort newest-first by modification time.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        to_delete = candidates[keep_last:]
        for old in to_delete:
            try:
                old.unlink()
                logger.info("Rotated out old checkpoint: %s", old)
            except OSError as exc:
                logger.warning("Could not delete old checkpoint %s: %s", old, exc)

    @staticmethod
    def find_latest_checkpoint(directory: str | Path) -> Path | None:
        """Find the most recent checkpoint file in *directory* (and fallback temp dir).

        Searches for files matching ``ckpt_*`` or ``ensemble_best.*`` with
        extensions ``.pt`` / ``.pth``.  Also searches the system temp
        directory fallback used when Windows Defender blocks the primary path.
        Returns the path with the newest modification time, or ``None`` if
        nothing is found.
        """
        import tempfile as _tf

        candidates: list[Path] = []
        search_dirs: list[Path] = [Path(directory)]

        # Also check the temp fallback directory.
        fallback = Path(_tf.gettempdir()) / "bot-cryptov2-checkpoints"
        if fallback.is_dir() and fallback.resolve() != Path(directory).resolve():
            search_dirs.append(fallback)

        _MIN_CHECKPOINT_BYTES = 1_000_000  # 1 MB — valid checkpoint threshold

        for d in search_dirs:
            if not d.is_dir():
                continue
            for pat in ("ckpt_*.pt", "ckpt_*.pth", "ensemble_best.pt", "ensemble_best.pth"):
                for p in d.glob(pat):
                    # Filter out phantom directories (Windows Defender) and
                    # corrupted partial writes (disk-full saves < 1 MB).
                    if p.is_file() and p.stat().st_size >= _MIN_CHECKPOINT_BYTES:
                        candidates.append(p)

        if not candidates:
            return None

        # Return the most recently modified file.
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        **model_kwargs: Any,
    ) -> tuple["EnsembleModel", dict[str, Any]]:
        """Load model from checkpoint.

        Args:
            path: path to .pt / .pth checkpoint.  If the exact file does not
                  exist, the parent directory is searched for the latest
                  checkpoint (``ckpt_*`` or ``ensemble_best.*``).
            device: torch device
            **model_kwargs: kwargs passed to EnsembleModel.__init__

        Returns:
            (model, checkpoint_dict)
        """
        path = Path(path)

        # If the exact path doesn't exist, search for the latest checkpoint.
        if not path.exists():
            resolved = cls.find_latest_checkpoint(path.parent)
            if resolved is None:
                raise FileNotFoundError(
                    f"No checkpoint found at {path} and none in {path.parent}"
                )
            logger.info(
                "Checkpoint %s not found; using latest: %s", path, resolved
            )
            path = resolved

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Restore architecture config from checkpoint (if saved).
        # Checkpoints saved before arch_config was added will only have
        # in_features — fall back to caller-supplied kwargs in that case.
        if "arch_config" in checkpoint:
            saved_arch = checkpoint["arch_config"]
            # Merge: saved_arch is the source of truth, but allow explicit
            # model_kwargs from the caller to override (e.g. in tests).
            merged = {**saved_arch, **model_kwargs}
            model_kwargs = merged
            logger.info(
                "Restored arch_config from checkpoint: in_features=%s, cnn_filters=%s, "
                "lstm_hidden=%s, d_model=%s",
                saved_arch.get("in_features"),
                saved_arch.get("cnn_filters"),
                saved_arch.get("lstm_hidden"),
                saved_arch.get("transformer_d_model"),
            )
        else:
            # Backward compat: old checkpoints only saved in_features
            if "in_features" in checkpoint and "in_features" not in model_kwargs:
                model_kwargs["in_features"] = checkpoint["in_features"]
            logger.warning(
                "Checkpoint lacks arch_config — using caller kwargs/defaults. "
                "Consider re-training to produce a checkpoint with full arch info."
            )

        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info("Model loaded from %s (epoch=%d)", path, checkpoint.get("epoch", 0))
        return model, checkpoint
