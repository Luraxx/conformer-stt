"""Training loop for the STT model."""

import os
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import jiwer

from ..model.model import STTModel
from ..preprocessing.tokenizer import CharTokenizer
from .loss import CTCLoss
from .scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the full training loop for an STT model."""

    def __init__(
        self,
        model: STTModel,
        tokenizer: CharTokenizer,
        config: dict,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        train_cfg = config.get("training", {})

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0001),
        )

        # Loss
        ctc_cfg = config.get("ctc", {})
        self.criterion = CTCLoss(
            blank_id=ctc_cfg.get("blank_id", 0),
            reduction=ctc_cfg.get("reduction", "mean"),
            zero_infinity=ctc_cfg.get("zero_infinity", True),
        )

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", False) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Training params
        self.epochs = train_cfg.get("epochs", 100)
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm", 5.0)
        self.accumulation_steps = train_cfg.get("accumulation_steps", 1)
        self.log_every = train_cfg.get("log_every_n_steps", 100)
        self.save_every = train_cfg.get("save_every_n_epochs", 5)
        self.keep_last_n = train_cfg.get("keep_last_n_checkpoints", 3)
        self.checkpoint_dir = Path(
            train_cfg.get("checkpoint_dir", "models/checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Scheduler (set up after we know total steps)
        self.scheduler = None
        self.warmup_steps = train_cfg.get("warmup_steps", 5000)
        self.min_lr = train_cfg.get("min_lr", 1e-5)

        # Tracking
        self.global_step = 0
        self.best_wer = float("inf")

    def _setup_scheduler(self, total_steps: int):
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
            min_lr=self.min_lr,
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None):
        """Run the full training loop."""
        total_steps = self.epochs * len(train_loader) // self.accumulation_steps
        self._setup_scheduler(total_steps)

        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}")

            # Validation
            if val_loader is not None:
                val_loss, val_wer = self._validate(val_loader)
                logger.info(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"Val Loss: {val_loss:.4f}, Val WER: {val_wer:.2%}"
                )

                # Save best model
                if val_wer < self.best_wer:
                    self.best_wer = val_wer
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"New best WER: {val_wer:.2%}")

            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        logger.info(f"Training complete. Best WER: {self.best_wer:.2%}")

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            features = batch["features"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            token_lengths = batch["token_lengths"].to(self.device)

            # Forward pass with mixed precision
            with torch.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                log_probs, output_lengths = self.model(features, feature_lengths)
                loss = self.criterion(
                    log_probs, tokens, output_lengths, token_lengths
                )
                loss = loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

            if self.global_step % self.log_every == 0 and self.global_step > 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_refs = []
        all_hyps = []

        for batch in val_loader:
            features = batch["features"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            token_lengths = batch["token_lengths"].to(self.device)

            log_probs, output_lengths = self.model(features, feature_lengths)
            loss = self.criterion(log_probs, tokens, output_lengths, token_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Greedy decode for WER
            predicted_ids = log_probs.argmax(dim=-1)  # (B, T)
            for i in range(predicted_ids.size(0)):
                length = output_lengths[i].item()
                pred_tokens = predicted_ids[i, :length].cpu().tolist()
                hyp = self.tokenizer.decode(pred_tokens)
                ref = batch["texts"][i].lower()

                all_refs.append(ref)
                all_hyps.append(hyp)

        avg_loss = total_loss / max(num_batches, 1)

        # Compute WER
        try:
            wer = jiwer.wer(all_refs, all_hyps)
        except Exception:
            wer = 1.0

        return avg_loss, wer

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_wer": self.best_wer,
            "config": self.config,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        # Clean up old checkpoints
        if not is_best:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            old.unlink()

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_wer = checkpoint.get("best_wer", float("inf"))
        logger.info(
            f"Resumed from epoch {checkpoint.get('epoch', '?')}, "
            f"step {self.global_step}, best WER: {self.best_wer:.2%}"
        )
