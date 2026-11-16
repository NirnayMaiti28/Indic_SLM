"""Reusable training loop: class weighting, grad accumulation, early stopping,
warmup + linear decay, gradient clipping, and optional soft-target distillation.

Checkpoint selection is on validation macro-F1 (the project's primary metric),
never on training loss.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LABELS = ["FAVOUR", "AGAINST", "NONE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class StanceDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 64, text_key: str = "text"):
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length
        self.text_key = text_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        enc = self.tok(str(r[self.text_key]), truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(LABEL2ID[str(r["label"])], dtype=torch.long)
        return item


@dataclass
class TrainConfig:
    lr: float = 2e-5
    epochs: int = 6
    batch_size: int = 8
    grad_accum: int = 4
    max_length: int = 64
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    patience: int = 2
    seed: int = 42
    loss: str = "ce"
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    temperature: float = 2.0
    alpha: float = 0.5
    beta: float = 0.5
    max_steps: int | None = None
    log_every: int = 10


@dataclass
class TrainHistory:
    epochs: list[dict] = field(default_factory=list)
    best_val_macro_f1: float = -1.0
    best_epoch: int = -1
    train_seconds: float = 0.0
    steps: int = 0

    def to_dict(self) -> dict:
        return {"epochs": self.epochs, "best_val_macro_f1": self.best_val_macro_f1,
                "best_epoch": self.best_epoch, "train_seconds": round(self.train_seconds, 2),
                "steps": self.steps}


def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: str | None = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    preds, golds, probs = [], [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out["logits"]
        p = torch.softmax(logits, dim=-1)
        probs.append(p.cpu().numpy())
        preds.extend(p.argmax(dim=-1).cpu().tolist())
        golds.extend(labels.cpu().tolist())
    y_true = [ID2LABEL[i] for i in golds]
    y_pred = [ID2LABEL[i] for i in preds]
    from sklearn.metrics import accuracy_score

    return {
        "macro_f1": _macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.concatenate(probs, axis=0) if probs else np.zeros((0, 3)),
    }


def train_model(
    model: nn.Module,
    train_records: list[dict],
    val_records: list[dict],
    tokenizer,
    config: TrainConfig,
    teacher=None,
    save_dir: str | Path | None = None,
    verbose: bool = True,
    device: str | None = None,
) -> tuple[nn.Module, TrainHistory]:
    torch.manual_seed(config.seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[*] Training on device: {device.upper()}")

    model.to(device)
    if teacher is not None:
        teacher.to(device)
        teacher.eval()

    train_ds = StanceDataset(train_records, tokenizer, config.max_length)
    val_ds = StanceDataset(val_records, tokenizer, config.max_length)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size * 2, shuffle=False)

    class_weights = None
    if config.use_class_weights:
        from .distillation import inverse_frequency_weights

        class_weights = inverse_frequency_weights([r["label"] for r in train_records], LABELS).to(device)

    if teacher is not None:
        from .distillation import DistillationLoss

        criterion = DistillationLoss(config.temperature, config.alpha, config.beta,
                                     class_weights, hard_loss=config.loss).to(device)
    else:
        from .distillation import build_loss

        criterion = build_loss(config.loss, class_weights, config.focal_gamma).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    total_steps = max(1, math.ceil(len(train_loader) / config.grad_accum) * config.epochs)
    if config.max_steps:
        total_steps = min(total_steps, config.max_steps)
    warmup = int(config.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = TrainHistory()
    best_state = None
    t_start = time.time()
    global_step = 0
    stop = False

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out["logits"]

            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(**batch)["logits"]
                parts = criterion(logits, t_logits, labels)
                loss = parts["loss"]
            else:
                loss = criterion(logits, labels)

            (loss / config.grad_accum).backward()
            running += float(loss.detach())
            n_batches += 1

            if (i + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1
                history.steps = global_step
                if verbose and global_step % config.log_every == 0:
                    print(f"  epoch {epoch+1} step {global_step}/{total_steps} loss {running/max(1,n_batches):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}")
                if config.max_steps and global_step >= config.max_steps:
                    stop = True
                    break

        val = evaluate_model(model, val_loader, device=device)
        rec = {"epoch": epoch + 1, "train_loss": round(running / max(1, n_batches), 4),
               "val_macro_f1": round(val["macro_f1"], 4), "val_accuracy": round(val["accuracy"], 4),
               "steps": global_step}
        history.epochs.append(rec)
        if verbose:
            print(f"epoch {epoch+1}: train_loss={rec['train_loss']:.4f} val_macro_f1={val['macro_f1']:.4f} "
                  f"val_acc={val['accuracy']:.4f}")

        if val["macro_f1"] > history.best_val_macro_f1:
            history.best_val_macro_f1 = val["macro_f1"]
            history.best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - history.best_epoch >= config.patience:
            if verbose:
                print(f"early stop at epoch {epoch+1} (best {history.best_epoch})")
            break

        if stop:
            break

    history.train_seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "train_history.json").write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")

    return model, history
