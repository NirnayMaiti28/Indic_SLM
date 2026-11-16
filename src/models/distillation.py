"""Loss functions: class-weighted CE, focal loss, and the distillation objective.

Every optimization the brief lists is a *separate* experiment; nothing is stacked
silently. `build_loss` therefore returns one named criterion at a time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Lin et al. (2017) focal loss with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def inverse_frequency_weights(labels: list[str], label_order: list[str]) -> torch.Tensor:
    """w_c = N / (K * n_c) - the standard balanced weighting."""
    from collections import Counter

    counts = Counter(labels)
    n = len(labels)
    k = len(label_order)
    return torch.tensor(
        [n / (k * max(1, counts.get(lbl, 0))) for lbl in label_order], dtype=torch.float32
    )


def build_loss(kind: str, class_weights: torch.Tensor | None = None, gamma: float = 2.0) -> nn.Module:
    kind = kind.lower()
    if kind in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(weight=class_weights)
    if kind == "focal":
        return FocalLoss(gamma=gamma, weight=class_weights)
    if kind == "ce_none":
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown loss: {kind}")


class DistillationLoss(nn.Module):
    """L = alpha * CE(student, gold) + beta * T^2 * KL(teacher || student).

    The T^2 factor restores the gradient magnitude that dividing by T removes
    (Hinton et al., 2015). Teacher logits are detached: the teacher is a fixed
    target, never updated by the student's objective.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, beta: float = 0.5,
                 class_weights: torch.Tensor | None = None, hard_loss: str = "ce"):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ce = build_loss(hard_loss, class_weights)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> dict[str, torch.Tensor]:
        ce = self.ce(student_logits, targets)
        soft_teacher = F.log_softmax(teacher_logits.detach() / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean", log_target=True)
        kl = kl * (self.T ** 2)
        total = self.alpha * ce + self.beta * kl
        return {"loss": total, "ce": ce.detach(), "kl": kl.detach()}


def feature_distillation_loss(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor,
                              student_mask: torch.Tensor) -> torch.Tensor:
    """Optional intermediate-representation KD (FitNets-style), masked mean MSE.

    Only valid when student and teacher share a hidden width; the caller must
    check that, and the trainer records it as a separate ablation.
    """
    if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
        raise ValueError(
            f"hidden mismatch {student_hidden.shape[-1]} vs {teacher_hidden.shape[-1]}; "
            "feature KD needs a projection layer, which this implementation does not add"
        )
    mask = student_mask.unsqueeze(-1).to(student_hidden.dtype)
    diff = (student_hidden - teacher_hidden.detach()) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1.0) / student_hidden.shape[-1]
