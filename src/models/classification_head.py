"""Classification heads and pooling strategies for stance detection.

Pooling is compared experimentally (CLS vs mean vs attention), not assumed:
social-media Indic text is short and clause-final in stance cues, which is a
hypothesis that mean pooling can beat the CLS token -- but only measurement
decides.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


class CLSPooling(nn.Module):
    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return last_hidden[:, 0]


class AttentionPooling(nn.Module):
    """Additive attention over tokens, masked so padding cannot attract weight."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.proj(last_hidden).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (last_hidden * weights).sum(dim=1)


def build_pooling(kind: str, hidden_size: int) -> nn.Module:
    kind = kind.lower()
    if kind == "cls":
        return CLSPooling()
    if kind == "mean":
        return MeanPooling()
    if kind == "attention":
        return AttentionPooling(hidden_size)
    raise ValueError(f"unknown pooling: {kind}")


class StanceHead(nn.Module):
    """pool -> dropout -> linear -> logits."""

    def __init__(self, hidden_size: int, num_labels: int = 3, dropout: float = 0.1, pooling: str = "cls"):
        super().__init__()
        self.pooling = build_pooling(pooling, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooling(last_hidden, attention_mask)
        return self.classifier(self.dropout(pooled))
