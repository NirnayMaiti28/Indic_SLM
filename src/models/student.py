"""Small Language Model (SLM) Architecture for Indic Climate Stance Detection.

This module provides the compact, multilingual student architecture:
Input text
  ↓
Unicode normalization
  ↓
Multilingual tokenizer (MuRIL / IndicBERTv2 shared vocabulary)
  ↓
Compact transformer encoder (layer-pruned or lightweight transformer)
  ↓
Pooling layer (CLS, Mean, Attention)
  ↓
Dropout
  ↓
Classification head
  ↓
Logits (FAVOUR, AGAINST, NONE)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .classification_head import StanceHead, build_pooling

LABELS = ["FAVOUR", "AGAINST", "NONE"]


@dataclass
class StudentConfig:
    encoder_name: str = "google/muril-base-cased"
    num_hidden_layers: int = 4
    hidden_size: int = 768
    num_labels: int = 3
    pooling: str = "mean"  # cls | mean | attention
    dropout: float = 0.1
    max_length: int = 64
    label_list: List[str] = None

    def __post_init__(self):
        if self.label_list is None:
            self.label_list = LABELS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StudentSLM(nn.Module):
    """Compact Student Small Language Model."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        self.encoder_name = config.encoder_name
        self.pooling_type = config.pooling
        self.num_labels = config.num_labels

        # Load transformer config and customize layers
        self.hf_config = AutoConfig.from_pretrained(config.encoder_name)
        if config.num_hidden_layers is not None:
            self.hf_config.num_hidden_layers = config.num_hidden_layers

        # Load base encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_name, config=self.hf_config, use_safetensors=True)
        
        # If pruning full model layers to target layer count
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            total_layers = len(self.encoder.encoder.layer)
            if config.num_hidden_layers < total_layers:
                self.encoder.encoder.layer = nn.ModuleList(
                    list(self.encoder.encoder.layer)[:config.num_hidden_layers]
                )
                self.hf_config.num_hidden_layers = config.num_hidden_layers

        hidden_dim = getattr(self.hf_config, "hidden_size", config.hidden_size)
        self.head = StanceHead(
            hidden_size=hidden_dim,
            num_labels=config.num_labels,
            dropout=config.dropout,
            pooling=config.pooling
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        encoder_out = self.encoder(**kwargs, output_hidden_states=output_hidden_states)
        last_hidden = encoder_out.last_hidden_state
        logits = self.head(last_hidden, attention_mask)

        out = {"logits": logits}
        if output_hidden_states:
            out["hidden_states"] = encoder_out.hidden_states
        if labels is not None:
            out["loss"] = nn.CrossEntropyLoss()(logits, labels)

        return out

    def get_parameter_summary(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.encoder.embeddings.parameters()) if hasattr(self.encoder, "embeddings") else 0
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": emb_params,
            "head_parameters": head_params,
            "layers": self.hf_config.num_hidden_layers,
            "pooling": self.pooling_type,
            "hidden_size": getattr(self.hf_config, "hidden_size", 768),
        }

    def save_pretrained(self, save_dir: str | Path, tokenizer=None, metadata: Optional[Dict] = None):
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict(),
                "hf_config": self.hf_config.to_dict(),
            },
            p / "student_model.pt"
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(p)
        if metadata:
            (p / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_student_slm(
    encoder_name: str = "google/muril-base-cased",
    num_hidden_layers: int = 4,
    pooling: str = "mean",
    dropout: float = 0.1,
    num_labels: int = 3,
) -> StudentSLM:
    """Build compact multilingual student SLM."""
    cfg = StudentConfig(
        encoder_name=encoder_name,
        num_hidden_layers=num_hidden_layers,
        pooling=pooling,
        dropout=dropout,
        num_labels=num_labels
    )
    return StudentSLM(cfg)


def load_student_slm(model_dir: str | Path, map_location: str = "cpu") -> StudentSLM:
    """Load saved student SLM from disk."""
    p = Path(model_dir)
    ckpt = torch.load(p / "student_model.pt", map_location=map_location, weights_only=False)
    cfg = StudentConfig(**ckpt["config"])
    model = StudentSLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model
