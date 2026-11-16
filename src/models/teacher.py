"""Teacher and student model factories.

Teacher: best-screened multilingual encoder (Phase 8 ranking) with a stance head.
Student: width-preserving, depth-reduced variant of the *same* encoder so the
student inherits the teacher's tokenizer -- Phase 8 showed that a DistilmBERT
student would inherit a tokenizer with 31 % truncation and 1.8 % UNK, which is a
capacity-independent handicap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .classification_head import StanceHead

LABELS = ["FAVOUR", "AGAINST", "NONE"]


class StanceModel(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int = 3, dropout: float = 0.1,
                 pooling: str = "cls", num_hidden_layers: int | None = None):
        super().__init__()
        self.encoder_name = encoder_name
        self.config = AutoConfig.from_pretrained(encoder_name)
        if num_hidden_layers is not None:
            self.config.num_hidden_layers = num_hidden_layers
            self.config.pruned_layers = list(range(num_hidden_layers, 12)) if num_hidden_layers < 12 else []
        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.config, use_safetensors=True)
        self.head = StanceHead(self.config.hidden_size, num_labels, dropout, pooling)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                output_hidden_states: bool = False):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs, output_hidden_states=output_hidden_states)
        logits = self.head(out.last_hidden_state, attention_mask)
        result = {"logits": logits}
        if output_hidden_states:
            result["hidden_states"] = out.hidden_states
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prune_layers(model: StanceModel, keep_layers: int) -> StanceModel:
    """Depth-prune an already-loaded encoder in place (keeps weights, drops layers).

    Depth pruning is the cheapest capacity reduction that preserves the tokenizer
    and the embedding matrix, which is exactly what a tokenizer-aligned student
    needs.
    """
    layers = model.encoder.encoder.layer
    if keep_layers >= len(layers):
        return model
    model.encoder.encoder.layer = nn.ModuleList(list(layers)[:keep_layers])
    model.config.num_hidden_layers = keep_layers
    return model


def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)


def build_teacher(encoder_name: str, pooling: str = "cls", dropout: float = 0.1) -> StanceModel:
    return StanceModel(encoder_name, pooling=pooling, dropout=dropout)


def build_student(encoder_name: str, keep_layers: int = 4, pooling: str = "mean",
                  dropout: float = 0.1) -> StanceModel:
    m = StanceModel(encoder_name, pooling=pooling, dropout=dropout)
    return prune_layers(m, keep_layers)


def save_model(model: StanceModel, tokenizer, out_dir: str, meta: dict) -> None:
    import json
    from pathlib import Path

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "encoder_name": model.encoder_name,
                "config": model.config.to_dict()}, p / "pytorch_model.bin")
    tokenizer.save_pretrained(p)
    (p / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_stance_model(model_dir: str, pooling: str = "cls", num_hidden_layers: int | None = None) -> nn.Module:
    import json
    from pathlib import Path

    p = Path(model_dir)
    bin_file = p / "pytorch_model.bin"
    if bin_file.exists():
        ckpt = torch.load(bin_file, map_location="cpu", weights_only=False)
        m = StanceModel(ckpt.get("encoder_name", "google/muril-base-cased"), pooling=pooling, num_hidden_layers=num_hidden_layers)
        state = {k: v for k, v in ckpt["state_dict"].items()}
        missing, unexpected = m.load_state_dict(state, strict=False)
        meta = {}
        if (p / "training_meta.json").exists():
            meta = json.loads((p / "training_meta.json").read_text(encoding="utf-8"))
        m.meta = meta  # type: ignore[attr-defined]
        m.load_report = {"missing": len(missing), "unexpected": len(unexpected)}  # type: ignore[attr-defined]
        return m

    # Check for LoRA or student adapters
    for cand in [p / "lora_adapter_kd.pt", p / "lora_adapter.pt", p / "lora_adapter_nokd.pt", p / "student_model.pt"]:
        if cand.exists():
            from src.training.phase9_lora_kd import LoRAStance
            adapter = torch.load(cand, map_location="cpu", weights_only=False)
            if "state_dict" in adapter:
                adapter = adapter["state_dict"]
            layers = num_hidden_layers or (4 if ("student" in str(p) or "kd" in cand.name) else 12)
            m = LoRAStance("google/muril-base-cased", pooling=pooling, keep_layers=layers if layers < 12 else None, r=8)
            missing, unexpected = m.load_state_dict(adapter, strict=False)
            meta = {}
            if (p / "model_card.json").exists():
                meta = json.loads((p / "model_card.json").read_text(encoding="utf-8"))
            m.meta = meta  # type: ignore[attr-defined]
            m.load_report = {"missing": len(missing), "unexpected": len(unexpected)}  # type: ignore[attr-defined]
            return m

    raise FileNotFoundError(f"No checkpoint (pytorch_model.bin or lora_adapter_*.pt) found in {model_dir}")
