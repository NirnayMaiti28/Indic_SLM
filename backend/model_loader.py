"""Model Loader & Inference Service for Indic Climate Stance Detection.

Loads the 4-layer Compact Student SLM distilled from MuRIL with LoRA adapters (models/student/lora_adapter_kd.pt).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import preprocess
from src.data.language_detection import analyze_language
from src.models.classification_head import StanceHead

LABELS = ["FAVOUR", "AGAINST", "NONE"]
ID2LABEL = {0: "FAVOUR", 1: "AGAINST", 2: "NONE"}
LABEL2ID = {"FAVOUR": 0, "AGAINST": 1, "NONE": 2}


class CompactStudentSLM(nn.Module):
    """4-Layer Depth-Pruned Student Small Language Model with LoRA and Mean-Pooling StanceHead."""

    def __init__(self, encoder_name: str = "google/muril-base-cased", keep_layers: int = 4, r: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder_name = encoder_name
        self.keep_layers = keep_layers

        # Load base encoder (safetensors format from local Hugging Face cache)
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            use_safetensors=True,
            local_files_only=True
        )

        # 1. Depth-prune transformer layers to 4 layers
        if keep_layers < len(self.encoder.encoder.layer):
            self.encoder.encoder.layer = nn.ModuleList(list(self.encoder.encoder.layer)[:keep_layers])

        # 2. Freeze all base encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 3. Inject LoRA adapters (r=8, alpha=16) on query and value attention projections
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=2 * r,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query", "value"]
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)

        # 4. Multilingual Stance Head with Mean Pooling
        self.head = StanceHead(
            hidden_size=self.encoder.config.hidden_size,
            num_labels=3,
            dropout=dropout,
            pooling="mean"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs).last_hidden_state
        return self.head(out, attention_mask)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class StanceInferenceEngine:
    """Production Inference Engine for Indic Climate Stance Detection."""

    _instance: Optional["StanceInferenceEngine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, checkpoint_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder_name = "google/muril-base-cased"
        self.checkpoint_path = Path(checkpoint_path or (ROOT / "models" / "student" / "lora_adapter_kd.pt"))

        print(f"[*] Initializing StanceInferenceEngine on device: {self.device.upper()}")
        t0 = time.time()

        # 1. Load Tokenizer
        print(f"[*] Loading tokenizer: {self.encoder_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, local_files_only=True)

        # 2. Build Student Architecture
        print(f"[*] Building 4-layer Compact Student SLM with LoRA (r=8)...")
        self.model = CompactStudentSLM(encoder_name=self.encoder_name, keep_layers=4, r=8)

        # 3. Load Trained LoRA Checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"[*] Loading trained weights from: {self.checkpoint_path}...")
        adapter_state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in adapter_state:
            adapter_state = adapter_state["state_dict"]

        missing, unexpected = self.model.load_state_dict(adapter_state, strict=False)
        print(f"[+] Adapter weights loaded: {len(adapter_state)} tensors (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

        self.model.to(self.device)
        self.model.eval()

        total_p, train_p = self.model.count_parameters()
        self.total_params = total_p
        self.trainable_params = train_p
        self.startup_time_sec = round(time.time() - t0, 2)
        print(f"[+] Engine ready in {self.startup_time_sec}s. Total params: {total_p:,}, Trainable: {train_p:,}")
        self._initialized = True

    def predict(self, text: str, explain: bool = False) -> Dict[str, Any]:
        """Classify stance of a single comment with linguistic metadata."""
        if text is None or not str(text).strip():
            return {
                "text": text or "",
                "clean_text": "",
                "predicted_label": None,
                "confidence": None,
                "probabilities": None,
                "language": None,
                "script": None,
                "is_code_mixed": False,
                "has_negation": False,
                "truncated": False,
                "latency_ms": 0.0,
                "device": self.device,
                "explanations": None,
                "error": "Input text cannot be empty or whitespace only."
            }

        raw_text = str(text).strip()
        t_start = time.perf_counter()

        # Preprocessing & Linguistic Analysis
        prep = preprocess(raw_text, variant="minimal")
        lang_profile = analyze_language(raw_text)
        clean_text = prep.text

        if not clean_text.strip():
            return {
                "text": raw_text,
                "clean_text": "",
                "predicted_label": None,
                "confidence": None,
                "probabilities": None,
                "language": lang_profile.language,
                "script": lang_profile.script,
                "is_code_mixed": lang_profile.is_code_mixed,
                "has_negation": prep.flags.get("has_negation", False),
                "truncated": False,
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
                "device": self.device,
                "explanations": None,
                "error": "Text is empty after standard preprocessing."
            }

        # Tokenization
        encoded = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Forward Pass with no_grad
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_label = ID2LABEL[pred_idx]
            confidence = float(probs[pred_idx])

        truncated = bool((attention_mask.sum().item()) >= 64)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        prob_dict = {
            "FAVOUR": round(float(probs[0]), 4),
            "AGAINST": round(float(probs[1]), 4),
            "NONE": round(float(probs[2]), 4)
        }

        # Optional Integrated Gradients Explainability (with gradients enabled)
        explanations = None
        if explain:
            with torch.enable_grad():
                explanations = self.compute_integrated_gradients(clean_text, pred_idx)

        return {
            "text": raw_text,
            "clean_text": clean_text,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "language": lang_profile.language,
            "script": lang_profile.script,
            "is_code_mixed": lang_profile.is_code_mixed,
            "has_negation": prep.flags.get("has_negation", False),
            "truncated": truncated,
            "latency_ms": latency_ms,
            "device": self.device,
            "explanations": explanations,
            "error": None
        }

    def compute_integrated_gradients(self, text: str, target_idx: int, steps: int = 16, max_length: int = 64) -> List[Dict[str, Any]]:
        """Compute token-level attribution scores using Integrated Gradients."""
        enc = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Retrieve word embedding layer from base encoder
        base_bert = self.model.encoder.base_model.model
        embedding_layer = base_bert.embeddings.word_embeddings
        orig_embeddings = embedding_layer(input_ids)

        baseline_embeddings = torch.zeros_like(orig_embeddings)
        accumulated_grads = torch.zeros_like(orig_embeddings)

        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = (baseline_embeddings + alpha * (orig_embeddings - baseline_embeddings)).detach().requires_grad_(True)

            # Pass interpolated embeddings through encoder & stance head
            encoder_out = base_bert(inputs_embeds=interpolated, attention_mask=attention_mask).last_hidden_state
            logits = self.model.head(encoder_out, attention_mask)
            target_score = logits[0, target_idx]
            target_score.backward()

            if interpolated.grad is not None:
                accumulated_grads += interpolated.grad.detach()

        avg_grads = accumulated_grads / steps
        attributions = ((orig_embeddings - baseline_embeddings) * avg_grads).sum(dim=-1)[0]
        attributions = attributions.detach().cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        results = []
        special_tokens = {self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token, "[CLS]", "[SEP]", "[PAD]"}

        for tok, score in zip(tokens, attributions):
            if tok in special_tokens:
                continue
            clean_tok = tok.replace("##", "")
            results.append({"token": clean_tok, "attribution": round(float(score), 4)})

        # Sort by absolute attribution impact
        results.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        return results[:10]

    def predict_batch(self, texts: List[str], explain: bool = False, batch_size: int = 32) -> Dict[str, Any]:
        """Batch inference for multiple inputs."""
        t_start = time.perf_counter()
        results = [self.predict(t, explain=explain) for t in texts]
        total_time = time.perf_counter() - t_start
        latency_ms = round(total_time * 1000, 2)
        throughput = round(len(texts) / max(total_time, 1e-6), 2)

        return {
            "results": results,
            "total": len(texts),
            "latency_ms": latency_ms,
            "throughput_items_per_sec": throughput
        }


def get_inference_engine() -> StanceInferenceEngine:
    return StanceInferenceEngine()
