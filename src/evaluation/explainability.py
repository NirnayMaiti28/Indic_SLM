"""Token-level explanations: Integrated Gradients (implemented directly, no
captum dependency) plus leave-one-token-out occlusion as a cross-check.

Both are attribution heuristics. They are used to *study* model behaviour and
errors, never presented as causal explanations of the model's decision.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)

from src.models.teacher import load_stance_model, load_tokenizer  # noqa: E402

LABELS = ["FAVOUR", "AGAINST", "NONE"]


def _forward_logits(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask)["logits"]


def integrated_gradients(model, tok, text: str, target_idx: int, steps: int = 24, max_length: int = 64) -> dict:
    enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    emb_layer = model.encoder.get_input_embeddings()
    emb = emb_layer(ids)
    baseline = torch.zeros_like(emb)
    total_grad = torch.zeros_like(emb)

    for a in range(1, steps + 1):
        alpha = a / steps
        x = (baseline + alpha * (emb - baseline)).detach().requires_grad_(True)
        logits = model.encoder(inputs_embeds=x, attention_mask=mask).last_hidden_state
        out = model.head(logits, mask)
        score = out[0, target_idx]
        score.backward()
        total_grad += x.grad.detach()

    avg_grad = total_grad / steps
    attributions = ((emb - baseline) * avg_grad).sum(-1)[0]
    tokens = tok.convert_ids_to_tokens(ids[0].tolist())
    pairs = [(t, float(a)) for t, a in zip(tokens, attributions.tolist())]
    # drop specials for readability
    pairs = [(t, a) for t, a in pairs if t not in (tok.cls_token, tok.sep_token, tok.pad_token)]
    pairs.sort(key=lambda x: -abs(x[1]))
    return {"tokens": [t for t, _ in pairs], "attributions": [round(a, 5) for _, a in pairs]}


def occlusion(model, tok, text: str, target_idx: int, max_length: int = 64) -> dict:
    """Drop each token, measure the drop in the target-class probability."""
    enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
    ids, mask = enc["input_ids"][0], enc["attention_mask"][0]
    toks = tok.convert_ids_to_tokens(ids.tolist())
    keep = [i for i, t in enumerate(toks) if t not in (tok.cls_token, tok.sep_token, tok.pad_token)]

    with torch.no_grad():
        base = torch.softmax(_forward_logits(model, ids.unsqueeze(0), mask.unsqueeze(0)), -1)[0, target_idx].item()
        deltas = []
        for i in keep:
            new_ids = [x for j, x in enumerate(ids.tolist()) if j != i]
            new_mask = [1] * len(new_ids)
            p = torch.softmax(_forward_logits(model, torch.tensor([new_ids]), torch.tensor([new_mask])), -1)[0, target_idx].item()
            deltas.append((toks[i], round(base - p, 5)))
    deltas.sort(key=lambda x: -abs(x[1]))
    return {"base_prob": round(base, 5), "drops": deltas}


def explain(model_dir: str, pooling: str, layers: int | None, examples: list[str]) -> dict:
    tok = load_tokenizer("google/muril-base-cased")
    model = load_stance_model(model_dir, pooling=pooling, num_hidden_layers=layers)
    model.eval()
    out = []
    for text in examples:
        enc = tok(text, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])["logits"]
            probs = torch.softmax(logits, -1)[0]
        pred = int(probs.argmax())
        ig = integrated_gradients(model, tok, text, pred)
        oc = occlusion(model, tok, text, pred)
        out.append({
            "text": text,
            "predicted": LABELS[pred],
            "confidence": round(float(probs[pred]), 4),
            "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(3)},
            "ig_top": [{"token": t, "attribution": a} for t, a in zip(ig["tokens"][:8], ig["attributions"][:8])],
            "occlusion_top": [{"token": t, "prob_drop": d} for t, d in oc["drops"][:6]],
        })
        print(f"{text[:60]:60s} -> {LABELS[pred]} ({float(probs[pred]):.3f}) "
              f"| IG top: {ig['tokens'][:5]}")
    p = ROOT / "experiments" / "results" / "explainability.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"examples": out}


EXAMPLES = [
    "जलवायु परिवर्तन एक गंभीर समस्या है।",
    "ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है।",
    "জলবায়ু পরিবর্তন একটি গুরুতর উদ্যেগের বিষয়।",
    "জলবায়ু পরিবর্তন কোনো গুরুতর সমস্যা নয়।",
    "सरकार की जलवायु नीति बिल्कुल फेल है, पैसा बर्बाद हो रहा है।",
]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "student"))
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args(argv)
    explain(args.model_dir, args.pooling, args.layers, EXAMPLES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
