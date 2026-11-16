"""Phase 9/10 under a hard 1.9 GB RAM ceiling: LoRA fine-tuning + offline KD.

Measured finding that shapes this script: AdamW on a 236M-parameter encoder needs
~16 bytes/param (weights+grads+moments) ~= 3.8 GB, which OOM-killed the naive run
on this box. Two consequences, both reported rather than hidden:
  1. PEFT is not a novelty here, it is the only way the teacher fits.
  2. Teacher and student cannot coexist in RAM, so KD uses *offline* soft targets
     (teacher probabilities exported once to disk), which is standard practice.
Checkpointing stores only the LoRA adapter, because cloning a 236M-param state
dict would itself cost ~1 GB.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None
    import psutil

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.data.dataset_loader import load_gwstance
from src.data.preprocessing import preprocess
from src.data.splitting import group_stratified_split
from src.evaluation.metrics import compute_metrics
from src.models.classification_head import StanceHead

LABELS = ["FAVOUR", "AGAINST", "NONE"]
L2I = {l: i for i, l in enumerate(LABELS)}
ENCODER = "google/muril-base-cased"
RES = os.path.join(str(ROOT), "experiments", "results")


def peak():
    if resource is not None:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 0)


class LoRAStance(torch.nn.Module):
    def __init__(self, encoder_name, pooling="cls", keep_layers=None, r=8, dropout=0.1):
        super().__init__()
        self.tok_free = True
        self.encoder_name = encoder_name
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(encoder_name, use_safetensors=True)
        if keep_layers:
            self.encoder.encoder.layer = torch.nn.ModuleList(list(self.encoder.encoder.layer)[:keep_layers])
        for p in self.encoder.parameters():
            p.requires_grad = False
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                         target_modules=["query", "value"])
        self.encoder = get_peft_model(self.encoder, cfg)
        self.head = StanceHead(self.encoder.config.hidden_size, 3, dropout, pooling)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        h = self.encoder(**kw).last_hidden_state
        return self.head(h, attention_mask)

    def n_total(self):
        return sum(p.numel() for p in self.parameters())

    def n_train(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def adapter_state(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()
                if ("lora" in k) or k.startswith("head.")}


def encode(rows, tok, max_len=64):
    enc = tok([str(r["text"]) for r in rows], truncation=True, max_length=max_len,
              padding="max_length", return_tensors="pt")
    y = torch.tensor([L2I[str(r["label"])] for r in rows], dtype=torch.long)
    return enc, y


@torch.no_grad()
def infer(model, rows, tok, bs=32):
    model.eval()
    probs = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc, _ = encode(chunk, tok)
        p = torch.softmax(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]), -1)
        probs.append(p)
    return torch.cat(probs).numpy()


def train(model, rows, tok, steps, lr=3e-4, bs=2, accum=2, soft=None, alpha=0.5, beta=0.5, T=2.0):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    ce = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)
    idx_all = np.arange(len(rows))
    t0 = time.time()
    gstep, losses = 0, []
    while gstep < steps:
        rng.shuffle(idx_all)
        for i in range(0, len(idx_all), bs):
            batch = [rows[j] for j in idx_all[i:i + bs]]
            enc, y = encode(batch, tok)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            loss = ce(logits, y)
            if soft is not None:
                ids = [idx_all[j] for j in range(i, min(i + bs, len(idx_all)))]
                t_logits = torch.tensor(np.log(np.clip(soft[ids], 1e-8, 1.0)), dtype=torch.float32)
                kl = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits / T, -1), t_logits, reduction="batchmean",
                    log_target=True) * T * T
                loss = alpha * loss + beta * kl
            (loss / accum).backward()
            if (i // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                gstep += 1
                losses.append(float(loss))
                if gstep % 12 == 0:
                    print(f"    step {gstep} loss {np.mean(losses[-12:]):.4f} rss {peak()}MB", flush=True)
                if gstep >= steps:
                    break
    return {"steps": gstep, "final_loss": round(float(np.mean(losses[-10:])), 4),
            "train_seconds": round(time.time() - t0, 1)}


def evaluate(model, rows, tok, name):
    p = infer(model, rows, tok)
    pred = [LABELS[i] for i in p.argmax(1)]
    gold = [str(r["label"]) for r in rows]
    m = compute_metrics(gold, pred)
    m["_probs"] = p
    print(f"  {name}: macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} rss={peak()}MB", flush=True)
    return m


def main():
    out = {}
    recs = load_gwstance(drop_screening=True)
    for r in recs:
        r["text"] = preprocess(r["text"], variant="minimal").text
    split, rep = group_stratified_split(recs, fractions=(0.70, 0.15, 0.15), seed=42)
    tr, va, te = split["train"], split["val"], split["test"]
    tok = AutoTokenizer.from_pretrained(ENCODER)
    print(f"split train={len(tr)} val={len(va)} test={len(te)} leak_safe={rep.leakage_safe}", flush=True)

    # ---------- teacher, LoRA ----------
    print("[teacher] loading MuRIL base (236M) + LoRA r=8 on q,v ...", flush=True)
    teacher = LoRAStance(ENCODER, pooling="cls", r=8)
    out["teacher_params"] = teacher.n_total()
    out["teacher_trainable"] = teacher.n_train()
    print(f"  total={teacher.n_total():,} trainable={teacher.n_train():,} ({100*teacher.n_train()/teacher.n_total():.3f}%)", flush=True)
    out["teacher_hist"] = train(teacher, tr, tok, steps=36, lr=3e-4, bs=2, accum=2)
    out["teacher_val"] = evaluate(teacher, va, tok, "teacher/val")
    out["teacher_test"] = evaluate(teacher, te, tok, "teacher/test")

    # export soft targets for offline KD (teacher alone in RAM)
    print("[teacher] exporting soft targets for offline KD ...", flush=True)
    soft = {}
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        soft[name] = infer(teacher, rows, tok, bs=32)
        print(f"  {name}: probs {soft[name].shape} rss {peak()}MB", flush=True)
    np.savez_compressed(os.path.join(RES, "teacher_soft_targets.npz"),
                        train=soft["train"], val=soft["val"], test=soft["test"])
    torch.save(teacher.adapter_state(), os.path.join(ROOT, "models", "teacher", "lora_adapter.pt"))
    with open(os.path.join(ROOT, "models", "teacher", "model_card.json"), "w") as f:
        json.dump({"base_model": ENCODER, "adapters": "LoRA r=8, alpha=16, q/v",
                   "trainable_params": teacher.n_train(), "total_params": teacher.n_total(),
                   "pooling": "cls", "test": {k: v for k, v in out["teacher_test"].items() if not k.startswith("_")},
                   "corpus": "GWStance/GWSD (English, derived labels)", "budget_steps": 36}, f, indent=2)
    t_probs = {k: v.copy() for k, v in soft.items()}
    del teacher, soft
    gc.collect()
    print(f"[teacher] freed. rss={peak()}MB", flush=True)

    # ---------- student, 4 layers, mean pooling ----------
    print("[student] 4-layer MuRIL + LoRA, mean pooling, no distillation ...", flush=True)
    student = LoRAStance(ENCODER, pooling="mean", keep_layers=4, r=8)
    out["student_params"] = student.n_total()
    out["student_trainable"] = student.n_train()
    print(f"  total={student.n_total():,} trainable={student.n_train():,}", flush=True)
    out["student_hist"] = train(student, tr, tok, steps=36, lr=3e-4, bs=2, accum=2)
    out["student_val"] = evaluate(student, va, tok, "student/val")
    out["student_test"] = evaluate(student, te, tok, "student/test")
    torch.save(student.adapter_state(), os.path.join(ROOT, "models", "student", "lora_adapter_nokd.pt"))
    s_nokd = {k: v for k, v in out["student_test"].items() if not k.startswith("_")}
    del student
    gc.collect()

    # ---------- student, distilled ----------
    print("[student-KD] same student, T=2 alpha=0.5 beta=0.5, offline soft targets ...", flush=True)
    kd = LoRAStance(ENCODER, pooling="mean", keep_layers=4, r=8)
    out["kd_hist"] = train(kd, tr, tok, steps=36, lr=3e-4, bs=2, accum=2,
                           soft=t_probs["train"], alpha=0.5, beta=0.5, T=2.0)
    out["kd_val"] = evaluate(kd, va, tok, "kd/val")
    out["kd_test"] = evaluate(kd, te, tok, "kd/test")
    torch.save(kd.adapter_state(), os.path.join(ROOT, "models", "student", "lora_adapter_kd.pt"))
    with open(os.path.join(ROOT, "models", "student", "model_card.json"), "w") as f:
        json.dump({"base_model": ENCODER, "layers": 4, "pooling": "mean",
                   "adapters": "LoRA r=8 q/v + head", "trainable_params": kd.n_train(),
                   "total_params": kd.n_total(), "distilled": True, "temperature": 2.0,
                   "alpha": 0.5, "beta": 0.5,
                   "test": {k: v for k, v in out["kd_test"].items() if not k.startswith("_")},
                   "no_kd_test": s_nokd, "budget_steps": 36}, f, indent=2)

    out["delta_kd_vs_nokd"] = round(out["kd_test"]["macro_f1"] - s_nokd["macro_f1"], 4)
    out["split"] = rep.to_dict()
    with open(os.path.join(RES, "phase9_10_results.json"), "w") as f:
        json.dump({k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
                   for k, v in out.items()}, f, indent=2)
    print("\n=== SUMMARY ===", flush=True)
    for k in ("teacher_val", "teacher_test", "student_val", "student_test", "kd_val", "kd_test"):
        m = out[k]
        print(f"{k:14s} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
              f"F1(F/A/N)={m['per_class']['FAVOUR']['f1']:.3f}/{m['per_class']['AGAINST']['f1']:.3f}/{m['per_class']['NONE']['f1']:.3f}")
    print(f"params teacher={out['teacher_params']:,} (trainable {out['teacher_trainable']:,})")
    print(f"params student={out['student_params']:,} (trainable {out['student_trainable']:,})")
    print(f"KD delta vs no-KD: {out['delta_kd_vs_nokd']:+.4f}")
    print(f"peak RSS: {peak()} MB", flush=True)


if __name__ == "__main__":
    main()
