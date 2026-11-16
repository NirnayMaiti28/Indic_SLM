"""Phase 9/10 driver: train teacher, student, and distilled student.

Every run is appended to the experiment registry with the metrics returned by the
trainer. Budgets are explicit CLI flags because this host has no GPU: `--max-steps`
exists so that a *reduced-budget* run can be reported honestly instead of a
fabricated full-training number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(2)

from src.data.augmentation import augment  # noqa: E402
from src.data.dataset_loader import load_custom, load_gwstance  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.data.splitting import group_stratified_split  # noqa: E402
from src.evaluation.metrics import compute_metrics, per_language_report  # noqa: E402
from src.models.teacher import build_student, build_teacher, load_tokenizer, save_model  # noqa: E402
from src.training.trainer import TrainConfig, evaluate_model, train_model, StanceDataset  # noqa: E402
from src.utils.registry import log_experiment  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ENCODER = "google/muril-base-cased"


def get_split(dataset_name: str = "indic", seed: int = 42):
    splits_path = ROOT / "data" / "splits" / "indic_climate_splits.json"
    if dataset_name == "indic" and splits_path.exists():
        with open(splits_path, encoding="utf-8") as f:
            data = json.load(f)
            return data["splits"], data["report"]
    recs = load_custom() if dataset_name == "indic" else load_gwstance(drop_screening=True)
    for r in recs:
        r["text_raw"] = r["text"]
        r["text"] = preprocess(r["text"], variant="minimal").text
    split, rep = group_stratified_split(recs, fractions=(0.70, 0.15, 0.15), seed=seed)
    return split, rep.to_dict()


def evaluate_split(model, records, tokenizer, max_length=64, batch_size=16) -> dict:
    ds = StanceDataset(records, tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    ev = evaluate_model(model, dl)
    m = compute_metrics(ev["y_true"], ev["y_pred"])
    m["_preds"] = ev["y_pred"]
    m["_probs"] = ev["probs"].tolist()
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher", "student", "distill"], required=True)
    ap.add_argument("--dataset", choices=["indic", "gwstance"], default="indic",
                    help="Dataset to train on (default: indic)")
    ap.add_argument("--tag", default=None, help="experiment tag for filenames/registry")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--layers", type=int, default=4, help="student depth")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal", "ce_none"])
    ap.add_argument("--class-weights", action="store_true", default=True)
    ap.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    ap.add_argument("--augment", default=None, help="comma list of ops, e.g. char_repeat,emoji")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--teacher-dir", default=str(ROOT / "models" / "teacher"))
    args = ap.parse_args(argv)

    tag = args.tag or f"{args.mode}_{args.dataset}"
    split, split_rep = get_split(dataset_name=args.dataset)
    tr, va, te = split["train"], split["val"], split["test"]
    tokenizer = load_tokenizer(ENCODER)

    aug_note = "none"
    if args.augment:
        ops = [o.strip() for o in args.augment.split(",") if o.strip()]
        tr = augment(tr, ops, copies=1, seed=42, keep_original=True)
        aug_note = "+".join(ops) + f" (x2 rows -> {len(tr)})"

    print(f"[{tag}] train={len(tr)} val={len(va)} test={len(te)} augment={aug_note}")

    if args.mode == "teacher":
        model = build_teacher(ENCODER, pooling=args.pooling)
    elif args.mode == "student":
        model = build_student(ENCODER, keep_layers=args.layers, pooling=args.pooling)
    else:
        model = build_student(ENCODER, keep_layers=args.layers, pooling=args.pooling)

    if args.lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                         target_modules=["query", "value"], task_type=None)
        model = get_peft_model(model, cfg)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{tag}] LoRA trainable params: {n_train:,}")

    teacher = None
    if args.mode == "distill":
        from src.models.teacher import load_stance_model

        teacher = load_stance_model(args.teacher_dir, pooling="cls")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[{tag}] teacher loaded from {args.teacher_dir} "
              f"({teacher.n_params():,} params)")

    cfg = TrainConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                      grad_accum=args.grad_accum, loss=args.loss,
                      use_class_weights=args.class_weights, max_steps=args.max_steps,
                      temperature=args.temperature, alpha=args.alpha, beta=args.beta,
                      seed=42, log_every=25)
    t0 = time.time()
    model, hist = train_model(model, tr, va, tokenizer, cfg, teacher=teacher,
                              save_dir=ROOT / "experiments" / "results" / f"hist_{tag}", verbose=True)
    wall = time.time() - t0

    val = evaluate_split(model, va, tokenizer)
    test = evaluate_split(model, te, tokenizer)
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = {
        "tag": tag, "mode": args.mode, "encoder": ENCODER,
        "params": n_params, "trainable_params": n_trainable,
        "pooling": args.pooling, "loss": args.loss, "augmentation": aug_note,
        "lora": args.lora, "max_steps": args.max_steps, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
        "split": split_rep, "history": hist.to_dict(), "wall_seconds": round(wall, 1),
        "val": {k: v for k, v in val.items() if not k.startswith("_")},
        "test": {k: v for k, v in test.items() if not k.startswith("_")},
        "test_preds": test["_preds"], "test_probs": test["_probs"],
        "per_language": per_language_report(te, [r["label"] for r in te], test["_preds"]),
    }
    res_dir = ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"run_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    per_lang = out["per_language"]
    hi_f1 = per_lang.get("Hindi", {}).get("macro_f1")
    bn_f1 = per_lang.get("Bengali", {}).get("macro_f1")
    en_f1 = per_lang.get("English", {}).get("macro_f1") or test["macro_f1"]

    log_experiment(model=f"{args.mode}:{ENCODER.split('/')[-1]}" + (f"+lora" if args.lora else "") +
                   (f"+{args.layers}L" if args.mode != "teacher" else ""),
                   model_params=n_params,
                   dataset=args.dataset, split_strategy=split_rep["strategy"],
                   n_train=len(tr), n_val=len(va), n_test=len(te), preprocessing="minimal",
                   augmentation=aug_note, loss=args.loss + ("+kd" if args.mode == "distill" else ""),
                   lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, seed=42,
                   distill_T=args.temperature if args.mode == "distill" else "",
                   distill_alpha=args.alpha if args.mode == "distill" else "",
                   distill_beta=args.beta if args.mode == "distill" else "",
                   peft="lora_r8" if args.lora else "none",
                   macro_f1=test["macro_f1"], accuracy=test["accuracy"], weighted_f1=test["weighted_f1"],
                   f1_favour=test["per_class"]["FAVOUR"]["f1"], f1_against=test["per_class"]["AGAINST"]["f1"],
                   f1_none=test["per_class"]["NONE"]["f1"],
                   hindi_f1=hi_f1, bengali_f1=bn_f1, english_f1=en_f1,
                   train_seconds=round(wall, 1),
                   notes=f"tag={tag}; val_macro_f1={val['macro_f1']}; best_epoch={hist.best_epoch}; "
                         f"steps={hist.steps}; pooling={args.pooling}; max_steps={args.max_steps}")

    if args.mode == "teacher" and not args.lora:
        save_model(model, tokenizer, str(ROOT / "models" / "teacher"),
                   {"encoder": ENCODER, "pooling": args.pooling, "test": out["test"],
                    "split": split_rep, "steps": hist.steps})
        print(f"saved teacher -> models/teacher")
    if args.mode in ("student", "distill") and args.layers != 12:
        save_model(model, tokenizer, str(ROOT / "models" / "student"),
                   {"encoder": ENCODER, "pooling": args.pooling, "layers": args.layers,
                    "test": out["test"], "kd": args.mode == "distill"})
        print("saved student -> models/student")

    print(f"[{tag}] val_macro_f1={val['macro_f1']:.4f} test_macro_f1={test['macro_f1']:.4f} "
          f"test_acc={test['accuracy']:.4f} params={n_params:,} wall={wall:.0f}s steps={hist.steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
