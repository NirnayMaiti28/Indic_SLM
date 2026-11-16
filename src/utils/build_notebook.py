"""Generates the interactive training notebook train_model.ipynb and notebooks/train_indic_stance_slm.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def make_cell(cell_type: str, src: str) -> dict:
    lines = [l + "\n" for l in src.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def build_training_notebook() -> dict:
    cells = []

    # ---------------- Cell 1: Intro ----------------
    cells.append(make_cell("markdown", """# 🌍 Indic Climate Stance Detection: SLM Training Notebook
### Train, Evaluate & Visualize a Compact Multilingual Small Language Model (SLM) for Hindi & Bengali

**Target Claim**: *"Climate change and global warming is a serious concern."*  
**Stance Classes**:
- **`FAVOUR`**: The text agrees with, supports, or expresses concern about climate change.
- **`AGAINST`**: The text dismisses, opposes, denies, or rejects climate change concern.
- **`NONE`**: The text is neutral, factual, ambiguous, or lacks an explicit stance.

---
### 💡 What this notebook does:
1. Loads the **preprocessed, leakage-free Indic Climate Stance Dataset** (Hindi, Bengali, Hinglish, Banglish).
2. Builds a **compact 4-layer Student SLM** based on `google/muril-base-cased` with **LoRA (Low-Rank Adaptation)** (~100k trainable parameters).
3. Executes a **live interactive training loop** with per-epoch loss, validation accuracy, and Macro-F1 tracking.
4. Plots **Training Loss** and **Macro-F1** learning curves side-by-side.
5. Evaluates on the held-out test split with a **Confusion Matrix Heatmap** and **per-language breakdown** (Hindi vs. Bengali).
6. Provides an **Interactive Stance Prediction Sandbox** where you can type any text and see the live prediction, confidence bar chart, and language/script analysis."""))

    # ---------------- Cell 2: Imports & Environment ----------------
    cells.append(make_cell("code", """import os
import sys
import json
import time
from pathlib import Path

# Automatically detect project root
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from src.models.classification_head import StanceHead

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hardware detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Execution Device: {device.type.upper()}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"   CPU Threads: {torch.get_num_threads()}")"""))

    # ---------------- Cell 3: Data Loading Markdown ----------------
    cells.append(make_cell("markdown", """## 1. Load Preprocessed Indic Climate Dataset
We load the pre-computed, group-stratified splits from `data/splits/indic_climate_splits.json`.
Group stratification guarantees **zero text leakage** across the splits even if duplicates exist in raw data."""))

    # ---------------- Cell 4: Data Loading Code ----------------
    cells.append(make_cell("code", """splits_path = ROOT / "data" / "splits" / "indic_climate_splits.json"
assert splits_path.exists(), f"Splits file not found at {splits_path}"

with open(splits_path, encoding="utf-8") as f:
    splits_data = json.load(f)

train_recs = splits_data["splits"]["train"]
val_recs = splits_data["splits"]["val"]
test_recs = splits_data["splits"]["test"]

print("📊 Dataset Summary:")
print(f"   • Train Split:      {len(train_recs):>5} samples")
print(f"   • Validation Split: {len(val_recs):>5} samples")
print(f"   • Test Split:       {len(test_recs):>5} samples")
print(f"   • Total Records:    {len(train_recs) + len(val_recs) + len(test_recs):>5} samples")

# Display sample records from the dataset
print("\\n🔍 Sample Training Records:")
df_train = pd.DataFrame(train_recs)
for _, row in df_train.sample(4, random_state=42).iterrows():
    lang = row.get("language", "Unknown")
    script = row.get("script", "Unknown")
    print(f"   [{row['label']:<7}] ({lang:<7} | {script:<10}) {row['text']}")

# Plot class distribution across splits
fig, ax = plt.subplots(figsize=(8, 3.5))
dist_df = pd.DataFrame({
    'Train': pd.Series([r['label'] for r in train_recs]).value_counts(),
    'Validation': pd.Series([r['label'] for r in val_recs]).value_counts(),
    'Test': pd.Series([r['label'] for r in test_recs]).value_counts()
}).fillna(0)
dist_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
plt.title("Class Distribution Across Dataset Splits", fontsize=12, fontweight='bold')
plt.xlabel("Stance Label")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""))

    # ---------------- Cell 5: Model Init Markdown ----------------
    cells.append(make_cell("markdown", """## 2. Model Architecture: Compact Student SLM with LoRA
We instantiate a compact Small Language Model (SLM) derived from `google/muril-base-cased`:
1. **Depth Pruning**: We keep the first **4 transformer layers** (out of 12).
2. **LoRA Adapters**: We freeze the 180M backbone weights and only train rank-8 attention adapter projections ($W_q, W_v$) + the stance classification head.
3. **Trainable Footprint**: Only **~100k parameters** (0.06% of the model), enabling fast training on CPU or GPU without RAM bottlenecks."""))

    # ---------------- Cell 6: Model Init Code ----------------
    cells.append(make_cell("code", """ENCODER_NAME = "google/muril-base-cased"
LABELS = ["FAVOUR", "AGAINST", "NONE"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

print(f"Loading tokenizer: {ENCODER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

class CompactStudentSLM(torch.nn.Module):
    def __init__(self, encoder_name=ENCODER_NAME, keep_layers=4, r=8, dropout=0.1):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name, use_safetensors=True)
        
        # 1. Depth-prune transformer layers to 4 layers
        if keep_layers < len(self.encoder.encoder.layer):
            self.encoder.encoder.layer = torch.nn.ModuleList(list(self.encoder.encoder.layer)[:keep_layers])
            
        # 2. Freeze base encoder parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        # 3. Inject LoRA adapters (r=8, alpha=16) on query and value projections
        lora_cfg = LoraConfig(
            r=r, 
            lora_alpha=2 * r, 
            lora_dropout=0.05, 
            bias="none",
            target_modules=["query", "value"]
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        
        # 4. Multilingual Stance Head with Mean Pooling
        self.head = StanceHead(self.encoder.config.hidden_size, num_labels=3, dropout=dropout, pooling="mean")
        
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(out, attention_mask)
        
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

student_model = CompactStudentSLM(ENCODER_NAME, keep_layers=4, r=8).to(device)
total_params, trainable_params = student_model.count_parameters()

print(f"\\n✅ Compact Student SLM Initialized:")
print(f"   • Total Parameters:     {total_params:>12,}")
print(f"   • Trainable Parameters: {trainable_params:>12,} ({trainable_params / total_params * 100:.2f}% of total)")
print(f"   • Frozen Base Weights:  {total_params - trainable_params:>12,}")"""))

    # ---------------- Cell 7: Dataset & DataLoader Markdown ----------------
    cells.append(make_cell("markdown", """## 3. Dataset Tokenization & PyTorch DataLoaders
Tokenizes text inputs with max length 64 and packages them into batched PyTorch DataLoaders."""))

    # ---------------- Cell 8: Dataset & DataLoader Code ----------------
    cells.append(make_cell("code", """class IndicStanceDataset(Dataset):
    def __init__(self, records, tok, max_length=64):
        self.records = records
        self.tok = tok
        self.max_length = max_length
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        item = self.records[idx]
        enc = self.tok(
            str(item["text"]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(L2I[str(item["label"])], dtype=torch.long)
        }

BATCH_SIZE = 8
train_loader = DataLoader(IndicStanceDataset(train_recs, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(IndicStanceDataset(val_recs, tokenizer), batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader = DataLoader(IndicStanceDataset(test_recs, tokenizer), batch_size=BATCH_SIZE * 2, shuffle=False)

print(f"DataLoaders prepared (batch_size={BATCH_SIZE}):")
print(f"   • Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")"""))

    # ---------------- Cell 9: Training Loop Markdown ----------------
    cells.append(make_cell("markdown", """## 4. Execute Model Training
Run the training loop directly in this cell. You will see:
- Live step/epoch loss
- Validation Macro-F1 after every epoch
- Automatic checkpoint tracking for the best model"""))

    # ---------------- Cell 10: Training Loop Code ----------------
    cells.append(make_cell("code", """from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

# --- Hyperparameters ---
EPOCHS = 4             # Adjust number of epochs (recommended: 4 to 6)
LR = 3e-4              # Learning rate for LoRA adapters
GRAD_ACCUM = 2         # Gradient accumulation steps

# Inverse-frequency class weights to balance minority classes
class_counts = Counter([r["label"] for r in train_recs])
weights = torch.tensor([1.0 / max(1, class_counts[lbl]) for lbl in LABELS], dtype=torch.float).to(device)
weights = weights / weights.sum() * 3.0
criterion = torch.nn.CrossEntropyLoss(weight=weights)

# Optimizer
optimizer = torch.optim.AdamW(
    [p for p in student_model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=0.01
)

history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": []}
best_f1 = -1.0
best_state = None
t_start = time.time()

print("=" * 66)
print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Macro-F1':^14} | {'Val Acc':^9}")
print("=" * 66)

for epoch in range(1, EPOCHS + 1):
    student_model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        logits = student_model(input_ids, attention_mask)
        loss = criterion(logits, labels) / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM
        
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
    epoch_train_loss = running_loss / len(train_loader)
    
    # Evaluate on Validation split
    student_model.eval()
    val_loss = 0.0
    val_preds, val_golds = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = student_model(input_ids, attention_mask)
            v_loss = criterion(logits, labels)
            val_loss += v_loss.item()
            
            preds = logits.argmax(dim=-1).cpu().tolist()
            val_preds.extend(preds)
            val_golds.extend(labels.cpu().tolist())
            
    epoch_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_golds, val_preds, average="macro", zero_division=0)
    val_acc = accuracy_score(val_golds, val_preds)
    
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["val_macro_f1"].append(val_f1)
    history["val_acc"].append(val_acc)
    
    star = " 🌟 (best)" if val_f1 > best_f1 else ""
    print(f" {epoch:^5} | {epoch_train_loss:^12.4f} | {epoch_val_loss:^10.4f} | {val_f1:^14.4f} | {val_acc:^9.4f}{star}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.detach().clone().cpu() for k, v in student_model.state_dict().items()}

wall_time = time.time() - t_start
print("=" * 66)
print(f"🎉 Training Complete in {wall_time:.1f}s | Best Validation Macro-F1: {best_f1:.4f}")

# Restore best checkpoint
if best_state is not None:
    student_model.load_state_dict(best_state)"""))

    # ---------------- Cell 11: Visualizations Markdown ----------------
    cells.append(make_cell("markdown", """## 5. Training & Validation Curves
Plots Loss and Macro-F1 curves to inspect convergence and ensure there is no overfitting."""))

    # ---------------- Cell 12: Visualizations Code ----------------
    cells.append(make_cell("code", """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
epochs_range = range(1, len(history["train_loss"]) + 1)

# Loss Plot
ax1.plot(epochs_range, history["train_loss"], marker='o', label="Train Loss", color="#1f77b4", linewidth=2)
ax1.plot(epochs_range, history["val_loss"], marker='s', label="Val Loss", color="#ff7f0e", linewidth=2)
ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_xticks(epochs_range)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Macro-F1 Plot
ax2.plot(epochs_range, history["val_macro_f1"], marker='^', label="Val Macro-F1", color="#2ca02c", linewidth=2)
ax2.plot(epochs_range, history["val_acc"], marker='d', label="Val Accuracy", color="#9467bd", linestyle="--", linewidth=1.5)
ax2.set_title("Validation Macro-F1 & Accuracy", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_xticks(epochs_range)
ax2.set_ylim([0.0, 1.05])
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()"""))

    # ---------------- Cell 13: Test Eval Markdown ----------------
    cells.append(make_cell("markdown", """## 6. Comprehensive Test Set Evaluation
Evaluates the best checkpoint on the held-out test split (235 items) and calculates:
- Overall **Macro-F1**, **Accuracy**, and **Weighted-F1**
- Per-class precision, recall, and F1 scores
- Per-language breakdown (Hindi vs Bengali)
- Confusion Matrix Heatmap"""))

    # ---------------- Cell 14: Test Eval Code ----------------
    cells.append(make_cell("code", """from sklearn.metrics import classification_report, confusion_matrix
from src.evaluation.metrics import compute_metrics, per_language_report

student_model.eval()
test_preds, test_golds, test_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        logits = student_model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        
        test_probs.extend(probs.cpu().tolist())
        test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        test_golds.extend(labels.cpu().tolist())

gold_labels = [I2L[i] for i in test_golds]
pred_labels = [I2L[i] for i in test_preds]

# 1. Primary Metrics
metrics = compute_metrics(gold_labels, pred_labels)
print(f"🏆 Final Test Set Results (n={metrics['n']}):")
print(f"   • Macro-F1:          {metrics['macro_f1']:.4f}")
print(f"   • Accuracy:          {metrics['accuracy']:.4f}")
print(f"   • Weighted-F1:       {metrics['weighted_f1']:.4f}")
print(f"   • Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")

# 2. Per-Language Breakdown
lang_rep = per_language_report(test_recs, gold_labels, pred_labels)
print("\\n🌐 Per-Language Breakdown:")
for lang, m in lang_rep.items():
    print(f"   • {lang:<10} Macro-F1: {m['macro_f1']:.4f} | Accuracy: {m['accuracy']:.4f} (n={m['n']})")

# 3. Detailed Class Report
print("\\n📋 Detailed Classification Report:")
print(classification_report(gold_labels, pred_labels, target_names=LABELS, digits=4))

# 4. Confusion Matrix Heatmap
cm = confusion_matrix(gold_labels, pred_labels, labels=LABELS)
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS, ax=ax)
plt.title(f"Confusion Matrix (Test Split, Macro-F1: {metrics['macro_f1']:.4f})", fontweight='bold')
plt.xlabel("Predicted Stance Label")
plt.ylabel("Gold True Label")
plt.tight_layout()
plt.show()"""))

    # ---------------- Cell 15: Interactive Sandbox Markdown ----------------
    cells.append(make_cell("markdown", """## 7. Interactive Stance Inference Sandbox 🌍
Test the trained model interactively on custom Hindi, Bengali, Hinglish, Banglish, or English sentences."""))

    # ---------------- Cell 16: Interactive Sandbox Code ----------------
    cells.append(make_cell("code", """from src.data.preprocessing import preprocess
from src.data.language_detection import detect_language, detect_script

def predict_stance_interactive(text: str):
    clean = preprocess(text, variant="minimal").text
    enc = tokenizer(clean, truncation=True, max_length=64, return_tensors="pt")
    
    student_model.eval()
    with torch.no_grad():
        logits = student_model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        
    pred_label = I2L[pred_idx]
    conf = float(probs[pred_idx])
    lang = detect_language(text)
    script = detect_script(text)
    
    badges = {"FAVOUR": "🟢 FAVOUR", "AGAINST": "🔴 AGAINST", "NONE": "⚪ NONE"}
    print("=" * 60)
    print(f'Input:       "{text}"')
    print(f"Prediction:  {badges.get(pred_label, pred_label)} (Confidence: {conf * 100:.1f}%)")
    print(f"Language:    {lang} | Script: {script}")
    print(f"Probabilities: FAVOUR: {probs[0]:.4f} | AGAINST: {probs[1]:.4f} | NONE: {probs[2]:.4f}")
    
    fig, ax = plt.subplots(figsize=(5, 1.4))
    bar_colors = ["#2ca02c" if l == pred_label else "#aec7e8" for l in LABELS]
    ax.barh(LABELS, probs, color=bar_colors, edgecolor='black', height=0.5)
    ax.set_xlim([0, 1.0])
    ax.set_xlabel("Probability")
    plt.title(f"Stance: {pred_label} ({conf * 100:.1f}%)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Test benchmark examples
samples = [
    "जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।",
    "ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है, यह सिर्फ पश्चिमी देशों का प्रोपेगैंडा है।",
    "জলবায়ু পরিবর্তন একটি বাস্তব সংকট, এখনই কার্বন নির্গমন কমাতে হবে।",
    "গ্লোবাল ওয়ার্মিং কোনো গুরুতর समस्या নয়, এটি একটি সাজানো নাটক।",
    "Climate change sach me ek bahut serious concern hai bhai.",
    "सौर ऊर्जा लगाने में कितना खर्च आता है, कोई जानकारी दे सकता है?",
]

for s in samples:
    predict_stance_interactive(s)"""))

    # ---------------- Cell 17: Custom User Input Markdown ----------------
    cells.append(make_cell("markdown", """### ✍️ Test Your Own Statement
Change `my_text` below and run the cell to classify any statement:"""))

    # ---------------- Cell 18: Custom User Input Code ----------------
    cells.append(make_cell("code", """# 👉 Enter any sentence here:
my_text = "ग्लोबल वार्मिंग कोई समस्या नहीं है"

predict_stance_interactive(my_text)"""))

    # ---------------- Cell 19: Save Model Markdown ----------------
    cells.append(make_cell("markdown", """## 8. Export Trained LoRA Checkpoint
Saves your trained weights to `models/student/lora_adapter_kd.pt` so other project modules (Streamlit app, CLI, API) immediately use your trained model."""))

    # ---------------- Cell 20: Save Model Code ----------------
    cells.append(make_cell("code", """save_dir = ROOT / "models" / "student"
save_dir.mkdir(parents=True, exist_ok=True)
adapter_path = save_dir / "lora_adapter_kd.pt"

# Extract adapter and classification head weights
adapter_state = {
    k: v.detach().clone().cpu() for k, v in student_model.state_dict().items() 
    if ("lora" in k) or k.startswith("head.")
}

torch.save(adapter_state, adapter_path)
print(f"💾 Trained adapter saved successfully to:")
print(f"   {adapter_path}")
print(f"   Size: {adapter_path.stat().st_size / 1024:.1f} KB ({len(adapter_state)} weight tensors)")"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return nb


def main():
    nb = build_training_notebook()
    p1 = ROOT / "notebooks" / "train_indic_stance_slm.ipynb"
    p2 = ROOT / "train_model.ipynb"

    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully generated:\n  - {p1}\n  - {p2}")


if __name__ == "__main__":
    main()
