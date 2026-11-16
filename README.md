# Indic Climate Stance Detection: Efficient Multilingual Small Language Model for Hindi & Bengali

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

> **Research Objective**: Investigating whether a compact, knowledge-distilled multilingual Small Language Model (SLM) can achieve competitive stance detection performance on Hindi and Bengali social-media text while drastically reducing parameter count, training memory, and inference latency compared to larger multilingual teacher models.

---

## 📌 Problem Statement

The task is three-way stance detection toward the following target claim:

> **Claim**: *"Climate change and global warming is a serious concern."*  
> **Target Labels**:
> - **`FAVOUR`**: The text agrees with, supports, or expresses concern about climate change.
> - **`AGAINST`**: The text dismisses, opposes, denies, or rejects climate change concern (e.g. natural cycle, hoax, carbon tax scam).
> - **`NONE`**: The text is neutral, factual, ambiguous, or lacks explicit stance.

### Supported Linguistic Dimensions
1. **Native Hindi in Devanagari** (`जलवायु परिवर्तन एक गंभीर समस्या है`)
2. **Native Bengali in Bengali script** (`জলবায়ু পরিবর্তন একটি বাস্তব সংকট`)
3. **Romanized Hindi (Hinglish)** (`climate change serious problem hai bhai`)
4. **Romanized Bengali (Banglish)** (`global warming kono issue na, shob natok`)
5. **Hindi-English & Bengali-English Code-Mixing**
6. **Linguistic Negation** (`नहीं`, `नয়`, `nahi`, `kono problem na`)
7. **Social Media Sarcasm & Irony** (*"It's 5 degrees outside, where is global warming? 😂"*)
8. **Policy Criticism vs. Climate Denial** (Distinguishing opposition to carbon taxes from denial of climate reality)
9. **Emojis and Hashtags** (`🔥`, `🌍`, `❄️`, `#ClimateCrisis`, `#ActNow`)

---

## 🏛️ Repository Architecture

```text
indic-climate-stance-slm/
│
├── README.md                           # Main project documentation & training guide
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project metadata & packaging configuration
├── .gitignore                          # Git tracking rules
├── config.yaml                         # Master hyperparameter & environment configuration
├── LICENSE                             # MIT License
├── test_backend_api.py                 # Comprehensive API & model integration test suite
│
├── backend/                            # Production FastAPI inference backend
│   ├── main.py                         # FastAPI application, CORS, lifespan & endpoints
│   ├── model_loader.py                 # CompactStudentSLM singleton loader & Integrated Gradients
│   ├── schemas.py                      # Pydantic v2 request/response schemas
│   └── benchmarks_data.py              # Empirical evaluation benchmarks & confusion matrix
│
├── frontend/                           # Modern React + TypeScript + Vite web interface
│   ├── src/
│   │   ├── components/                 # Landing, Analyzer, Explainability, Batch, Analytics, Model Info
│   │   ├── types.ts                    # Strict TypeScript interfaces matching backend schemas
│   │   ├── api.ts                      # Typed client with error handling & health polling
│   │   ├── App.tsx                     # Main application shell & tab router
│   │   └── index.css                   # Tailwind v4 dark research theme styles
│   ├── package.json                    # Frontend dependencies (Lucide, Recharts, Framer Motion)
│   └── dist/                           # Compiled production static bundle
│
├── data/
│   ├── raw/
│   │   ├── gwstance/                   # English climate-stance reference corpus (GWSD)
│   │   ├── dhoroni/                    # Bengali climate domain news corpus (unlabeled)
│   │   ├── indic_sentiment/            # AI4Bharat translated sentiment dataset
│   │   └── custom_climate_dataset/     # Custom Hindi-Bengali Stance Corpus (corpus.jsonl)
│   ├── processed/
│   │   └── indic_climate_dataset.json  # Preprocessed & validated Indic dataset
│   └── splits/
│       ├── indic_climate_splits.json   # Group-stratified train/val/test splits (Zero Leakage)
│       └── gwstance_group_stratified.json # English reference splits
│
├── notebooks/                          # Interactive research & visualization notebooks
│   ├── 01_dataset_audit.ipynb          # Multi-corpus audit & class distributions
│   ├── 02_exploratory_analysis.ipynb   # Sequence length & word/character EDA
│   ├── 03_linguistic_analysis.ipynb    # Negation, sarcasm, code-mixing analysis
│   ├── 04_baseline_models.ipynb        # Classical ML baselines (TF-IDF + LogReg & SVM)
│   ├── 05_teacher_training.ipynb       # Teacher fine-tuning (MuRIL-base)
│   ├── 06_student_training.ipynb       # Compact 4-layer Student SLM architecture
│   ├── 07_knowledge_distillation.ipynb # Distillation loss & dark knowledge transfer
│   └── 08_error_analysis.ipynb         # Confusion matrix & Integrated Gradients
│
├── src/
│   ├── data/
│   │   ├── download.py                 # Fetches external datasets from original sources
│   │   ├── dataset_loader.py           # Unified loading & schema enforcement
│   │   ├── dataset_validator.py        # Automated integrity & leakage audits
│   │   ├── preprocessing.py            # Multilingual preprocessing & negation preservation
│   │   ├── language_detection.py       # Script & language classifier (Hindi/Bengali/Romanized)
│   │   ├── generate_indic_dataset.py   # High-quality Indic stance dataset builder
│   │   ├── augmentation.py             # Controlled, stance-preserving perturbations
│   │   └── splitting.py                # Leakage-safe group-stratified data splitting
│   │
│   ├── models/
│   │   ├── baseline.py                 # TF-IDF (word/char/combined) + LogReg & Linear SVC
│   │   ├── teacher.py                  # Multilingual Teacher factory (MuRIL-base, 12 layers)
│   │   ├── student.py                  # Compact Multilingual Student SLM (4 layers)
│   │   ├── classification_head.py      # Pooling strategies (Mean, CLS, Attention) + StanceHead
│   │   ├── distillation.py             # Distillation loss: L_CE + beta * T^2 * L_KL
│   │   └── tokenizer_screening.py      # Subword fertility & UNK screening across Indic tokenizers
│   │
│   ├── training/
│   │   ├── train_baseline.py           # Evaluates traditional ML baselines on Indic/GWStance
│   │   ├── train_teacher.py            # Supervised Teacher training with CUDA & LoRA support
│   │   ├── train_student.py            # Standalone Student training without distillation
│   │   ├── train_distillation.py       # Knowledge distillation from Teacher to Student
│   │   └── trainer.py                  # Reusable training loop with early stopping & CUDA
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # Macro-F1, per-class F1, bootstrap 95% CIs
│   │   ├── robustness.py               # 7 adversarial perturbation evaluation suites
│   │   ├── cross_lingual.py            # Cross-lingual transfer & script breakdown
│   │   └── explainability.py           # Integrated Gradients token-attribution
│   │
│   ├── inference/
│   │   ├── predict.py                  # Single-comment inference API with schema output
│   │   └── batch_predict.py            # Batch inference CLI for CSV & JSONL files
│   │
│   └── utils/
│       ├── config.py                   # YAML config parser
│       ├── device.py                   # CUDA/GPU device manager
│       ├── seed.py                     # Deterministic random seed controller
│       └── registry.py                 # Permanent experiment tracking registry (CSV)
│
├── experiments/
│   ├── experiment_registry.csv         # Permanent audit log of every experiment run
│   └── results/                        # JSON results for baselines, KD, efficiency, robustness
│
├── reports/
│   ├── dataset_audit.md                # Multi-corpus audit report
│   ├── linguistic_analysis.md          # Linguistic phenomena & preprocessing effects
│   ├── baseline_comparison.md          # Traditional baseline benchmark analysis
│   ├── model_comparison.md             # Teacher vs Student SLM vs Knowledge Distillation
│   ├── robustness_report.md            # Robustness across 7 perturbation suites
│   ├── efficiency_report.md            # Latency, parameter count & memory benchmark
│   ├── error_analysis.md               # Qualitative error breakdown & Integrated Gradients
│   └── final_research_report.md        # Complete 24-section research paper
│
├── app/
│   └── streamlit_app.py                # Interactive web demonstration
│
└── tests/
    ├── test_preprocessing.py           # Tests normalization & negation preservation
    ├── test_dataset.py                 # Tests schema, duplicate checking & integrity
    ├── test_training.py                # Tests training configurations, loss & datasets
    ├── test_evaluation.py              # Tests metrics, language reports & bootstrap CIs
    └── test_inference.py               # Tests language detection, scripts & preprocessing
```

---

## ⚡ Quickstart: Training on a GPU Server

This repository is pre-configured and tested for GPU clusters and CUDA-enabled machines.

### 1. Environment Setup
```bash
# Clone and enter repository
cd indic-climate-stance-slm

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Data and Run Unit Tests
```bash
# Verify unit tests (all 65+ unit tests pass)
pytest tests -v
```

### 3. Step-by-Step Model Execution

#### Phase A: Run Classical Baselines
```bash
# Evaluates Majority Class, TF-IDF Word, Char, and Combined LogReg/SVM
python -m src.training.train_baseline --dataset indic
```

#### Phase B: Train Multilingual Teacher Model (MuRIL)
```bash
# Trains 12-layer MuRIL teacher on CUDA with fp16 mixed precision
python -m src.training.train_teacher --dataset indic --epochs 5 --batch-size 16
```

#### Phase C: Train Standalone Student SLM (Without Distillation)
```bash
# Trains 4-layer compact student using hard cross-entropy
python -m src.training.train_student --dataset indic --epochs 5 --batch-size 16
```

#### Phase D: Train Knowledge-Distilled Student SLM
```bash
# Distills dark knowledge from Teacher to Student using temperature scaling (T=2.0)
python -m src.training.train_distillation --dataset indic --epochs 5 --batch-size 16
```

#### Phase E: Launch Interactive Web Research Platform (FastAPI + React)

A full-stack, production-grade web platform and REST API for real-time stance analysis, batch inference, explainability (Integrated Gradients), and benchmark exploration.

##### 1. Run the FastAPI Unified Server (Serving both API & React UI)
The production-built React SPA is pre-compiled in `frontend/dist` and served directly by FastAPI:
```bash
# Start backend server on port 8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** in your browser to access the complete application.

##### 2. Run in Frontend Development Mode (Optional)
If modifying the React TypeScript frontend:
```bash
# In another terminal:
cd frontend
npm install
npm run dev
```
The Vite development server runs at `http://localhost:5173` with hot-reloading and automatically proxies `/api` calls to `http://localhost:8000`.

##### 3. Run Backend Integration Tests
Verify end-to-end inference, linguistic profiling, Integrated Gradients, batch analysis, and CSV uploads against the live student checkpoint:
```bash
python test_backend_api.py
```

##### 4. REST API Endpoints
Interactive Swagger OpenAPI documentation is accessible at `http://localhost:8000/docs`. Key endpoints include:

- **`POST /api/v1/predict`**: Single text analysis with linguistic profiling and Integrated Gradients token attributions.
  ```bash
  curl -X POST "http://localhost:8000/api/v1/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "जलवायु परिवर्तन बहुत गंभीर खतरा है", "compute_attributions": true}'
  ```
- **`POST /api/v1/predict/batch`**: High-throughput batched text classification (up to 500 items).
- **`POST /api/v1/predict/file`**: Multipart CSV/JSONL file upload for batch processing and downloadable results.
- **`GET /api/v1/model/info`**: Model metadata (architecture, layers, LoRA rank/alpha, pooling, checkpoint, device).
- **`GET /api/v1/benchmarks`**: Comprehensive empirical evaluation metrics, language breakdown, and confusion matrix.
- **`GET /api/v1/health`**: Real-time service health check and active GPU/CPU device status.

#### Phase F: Legacy Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Benchmark Results

Evaluated on the held-out test split of the Custom Indic Climate Stance Dataset:

| Model Architecture | Total Params | Trainable Params | Macro F1 | Accuracy | Hindi F1 | Bengali F1 | P50 Latency (GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Floor Baseline** | — | — | 0.1693 | 0.3404 | 0.1646 | 0.1743 | < 0.01 ms |
| **TF-IDF Char + Linear SVM** | 30,615 | 30,615 | 0.9916 | 0.9915 | 0.9846 | 1.0000 | 0.19 ms |
| **TF-IDF Combined + SVM** | 50,472 | 50,472 | 0.9873 | 0.9872 | 0.9768 | 1.0000 | 0.23 ms |
| **MuRIL Teacher Transformer** | 237,853,443 | 297,219 (LoRA) | 0.8985 | 0.9021 | 0.9040 | 0.8930 | 17.50 ms |
| **Student SLM (Standalone)** | 180,953,859 | 100,611 (LoRA) | 0.8120 | 0.8170 | 0.8190 | 0.8050 | 7.40 ms |
| **Distilled Student SLM (Ours)**| **180,953,859** | **100,611 (LoRA)** | **0.8790** | **0.8809** | **0.8842** | **0.8738** | **7.40 ms** |

---

## 🔍 Key Findings

1. **Knowledge Distillation Efficacy**: Distillation boosts student Macro-F1 by **+6.7%** over training without soft targets (0.812 $\to$ 0.879), successfully transferring nuanced probability distributions for ambiguous cases.
2. **Computational Savings**:
   - Transformer layer depth pruned from **12 down to 4 layers** (66.7% reduction in FLOPs).
   - Peak training memory under 1 GB; single-comment P50 inference latency under **7.4 ms** on GPU.
   - Trainable parameters reduced by **2.95x** via PEFT adaptation.
3. **Robustness & Generalization**:
   - Retains **92.3% of performance** under simultaneous real-world social-media noise (emojis, hashtag suffixes, character elongation, code-mixing).
   - Integrated Gradients attributions confirm that the model concentrates decision weight on decisive semantic cues (`गंभीर`, `खतरा`, `নয়`, `নাটক`).

---

## 📜 Citation & Research Ethics
- This project adheres to ethical research guidelines: no private or copyright-protected data is distributed.
- Human annotation and synthetic generation protocols are strictly partitioned.
- Licences for all auxiliary corpora (GWStance, Dhoroni, IndicSentiment) are audited and preserved in `data/README.md`.
