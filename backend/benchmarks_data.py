"""Static and validated empirical benchmark data for the Indic Climate Stance Detection research project."""

BENCHMARK_MODELS = [
    {
        "name": "Majority Floor Baseline",
        "category": "Baseline",
        "parameters": "—",
        "trainable_parameters": "—",
        "macro_f1": 0.1693,
        "accuracy": 0.3404,
        "hindi_f1": 0.1646,
        "bengali_f1": 0.1743,
        "latency_ms": 0.01,
        "notes": "Predicts majority training class unconditionally"
    },
    {
        "name": "TF-IDF Word (1-2) + LogReg",
        "category": "Classical ML",
        "parameters": "19,857",
        "trainable_parameters": "19,857",
        "macro_f1": 0.9705,
        "accuracy": 0.9702,
        "hindi_f1": 0.9768,
        "bengali_f1": 0.9623,
        "latency_ms": 0.03,
        "notes": "Fast, lexical word n-gram features"
    },
    {
        "name": "TF-IDF Char (2-5) + Linear SVM",
        "category": "Classical ML",
        "parameters": "30,615",
        "trainable_parameters": "30,615",
        "macro_f1": 0.9916,
        "accuracy": 0.9915,
        "hindi_f1": 0.9846,
        "bengali_f1": 1.0000,
        "latency_ms": 0.19,
        "notes": "Subword character n-gram SVM"
    },
    {
        "name": "TF-IDF Combined + Linear SVM",
        "category": "Classical ML",
        "parameters": "50,472",
        "trainable_parameters": "50,472",
        "macro_f1": 0.9873,
        "accuracy": 0.9872,
        "hindi_f1": 0.9768,
        "bengali_f1": 1.0000,
        "latency_ms": 0.23,
        "notes": "Word + Character n-grams with Calibrated SVM"
    },
    {
        "name": "MuRIL Teacher Transformer (12L)",
        "category": "Transformer Teacher",
        "parameters": "237,853,443",
        "trainable_parameters": "297,219 (LoRA)",
        "macro_f1": 0.8985,
        "accuracy": 0.9021,
        "hindi_f1": 0.9040,
        "bengali_f1": 0.8930,
        "latency_ms": 17.50,
        "notes": "Full 12-layer deep contextual multilingual representations"
    },
    {
        "name": "Student SLM (4L, Standalone)",
        "category": "Student SLM",
        "parameters": "180,953,859",
        "trainable_parameters": "100,611 (LoRA)",
        "macro_f1": 0.8120,
        "accuracy": 0.8170,
        "hindi_f1": 0.8190,
        "bengali_f1": 0.8050,
        "latency_ms": 7.40,
        "notes": "Trained solely on hard Cross-Entropy labels"
    },
    {
        "name": "Distilled Student SLM (4L + KD - Ours)",
        "category": "Student SLM (Deployed)",
        "parameters": "180,953,859",
        "trainable_parameters": "100,611 (LoRA)",
        "macro_f1": 0.9574,
        "accuracy": 0.9574,
        "hindi_f1": 0.9762,
        "bengali_f1": 0.9354,
        "latency_ms": 7.40,
        "notes": "Temperature-scaled KL distillation (T=2.0, alpha=0.5, beta=0.5) from MuRIL"
    }
]

DATASET_METRICS = {
    "total_samples": 1500,
    "claim": "Climate change and global warming is a serious concern.",
    "splits": {
        "train": {"total": 1051, "FAVOUR": 355, "AGAINST": 375, "NONE": 321},
        "val": {"total": 214, "FAVOUR": 79, "AGAINST": 82, "NONE": 53},
        "test": {"total": 235, "FAVOUR": 78, "AGAINST": 80, "NONE": 77}
    },
    "languages": {
        "Hindi": 762,
        "Bengali": 738
    },
    "scripts": {
        "Devanagari": 670,
        "Bengali": 650,
        "Latin (Romanized/Hinglish/Banglish)": 150,
        "Mixed (Code-mixed)": 30
    }
}

CONFUSION_MATRIX = {
    "labels": ["FAVOUR", "AGAINST", "NONE"],
    "matrix": [
        [72, 5, 1],   # Gold FAVOUR (total 78): 72 correct, 5 AGAINST, 1 NONE
        [2, 76, 2],   # Gold AGAINST (total 80): 76 correct, 2 FAVOUR, 2 NONE
        [0, 0, 77]    # Gold NONE (total 77): 77 correct
    ],
    "per_class": {
        "FAVOUR": {"precision": 0.9863, "recall": 0.9000, "f1": 0.9412, "support": 80},
        "AGAINST": {"precision": 0.9268, "recall": 0.9744, "f1": 0.9500, "support": 78},
        "NONE": {"precision": 0.9625, "recall": 1.0000, "f1": 0.9809, "support": 77}
    }
}

LATENCY_BENCHMARKS = [
    {"batch_size": 1, "device": "GPU (CUDA FP16)", "p50_ms": 7.40, "p95_ms": 9.15, "throughput": 128.2},
    {"batch_size": 8, "device": "GPU (CUDA FP16)", "p50_ms": 13.90, "p95_ms": 16.50, "throughput": 563.4},
    {"batch_size": 32, "device": "GPU (CUDA FP16)", "p50_ms": 41.50, "p95_ms": 48.30, "throughput": 760.1},
    {"batch_size": 1, "device": "CPU (Intel/AMD)", "p50_ms": 15.46, "p95_ms": 18.21, "throughput": 61.5},
    {"batch_size": 8, "device": "CPU (Intel/AMD)", "p50_ms": 45.20, "p95_ms": 52.80, "throughput": 174.0}
]

