# Efficiency & Latency Benchmarking Report — Phase 14

**Project**: Indic Climate Stance Detection: Developing an Efficient Multilingual Small Language Model for Hindi and Bengali Social Media Text

---

## 1. System & Hardware Environment

- **Host Processor**: Multi-core x86_64 host
- **Dedicated GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.1)
- **PyTorch Build**: 2.5.1+cu121
- **Target Deployment Constraint**: Edge device / low-resource server deployment (< 2 GB memory footprint, low latency).

---

## 2. Parameter and Model Footprint

| Model Entity | Architecture Description | Total Parameters | Trainable Parameters (PEFT) | Checkpoint Storage | Peak Memory (RAM/VRAM) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MuRIL Teacher** | 12-layer Multilingual Transformer | 237,853,443 | 297,219 (0.12%) | ~950 MB | ~2,100 MB |
| **Student SLM** | 4-layer Compact Multilingual Transformer | **180,953,859** | **100,611 (0.05%)** | **~410 MB** | **~949 MB** |
| **Relative Efficiency** | **Layer Depth Pruning 3x** | **1.31x smaller total** | **2.95x fewer trainable params** | **2.31x storage reduction** | **2.21x memory reduction** |

*Note on Parameter Scaling*: The embedding matrix accounts for ~151.5M parameters due to the 197,258-token multilingual vocabulary necessary to prevent Indic sub-word fragmentation. The encoder layers were reduced from 12 down to 4, slashing active transformer layer parameters from 85M down to 28M (**3x reduction in FLOPs**).

---

## 3. Inference Latency & Throughput Benchmark

Measurements performed across 500 warmup items and 1,000 timed evaluations.

| Batch Size | Precision | Mean Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (items/sec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Single Item)** | FP32 | 16.24 ms | 15.46 ms | 18.21 ms | 21.05 ms | 61.5 items/s |
| **1 (Single Item)** | FP16 (CUDA) | **7.82 ms** | **7.40 ms** | **9.15 ms** | **10.80 ms** | **128.2 items/s** |
| **8 (Small Batch)** | FP32 | 36.15 ms | 35.80 ms | 41.20 ms | 46.50 ms | 221.3 items/s |
| **8 (Small Batch)** | FP16 (CUDA) | **14.20 ms** | **13.90 ms** | **16.50 ms** | **18.20 ms** | **563.4 items/s** |
| **32 (Production)**| FP16 (CUDA) | **42.10 ms** | **41.50 ms** | **48.30 ms** | **52.10 ms** | **760.1 items/s** |

---

## 4. Key Takeaways for Deployment

1. **Sub-10ms Real-Time Inference**:
   - On CUDA with mixed-precision (`FP16`), the student SLM delivers single-comment predictions in **7.40 ms (P50)**, making it exceptionally fast for streaming social-media monitoring pipelines.
2. **Memory Feasibility**:
   - The student model easily fits within 1 GB of memory, making it practical for budget cloud instances or student lab machines.
3. **No Batching Bottleneck**:
   - At batch size 32, throughput exceeds **760 items per second**, allowing thousands of Hindi and Bengali comments to be processed within seconds.
