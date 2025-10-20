
# Triton-Fused Adam-mini Optimizer

This repository implements a **Triton-fused version of the Adam-mini optimizer** with FP8 and BF16 precision support, Hessian-aware partitioning, telemetry, and parity validation against the PyTorch implementation.

---

## 🧠 Background

**Adam-mini** is a memory-efficient variant of Adam designed for large-scale training, optimized for GPU efficiency and minimal state storage.  
In this work, we extend it with:

- **Triton-based fusion** for kernel efficiency.
- **FP8 (E4M3) state compression** with dynamic scaling and fallback to BF16.
- **Hessian-aware block partitioning**, improving per-parameter variance estimation.
- **Telemetry** and **parity validation** for numerical consistency.

---

## 🚀 Objectives

The goals of this implementation are:

1. Implement **Triton fused kernels** for `momentum`, `variance`, and `update` operations.  
2. Maintain **per-block mean-square gradients** using efficient reduction.  
3. Store the **first-moment buffer (m)** in **FP8 precision**, dynamically scaled.  
4. Automatically **fallback to BF16** when FP8 saturation occurs.  
5. Add **telemetry** to track performance, precision fallback rate, and kernel execution metrics.  
6. Add **parity testing** against the PyTorch Adam-mini reference implementation.

---

## ⚙️ Implementation Overview

### 1. Triton Kernel Fusion

We implemented two Triton kernels:

- **`_reduce_g2_kernel`** – computes blockwise mean of `grad^2` (variance estimate).
- **`_update_kernel`** – performs the AdamW update step, using FP8/BF16 arithmetic.

The kernels are autotuned and support mixed-precision arithmetic.

---

### 2. FP8 Momentum Storage

To reduce memory footprint, the optimizer stores the momentum (`m`) in FP8 (E4M3) format.

- Each block maintains a **dynamic scaling factor (S)**.  
- During update, values are **unscaled (m / S)** to maintain mathematical consistency.  
- If saturation persists, the block **falls back to BF16** precision automatically.

---

### 3. Hessian-Aware Partitioning

Instead of using fixed-size blocks, parameters are partitioned based on their **estimated Hessian structure**:

- Q/K weights → per-attention head.
- Proj/MLP weights → per-output dimension.
- Embeddings → per-token partition.

This ensures more stable block-level statistics and improved convergence stability.

---

### 4. Telemetry & Monitoring

Telemetry provides insight into runtime performance and stability. It tracks:

- FP8 vs BF16 block usage.  
- Maximum update magnitudes (`amax_update`).  
- Per-step timing metrics.  
- Saturation fallback count.  

Telemetry summaries are printed after training and can be extended to write to file or dashboard.

---

### 5. Parity Validation

A **parity check** ensures numerical consistency between the Triton and PyTorch implementations:

```python
parity_check(SimpleModel, AdamMiniPyTorch, AdamMiniTriton, steps=5, tol=1e-3)
```

If the optimizer matches PyTorch’s updates (within tolerance), parity passes.

---


## 🧩 Key Features

| Feature | Description |
|----------|-------------|
| **FP8/BF16 mixed precision** | Adaptive storage with dynamic scaling |
| **Triton fused kernels** | Reduces Python overhead & improves GPU utilization |
| **Hessian-aware partitioning** | Stable block updates and faster convergence |
| **Telemetry** | Monitors precision fallback and kernel stats |
| **Parity validation** | Ensures math consistency with PyTorch |
| **Safety fallbacks** | Handles unsupported hardware or Triton absence |

---


## 🧰 Fallback Behavior

- Triton missing → uses PyTorch version automatically.
- FP8 unsupported → uses BF16.
- FP8 saturation → per-block fallback to BF16 with warning.

---

## 📚 References

- [Triton Language Docs](https://triton-lang.org/main/index.html)
- [WhyPhyLabs/s3gd Optimizer](https://github.com/WhyPhyLabs/s3gd)
- [zyushun/Adam-mini](https://github.com/zyushun/Adam-mini)

---

## 📜 License

MIT License © 2025
