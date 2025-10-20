
# Adam-mini Triton Fusion — README

## Overview

This repository contains two single-file example implementations demonstrating a Triton-fused “Adam-mini” style optimizer and a reference PyTorch version. The goal is to:

- Implement an Adam-mini optimizer that keeps a single block-level `v` statistic (mean of `g^2`) and a per-parameter first moment `m`.
- Provide a Triton-fused kernel (`_update_kernel`) that updates parameters and `m` in a single pass for performance.
- Support optional FP8 state storage with an adaptive `state_scale` and fallback to BF16.
- Provide a Hessian-aware partitioner (simple Transformer heuristic) to create blocks.
- Add telemetry and parity checks between the PyTorch reference and the Triton-fused implementation.
- Benchmark against `torch.optim.AdamW` and optionally the “official” Adam-mini package.

---

## Requirements

- Python 3.8+
- PyTorch with CUDA (for Triton runs; CPU-only will run the reference code but not the Triton kernels)
- Triton (optional; if not installed the fused kernel is disabled and a fallback behaviour is used)

Install the recommended packages:

```bash
# Basic dependencies
pip install adam-mini
pip install torch torchvision

# Triton (only if you want to run the fused kernels)
# Follow Triton official install instructions for your CUDA version.
# Example (may vary across environments):
pip install triton

```

---

## What’s included and how to run

### Code 1 (Parity + Telemetry)
- Purpose: Small toy parity check and telemetry example comparing the PyTorch reference (`AdamMiniPyTorch`) to the Triton-accelerated (`AdamMiniTriton`).
- How to run:
  1. Run the first block of code.

- Observed output:
```
Running parity test between AdamMiniPyTorch and AdamMiniTriton...

Param linear.weight: max diff 2.246886e-03

  [Telemetry] Steps=5, max_update_A=[0.004441387485712767, 0.0023785014636814594, 0.00216480134986341, 0.0015761416871100664, 0.0015192623250186443], max_m_A=[0.02251160517334938, 0.026076965034008026, 0.03351600840687752, 0.029425164684653282, 0.03513770550489426], dtype_A=torch.float32
              Steps=5, max_update_B=[0.004538297653198242, 0.002416654722765088, 0.002185906982049346, 0.001596447778865695, 0.0015406586462631822], max_m_B=[0.02294921875, 0.0264892578125, 0.033935546875, 0.02978515625, 0.03564453125], dtype_B=torch.bfloat16

Param linear.bias: max diff 1.433402e-03

  [Telemetry] Steps=5, max_update_A=[0.002155275084078312, 0.001265847124159336, 0.001437773578800261, 0.0010939586209133267, 0.0008378218044526875], max_m_A=[0.009643001481890678, 0.015522426925599575, 0.023487450554966927, 0.022799750789999962, 0.021312423050403595], dtype_A=torch.float32
              Steps=5, max_update_B=[0.002181529998779297, 0.001284872181713581, 0.0014582243748009205, 0.0011055630166083574, 0.0008453609189018607], max_m_B=[0.009765625, 0.0157470703125, 0.0238037109375, 0.0230712890625, 0.021484375], dtype_B=torch.bfloat16
```

**Notes**:
- Parity differences are small and consistent with BF16 rounding; telemetry provides `max_update` and `max_m` traces for each parameter for debugging and comparison.

---

### Code 2 (Full Benchmark + Hessian-aware partitioner)
- Purpose: Full benchmark harness that:
  - Optionally imports the official `adam-mini` package for comparison.
  - Implements a simple Hessian-aware partitioner tuned for Transformer-like modules.
  - Runs benchmarks comparing PyTorch AdamW, Official Adam-mini, and the Triton-fused Adam-mini (BF16 or FP8 state).
- How to run:
  1. Run the second block of code.

- Observed output:
```
Collecting adam-mini
  Downloading adam_mini-1.1.1-py3-none-any.whl.metadata (2.9 kB)
Downloading adam_mini-1.1.1-py3-none-any.whl (13 kB)
Installing collected packages: adam-mini
Successfully installed adam-mini-1.1.1


==================================================
          Running Full Optimizer Benchmark
==================================================

--- Benchmarking AdamW (PyTorch) ---
Total Time (5000 steps): 31.564 s | Peak Memory: 153.57 MB
Adam-mini found the param block with name: embed.weight torch.Size([1024, 512])
...
--- Benchmarking Adam-mini (Official) ---
Adam-mini found 1 embedding layers, 1 output layers; 2 Querys and Keys;  1 Values;  1 attn_proj;  2 MLPs;
Total Time (5000 steps): 30.385 s | Peak Memory: 115.83 MB

--- Benchmarking Adam-mini (Triton, BF16 State) ---
Hessian-aware partitioner created 9 blocks.
Total Time (5000 steps): 46.129 s | Peak Memory: 99.13 MB

--- Benchmarking Adam-mini (Triton, FP8 State) ---
Hessian-aware partitioner created 9 blocks.
/tmp/ipython-input-4233490169.py:35: UserWarning: PyTorch version lacks fp8e4. Falling back to bf16.
  warnings.warn(msg); _warn_once_cache.add(msg)
Total Time (5000 steps): 46.086 s | Peak Memory: 99.13 MB
```

**Notes**:
- The Triton-fused implementation with BF16/FP8-state shows higher *time* in this environment but lower *peak memory* in the benchmark because of differences in kernel layout, partitioning and environment. Performance will vary based on GPU, Triton tuning configs, and kernel optimization.
- FP8 state required both Triton support and PyTorch support; absence of FP8 in your environment triggers a BF16 fallback (see the warning printed).

---

## Implementation & Fixes mapping to client feedback

Below is a concise checklist that maps the original client feedback to what was implemented or improved in the provided code:

1. **Fix grad_scaler semantics (major correctness)**  
   - Implemented: introduced `state_scale` per-block and used an *unscale/re-scale* approach inside the Triton kernel. That is, momentum is stored scaled (for FP8) and unscaled inside the kernel to compute updates so that state scaling does not change the numeric magnitude of the update.

2. **Respect param groups / API correctness**  
   - Implemented: partitioner and `_init_groups_and_states()` attach each block to the corresponding param group and use the group's hyperparameters when updating that block.

3. **Hessian-structure-aware partitioner (Transformers)**  
   - Implemented a heuristic partitioner (`adam_mini_transformer_partition`) that groups parameters into blocks by name patterns commonly found in Transformer modules (q/k/v, mlp, embeddings, norms, output). This is a best-effort rule-based partitioner (not a perfect replacement for paper-level partitioning, but a practical start).

4. **Parallel reduce kernel & autotune**  
   - Implemented `_reduce_g2_kernel_parallel` which runs a parallel grid (one tile per triton program) and atomically accumulates into the block-level accumulator. `_update_kernel` has an autotune decorator (two BLOCK_SIZE candidates) in the benchmark code.

5. **Stability defaults**  
   - eps was set to `1e-6` in the benchmark harness for BF16-friendly defaults (configurable). `betas` default uses `0.9, 0.999` (configurable).

6. **FP8 handling and fallback**  
   - Implemented: adaptive `state_scale` per block. After each step we monitor `amax` and update the scale toward a target range. If PyTorch / hardware doesn't support FP8, the optimizer falls back to BF16 and prints a warning.

7. **Telemetry**  
   - Implemented: per-parameter telemetry stored in `optimizer.state[p]['telemetry']` (Code 1 parity example), and per-block telemetry in `AdamMiniTriton.telemetry` (Code 2 benchmark) storing `amax_m`, `scale`, and `state_dtype`. Parity test prints both optimizer telemetry summaries.

8. **Parity harness**  
   - Implemented: `parity_check` compares final parameters and prints max absolute diffs per-parameter and prints telemetry from both implementations.

---

## Known limitations & recommended next steps

- **Small numeric differences**: Parity diffs on the order of e-3 are expected when the fused implementation uses BF16 or FP8 states vs the full FP32 reference. These are typically acceptable but should be validated on a per-task basis.
- **True FP8 production support** requires compatible PyTorch + CUDA (SM90+) and Triton dtypes; otherwise the code falls back to BF16.
- **Partitioner** is heuristic-based and not a drop-in match for the Adam-mini paper’s exact Hessian partitioning rules. For best results on Transformers, implement the full paper partitioner for block formation (this code uses a practical rule set).
- **Non-contiguous / sparse gradients**: kernels assume contiguous dense layouts. The code warns on non-leaf gradients and requires gradients to be unscaled if using `torch.cuda.amp`’s GradScaler (call `scaler.unscale_(optimizer)` beforehand).
- **More autotuning & kernel optimization**: Add more `BLOCK_SIZE` configs and different `num_warps` to find the best kernel configuration for your GPU.

---

## Quick Troubleshooting

- If Triton kernel compilation errors appear, ensure Triton is installed and matched to your CUDA toolkit and GPU architecture.
- If you see the FP8 fallback warning, your PyTorch build or GPU does not yet support the expected FP8 dtype — BF16 fallback is used and training remains numerically safe.
- If parity diffs are large (>1e-2), re-run with `state_dtype="bf16"` and `eps=1e-6` to see if numerical differences shrink; also ensure no other global randomness/seed differences and that both models start with identical weights.

---


