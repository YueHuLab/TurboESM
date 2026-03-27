# TurboESM

**TurboESM** ports Google's TurboQuant quantization technique to the ESM-2 protein language model. It eliminates numerical outliers in the KV Cache via orthogonal rotation matrices, combines 3-bit Lloyd-Max quantization with QJL residual correction, and achieves substantial memory reduction with minimal accuracy loss.

## What We Built

### Core Technical Contributions

**1. TurboQuant × ESM-2 Adaptation**

Google TurboQuant was originally designed for large language models. ESM-2 uses Rotary Position Embeddings (RoPE), which conflicts with TurboQuant's orthogonal rotation matrix $\Pi$ in terms of application order. We derived and validated the correct execution order — apply RoPE first, then $\Pi$ — leveraging the inner-product invariance of orthogonal matrices to guarantee attention equivalence:

$$Q' K'^T = (Q_{rope} \Pi^T)(\Pi K_{rope}^T) = Q_{rope} K_{rope}^T$$

**2. Data-Driven $\Pi$ Matrix Calibration**

For each layer and each attention head, we run SVD on post-RoPE activations from real protein sequences to obtain the optimal orthogonal rotation matrix. This makes the rotated K distribution approximate an isotropic Gaussian — the key prerequisite for high-quality 3-bit quantization.

**3. Separate K/V LUT Design**

K and V have significantly different numerical distributions (K is heavy-tailed; V is more uniform). Using a shared LUT drops V's SNR from 4x to 1.9x. We calibrate independent 8-point Lloyd-Max lookup tables for K (rotation space) and V (original space).

**4. QJL 1-bit Residual Correction**

We store the sign bit of the quantization residual (1-bit). At decode time, a first-order correction is applied using residual mean magnitude × sign, further reducing quantization error with negligible memory overhead (1-bit is almost free relative to 3-bit).

**5. Mac MPS / CPU Full Compatibility**

All quantization, packing, and unpacking operations are implemented in pure PyTorch for environments without CUDA/Triton, ensuring full functional correctness. The Triton kernel acceleration path is enabled automatically when CUDA is available.

**6. Triton Fused Decode Attention Kernel**

On CUDA, we implement a complete fused decode attention kernel that merges "3-bit unpack → QJL residual correction → Flash Attention-style online softmax → V weighted sum" into a single Triton kernel, eliminating intermediate tensor allocations. Validated layer-by-layer across all 33 layers; output error vs. the PyTorch two-step path is < 1e-6.

**7. Vectorized Quantization**

Rewrote `_pack_3bit`, `_unpack_3bit`, and `_pack_1bit` from Python for-loops to PyTorch vectorized operations (`torch.arange` + broadcast bitshift). Merged quantization (argmin) and residual sign computation into a single forward pass, eliminating redundant distance computations. Prefill quantization overhead reduced from ~63ms to ~21ms (33 layers, 143-token sequence).

---

### Accuracy Results (ESM-2 650M, Mac MPS)

| Sequence Type | Length | Prefill Similarity | Decode Similarity |
|---------------|--------|--------------------|-------------------|
| Short peptide (Insulin B-chain) | 32 | 1.0000 | 0.9603 |
| Medium (Hemoglobin α) | 143 | 1.0000 | 0.9639 |
| Hydrophobic transmembrane | 39 | 1.0000 | 0.9710 |
| Low-complexity repeat | 38 | 1.0000 | 0.9735 |
| Enzyme active site fragment | 51 | 1.0000 | 0.9641 |
| IDR disordered long sequence | 165 | 1.0000 | 0.9757 |

- Prefill: similarity **1.0000** (original-precision KV, zero loss)
- Decode: similarity **> 0.96** (3-bit quantized KV Cache, target > 0.95)
- Theoretical memory savings: **~5x** (FP32 → 3-bit, KV Cache portion)

### Accuracy Results (ESM-2 650M, NVIDIA GPU, Colab)

All sequences achieve exact Prefill similarity of 1.000000 on CUDA:

| Sequence Type | Length | Prefill Similarity |
|---------------|--------|--------------------|
| Short peptide (Insulin B-chain) | 32 | **1.000000** |
| Medium (Hemoglobin α) | 143 | **1.000000** |
| Long sequence (IDR disordered) | 165 | **1.000000** |

Triton fused decode attention kernel validated across all 33 layers:

| Metric | Value |
|--------|-------|
| Mean cosine similarity | 1.000000 |
| Max absolute error | < 1e-6 (FP32 precision floor) |

---

## Quick Start

### Installation

```bash
pip install torch transformers scipy
# CUDA environments only (not needed on Mac)
# pip install triton
```

### Step 1: Calibrate and Generate Weights

```bash
cd esm_turbo_sdk
python run_calibrate.py
```

Calibration runs SVD + K-means on real protein activations for each layer and head. Takes ~2–5 minutes and produces `weights/esm2_650M_turbo.pt`.

### Step 2: Inference

```python
from esm_turbo.turbo_esm import TurboESM

model = TurboESM(
    model_dir="weights/esm2_650M",
    checkpoint_path="weights/esm2_650M_turbo.pt"
)

features = model.generate("MKTVRQERLKSIVVLGAGGVGSAVADYLRQKGIPVT")
print(f"Output shape: {features.shape}")  # (1, seq_len, 1280)
```

### Step 3: Accuracy Validation

```bash
python run_compare.py
```

Outputs per-sequence Prefill/Decode cosine similarity compared against the original ESM-2.

### Custom Calibration Sequences

Edit the `calibration_sequences` list in `esm_turbo/calibrate.py` to include representative sequences from your target protein family, improving quantization accuracy for that domain.

---

## Colab

The project includes dedicated Colab scripts that handle HuggingFace model path compatibility with `os.path.abspath`:

```python
# Initialize (run after every Runtime restart)
exec(open('/content/colab_setup.py').read())

# Accuracy validation (6 sequences vs. original model)
exec(open('/content/colab_alignment_test.py').read())

# CUDA performance benchmark (speed + memory)
exec(open('/content/colab_benchmark_cuda.py').read())

# Decode scenario benchmark (Triton kernel vs. PyTorch)
exec(open('/content/colab_benchmark_decode.py').read())

# Triton fused kernel correctness verification
exec(open('/content/colab_verify_fused_kernel.py').read())
```

---

## Project Structure

```
esm_turbo_sdk/
├── esm_turbo/
│   ├── modeling_esm_turbo.py   # TurboEsmSelfAttention: Pi rotation + RoPE adaptation
│   ├── kv_cache.py             # 3-bit KV Cache management with QJL residual correction (vectorized)
│   ├── calibrate.py            # SVD + K-means data-driven calibration
│   ├── builder.py              # Random weight initialization (for quick validation)
│   ├── turbo_esm.py            # TurboESM unified entry point
│   └── triton_kernels.py       # Triton CUDA kernel (fused dequant + decode attention)
├── weights/
│   ├── esm2_650M/              # Original ESM-2 650M model files (not in git)
│   └── esm2_650M_turbo.pt      # Turbo calibration weights: Pi matrix + LUT + residual scale (not in git)
├── colab_setup.py
├── colab_alignment_test.py
├── colab_benchmark_cuda.py
├── colab_benchmark_decode.py
├── colab_verify_fused_kernel.py
├── run_calibrate.py
├── run_compare.py
└── run_benchmark.py            # Speed benchmark (CUDA environments)
```

---

## Technical Background

### Why $\Pi$ Rotation?

ESM-2's KV activations exhibit significant numerical outliers. Direct 3-bit quantization wastes most LUT points covering those outliers, severely degrading quantization quality for the main distribution. The orthogonal rotation matrix $\Pi$ redistributes energy across dimensions, pushing the distribution toward an isotropic Gaussian. With outliers eliminated, 3-bit quantization SNR improves dramatically.

### Prefill vs. Decode

- **Prefill** (`seq_len > 1`): Attention is computed in original FP32 precision. KV is simultaneously quantized and packed into the cache. Output is identical to the original model (similarity 1.0000).
- **Decode** (`seq_len == 1`): KV is unpacked from the 3-bit cache, corrected with QJL residual, then used for attention. Similarity > 0.96.

### Correspondence to TurboQuant Paper

| Paper Component | This Implementation |
|-----------------|---------------------|
| Orthogonal rotation $\Pi$ (SVD calibration) | `calibrate.py` SVD + `pi_matrix` buffer |
| RoPE compatibility | `modeling_esm_turbo.py`: RoPE before $\Pi$ |
| 3-bit Lloyd-Max LUT | `lut_k` / `lut_v` independently calibrated |
| QJL 1-bit residual correction | `qjl_cache` + `residual_scale_k` |
| Triton kernel fusion | `triton_kernels.py` ✅ implemented and validated |

---

## CUDA Performance (ESM-2 650M, NVIDIA GPU, Colab)

### Prefill Latency

| Sequence | Tokens | Original | TurboESM | Note |
|----------|--------|----------|----------|------|
| Short peptide | 32 | 31 ms | 57 ms | KV quantization pack overhead |
| Medium | 143 | 77 ms | 104 ms | Overhead fraction decreases with length |
| Long sequence | 165 | 82 ms | 103 ms | ~21 ms overhead (33-layer pack) |

> TurboESM is slower during Prefill due to the additional KV quantization and packing step.
> TurboAttention itself (Pi rotation + attention) adds only ~2ms, negligible.

### KV Cache Memory (max_seq=1024)

| Metric | Value |
|--------|-------|
| FP32 KV Cache | **330.0 MB** |
| 3-bit packed K+V | 41.2 MB |
| 1-bit QJL signs | 5.2 MB |
| **Turbo total** | **46.6 MB** |
| **Actual compression ratio** | **7.1x** |

### Triton Fused Kernel Performance

| Operation | PyTorch | Triton | Speedup |
|-----------|---------|--------|---------|
| fetch_unpacked (143 tok) | 1.19 ms | 0.61 ms | **1.96x** |
| decode attention (single layer) | — | single fused kernel | eliminates intermediate memory |

---

## ESM-2 15B Analysis

ESM-2 15B (`esm2_t48_15B_UR50D`): 48 layers, 20 heads, head_dim=320.

| Metric | ESM-2 650M | ESM-2 15B |
|--------|------------|-----------|
| Model parameters (FP16) | ~1.2 GB | ~30 GB |
| FP32 KV Cache (seq=1024) | 330 MB | **7.7 GB** |
| Turbo KV Cache (seq=1024) | 47 MB | **1.1 GB** |
| **KV Cache savings** | 284 MB | **6.6 GB** |
| Compression ratio | 7.1x | 7.2x |

**Practical impact on A100 40GB:**
- Original 15B (FP16) + FP32 KV Cache = 30 + 7.7 = **37.7 GB** → batch=1 only
- Original 15B (FP16) + Turbo KV Cache = 30 + 1.1 = **31.1 GB** → **batch=4–6**

> KV Cache compression provides limited benefit for 650M (330MB), but is critical for 15B (7.7GB) — directly determining whether the model fits on a single GPU and the maximum batch size.

---

## Use Cases

| Scenario | Suitability | Notes |
|----------|-------------|-------|
| ESM-2 15B single-GPU deployment | ✅ Strongly recommended | KV Cache 7.7GB → 1.1GB; fits on A100 40G |
| ESM-2 autoregressive protein generation | ✅ Recommended | Real decode workload; Triton kernel accelerates |
| Long-sequence (>512 aa) sliding-window inference | ✅ Recommended | Historical KV must be retained; compression pays off |
| ESM-2 650M standard embedding extraction | ⚠️ Limited benefit | Small KV Cache; slightly slower than original during Prefill |

---

## Roadmap

- [x] Mac MPS / CPU full validation path
- [x] Separate K/V LUT calibration
- [x] QJL residual correction
- [x] Triton CUDA kernel (dequant + fused decode attention)
- [x] CUDA accuracy validation (all 33 layers pass, error < 1e-6)
- [x] Vectorized quantization (pack/unpack/quantize)
- [x] KV Cache compression ratio validation (7.1x measured)
- [ ] ESM-2 15B end-to-end validation on real hardware
- [ ] Autoregressive protein generation end-to-end test
- [ ] Systematic speed/accuracy comparison against INT8 baseline

---

## weights download 

# https://huggingface.co/YueHuLab/TurboESM/blob/main/esm2_650M_turbo.pt


## Acknowledgements

- **ESM-2**: This project builds on [ESM-2](https://github.com/facebookresearch/esm) by Meta FAIR Protein Team, licensed under Apache 2.0.
- **TurboQuant**: Quantization methodology is an independent implementation based on the paper *TurboQuant: Accurate KV Cache Quantization with Rotation and Outlier-Free Quantization* (arXiv:2504.19874) by Google Research, adapted for ESM-2's RoPE architecture.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Use of this project is also subject to the ESM-2 [Apache 2.0 License](https://github.com/facebookresearch/esm/blob/main/LICENSE).
