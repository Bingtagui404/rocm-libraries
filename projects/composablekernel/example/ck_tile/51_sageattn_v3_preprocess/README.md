# 51 — SageAttention V3 Preprocessing

Benchmark for the SageAttention V3 (SA3) preprocessing pipeline — a sequence of
four HIP kernels that prepares Q, K, and V for the SA3 attention forward pass.

**Architecture requirement:** gfx950 (MI350) — requires native MXFP4 hardware support.

---

## What This Example Does

`sageattn_v3_preprocess_run()` launches the following kernels on a single HIP stream:

| # | Kernel | Output |
|---|--------|--------|
| 0 | **KMean** (`SageAttnV3KMeanKernel`) | `k_mean[B, H, D]` = column mean of K over seqlen_k |
| 1 | **Preprocess Q/K** (`SageAttnV3PreprocessKernel`) | Q mean → MXFP4; K' = K − k_mean → MXFP4 |
| 1b | **V preprocess** (`SageAttnV3VPreprocessKernel`) | V transposed → MXFP4 `[B,H,D,Sk]` |
| 2 | **delta_s GEMM** (batched GEMM) | `delta_s[B,H,T_q,Sk]` = q_mean @ K'ᵀ |

Outputs produced:
- `q_hat`, `q_scale` — MXFP4-quantised Q (smoothed: Q − q_mean)
- `q_mean` — per-tile column mean of Q
- `k_hat`, `k_scale` — MXFP4-quantised K' = K − k_mean
- `v_hat`, `v_scale` — MXFP4-quantised V in transposed layout
- `delta_s` — float32 correction term for the attention logits

---

## Build

```bash
cd /path/to/composablekernel/build
cmake -DGPU_TARGETS=gfx950 ..
cmake --build . --target tile_example_sageattn_v3_preprocess -j$(nproc)
```

---

## Run

### Single shape
```bash
./bin/tile_example_sageattn_v3_preprocess \
    -b 1 -h 32 -q 1024 -k 4096 -d 128 -t fp16 -w 5 -r 50
```

### Built-in benchmark suite (fp16 + fp32, multiple shapes)
```bash
./bin/tile_example_sageattn_v3_preprocess
```

### CSV output
```bash
./bin/tile_example_sageattn_v3_preprocess --csv
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `-b` | 4 | Batch size |
| `-h` | 16 | Number of heads |
| `-q` | 1024 | Sequence length Q |
| `-k` | 4096 | Sequence length K (= V) |
| `-d` | 128 | Head dimension (128 or 256) |
| `-t` | fp16 | Input type: `fp16` or `fp32` |
| `-w` | 5 | Warmup iterations |
| `-r` | 50 | Measurement iterations |
| `--csv` | off | Print CSV header + rows (one row per kernel + total) |

---

## Output

Benchmark mode always reports a per-kernel breakdown plus a total (all four kernels
timed end-to-end):

```
dtype=fp16  B=1 H=32 Sq=1024 Sk=4096 D=128
  [0] KMean            0.042 ms    123.4 GB/s  (2048 MB)
  [1] Prep Q/K         0.210 ms    567.8 GB/s  (... MB)
  [1b] Prep V          0.089 ms    234.5 GB/s  (... MB)
  [2] delta_s GEMM     0.180 ms    456.7 GB/s  (... MB)
  --------------------------------------------------
  Total                0.521 ms    987.6 GB/s  (... MB)
```

### HBM accounting per kernel

| Kernel | Reads | Writes |
|--------|-------|--------|
| [0] KMean | K [B,H,Sk,D] | k_mean [B,H,D] |
| [1] Prep Q/K | Q [B,H,Sq,D], K [B,H,Sk,D] (2nd pass), k_mean | q_mean, q_hat, q_scale, K', k_hat, k_scale |
| [1b] Prep V | V [B,H,Sk,D] | v_hat [B,H,D,Sk/2], v_scale [B,H,D,Sk/G] |
| [2] delta_s GEMM | q_mean, K' | delta_s [B,H,T_q,Sk] float32 |

Note: K is read twice (KMean + Preprocess). q_mean and K' are written then re-read by
the GEMM. Each kernel is timed independently, with the full pipeline used for warmup.

---

## Implementation Notes

### MXFP4 quantization
Each group of 32 elements shares one e8m0 (power-of-two) scale factor. The native
`__builtin_amdgcn_cvt_scalef32_pk_fp4_f32` intrinsic packs 2 FP4 values per byte.

### Thread layout (Q/K preprocess)
`kBlockSize = kRows × (kCols / 32)` threads, one per (row, MXFP4-group) pair →
100% thread utilisation in quantize steps.

### V LDS transpose
A `[kVGroup × kVHdimTile]` tile is loaded coalesced into shared memory with +1 float
padding per row (bank-conflict-free for `ds_read_b32` on gfx950), then quantized
column-wise.

### last-CTA normalization (KMean)
Each tile CTA `atomicAdd`s its partial column-sum; the last CTA to finish (detected via
a completion counter) normalises and stores `k_mean` as InputT — no second kernel launch
required.

---

## Correctness Testing

End-to-end GPU tests against CPU float reference:

```bash
cmake --build . --target test_ck_tile_fmha_sageattn_v3_preprocess -j$(nproc)
ctest -R test_ck_tile_fmha_sageattn_v3_preprocess
```

[Back to CK Tile Examples](../README.md)
