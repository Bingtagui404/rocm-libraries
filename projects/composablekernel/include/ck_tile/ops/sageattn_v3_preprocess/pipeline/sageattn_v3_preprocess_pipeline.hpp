// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_mxfp4_pack.hpp"

namespace ck_tile {

// SageAttnV3PreprocessPipeline: per-CTA computation for one tile of Q and K.
//
// Each CTA (identified by tile_x, head, batch) performs:
//
//   1. RunQMean:              compute column mean of Q tile → smem + global q_mean (InputT)
//   2. RunQQuantize:          Q_smooth = Q - q_mean → MXFP4
//   3. RunKSmoothAndQuantize: K' = K - k_mean → write k_prime; K' → MXFP4
//
// delta_s = q_mean @ K'^T is computed separately via a batched GEMM kernel.
// V preprocessing is handled by SageAttnVPreprocessKernel (separate launch).
//
// RunKMeanPartial: used by SageAttnKMeanKernel (separate launch, different grid).
//   Each CTA accumulates a partial column-sum of its K tile into k_mean_partial (float atomicAdd).
//   The last CTA to finish normalises and stores k_mean as InputT.
//
// Thread layout (Steps 2 & 3 — quantize):
//   kNumGroups = kCols / kScaleGranularity
//   kBlockSize = kRows * kNumGroups   ← one thread per (row, group) pair
//   row_idx    = tid / kNumGroups     (0 .. kRows-1)
//   grp_idx    = tid % kNumGroups     (0 .. kNumGroups-1)
//   Each thread processes one MXFP4 group of kScaleGranularity(=32) elements.
//   → 100% thread utilisation for Steps 2 and 3.
//
// Thread layout (Step 1 — mean):
//   thread d (d < kCols) computes mean of column d over kRows rows.
//   Threads d >= kCols are idle during this step (kBlockSize > kCols when kRows > 1).
//
// Template parameters:
//   InputT_:           input element type (fp16_t, bf16_t, or float)
//   kRows_:            tile rows (kM0 == kN0, used for Q and K tiles)
//   kCols_:            tile cols (hdim)
//   kScaleGranularity: MXFP4 group size (must be 32)
//   kBlockSize_:       threads per CTA; default = kRows * (kCols / kScaleGranularity)

template <typename InputT_,
          index_t kRows_,
          index_t kCols_,
          index_t kScaleGranularity_ = 32,
          index_t kBlockSize_        = 256>
struct SageAttnV3PreprocessPipeline
{
    using InputT = InputT_;

    static constexpr index_t kRows             = kRows_;
    static constexpr index_t kCols             = kCols_;
    static constexpr index_t kScaleGranularity = kScaleGranularity_;
    static constexpr index_t kBlockSize        = kBlockSize_;

    static_assert(kScaleGranularity == 32, "MXFP4 scale granularity must be 32");
    static_assert(kCols % kScaleGranularity == 0,
                  "kCols must be divisible by kScaleGranularity for MXFP4");

    // Shared memory: kCols float32 elements for q_mean broadcast.
    // Stored as float (not InputT) so RunQQuantize can read it directly without
    // an InputT→float cast, and to accommodate the float mean computation.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kCols * sizeof(float); }

    // -------------------------------------------------------------------------
    // Step 1: compute column mean of [kRows, kCols] Q tile.
    //   Thread d (d < kCols) accumulates src[0..kRows-1][d] in float, stores
    //   the mean as float in smem and as InputT in q_mean_ptr.
    //   Threads d >= kCols are idle (kBlockSize = kRows*kCols/32 > kCols).
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQMean(const InputT* __restrict__ src_ptr,
                                 InputT* __restrict__ q_mean_ptr,
                                 void* smem,
                                 index_t n_rows_valid) const
    {
        float* smem_f     = reinterpret_cast<float*>(smem);
        const index_t tid = get_thread_id();

        for(index_t d = tid; d < kCols; d += kBlockSize)
        {
            float sum = 0.0f;
            for(index_t r = 0; r < n_rows_valid; r++)
                sum += static_cast<float>(src_ptr[r * kCols + d]);
            const float mean = sum / static_cast<float>(n_rows_valid);
            smem_f[d]        = mean;
            q_mean_ptr[d]    = static_cast<InputT>(mean);
        }
        // Caller issues block_sync_lds() before RunQQuantize.
    }

    // -------------------------------------------------------------------------
    // Step 2: quantize Q_smooth = Q_tile - q_mean (smem) → MXFP4.
    //   Reads q_mean from smem (set by RunQMean).
    //
    //   Thread layout: one thread per (row, group) pair.
    //     row_idx = tid / kNumGroups  (0 .. kRows-1)
    //     grp_idx = tid % kNumGroups  (0 .. kNumGroups-1)
    //   → 100% thread utilisation.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQQuantize(const InputT* __restrict__ src_ptr,
                                     uint8_t* __restrict__ dst_hat_ptr,
                                     uint8_t* __restrict__ dst_scale_ptr,
                                     const void* smem,
                                     index_t n_rows_valid) const
    {
        const float* smem_mean_f = reinterpret_cast<const float*>(smem);
        const index_t tid        = get_thread_id();

        constexpr index_t kNumGroups = kCols / kScaleGranularity;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            // pad row — zero hat and scale
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        const InputT* src_row = src_ptr + row_idx * kCols;

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;
        for(index_t j = 0; j < kScaleGranularity; j++)
        {
            const float val = static_cast<float>(src_row[d_start + j]) - smem_mean_f[d_start + j];
            group_data[j]   = val;
            max_abs         = max(max_abs, abs(val));
        }

        const float scale = bit_cast<float>(
            (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
            numeric_traits<float>::head_mask);

        dst_scale_ptr[row_idx * kNumGroups + grp_idx] =
            static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

        PackFP4Group<kScaleGranularity>(group_data,
                                        dst_hat_ptr + row_idx * (kCols / 2) +
                                            grp_idx * (kScaleGranularity / 2),
                                        scale);
    }

    // -------------------------------------------------------------------------
    // Step 3: compute K' = K_tile - k_mean, write to k_prime_ptr in natural
    //   row-major layout [kRows, kCols], then quantize K' → MXFP4.
    //   k_mean_ptr:     [kCols] InputT per-head channel mean for this (batch, head).
    //   k_prime_ptr:    pointer to K'[row_start, 0] in [seqlen_k, hdim] global layout.
    //   k_prime_stride: hdim (row stride of K').
    //   smem:           shared memory (kCols floats) — reused from RunQMean to cache k_mean.
    //                   Caller may pass the same smem buffer; q_mean data is no longer
    //                   needed after RunQQuantize completes.
    //
    //   Thread layout: one thread per (row, group) pair.
    //     row_idx = tid / kNumGroups  (0 .. kRows-1)
    //     grp_idx = tid % kNumGroups  (0 .. kNumGroups-1)
    //   → 100% thread utilisation. No kprime_row[kCols] stack array needed.
    //
    //   k_mean caching: threads 0..kCols-1 (one per channel) preload k_mean into
    //   smem as float, followed by block_sync_lds(), so the compute loop reads from
    //   LDS instead of global memory (avoids kRows redundant L2/HBM accesses per thread).
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunKSmoothAndQuantize(const InputT* __restrict__ src_ptr,
                                              const InputT* __restrict__ k_mean_ptr,
                                              InputT* __restrict__ k_prime_ptr,
                                              index_t k_prime_stride,
                                              uint8_t* __restrict__ dst_hat_ptr,
                                              uint8_t* __restrict__ dst_scale_ptr,
                                              void* smem,
                                              index_t n_rows_valid) const
    {
        float* smem_f     = reinterpret_cast<float*>(smem);
        const index_t tid = get_thread_id();

        // Cache k_mean from global → LDS. Threads 0..kCols-1 each load one channel.
        // kBlockSize >= kCols so the loop runs at most once per thread.
        for(index_t d = tid; d < kCols; d += kBlockSize)
            smem_f[d] = static_cast<float>(k_mean_ptr[d]);
        block_sync_lds();

        constexpr index_t kNumGroups = kCols / kScaleGranularity;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            // pad row — zero k_prime, hat, scale
            InputT* dst_row = k_prime_ptr + row_idx * k_prime_stride;
            for(index_t j = 0; j < kScaleGranularity; j++)
                dst_row[d_start + j] = static_cast<InputT>(0);
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        const InputT* src_row = src_ptr + row_idx * kCols;
        InputT* dst_row       = k_prime_ptr + row_idx * k_prime_stride;

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;
        for(index_t j = 0; j < kScaleGranularity; j++)
        {
            // Read k_mean from LDS instead of global memory.
            const float val      = static_cast<float>(src_row[d_start + j]) - smem_f[d_start + j];
            group_data[j]        = val;
            dst_row[d_start + j] = static_cast<InputT>(val);
            max_abs              = max(max_abs, abs(val));
        }

        const float scale = bit_cast<float>(
            (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
            numeric_traits<float>::head_mask);

        dst_scale_ptr[row_idx * kNumGroups + grp_idx] =
            static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

        PackFP4Group<kScaleGranularity>(group_data,
                                        dst_hat_ptr + row_idx * (kCols / 2) +
                                            grp_idx * (kScaleGranularity / 2),
                                        scale);
    }

    // -------------------------------------------------------------------------
    // RunKMeanPartial: called from SageAttnKMeanKernel (BlockSize = kCols).
    //   Each thread d accumulates K[row_start..row_end, d] into k_mean_partial[d]
    //   via float atomicAdd.
    //
    //   k_tile_ptr:      K[row_start, 0] for this (batch, head, tile_x)
    //   n_rows:          actual rows in this tile (handles tail padding)
    //   stride_k:        hdim (columns per row)
    //   k_mean_partial:  [kCols] float accumulator for this (batch, head), pre-zeroed
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunKMeanPartial(const InputT* __restrict__ k_tile_ptr,
                                        index_t n_rows,
                                        index_t stride_k,
                                        float* __restrict__ k_mean_partial) const
    {
        // One thread per d-channel (BlockSize == kCols in SageAttnKMeanKernel).
        const index_t d = get_thread_id();

        float acc = 0.0f;
        for(index_t r = 0; r < n_rows; r++)
            acc += static_cast<float>(k_tile_ptr[r * stride_k + d]);

        atomicAdd(&k_mean_partial[d], acc);
    }
};

} // namespace ck_tile
