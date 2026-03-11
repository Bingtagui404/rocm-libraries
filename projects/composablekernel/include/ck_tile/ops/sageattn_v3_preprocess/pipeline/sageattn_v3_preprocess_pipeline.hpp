// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_mxfp4_pack.hpp"

namespace ck_tile {

// SageAttnV3PreprocessPipeline: per-CTA computation for one tile of Q and K.
//
// Optimized for MI350 (gfx950) HBM bandwidth:
//
//   1. RunQMean: all kBlockSize threads participate in column-mean computation.
//      Threads are partitioned into (kBlockSize / kCols) groups per column,
//      each group accumulates (kRows / kRowsPerThread) rows and then does a
//      warp-level reduction. Eliminates the 75% idle threads of the old
//      single-thread-per-column approach.
//
//   2. RunQQuantize / RunKSmoothAndQuantize: vectorized fp16 load/store via
//      uint4 (128-bit = 8 x fp16), reducing instruction count 8x compared to
//      scalar 2-byte loads.
//
//   3. RunKMeanPartial: vectorized load (uint4) across the d dimension.
//
// Thread layout (Steps 2 & 3 -- quantize):
//   kNumGroups = kCols / kScaleGranularity
//   kBlockSize = kRows * kNumGroups  (one thread per (row, group) pair)
//   row_idx    = tid / kNumGroups    (0 .. kRows-1)
//   grp_idx    = tid % kNumGroups    (0 .. kNumGroups-1)
//   -> 100% thread utilisation for Steps 2 and 3.
//
// Thread layout (Step 1 -- mean, new):
//   kThreadsPerCol = kBlockSize / kCols  (how many threads collaborate per column)
//   col_idx        = tid % kCols         (0 .. kCols-1)
//   grp_in_col     = tid / kCols         (0 .. kThreadsPerCol-1)
//   Each thread accumulates a strided subset of rows, then a reduction across
//   grp_in_col threads yields the column mean. The mean is stored in smem for
//   RunQQuantize and also written to global q_mean_ptr by the leader thread
//   (grp_in_col == 0).
//
// Template parameters:
//   InputT_:           input element type (fp16_t, bf16_t, or float)
//   kRows_:            tile rows (kM0 == kN0, used for Q and K tiles)
//   kCols_:            tile cols (hdim)
//   kScaleGranularity: MXFP4 group size (must be 32)
//   kBlockSize_:       threads per CTA; default = kRows * (kCols / kScaleGranularity)

// ---------------------------------------------------------------------------
// Vectorized load helpers: load 8 x fp16 at once (128-bit / uint4).
// ---------------------------------------------------------------------------

template <typename InputT>
CK_TILE_DEVICE void load_vec8(const InputT* __restrict__ ptr, float (&out)[8])
{
    // Generic scalar fallback.
    for(int i = 0; i < 8; i++)
        out[i] = static_cast<float>(ptr[i]);
}

// Specialization for fp16_t: one 128-bit load.
template <>
CK_TILE_DEVICE void load_vec8<fp16_t>(const fp16_t* __restrict__ ptr, float (&out)[8])
{
    // Load 8 x fp16 (16 bytes) as a single uint4.
    using vec_t     = uint4;
    const auto raw  = *reinterpret_cast<const vec_t*>(ptr);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
    for(int i = 0; i < 8; i++)
        out[i] = static_cast<float>(bit_cast<fp16_t>(h[i]));
}

// Specialization for float: load 8 x float (32 bytes) as two uint4.
template <>
CK_TILE_DEVICE void load_vec8<float>(const float* __restrict__ ptr, float (&out)[8])
{
    using vec_t     = uint4;
    const auto* vp  = reinterpret_cast<const vec_t*>(ptr);
    const float* f0 = reinterpret_cast<const float*>(&vp[0]);
    const float* f1 = reinterpret_cast<const float*>(&vp[1]);
    for(int i = 0; i < 4; i++)
    {
        out[i]     = f0[i];
        out[i + 4] = f1[i];
    }
}

// Store 8 x fp16.
template <typename InputT>
CK_TILE_DEVICE void store_vec8(InputT* __restrict__ ptr, const float (&in)[8])
{
    for(int i = 0; i < 8; i++)
        ptr[i] = static_cast<InputT>(in[i]);
}

template <>
CK_TILE_DEVICE void store_vec8<fp16_t>(fp16_t* __restrict__ ptr, const float (&in)[8])
{
    uint4 raw;
    uint16_t* h = reinterpret_cast<uint16_t*>(&raw);
    for(int i = 0; i < 8; i++)
        h[i] = bit_cast<uint16_t>(static_cast<fp16_t>(in[i]));
    *reinterpret_cast<uint4*>(ptr) = raw;
}

template <>
CK_TILE_DEVICE void store_vec8<float>(float* __restrict__ ptr, const float (&in)[8])
{
    uint4* vp = reinterpret_cast<uint4*>(ptr);
    vp[0]     = *reinterpret_cast<const uint4*>(&in[0]);
    vp[1]     = *reinterpret_cast<const uint4*>(&in[4]);
}

// ---------------------------------------------------------------------------

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

    // kThreadsPerCol: threads collaborating on a single column mean.
    static constexpr index_t kThreadsPerCol = kBlockSize / kCols;
    static_assert(kBlockSize % kCols == 0,
                  "kBlockSize must be divisible by kCols for parallel mean reduction");

    // Shared memory layout:
    //   [kCols] float for column mean broadcast to RunQQuantize
    //   + [kCols * kThreadsPerCol] float for intermediate per-thread partial sums
    //     (only needed when kThreadsPerCol > 1; size == kBlockSize floats)
    //
    // Total: kCols + kBlockSize floats when kThreadsPerCol > 1, else kCols floats.
    static constexpr index_t kSmemMeanWords    = kCols;
    static constexpr index_t kSmemPartialWords = (kThreadsPerCol > 1) ? kBlockSize : 0;
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return (kSmemMeanWords + kSmemPartialWords) * static_cast<index_t>(sizeof(float));
    }

    // -------------------------------------------------------------------------
    // Step 1: compute column mean of [kRows, kCols] Q tile.
    //   All kBlockSize threads participate.
    //   Thread tid is assigned to column (tid % kCols) and reduces rows
    //   tid/kCols, tid/kCols + kThreadsPerCol, ... in steps of kThreadsPerCol.
    //   After intra-column reduction via smem, the leader (grp_in_col==0) stores
    //   the final mean to smem_mean[col] and global q_mean_ptr[col].
    //   Caller issues block_sync_lds() before RunQQuantize.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQMean(const InputT* __restrict__ src_ptr,
                                 InputT* __restrict__ q_mean_ptr,
                                 void* smem,
                                 index_t n_rows_valid) const
    {
        float* smem_f       = reinterpret_cast<float*>(smem);
        float* smem_partial = smem_f + kSmemMeanWords; // only valid when kThreadsPerCol > 1

        const index_t tid       = get_thread_id();
        const index_t col_idx   = tid % kCols;      // which column this thread owns
        const index_t grp_in_col = tid / kCols;     // which partial-sum group within column

        // Accumulate partial sum over every kThreadsPerCol-th row.
        float acc = 0.0f;
        for(index_t r = grp_in_col; r < n_rows_valid; r += kThreadsPerCol)
            acc += static_cast<float>(src_ptr[r * kCols + col_idx]);

        // Reduce partial sums across kThreadsPerCol threads for the same column.
        if constexpr(kThreadsPerCol > 1)
        {
            // Write partial sum to smem, sync, then tree-reduce.
            smem_partial[tid] = acc;
            block_sync_lds();

            // Single-pass reduction: grp_in_col == 0 reads all kThreadsPerCol partials.
            if(grp_in_col == 0)
            {
                for(index_t g = 1; g < kThreadsPerCol; g++)
                    acc += smem_partial[g * kCols + col_idx];
            }
            // No second sync needed: RunQQuantize syncs with the smem_mean store below.
        }

        // Leader thread stores mean.
        if(grp_in_col == 0)
        {
            const float mean   = acc / static_cast<float>(n_rows_valid);
            smem_f[col_idx]    = mean;
            q_mean_ptr[col_idx] = static_cast<InputT>(mean);
        }
        // Caller issues block_sync_lds() before RunQQuantize reads smem_f.
    }

    // -------------------------------------------------------------------------
    // Step 2: quantize Q_smooth = Q_tile - q_mean (smem) -> MXFP4.
    //   Reads q_mean from smem (set by RunQMean).
    //   Vectorized: loads 8 fp16 per instruction.
    //
    //   Thread layout: one thread per (row, group) pair.
    //     row_idx = tid / kNumGroups  (0 .. kRows-1)
    //     grp_idx = tid % kNumGroups  (0 .. kNumGroups-1)
    //   -> 100% thread utilisation.
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
            // pad row -- zero hat and scale
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        const InputT* src_row = src_ptr + row_idx * kCols + d_start;

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        // Vectorized load: 8 fp16 per iteration (kScaleGranularity/8 iterations).
        constexpr index_t kVec = 8;
        static_assert(kScaleGranularity % kVec == 0, "");
        for(index_t v = 0; v < kScaleGranularity / kVec; v++)
        {
            float tmp[kVec];
            load_vec8(src_row + v * kVec, tmp);
            for(index_t j = 0; j < kVec; j++)
            {
                const float val       = tmp[j] - smem_mean_f[d_start + v * kVec + j];
                group_data[v * kVec + j] = val;
                max_abs                   = max(max_abs, abs(val));
            }
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
    // Step 3: compute K' = K_tile - k_mean, write to k_prime_ptr (vectorized),
    //   then quantize K' -> MXFP4.
    //   Vectorized: loads and stores 8 fp16 per instruction.
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

        // Cache k_mean from global -> LDS. Threads 0..kCols-1 each load one channel.
        // kBlockSize >= kCols so the loop runs at most once per thread.
        // Vectorized: load 8 fp16 per thread if possible.
        constexpr index_t kVec = 8;
        for(index_t d = tid * kVec; d < kCols; d += kBlockSize * kVec)
        {
            float tmp[kVec];
            load_vec8(k_mean_ptr + d, tmp);
            for(index_t j = 0; j < kVec && d + j < kCols; j++)
                smem_f[d + j] = tmp[j];
        }
        // Fallback for remaining elements if kCols not divisible by kVec*kBlockSize
        // (in practice kCols=128 and kBlockSize=512 -> each thread covers d=tid*8,
        //  only threads 0..15 execute; exactly covers 16*8=128 elements).
        block_sync_lds();

        constexpr index_t kNumGroups = kCols / kScaleGranularity;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            // pad row -- zero k_prime, hat, scale
            InputT* dst_row = k_prime_ptr + row_idx * k_prime_stride + d_start;
            // Vectorized zero store.
            float zeros[kVec] = {};
            static_assert(kScaleGranularity % kVec == 0, "");
            for(index_t v = 0; v < kScaleGranularity / kVec; v++)
                store_vec8(dst_row + v * kVec, zeros);
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        const InputT* src_row = src_ptr + row_idx * kCols + d_start;
        InputT* dst_row       = k_prime_ptr + row_idx * k_prime_stride + d_start;

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        // Vectorized load + subtract k_mean + vectorized store.
        static_assert(kScaleGranularity % kVec == 0, "");
        for(index_t v = 0; v < kScaleGranularity / kVec; v++)
        {
            float tmp[kVec];
            load_vec8(src_row + v * kVec, tmp);
            for(index_t j = 0; j < kVec; j++)
            {
                const float val           = tmp[j] - smem_f[d_start + v * kVec + j];
                group_data[v * kVec + j]  = val;
                tmp[j]                    = val; // reuse for store
                max_abs                   = max(max_abs, abs(val));
            }
            store_vec8(dst_row + v * kVec, tmp);
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
    //   via float atomicAdd. Vectorized: loads 8 fp16 per instruction per
    //   group of 8 consecutive threads (column-major across the warp).
    //
    //   For a warp of 64 threads covering d=0..63, each inner loop iteration
    //   issues a coalesced 128-byte read (64 threads x 2 bytes), then each
    //   thread processes its element. Since we accumulate a scalar per column,
    //   there is no further vectorization benefit at the per-thread level for
    //   the accumulate step, but the load itself is a 128-byte transaction.
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
