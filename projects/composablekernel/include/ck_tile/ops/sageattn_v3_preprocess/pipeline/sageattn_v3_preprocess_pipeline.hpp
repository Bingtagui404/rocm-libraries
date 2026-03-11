// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_mxfp4_pack.hpp"

namespace ck_tile {

// SageAttnV3PreprocessPipeline: per-CTA computation for one tile of Q and K.
//
// Optimizations for MI350 (gfx950) HBM bandwidth:
//
//   1. Q tile LDS cache (kUseLdsQ=true): Q is loaded from HBM once into shared
//      memory using vectorized 128-bit loads, then RunQMean and RunQQuantize both
//      read from LDS instead of re-reading HBM. Eliminates the second global Q read.
//      Enabled only when the Q tile fits in LDS (kQTileBytes <= kMaxSmemForLdsQ).
//      For fp16/bf16 with kCols<=256: enabled. For float with kCols=256: disabled.
//
//   2. Vectorized fp16 load/store: uint4 (128-bit = 8 x fp16) throughout.
//      load_vec8 / store_vec8 reduce instruction count 8x vs scalar 2-byte.
//
//   3. Parallel RunQMean: all kBlockSize threads compute the column mean
//      (kThreadsPerCol threads per column), eliminating idle threads.
//
//   4. RunKSmoothAndQuantize: vectorized load + store for K and k_mean.
//
// Smem layout when kUseLdsQ=true (bytes):
//   [0 .. kQTileBytes):                 Q tile as InputT (kRows x kCols)
//   [kQTileBytes .. +kSmemMeanBytes):   column means as float32 (kCols)
//   [kQTileBytes+kSmemMeanBytes .. +kSmemPartialBytes): partial sums (kBlockSize)
//
// Smem layout when kUseLdsQ=false (bytes):
//   [0 .. kSmemMeanBytes):              column means as float32 (kCols)
//   [kSmemMeanBytes .. +kSmemPartialBytes): partial sums (kBlockSize)
//
// The K step (RunKSmoothAndQuantize) reuses smem[0..kCols*4] for the k_mean
// cache after the Q steps are done (Q tile bytes are no longer needed).
//
// Template parameters:
//   InputT_:           input element type (fp16_t, bf16_t, or float)
//   kRows_:            tile rows (kM0 == kN0, used for Q and K tiles)
//   kCols_:            tile cols (hdim)
//   kScaleGranularity: MXFP4 group size (must be 32)
//   kBlockSize_:       threads per CTA; default = kRows * (kCols / kScaleGranularity)

// ---------------------------------------------------------------------------
// Vectorized load/store helpers: 8 x fp16 per 128-bit instruction.
// ---------------------------------------------------------------------------

template <typename InputT>
CK_TILE_DEVICE void load_vec8(const InputT* __restrict__ ptr, float (&out)[8])
{
    for(int i = 0; i < 8; i++)
        out[i] = static_cast<float>(ptr[i]);
}

template <>
CK_TILE_DEVICE void load_vec8<fp16_t>(const fp16_t* __restrict__ ptr, float (&out)[8])
{
    using vec_t       = uint4;
    const auto raw    = *reinterpret_cast<const vec_t*>(ptr);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
    for(int i = 0; i < 8; i++)
        out[i] = static_cast<float>(bit_cast<fp16_t>(h[i]));
}

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

    // ---------------------------------------------------------------------------
    // Smem layout
    // ---------------------------------------------------------------------------
    // Q tile size: kRows x kCols elements of InputT.
    static constexpr index_t kQTileWords = kRows * kCols;
    static constexpr index_t kQTileBytes =
        kQTileWords * static_cast<index_t>(sizeof(InputT));

    // Enable LDS Q caching only when the Q tile fits comfortably.
    // Reserve 16 KB headroom for k_mean cache (kCols*4) + mean/partial buffers.
    static constexpr index_t kMaxSmemForLdsQ = 49152; // 48 KB
    static constexpr bool kUseLdsQ           = (kQTileBytes <= kMaxSmemForLdsQ);

    // Slot: column means (float32, kCols).
    // Slot: partial sums (float32, kBlockSize) -- only when kThreadsPerCol > 1.
    static constexpr index_t kSmemMeanWords    = kCols;
    static constexpr index_t kSmemPartialWords = (kThreadsPerCol > 1) ? kBlockSize : 0;

    // Byte offsets within smem.
    // When kUseLdsQ: Q tile occupies smem[0..kQTileBytes), then mean, then partial.
    // When !kUseLdsQ: mean starts at 0, partial follows.
    static constexpr index_t kSmemMeanOffset =
        kUseLdsQ ? kQTileBytes : 0;
    static constexpr index_t kSmemPartialOffset = kSmemMeanOffset + kSmemMeanWords * 4;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kSmemPartialOffset + kSmemPartialWords * 4;
    }

    // -------------------------------------------------------------------------
    // Step 0 (kUseLdsQ only): Load Q tile from global into LDS.
    //   Vectorized 128-bit reads, all kBlockSize threads participate.
    //   Bounds-checked: padded rows (row >= n_rows_valid) are zeroed.
    //   Caller must issue block_sync_lds() before RunQMean reads LDS.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunLoadQTile(const InputT* __restrict__ src_ptr,
                                     void* smem,
                                     index_t n_rows_valid) const
    {
        static_assert(kUseLdsQ, "RunLoadQTile requires kUseLdsQ=true");
        InputT* smem_q         = reinterpret_cast<InputT*>(smem);
        const index_t tid      = get_thread_id();
        constexpr index_t kVec = 8;
        static_assert(kQTileWords % kVec == 0, "Q tile must be divisible by kVec");

        for(index_t base = tid * kVec; base < kQTileWords; base += kBlockSize * kVec)
        {
            const index_t row = base / kCols;
            if(row < n_rows_valid)
            {
                float tmp[kVec];
                load_vec8(src_ptr + base, tmp);
                store_vec8(smem_q + base, tmp);
            }
            else
            {
                float zeros[kVec] = {};
                store_vec8(smem_q + base, zeros);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 1a: Compute column mean from LDS Q tile (kUseLdsQ=true path).
    //   Reads smem_q loaded by RunLoadQTile. No HBM traffic.
    //   Leader stores mean to smem_mean and q_mean_ptr.
    //   Caller issues block_sync_lds() before RunQQuantize.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQMean(const void* smem,
                                 InputT* __restrict__ q_mean_ptr,
                                 index_t n_rows_valid) const
    {
        static_assert(kUseLdsQ, "RunQMean(smem) requires kUseLdsQ=true");
        const InputT* smem_q = reinterpret_cast<const InputT*>(smem);
        float* smem_mean =
            reinterpret_cast<float*>(reinterpret_cast<char*>(const_cast<void*>(smem)) +
                                     kSmemMeanOffset);
        float* smem_partial = smem_mean + kSmemMeanWords;

        const index_t tid        = get_thread_id();
        const index_t col_idx    = tid % kCols;
        const index_t grp_in_col = tid / kCols;

        float acc = 0.0f;
        for(index_t r = grp_in_col; r < n_rows_valid; r += kThreadsPerCol)
            acc += static_cast<float>(smem_q[r * kCols + col_idx]);

        if constexpr(kThreadsPerCol > 1)
        {
            smem_partial[tid] = acc;
            block_sync_lds();
            if(grp_in_col == 0)
                for(index_t g = 1; g < kThreadsPerCol; g++)
                    acc += smem_partial[g * kCols + col_idx];
        }

        if(grp_in_col == 0)
        {
            const float mean    = acc / static_cast<float>(n_rows_valid);
            smem_mean[col_idx]  = mean;
            q_mean_ptr[col_idx] = static_cast<InputT>(mean);
        }
    }

    // -------------------------------------------------------------------------
    // Step 1b: Compute column mean from global Q tile (kUseLdsQ=false path).
    //   Reads Q from global memory. Stores mean to smem_mean and q_mean_ptr.
    //   Caller issues block_sync_lds() before RunQQuantizeGlobal.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQMeanGlobal(const InputT* __restrict__ q_ptr,
                                       void* smem,
                                       InputT* __restrict__ q_mean_ptr,
                                       index_t n_rows_valid) const
    {
        static_assert(!kUseLdsQ, "RunQMeanGlobal requires kUseLdsQ=false");
        float* smem_mean =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + kSmemMeanOffset);
        float* smem_partial = smem_mean + kSmemMeanWords;

        const index_t tid        = get_thread_id();
        const index_t col_idx    = tid % kCols;
        const index_t grp_in_col = tid / kCols;

        constexpr index_t kVec = 8;
        float acc = 0.0f;
        // Strided load: each thread group handles its row subset.
        for(index_t r = grp_in_col; r < n_rows_valid; r += kThreadsPerCol)
        {
            // Scalar load per element (global path, any alignment).
            acc += static_cast<float>(q_ptr[r * kCols + col_idx]);
        }

        if constexpr(kThreadsPerCol > 1)
        {
            smem_partial[tid] = acc;
            block_sync_lds();
            if(grp_in_col == 0)
                for(index_t g = 1; g < kThreadsPerCol; g++)
                    acc += smem_partial[g * kCols + col_idx];
        }

        if(grp_in_col == 0)
        {
            const float mean    = acc / static_cast<float>(n_rows_valid);
            smem_mean[col_idx]  = mean;
            q_mean_ptr[col_idx] = static_cast<InputT>(mean);
        }
        (void)kVec;
    }

    // -------------------------------------------------------------------------
    // Step 2a: Quantize Q from LDS (kUseLdsQ=true path).
    //   Q data and mean both read from LDS. No HBM Q read.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQQuantize(const void* smem,
                                     uint8_t* __restrict__ dst_hat_ptr,
                                     uint8_t* __restrict__ dst_scale_ptr,
                                     index_t n_rows_valid) const
    {
        static_assert(kUseLdsQ, "RunQQuantize(smem) requires kUseLdsQ=true");
        const InputT* smem_q = reinterpret_cast<const InputT*>(smem);
        const float* smem_mean =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(smem) +
                                           kSmemMeanOffset);

        const index_t tid            = get_thread_id();
        constexpr index_t kNumGroups = kCols / kScaleGranularity;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;
        const InputT* src_row       = smem_q + row_idx * kCols + d_start;

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        constexpr index_t kVec = 8;
        static_assert(kScaleGranularity % kVec == 0, "");
        for(index_t v = 0; v < kScaleGranularity / kVec; v++)
        {
            float tmp[kVec];
            load_vec8(src_row + v * kVec, tmp);
            for(index_t j = 0; j < kVec; j++)
            {
                const float val          = tmp[j] - smem_mean[d_start + v * kVec + j];
                group_data[v * kVec + j] = val;
                max_abs                  = max(max_abs, abs(val));
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
    // Step 2b: Quantize Q from global (kUseLdsQ=false path).
    //   Q data read from global; mean read from smem_mean.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunQQuantizeGlobal(const InputT* __restrict__ q_ptr,
                                           const void* smem,
                                           uint8_t* __restrict__ dst_hat_ptr,
                                           uint8_t* __restrict__ dst_scale_ptr,
                                           index_t n_rows_valid) const
    {
        static_assert(!kUseLdsQ, "RunQQuantizeGlobal requires kUseLdsQ=false");
        const float* smem_mean =
            reinterpret_cast<const float*>(reinterpret_cast<const char*>(smem) +
                                           kSmemMeanOffset);

        const index_t tid            = get_thread_id();
        constexpr index_t kNumGroups = kCols / kScaleGranularity;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max  = 1.0f / 6.0f;
        const InputT* src_row        = q_ptr + row_idx * kCols + d_start;

        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        constexpr index_t kVec = 8;
        static_assert(kScaleGranularity % kVec == 0, "");
        for(index_t v = 0; v < kScaleGranularity / kVec; v++)
        {
            float tmp[kVec];
            load_vec8(src_row + v * kVec, tmp);
            for(index_t j = 0; j < kVec; j++)
            {
                const float val          = tmp[j] - smem_mean[d_start + v * kVec + j];
                group_data[v * kVec + j] = val;
                max_abs                  = max(max_abs, abs(val));
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
    // Step 3: K' = K_tile - k_mean -> k_prime_ptr (vectorized store), then
    //   quantize K' -> MXFP4.
    //   k_mean is cached into smem[0..kCols*4] (reuses Q tile LDS space or
    //   the beginning of smem when kUseLdsQ=false).
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
        // Reuse smem[0..kCols*sizeof(float)] for the k_mean float cache.
        // This is safe because: when kUseLdsQ=true, Q tile bytes are done;
        // when kUseLdsQ=false, smem starts at mean offset which is 0.
        float* smem_f     = reinterpret_cast<float*>(smem);
        const index_t tid = get_thread_id();

        constexpr index_t kVec = 8;
        for(index_t d = tid * kVec; d < kCols; d += kBlockSize * kVec)
        {
            float tmp[kVec];
            load_vec8(k_mean_ptr + d, tmp);
            for(index_t j = 0; j < kVec && d + j < kCols; j++)
                smem_f[d + j] = tmp[j];
        }
        block_sync_lds();

        constexpr index_t kNumGroups = kCols / kScaleGranularity;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            InputT* dst_row   = k_prime_ptr + row_idx * k_prime_stride + d_start;
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

        static_assert(kScaleGranularity % kVec == 0, "");
        for(index_t v = 0; v < kScaleGranularity / kVec; v++)
        {
            float tmp[kVec];
            load_vec8(src_row + v * kVec, tmp);
            for(index_t j = 0; j < kVec; j++)
            {
                const float val          = tmp[j] - smem_f[d_start + v * kVec + j];
                group_data[v * kVec + j] = val;
                tmp[j]                   = val;
                max_abs                  = max(max_abs, abs(val));
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
    // RunKMeanPartial: kept for API compatibility.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunKMeanPartial(const InputT* __restrict__ k_tile_ptr,
                                        index_t n_rows,
                                        index_t stride_k,
                                        float* __restrict__ k_mean_partial) const
    {
        const index_t d = get_thread_id();
        float acc       = 0.0f;
        for(index_t r = 0; r < n_rows; r++)
            acc += static_cast<float>(k_tile_ptr[r * stride_k + d]);
        atomicAdd(&k_mean_partial[d], acc);
    }
};

} // namespace ck_tile
