// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_mxfp4_pack.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_pipeline.hpp"

namespace ck_tile {

// ============================================================================
// SageAttnV3PreprocessKernel
//
// Grid: (num_tiles, nhead, batch)
//   num_tiles = max(num_q_tiles, num_k_tiles)
//
// Each CTA at tile_x processes (with bounds checking):
//   - Q tile[tile_x]: load into LDS, compute mean, quantize to MXFP4
//   - K tile[tile_x]: K' = K - k_mean -> k_prime; quantize K' to MXFP4
//
// Three-step Q pipeline (eliminates double HBM read):
//   0. RunLoadQTile: load Q[kRows x kCols] from HBM into LDS once (vectorized).
//   1. RunQMean:     compute column mean from LDS (fast, no HBM traffic).
//   2. RunQQuantize: quantize Q from LDS - mean (no HBM traffic).
// Then K step rewrites LDS for k_mean cache.
// ============================================================================

template <typename InputT>
struct SageAttnV3PreprocessArgs
{
    // --- Q: [batch, nhead, seqlen_q, hdim] InputT ---
    const InputT* q_ptr;
    index_t seqlen_q;
    index_t hdim;
    index_t stride_q;
    index_t nhead_stride_q;
    index_t batch_stride_q;

    // Q hat: [batch, nhead, seqlen_q, hdim/2] uint8
    uint8_t* q_hat_ptr;
    index_t stride_q_hat;
    index_t nhead_stride_q_hat;
    index_t batch_stride_q_hat;

    // Q scale: [batch, nhead, seqlen_q, hdim/32] uint8
    uint8_t* q_scale_ptr;
    index_t stride_q_scale;
    index_t nhead_stride_q_scale;
    index_t batch_stride_q_scale;

    // Q mean: [batch, nhead, num_q_tiles, hdim] InputT
    InputT* q_mean_ptr;
    index_t q_tile_size;         // kM0
    index_t stride_q_mean;       // = hdim
    index_t nhead_stride_q_mean; // = num_q_tiles * hdim
    index_t batch_stride_q_mean;

    // --- K: [batch, nhead, seqlen_k, hdim] InputT ---
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t stride_k;
    index_t nhead_stride_k;
    index_t batch_stride_k;

    // K hat: [batch, nhead, seqlen_k, hdim/2] uint8
    uint8_t* k_hat_ptr;
    index_t stride_k_hat;
    index_t nhead_stride_k_hat;
    index_t batch_stride_k_hat;

    // K scale: [batch, nhead, seqlen_k, hdim/32] uint8
    uint8_t* k_scale_ptr;
    index_t stride_k_scale;
    index_t nhead_stride_k_scale;
    index_t batch_stride_k_scale;

    // K mean: [batch, nhead, hdim] InputT
    const InputT* k_mean_ptr;
    index_t nhead_stride_k_mean;
    index_t batch_stride_k_mean;

    // K prime: [batch, nhead, seqlen_k, hdim] InputT row-major
    InputT* k_prime_ptr;
    index_t stride_k_prime;
    index_t nhead_stride_k_prime;
    index_t batch_stride_k_prime;

    // --- V: [batch, nhead, seqlen_k, hdim] InputT ---
    const InputT* v_ptr;
    index_t nhead_stride_v;
    index_t batch_stride_v;

    // V hat: [batch, nhead, hdim, seqlen_k/2] uint8 (transposed)
    uint8_t* v_hat_ptr;
    index_t stride_v_hat;
    index_t nhead_stride_v_hat;
    index_t batch_stride_v_hat;

    // V scale: [batch, nhead, hdim, seqlen_k/32] uint8
    uint8_t* v_scale_ptr;
    index_t stride_v_scale;
    index_t nhead_stride_v_scale;
    index_t batch_stride_v_scale;

    // --- Dimensions ---
    index_t batch;
    index_t nhead;
    index_t num_q_tiles;
    index_t num_k_tiles;
};

template <typename InputT>
using SageAttnV3PreprocessKargs = SageAttnV3PreprocessArgs<InputT>;
template <typename InputT>
using SageAttnV3PreprocessHostArgs = SageAttnV3PreprocessArgs<InputT>;

// kBlockSize = kRows * (kCols / 32): one thread per (row, MXFP4-group) pair.
// RunQQuantize and RunKSmoothAndQuantize achieve 100% thread utilisation.
template <typename InputT_,
          index_t kRows_,
          index_t kCols_,
          index_t kBlockSize_ = kRows_ * (kCols_ / 32)>
struct SageAttnV3PreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kBlockSize_;

    using Pipeline =
        SageAttnV3PreprocessPipeline<InputT, kRows, kCols, /*kScaleGranularity=*/32, kBlockSize>;

    using Kargs = SageAttnV3PreprocessArgs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        const index_t num_tiles = max(k.num_q_tiles, k.num_k_tiles);
        return dim3(num_tiles, k.nhead, k.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_x    = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        const bool do_q = tile_x < kargs.num_q_tiles;
        const bool do_k = tile_x < kargs.num_k_tiles;

        __shared__ char smem[GetSmemSize()];

        Pipeline pipeline{};

        // Compute Q tile pointers and dims once.
        const index_t row_start_q = tile_x * kargs.q_tile_size;
        const index_t n_rows_q =
            do_q ? min(static_cast<index_t>(kRows), kargs.seqlen_q - row_start_q) : 0;
        const InputT* src_q = kargs.q_ptr + batch_idx * kargs.batch_stride_q +
                              head_idx * kargs.nhead_stride_q + row_start_q * kargs.stride_q;
        InputT* q_mean = kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
                         head_idx * kargs.nhead_stride_q_mean + tile_x * kargs.stride_q_mean;
        uint8_t* dst_q_hat = kargs.q_hat_ptr + batch_idx * kargs.batch_stride_q_hat +
                             head_idx * kargs.nhead_stride_q_hat + row_start_q * kargs.stride_q_hat;
        uint8_t* dst_q_scale = kargs.q_scale_ptr + batch_idx * kargs.batch_stride_q_scale +
                               head_idx * kargs.nhead_stride_q_scale +
                               row_start_q * kargs.stride_q_scale;

        if constexpr(Pipeline::kUseLdsQ)
        {
            // LDS path: Q tile read once from HBM, mean and quantize from LDS.
            // ---- Step 0: Load Q tile into LDS ----
            if(do_q)
                pipeline.RunLoadQTile(src_q, smem, n_rows_q);
            block_sync_lds();

            // ---- Step 1: Q mean from LDS ----
            if(do_q)
                pipeline.RunQMean(smem, q_mean, n_rows_q);
            block_sync_lds();

            // ---- Step 2: Q quantize from LDS ----
            if(do_q)
                pipeline.RunQQuantize(smem, dst_q_hat, dst_q_scale, n_rows_q);
        }
        else
        {
            // Global path: Q tile too large for LDS; read Q twice from HBM.
            // ---- Step 1: Q mean from global ----
            if(do_q)
                pipeline.RunQMeanGlobal(src_q, smem, q_mean, n_rows_q);
            block_sync_lds();

            // ---- Step 2: Q quantize from global (mean in smem) ----
            if(do_q)
                pipeline.RunQQuantizeGlobal(src_q, smem, dst_q_hat, dst_q_scale, n_rows_q);
        }

        // ---- Step 3: K smooth + quantize ------------------------------------
        // smem is reused: RunKSmoothAndQuantize loads k_mean into smem[0..kCols*4].
        if(do_k)
        {
            const index_t row_start_k = tile_x * kRows;
            const index_t n_rows_k =
                min(static_cast<index_t>(kRows), kargs.seqlen_k - row_start_k);

            const InputT* src_k = kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                                  head_idx * kargs.nhead_stride_k + row_start_k * kargs.stride_k;

            const InputT* k_mean = kargs.k_mean_ptr + batch_idx * kargs.batch_stride_k_mean +
                                   head_idx * kargs.nhead_stride_k_mean;

            InputT* k_prime = kargs.k_prime_ptr + batch_idx * kargs.batch_stride_k_prime +
                              head_idx * kargs.nhead_stride_k_prime +
                              row_start_k * kargs.stride_k_prime;

            uint8_t* dst_k_hat = kargs.k_hat_ptr + batch_idx * kargs.batch_stride_k_hat +
                                 head_idx * kargs.nhead_stride_k_hat +
                                 row_start_k * kargs.stride_k_hat;

            uint8_t* dst_k_scale = kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
                                   head_idx * kargs.nhead_stride_k_scale +
                                   row_start_k * kargs.stride_k_scale;

            pipeline.RunKSmoothAndQuantize(src_k,
                                           k_mean,
                                           k_prime,
                                           kargs.stride_k_prime,
                                           dst_k_hat,
                                           dst_k_scale,
                                           smem,
                                           n_rows_k);
        }
    }
};

// ============================================================================
// SageAttnV3KMeanKernel
//
// Computes k_mean[batch, nhead, hdim] = mean over seqlen_k of K[batch, nhead, :, hdim].
//
// Grid:      (1, nhead, batch) -- single-pass, no atomics
// BlockSize: kColGroups * kRowsPerGroup (up to 1024 threads = 16 wavefronts)
//
// Algorithm (vectorized, no cross-CTA atomics):
//   Each thread handles kVec=8 consecutive columns (one uint4 load per row).
//   Thread layout:
//     col_grp  = tid % kColGroups   (which 8-column group)
//     row_grp  = tid / kColGroups   (which row subset)
//   Inner loop: load 8 fp16 at once, accumulate 8 float partial sums.
//   Intra-block reduce via smem: leader (row_grp==0) sums kRowsPerGroup partials.
//   Vectorized store of k_mean and k_mean_partial.
//
// Eliminating cross-CTA atomics (vs. the old tiled approach) removes the
// main latency bottleneck at large batch x nhead counts.
// ============================================================================

template <typename InputT>
struct SageAttnV3KMeanKargs
{
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t hdim;
    index_t stride_k;
    index_t nhead_stride_k;
    index_t batch_stride_k;

    float* k_mean_partial_ptr; // [batch, nhead, hdim] float, kept for API compat
    InputT* k_mean_ptr;        // [batch, nhead, hdim] InputT, output
    int32_t* counter_ptr;      // unused in new kernel, kept for API compat

    index_t nhead_stride_kmean;
    index_t batch_stride_kmean;

    index_t num_k_tiles; // unused in new kernel (processes all rows in one CTA)
    index_t nhead;
    index_t batch;
};

template <typename InputT_, index_t kRows_, index_t kCols_>
struct SageAttnV3KMeanKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows = kRows_;
    static constexpr index_t kCols = kCols_;

    // Vectorized: kVec=8 fp16 loaded per instruction.
    // kColGroups: number of 8-column groups covering kCols.
    // kRowsPerGroup: threads collaborating on the same 8-column group.
    // kBlockSize = kColGroups * kRowsPerGroup.
    //
    // For kCols=128: kColGroups=16, kRowsPerGroup=16, kBlockSize=256 (4 wavefronts).
    // For kCols=256: kColGroups=32, kRowsPerGroup=8,  kBlockSize=256 (4 wavefronts).
    // kBlockSize=256 allows up to 32/4=8 CTAs per CU simultaneously on gfx950
    // (max 32 wavefronts per CU), maximising occupancy.
    static constexpr index_t kVec         = 8;
    static constexpr index_t kColGroups   = kCols / kVec;
    static constexpr index_t kRowsPerGroup = 256 / kColGroups; // target kBlockSize=256
    static constexpr index_t kBlockSize   = kColGroups * kRowsPerGroup;

    static_assert(kCols % kVec == 0, "kCols must be divisible by kVec=8");

    // Pipeline kept as alias for backward compat (not called by this kernel).
    using Pipeline = SageAttnV3PreprocessPipeline<InputT, kRows, kCols>;

    using Kargs = SageAttnV3KMeanKargs<InputT>;

    // Grid: one CTA per (head, batch) -- processes all seqlen_k rows.
    CK_TILE_HOST static dim3 GridSize(const Kargs& k) { return dim3(1, k.nhead, k.batch); }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // smem: kBlockSize * kVec floats for partial sum reduction.
    // = 256 * 8 * 4 = 8192 bytes for kCols=128 or kCols=256.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kBlockSize * kVec * static_cast<index_t>(sizeof(float));
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;
        const index_t tid       = get_thread_id();

        // Thread mapping: col_grp covers kVec consecutive columns.
        const index_t col_grp   = tid % kColGroups; // 0 .. kColGroups-1
        const index_t row_grp   = tid / kColGroups; // 0 .. kRowsPerGroup-1
        const index_t col_start = col_grp * kVec;

        const InputT* k_head = kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                               head_idx * kargs.nhead_stride_k;

        // --- Step 1: vectorized accumulation over strided rows ---
        // Each thread loads 8 fp16 at once (128-bit), accumulates 8 floats.
        float acc[kVec] = {};
        for(index_t r = row_grp; r < kargs.seqlen_k; r += kRowsPerGroup)
        {
            float tmp[kVec];
            load_vec8(k_head + r * kargs.stride_k + col_start, tmp);
            for(int i = 0; i < kVec; i++)
                acc[i] += tmp[i];
        }

        // --- Step 2: intra-block reduce via smem ---
        // Layout: smem[tid * kVec + i] so that leader accesses are strided.
        __shared__ float smem_arr[kBlockSize * kVec];
        float* smem_f = smem_arr;
        for(int i = 0; i < kVec; i++)
            smem_f[tid * kVec + i] = acc[i];
        block_sync_lds();

        // Leader (row_grp==0) accumulates all kRowsPerGroup partial sums.
        if(row_grp == 0)
        {
            for(index_t g = 1; g < kRowsPerGroup; g++)
            {
                const index_t src_tid = g * kColGroups + col_grp;
                for(int i = 0; i < kVec; i++)
                    acc[i] += smem_f[src_tid * kVec + i];
            }

            // --- Step 3: compute mean and write outputs ---
            float* partial = kargs.k_mean_partial_ptr +
                             batch_idx * kargs.batch_stride_kmean +
                             head_idx * kargs.nhead_stride_kmean;
            InputT* k_mean = kargs.k_mean_ptr + batch_idx * kargs.batch_stride_kmean +
                             head_idx * kargs.nhead_stride_kmean;

            float mean_f[kVec];
            for(int i = 0; i < kVec; i++)
                mean_f[i] = acc[i] / static_cast<float>(kargs.seqlen_k);

            // Vectorized store of float partial (for API compat).
            store_vec8(partial + col_start, mean_f);
            // Vectorized store of InputT k_mean.
            store_vec8(k_mean + col_start, mean_f);
        }
    }
};

// ============================================================================
// SageAttnV3VPreprocessKernel
//
// Quantises V (transposed layout) using an LDS-based 2-D tile transpose.
//
// Grid:      (seqlen_k / (kVGroup * kVGroupsPerBlock), hdim / kVHdimTile, batch * nhead)
// BlockSize: kVGroup * kVGroupsPerBlock  (kVGroupsPerBlock=8 -> 256 threads = 4 wavefronts)
//
// kVGroupsPerBlock=8:
//   LDS = 8 * 32 * 33 * 4 = 33792 bytes < 64 KB.
//   kBlockSize = 32 * 8 = 256 threads = 4 WAVE64 wavefronts.
//   seqlen_k_padded must be divisible by kVGroup * kVGroupsPerBlock = 256.
//   get_buffer_sizes enforces this by padding seqlen_k to a multiple of 256.
//
// LDS bank conflict analysis (gfx950):
//   smem[kVGroupsPerBlock][kVGroup][kVHdimTile + 1] as float32.
//   Row stride = kVHdimTile + 1 = 33 floats.
//   bank(j, d) = (j * 33 + d) % 32 = (j + d) % 32 (since 33 % 32 = 1).
//   Column access (fixed d, j=0..31): bank = (j + d) % 32 -> 32 distinct banks OK.
// ============================================================================

template <typename InputT>
struct SageAttnV3VPreprocessKargs
{
    const InputT* v_ptr;
    index_t seqlen_k;       // padded seqlen_k (used for GridSize)
    index_t seqlen_k_real;  // unpadded seqlen_k (used for input bounds check)
    index_t hdim;
    index_t nhead_stride_v;
    index_t batch_stride_v;

    uint8_t* v_hat_ptr;
    index_t stride_v_hat;
    index_t nhead_stride_v_hat;
    index_t batch_stride_v_hat;

    uint8_t* v_scale_ptr;
    index_t stride_v_scale;
    index_t nhead_stride_v_scale;
    index_t batch_stride_v_scale;

    index_t nhead;
    index_t batch;
};

template <typename InputT_,
          index_t kVGroup_          = 32,
          index_t kVHdimTile_       = 32,
          index_t kVGroupsPerBlock_ = 4>
struct SageAttnV3VPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kVGroup          = kVGroup_;
    static constexpr index_t kVHdimTile       = kVHdimTile_;
    static constexpr index_t kVGroupsPerBlock = kVGroupsPerBlock_;
    static constexpr index_t kScaleGranularity = 32;
    static constexpr index_t kBlockSize        = kVGroup * kVGroupsPerBlock;
    static constexpr index_t kLDSPad          = 1;

    static_assert(kVGroup == kScaleGranularity,
                  "kVGroup must equal kScaleGranularity (32) for MXFP4");
    static_assert(kVHdimTile % kVGroup == 0 || kVGroup % kVHdimTile == 0,
                  "kVHdimTile and kVGroup must be multiples of each other");
    static_assert(kVGroupsPerBlock >= 1, "kVGroupsPerBlock must be at least 1");

    using Kargs = SageAttnV3VPreprocessKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(
            k.seqlen_k / (kVGroup * kVGroupsPerBlock), k.hdim / kVHdimTile, k.batch * k.nhead);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kVGroupsPerBlock * kVGroup * (kVHdimTile + kLDSPad) * sizeof(float);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t d_idx     = blockIdx.y;
        const index_t bh_idx    = blockIdx.z;
        const index_t head_idx  = bh_idx % kargs.nhead;
        const index_t batch_idx = bh_idx / kargs.nhead;
        const index_t tid       = get_thread_id();

        const index_t grp_local = tid / kVGroup;
        const index_t row_local = tid % kVGroup;
        const index_t g_abs     = blockIdx.x * kVGroupsPerBlock + grp_local;

        const index_t col_start = d_idx * kVHdimTile;

        __shared__ float smem[kVGroupsPerBlock][kVGroup][kVHdimTile + kLDSPad];

        // ---- Step 1: vectorized load [kVGroup, kVHdimTile] from V -> LDS ----
        const index_t row = g_abs * kVGroup + row_local;
        const InputT* v_base =
            kargs.v_ptr + batch_idx * kargs.batch_stride_v + head_idx * kargs.nhead_stride_v;

        if(row < kargs.seqlen_k_real)
        {
            const InputT* v_row = v_base + row * kargs.hdim + col_start;
            constexpr index_t kVec = 8;
            static_assert(kVHdimTile % kVec == 0, "kVHdimTile must be divisible by 8");
            for(index_t v = 0; v < kVHdimTile / kVec; v++)
            {
                float tmp[kVec];
                ck_tile::load_vec8(v_row + v * kVec, tmp);
                for(index_t j = 0; j < kVec; j++)
                    smem[grp_local][row_local][v * kVec + j] = tmp[j];
            }
        }
        else
        {
            for(index_t d = 0; d < kVHdimTile; d++)
                smem[grp_local][row_local][d] = 0.0f;
        }
        block_sync_lds();

        // ---- Step 2: quantize per hdim channel from LDS ----
        constexpr float rcp_dst_max = 1.0f / 6.0f;

        uint8_t* v_hat_base = kargs.v_hat_ptr + batch_idx * kargs.batch_stride_v_hat +
                              head_idx * kargs.nhead_stride_v_hat;
        uint8_t* v_scale_base = kargs.v_scale_ptr + batch_idx * kargs.batch_stride_v_scale +
                                head_idx * kargs.nhead_stride_v_scale;

        for(index_t d_local = row_local; d_local < kVHdimTile; d_local += kVGroup)
        {
            const index_t d_global = col_start + d_local;

            float group_data[kVGroup];
            float max_abs = 0.0f;
            for(index_t j = 0; j < kVGroup; j++)
            {
                group_data[j] = smem[grp_local][j][d_local];
                max_abs       = max(max_abs, abs(group_data[j]));
            }

            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            v_scale_base[d_global * kargs.stride_v_scale + g_abs] =
                static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            uint8_t* hat_ptr = v_hat_base + d_global * kargs.stride_v_hat + g_abs * (kVGroup / 2);
            PackFP4Group<kVGroup>(group_data, hat_ptr, scale);
        }
    }
};

} // namespace ck_tile
