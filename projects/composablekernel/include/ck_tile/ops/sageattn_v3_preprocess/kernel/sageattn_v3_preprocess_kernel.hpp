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
//   num_tiles = max(num_q_tiles, num_k_tiles, hdim)
//
// Each CTA at tile_x processes (with bounds checking):
//   - Q tile[tile_x]:      Q mean → smem → Q quantize
//   - K tile[tile_x]:      K' = K - k_mean → k_prime; K' quantize
//   - V channel[tile_x]:   V transpose + quantize
//
// delta_s is produced by a separate batched GEMM (q_mean @ K'^T).
// k_mean must already be available (written by SageAttnV3KMeanKernel).
// ============================================================================

// SageAttnV3PreprocessArgs: unified host/kernel args (typed on InputT).
// Used as both the public host-side API and the kernel arg struct —
// eliminates the old void*-based Hargs + typed Kargs duplication.
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

    // K mean: [batch, nhead, hdim] InputT — produced by SageAttnV3KMeanKernel
    const InputT* k_mean_ptr;
    index_t nhead_stride_k_mean;
    index_t batch_stride_k_mean;

    // K prime: [batch, nhead, seqlen_k, hdim] InputT row-major — K' = K - k_mean
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

// Backward-compatible aliases.
template <typename InputT>
using SageAttnV3PreprocessKargs = SageAttnV3PreprocessArgs<InputT>;
template <typename InputT>
using SageAttnV3PreprocessHostArgs = SageAttnV3PreprocessArgs<InputT>;

// kBlockSize = kRows * (kCols / 32): one thread per (row, MXFP4-group) pair.
// RunQQuantize and RunKSmoothAndQuantize achieve 100% thread utilisation.
// RunQMean uses only kCols of the kBlockSize threads (column-per-thread reduction).
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

    // SageAttnV3PreprocessArgs<InputT> serves as both host and kernel args — no conversion needed.
    using Kargs = SageAttnV3PreprocessArgs<InputT>;

    // Grid: (num_tiles, nhead, batch)
    //   num_tiles = max(num_q_tiles, num_k_tiles); V is handled by SageAttnV3VPreprocessKernel.
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

        // ---- Step 1: Q mean ------------------------------------------------
        if(do_q)
        {
            const index_t row_start = tile_x * kargs.q_tile_size;
            const index_t n_rows_q  = min(static_cast<index_t>(kRows), kargs.seqlen_q - row_start);

            const InputT* src_q = kargs.q_ptr + batch_idx * kargs.batch_stride_q +
                                  head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

            InputT* q_mean = kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
                             head_idx * kargs.nhead_stride_q_mean + tile_x * kargs.stride_q_mean;

            pipeline.RunQMean(src_q, q_mean, smem, n_rows_q);
        }
        block_sync_lds(); // smem q_mean visible for step 2

        // ---- Step 2: Q quantize --------------------------------------------
        if(do_q)
        {
            const index_t row_start = tile_x * kargs.q_tile_size;
            const index_t n_rows_q  = min(static_cast<index_t>(kRows), kargs.seqlen_q - row_start);

            const InputT* src_q = kargs.q_ptr + batch_idx * kargs.batch_stride_q +
                                  head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

            uint8_t* dst_q_hat = kargs.q_hat_ptr + batch_idx * kargs.batch_stride_q_hat +
                                 head_idx * kargs.nhead_stride_q_hat +
                                 row_start * kargs.stride_q_hat;

            uint8_t* dst_q_scale = kargs.q_scale_ptr + batch_idx * kargs.batch_stride_q_scale +
                                   head_idx * kargs.nhead_stride_q_scale +
                                   row_start * kargs.stride_q_scale;

            pipeline.RunQQuantize(src_q, dst_q_hat, dst_q_scale, smem, n_rows_q);
        }

        // ---- Step 3: K smooth + quantize -----------------------------------
        // smem is reused from RunQMean: q_mean data is no longer needed after
        // RunQQuantize. RunKSmoothAndQuantize will overwrite it with k_mean.
        if(do_k)
        {
            const index_t row_start = tile_x * kRows;
            const index_t n_rows_k  = min(static_cast<index_t>(kRows), kargs.seqlen_k - row_start);

            const InputT* src_k = kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                                  head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

            const InputT* k_mean = kargs.k_mean_ptr + batch_idx * kargs.batch_stride_k_mean +
                                   head_idx * kargs.nhead_stride_k_mean;

            InputT* k_prime = kargs.k_prime_ptr + batch_idx * kargs.batch_stride_k_prime +
                              head_idx * kargs.nhead_stride_k_prime +
                              row_start * kargs.stride_k_prime;

            uint8_t* dst_k_hat = kargs.k_hat_ptr + batch_idx * kargs.batch_stride_k_hat +
                                 head_idx * kargs.nhead_stride_k_hat +
                                 row_start * kargs.stride_k_hat;

            uint8_t* dst_k_scale = kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
                                   head_idx * kargs.nhead_stride_k_scale +
                                   row_start * kargs.stride_k_scale;

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
// Grid:      (num_k_tiles, nhead, batch)
// BlockSize: kCols  (one thread per d-channel)
//
// Algorithm: each CTA atomicAdds its partial column-sum into k_mean_partial (float).
// The last CTA to finish (detected via completion counter) normalises and casts to InputT.
//
// Caller must zero-initialise k_mean_partial_ptr and counter_ptr before launch.
// ============================================================================

// Unified kernel/host args for the K-mean kernel (typed on InputT).
template <typename InputT>
struct SageAttnV3KMeanKargs
{
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t hdim;
    index_t stride_k; // = hdim
    index_t nhead_stride_k;
    index_t batch_stride_k;

    float* k_mean_partial_ptr; // [batch, nhead, hdim] float, pre-zeroed
    InputT* k_mean_ptr;        // [batch, nhead, hdim] InputT, output
    int32_t* counter_ptr;      // [batch, nhead] int32, pre-zeroed

    index_t nhead_stride_kmean; // = hdim
    index_t batch_stride_kmean; // = nhead * hdim

    index_t num_k_tiles;
    index_t nhead;
    index_t batch;
};

template <typename InputT_, index_t kRows_, index_t kCols_>
struct SageAttnV3KMeanKernel
{
    using InputT = InputT_;

    static constexpr index_t kRows      = kRows_;
    static constexpr index_t kCols      = kCols_;
    static constexpr index_t kBlockSize = kCols; // one thread per d-channel

    // Pipeline reused only for RunKMeanPartial.
    using Pipeline = SageAttnV3PreprocessPipeline<InputT, kRows, kCols>;

    using Kargs = SageAttnV3KMeanKargs<InputT>;

    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(k.num_k_tiles, k.nhead, k.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // Smem: one int32 flag to broadcast "is_last" to all threads.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return sizeof(int32_t); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t tile_x    = get_block_id();
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        // Tile row range (handles tail).
        const index_t row_start = tile_x * kRows;
        const index_t row_end   = min(row_start + kRows, kargs.seqlen_k);
        const index_t n_rows    = row_end - row_start;

        const InputT* k_tile = kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                               head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

        float* partial = kargs.k_mean_partial_ptr + batch_idx * kargs.batch_stride_kmean +
                         head_idx * kargs.nhead_stride_kmean;

        // --- Step 1: accumulate partial column-sum via atomicAdd ---
        Pipeline pipeline{};
        pipeline.RunKMeanPartial(k_tile, n_rows, kargs.stride_k, partial);

        // Ensure our atomicAdds are globally visible before incrementing counter.
        __threadfence();

        // --- Step 2: completion counter (thread 0 only) ---
        __shared__ int32_t smem_is_last;
        if(get_thread_id() == 0)
        {
            int32_t* cnt = kargs.counter_ptr + batch_idx * kargs.nhead + head_idx;
            int32_t prev = atomicAdd(cnt, 1);
            smem_is_last = (prev == kargs.num_k_tiles - 1) ? 1 : 0;
        }
        block_sync_lds();

        // --- Step 3: last CTA normalises and stores k_mean as InputT ---
        if(smem_is_last)
        {
            const index_t d = get_thread_id();
            InputT* k_mean  = kargs.k_mean_ptr + batch_idx * kargs.batch_stride_kmean +
                             head_idx * kargs.nhead_stride_kmean;

            const float mean_f = partial[d] / static_cast<float>(kargs.seqlen_k);
            k_mean[d]          = static_cast<InputT>(mean_f);
        }
    }
};

// ============================================================================
// SageAttnV3VPreprocessKernel
//
// Quantises V (transposed layout) using an LDS-based 2-D tile transpose that
// avoids the non-coalesced column reads of the old per-channel approach.
//
// Grid:      (seqlen_k / (kVGroup * kVGroupsPerBlock), hdim / kVHdimTile, batch * nhead)
// BlockSize: kVGroup * kVGroupsPerBlock  (kVGroupsPerBlock = 2 by default → 64 threads,
//            filling one complete WAVE64 wavefront on gfx950)
//
// Each CTA processes kVGroupsPerBlock consecutive seqlen groups.
// Thread layout within the CTA:
//   grp_local = tid / kVGroup       (0 .. kVGroupsPerBlock-1)
//   row_local = tid % kVGroup       (0 .. kVGroup-1)
//   g_abs     = blockIdx.x * kVGroupsPerBlock + grp_local
//
// Algorithm per CTA (blockIdx.x, d_idx, bh_idx):
//   1. Each thread loads one row of V:
//        V[g_abs*kVGroup + row_local, d_idx*kVHdimTile .. +kVHdimTile]
//      into smem[grp_local][row_local][0..kVHdimTile-1] as float32 → coalesced.
//   2. After __syncthreads, thread row_local owns channel d_local=row_local
//      and reads column smem[grp_local][0..kVGroup-1][d_local] from LDS.
//   3. Computes MXFP4 scale and packs the group, then writes to V_hat /
//      V_scale in transposed layout [hdim, seqlen_k/2|seqlen_k/32].
//
// LDS bank conflict analysis (gfx950 / MI350):
//   gfx950 has 64 physical banks, but ds_read_b32 conflict resolution acts
//   as-if there are 32 banks (64 banks only matter for ds_read_b64/b128).
//   Reference: "LDS Analysis gfx942/950" (Kerry Wang, internal Confluence).
//
//   smem[kVGroupsPerBlock][kVGroup][kVHdimTile + kLDSPad] as float32, kLDSPad = 1.
//   Within each group slice, row stride = kVHdimTile + 1 = 33 floats.
//   bank(j, d) = (j * 33 + d) % 32 = (j + d) % 32   [since 33 % 32 = 1]
//   Column access (fixed d, j = 0..31): bank = (j + d) % 32 → 32 distinct banks ✓
//   Between group slices the offset is kVGroup*(kVHdimTile+1) = 32*33 = 1056 floats,
//   which is 1056 % 32 = 0 → same bank pattern per slice, still conflict-free ✓
//
// Caller requirements:
//   seqlen_k % (kVGroup * kVGroupsPerBlock) == 0
//   hdim     % kVHdimTile                   == 0
// ============================================================================

// Unified kernel/host args for the V preprocess kernel (typed on InputT).
template <typename InputT>
struct SageAttnV3VPreprocessKargs
{
    // V source: [batch, nhead, seqlen_k, hdim] InputT row-major
    const InputT* v_ptr;
    index_t seqlen_k;       // padded seqlen_k (used for GridSize)
    index_t seqlen_k_real;  // unpadded seqlen_k (used for input bounds check)
    index_t hdim;           // also the row stride of V
    index_t nhead_stride_v; // = seqlen_k * hdim
    index_t batch_stride_v; // = nhead * seqlen_k * hdim

    // V hat: [batch, nhead, hdim, seqlen_k/2] uint8 (transposed)
    uint8_t* v_hat_ptr;
    index_t stride_v_hat; // = seqlen_k / 2
    index_t nhead_stride_v_hat;
    index_t batch_stride_v_hat;

    // V scale: [batch, nhead, hdim, seqlen_k/32] uint8
    uint8_t* v_scale_ptr;
    index_t stride_v_scale; // = seqlen_k / kScaleGranularity
    index_t nhead_stride_v_scale;
    index_t batch_stride_v_scale;

    index_t nhead;
    index_t batch;
};

template <typename InputT_,
          index_t kVGroup_          = 32,
          index_t kVHdimTile_       = 32,
          index_t kVGroupsPerBlock_ = 2>
struct SageAttnV3VPreprocessKernel
{
    using InputT = InputT_;

    static constexpr index_t kVGroup           = kVGroup_;
    static constexpr index_t kVHdimTile        = kVHdimTile_;
    static constexpr index_t kVGroupsPerBlock  = kVGroupsPerBlock_;
    static constexpr index_t kScaleGranularity = 32;
    // kVGroupsPerBlock groups per CTA → full WAVE64 wavefronts on gfx950.
    static constexpr index_t kBlockSize = kVGroup * kVGroupsPerBlock;
    // Pad LDS rows by 1 float32. For ds_read_b32 on gfx950 the effective bank
    // count is 32. Column stride = (kVHdimTile+1) % 32 = 1 → coprime with 32
    // → zero bank conflicts (see struct comment for full derivation).
    static constexpr index_t kLDSPad = 1;

    static_assert(kVGroup == kScaleGranularity,
                  "kVGroup must equal kScaleGranularity (32) for MXFP4");
    static_assert(kVHdimTile % kVGroup == 0 || kVGroup % kVHdimTile == 0,
                  "kVHdimTile and kVGroup must be multiples of each other");
    static_assert(kVGroupsPerBlock >= 1, "kVGroupsPerBlock must be at least 1");

    using Kargs = SageAttnV3VPreprocessKargs<InputT>;

    // Grid: (seqlen_k / (kVGroup * kVGroupsPerBlock), hdim / kVHdimTile, batch * nhead)
    CK_TILE_HOST static dim3 GridSize(const Kargs& k)
    {
        return dim3(
            k.seqlen_k / (kVGroup * kVGroupsPerBlock), k.hdim / kVHdimTile, k.batch * k.nhead);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // LDS: kVGroupsPerBlock × kVGroup × (kVHdimTile + kLDSPad) float32.
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
        const index_t tid       = get_thread_id(); // 0 .. kBlockSize-1

        // Map thread to its seqlen group within the CTA and its row within that group.
        const index_t grp_local = tid / kVGroup; // 0 .. kVGroupsPerBlock-1
        const index_t row_local = tid % kVGroup; // 0 .. kVGroup-1
        const index_t g_abs     = blockIdx.x * kVGroupsPerBlock + grp_local;

        const index_t col_start = d_idx * kVHdimTile;

        // LDS: smem[kVGroupsPerBlock][kVGroup][kVHdimTile + kLDSPad].
        // bank(j, d) = (j * 33 + d) % 32 = (j + d) % 32 → zero bank conflicts ✓
        __shared__ float smem[kVGroupsPerBlock][kVGroup][kVHdimTile + kLDSPad];

        // ---- Step 1: coalesced load [kVGroup, kVHdimTile] from V → float LDS ----
        const index_t row = g_abs * kVGroup + row_local;

        const InputT* v_base =
            kargs.v_ptr + batch_idx * kargs.batch_stride_v + head_idx * kargs.nhead_stride_v;

        if(row < kargs.seqlen_k_real)
        {
            const InputT* v_row = v_base + row * kargs.hdim + col_start;
            for(index_t d = 0; d < kVHdimTile; d++)
                smem[grp_local][row_local][d] = static_cast<float>(v_row[d]);
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

        // row_local acts as the channel index within each group (kVHdimTile == kVGroup).
        for(index_t d_local = row_local; d_local < kVHdimTile; d_local += kVGroup)
        {
            const index_t d_global = col_start + d_local;

            // Read one column of the LDS tile for this group (no bank conflicts).
            float group_data[kVGroup];
            float max_abs = 0.0f;
            for(index_t j = 0; j < kVGroup; j++)
            {
                group_data[j] = smem[grp_local][j][d_local];
                max_abs       = max(max_abs, abs(group_data[j]));
            }

            // MXFP4 scale (e8m0, round-up to next power-of-two exponent).
            const float scale = bit_cast<float>(
                (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
                numeric_traits<float>::head_mask);

            // V_scale[d_global, g_abs]  (transposed layout)
            v_scale_base[d_global * kargs.stride_v_scale + g_abs] =
                static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

            // V_hat[d_global, g_abs*(kVGroup/2) ..]
            uint8_t* hat_ptr = v_hat_base + d_global * kargs.stride_v_hat + g_abs * (kVGroup / 2);
            PackFP4Group<kVGroup>(group_data, hat_ptr, scale);
        }
    }
};

} // namespace ck_tile
