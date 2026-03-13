// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
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
struct SageAttnV3PreprocessKargs
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

    // --- Dimensions ---
    index_t batch;
    index_t nhead;
    index_t num_q_tiles;
    index_t num_k_tiles;
};

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

    using Kargs = SageAttnV3PreprocessKargs<InputT>;

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
// BlockSize: kWarps*64 = 256 threads (4 wavefronts)
//
// Algorithm (vectorized, no cross-CTA atomics):
//   Each thread handles kVec=8 consecutive columns (one uint4 load per row).
//   Thread layout (NDimP=2: warp_id, lane_id):
//     col_grp = warp_id * kColsPerWarp + lane_id / kRowsPerGroup
//     row_grp = lane_id % kRowsPerGroup
//   Inner loop: load 8 fp16 at once, accumulate 8 float partial sums.
//   Intra-warp reduce via XOR butterfly (warp_shuffle): all lanes in a col_grp
//   group receive the total sum -- no shared memory needed.
//   Vectorized store of k_mean (all threads write same value; idempotent).
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

    InputT* k_mean_ptr; // [batch, nhead, hdim] InputT, output

    index_t nhead_stride_kmean;
    index_t batch_stride_kmean;

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
    // kWarps: fixed at 4 wavefronts (256 threads / 64 lanes per wavefront).
    // kColsPerWarp: col_grp groups assigned per wavefront.
    // kRowsPerGroup: lanes within one wavefront collaborating on the same col_grp.
    // kBlockSize = kWarps * 64 = 256.
    //
    // For kCols=128: kColGroups=16, kWarps=4, kColsPerWarp=4, kRowsPerGroup=16.
    // For kCols=256: kColGroups=32, kWarps=4, kColsPerWarp=8, kRowsPerGroup=8.
    // kBlockSize=256 allows up to 32/4=8 CTAs per CU simultaneously on gfx950
    // (max 32 wavefronts per CU), maximising occupancy.
    static constexpr index_t kVec          = 8;
    static constexpr index_t kColGroups    = kCols / kVec;
    static constexpr index_t kWarps        = 4; // wavefronts per CTA (fixed)
    static constexpr index_t kColsPerWarp  = kColGroups / kWarps;
    static constexpr index_t kRowsPerGroup = 64 / kColsPerWarp; // lanes per col_grp group
    static constexpr index_t kBlockSize    = kWarps * 64;

    static_assert(kCols % kVec == 0, "kCols must be divisible by kVec=8");
    static_assert(kColGroups % kWarps == 0,
                  "kColGroups must be divisible by kWarps=4 for the tile distribution");
    static_assert((kRowsPerGroup & (kRowsPerGroup - 1)) == 0,
                  "kRowsPerGroup must be a power of 2 for XOR butterfly reduction");

    // Pipeline kept as alias for backward compat (not called by this kernel).
    using Pipeline = SageAttnV3PreprocessPipeline<InputT, kRows, kCols>;

    using Kargs = SageAttnV3KMeanKargs<InputT>;

    // Grid: one CTA per (head, batch) -- processes all seqlen_k rows.
    CK_TILE_HOST static dim3 GridSize(const Kargs& k) { return dim3(1, k.nhead, k.batch); }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // No smem needed: cross-thread reduction uses warp XOR shuffles.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t head_idx  = blockIdx.y;
        const index_t batch_idx = blockIdx.z;

        const InputT* k_head = kargs.k_ptr + batch_idx * kargs.batch_stride_k +
                               head_idx * kargs.nhead_stride_k;

        // ----------------------------------------------------------------
        // Thread layout (block-level, NDimP=2):
        //   warp_id in 0..kWarps-1, lane_id in 0..63
        //   col_grp  = warp_id * kColsPerWarp + lane_id / kRowsPerGroup
        //   row_grp  = lane_id % kRowsPerGroup
        //
        // NDimP=2 is required for block-level distributions; NDimP=1 would use
        // only get_lane_id() (0..63) as the partition, making all warps cover
        // the same col_grps and producing wrong results for warps 1..3.
        //
        // Accumulator distribution: 1D [kCols] with R=kRowsPerGroup.
        //   R[0] = kRowsPerGroup: reduced by XOR shuffle within each wavefront.
        //   H[X=0] = [kWarps, kColsPerWarp, kVec]:
        //     H[X=0][0] = kWarps      <- P[0]=warp_id (slow)
        //     H[X=0][1] = kColsPerWarp <- P[1] outer (col_grp within warp)
        //     H[X=0][2] = kVec         <- Y[0] (register)
        //   P[1] inner -> R[0]=kRowsPerGroup (fast within wavefront).
        //   XOR shuffles with masks 1..kRowsPerGroup/2 stay within aligned
        //   kRowsPerGroup-wide lane groups inside each wavefront.
        // ----------------------------------------------------------------
        constexpr auto acc_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<kRowsPerGroup>,
                tuple<sequence<kWarps, kColsPerWarp, kVec>>,
                tuple<sequence<1>, sequence<1, 0>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<1>,
                sequence<2>>{});

        // K load tile distribution: 2D [kRowsPerGroup, kCols].
        //   H[X=0] = [kRowsPerGroup]: row dimension.
        //   H[X=1] = [kWarps, kColsPerWarp, kVec]: col dimension.
        //   P[0]=warp_id -> H[X=1][0]=kWarps (rh_major=2, rh_minor=0).
        //   P[1]=lane_id outer -> H[X=1][1]=kColsPerWarp (rh_major=2, rh_minor=1).
        //   P[1]=lane_id inner -> H[X=0][0]=kRowsPerGroup (rh_major=1, rh_minor=0).
        //   Y[0] -> H[X=1][2]=kVec (rh_major=2, rh_minor=2).
        constexpr auto k_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kRowsPerGroup>, sequence<kWarps, kColsPerWarp, kVec>>,
                tuple<sequence<2>, sequence<2, 1>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2>,
                sequence<2>>{});

        // ----------------------------------------------------------------
        // Global K view: [seqlen_k, kCols] with strides [stride_k, 1].
        // Vectorized along the last dimension (kCols, stride=1), kVec per load.
        const auto k_view = make_naive_tensor_view<address_space_enum::global>(
            k_head,
            make_tuple(kargs.seqlen_k, number<kCols>{}),
            make_tuple(kargs.stride_k, number<1>{}),
            number<kVec>{},
            number<1>{});

        // ----------------------------------------------------------------
        // Accumulator: float column sums, initially zero.
        auto acc = make_static_distributed_tensor<float>(acc_dstr);
        clear_tile(acc);

        // ----------------------------------------------------------------
        // Main loop: process full kRowsPerGroup-row chunks from K.
        // Each iteration loads [kRowsPerGroup, kCols] and accumulates.
        auto k_window = make_tile_window(
            k_view,
            make_tuple(number<kRowsPerGroup>{}, number<kCols>{}),
            {0, 0},
            k_dstr);

        const index_t num_full = kargs.seqlen_k / kRowsPerGroup;

        for(index_t chunk = 0; chunk < num_full; chunk++)
        {
            auto k_tile = load_tile(k_window);

            // NOTE: acc_dstr and k_dstr have different structures (acc has R=kRowsPerGroup
            // replication; k_tile has a 2D P partition). tile_elementwise_inout operates on
            // raw thread buffers element-by-element, so it requires that both tensors have the
            // same thread_buffer_size AND that thread-buffer slot i maps to the same logical
            // column (col_grp*kVec + i) in both distributions. This holds here because:
            //   acc_dstr:  slot i = Y[1] = i-th element within col_grp's kVec group
            //   k_dstr:    slot i = same mapping (kVec inner dim of X[1] col groups)
            // If either distribution is changed, this must be verified again.
            tile_elementwise_inout(
                [](float& a, const InputT& b) { a += type_convert<float>(b); }, acc, k_tile);

            move_tile_window(k_window, {kRowsPerGroup, 0});
        }

        // ----------------------------------------------------------------
        // Tail rows: seqlen_k % kRowsPerGroup rows remain.
        // The k_dstr assigns row_grp = lane_id % kRowsPerGroup within each wavefront.
        // For tail rows [num_full*kRowsPerGroup .. seqlen_k-1], only the thread
        // whose row_grp equals the tail-row index r should accumulate, so we use a
        // scalar conditional add over the kVec elements this thread owns.
        //
        // Thread-to-column mapping mirrors k_dstr:
        //   warp_id     = get_warp_id()                   (0..kWarps-1)
        //   lane_id     = get_lane_id()                   (0..63)
        //   col_grp_l   = lane_id / kRowsPerGroup         column group within warp
        //   row_grp     = lane_id % kRowsPerGroup         row index within group
        //   col_grp     = warp_id * kColsPerWarp + col_grp_l
        //   col_base    = col_grp * kVec                  first column owned by this thread
        //
        // Accumulator thread buffer slot j corresponds to column (col_base + j).
        const index_t tail     = kargs.seqlen_k % kRowsPerGroup;
        const index_t lane_id  = get_lane_id();
        const index_t warp_id  = get_warp_id();
        const index_t row_grp  = lane_id % kRowsPerGroup;
        const index_t col_grp_l = lane_id / kRowsPerGroup;
        const index_t col_grp  = warp_id * kColsPerWarp + col_grp_l;
        const index_t col_base = col_grp * kVec;

        if(tail > 0)
        {
            for(index_t r = 0; r < tail; r++)
            {
                if(row_grp == r)
                {
                    const index_t abs_row = num_full * kRowsPerGroup + r;
                    const InputT* row_ptr = k_head + abs_row * kargs.stride_k + col_base;
                    static_for<0, kVec, 1>{}([&](auto j) {
                        acc.get_thread_buffer()(j) += type_convert<float>(row_ptr[j.value]);
                    });
                }
            }
        }

        // ----------------------------------------------------------------
        // Cross-thread reduction: XOR butterfly within each col_grp's lane group.
        // Each col_grp has kRowsPerGroup consecutive lanes within one wavefront:
        //   col_grp_l = lane_id / kRowsPerGroup  (0..kColsPerWarp-1 per wavefront)
        //   row_grp   = lane_id % kRowsPerGroup  (fast, 0..kRowsPerGroup-1)
        // XOR masks 1..kRowsPerGroup/2 stay within the aligned kRowsPerGroup-wide
        // group. After the butterfly, all threads in a col_grp hold the total sum.
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };
        block_tile_reduce_xor_sync(acc, f_sum);

        // ----------------------------------------------------------------
        // Compute mean (divide by seqlen_k) then store via store_tile.
        // After block_tile_reduce_xor_sync all kRowsPerGroup threads in each
        // col_grp hold the same reduced sum, so all threads write the same value
        // to the same memory locations -- redundant but correct (idempotent writes).
        tile_elementwise_inout(
            [scale = 1.0f / static_cast<float>(kargs.seqlen_k)](float& a) { a *= scale; }, acc);

        InputT* k_mean_out = kargs.k_mean_ptr + batch_idx * kargs.batch_stride_kmean +
                             head_idx * kargs.nhead_stride_kmean;

        // Write InputT k_mean using scalar indexed stores.
        // Each thread owns kVec consecutive columns starting at col_base.
        // acc thread-buffer slot j corresponds to column (col_base + j).
        // Scalar stores avoid make_naive_tensor_view<global, constant<kCols>> which
        // would trigger buffer_view<global, T, constant<kCols>> and the constexpr
        // aggregate-initializer excess-elements error on the divide-by-PackedSize path.
        static_for<0, kVec, 1>{}([&](auto j) {
            const float val          = acc.get_thread_buffer()[j];
            k_mean_out[col_base + j] = type_convert<InputT>(val);
        });
    }
};

// ============================================================================
// SageAttnV3VPreprocessKernel
//
// Quantises V (transposed layout) using an LDS-based 2-D tile transpose with
// a coalesced write-back path.
//
// Grid:      (seqlen_k / (kVGroup * kVGroupsPerBlock), hdim / kVHdimTile, batch * nhead)
// BlockSize: kVGroup * kVGroupsPerBlock = 128 threads = 2 WAVE64 wavefronts
//
// Four-phase pipeline (looped single-group smem_v for high occupancy):
//   Loop over grp_iter = 0 .. kVGroupsPerBlock - 1:
//     Phase 1: Load ONE group's V tile [kVGroup, kVHdimTile] from global -> smem_v
//              (smem_v reused each iteration; only 1 group's worth of LDS needed).
//     Phase 2: Quantize from smem_v -> stage into smem_fp4[grp_iter][...] /
//              smem_scale[grp_iter][...].  No sync needed after stage since each
//              thread writes its own exclusive slot.
//   After loop:
//   Phase 3: barrier (all groups staged).
//   Phase 4: Write-out with thread remapping (coalesced):
//            tid -> (write_d_idx = tid/kVGroupsPerBlock, write_g_idx = tid%kVGroupsPerBlock)
//            Consecutive threads share d_idx, differ in g_idx -> 16-byte stride writes.
//
// LDS layout (bytes, kVGroupsPerBlock=4, kVHdimTile=32):
//   smem_v:     1 * 32 * 33 * 4 = 4224  (float32, 1 group at a time, +1 pad/row)
//   smem_fp4:  32 *  4 * 16     = 2048  (all groups staged)
//   smem_scale: 32 *  4          =  128  (all groups staged)
//   Total: 6400 bytes  (vs. 19072 previously)
//   gfx950 LDS = 160 KiB/CU -> LDS allows floor(163840/6400) = 25 CTAs/CU.
//   Wavefront limit = 32 wavefronts / 2 per CTA = 16 CTAs/CU (binding constraint).
//   -> 16 CTAs per CU (wavefront-limited) vs. 3 previously => 5.3x occupancy improvement.
//
// smem_v bank conflict analysis (gfx950, 32 banks x 4 bytes):
//   Row stride = kVHdimTile + 1 = 33 floats.
//   Column access (fixed d, j=0..31): bank = (j*33+d) % 32 = (j+d) % 32
//   -> 32 distinct banks, zero conflicts.
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

    static_assert(kBlockSize == kVHdimTile * kVGroupsPerBlock,
                  "kBlockSize (kVGroup*kVGroupsPerBlock) must equal kVHdimTile*kVGroupsPerBlock "
                  "for Phase 3 write-out coverage; requires kVGroup == kVHdimTile");

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    // LDS: smem_v (1 group at a time, reused) + smem_fp4/scale staging (all groups).
    //   smem_v:     1 * kVGroup * (kVHdimTile + kLDSPad) * 4 bytes  [reused per iteration]
    //   smem_fp4:   kVHdimTile * kVGroupsPerBlock * (kVGroup / 2) bytes  [all groups]
    //   smem_scale: kVHdimTile * kVGroupsPerBlock bytes                   [all groups]
    static constexpr index_t kSmemVBytes =
        kVGroup * (kVHdimTile + kLDSPad) * static_cast<index_t>(sizeof(float));
    static constexpr index_t kSmemFp4Bytes  = kVHdimTile * kVGroupsPerBlock * (kVGroup / 2);
    static constexpr index_t kSmemScaleBytes = kVHdimTile * kVGroupsPerBlock;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kSmemVBytes + kSmemFp4Bytes + kSmemScaleBytes;
    }

    // -------------------------------------------------------------------------
    // Phase 1 tile distribution: load V [kVGroup, kVHdimTile] from global -> LDS.
    //
    // kBlockSize = kVGroup * kVGroupsPerBlock = 128 threads = 2 warps.
    // kVec = 8 elements per thread (vectorized load).
    // kDPacks = kVHdimTile / kVec = 4 (D-column packs per row).
    // kVWarps = kBlockSize / 64 = 2.
    // kDPacksPerWarp = kDPacks / kVWarps = 2.
    //
    // Thread layout:
    //   load_row    = tid % kVGroup  = lane_id % kVGroup             (0..31, FAST axis)
    //   load_d_pack = tid / kVGroup  = warp_id*kDPacksPerWarp + lane_id/kVGroup (SLOW axis)
    //
    // Distribution encoding [kVGroup=32, kVHdimTile=32]:
    //   H[X=0] = [kVGroup]                        (row dimension, single factor)
    //   H[X=1] = [kVWarps, kDPacksPerWarp, kVec]  (col dimension, 3 factors)
    //
    //   P[0]=warp_id  -> H[X=1][0]=kVWarps (rh_major=2, rh_minor=0)
    //   P[1]=lane_id  -> H[X=1][1]=kDPacksPerWarp (rh_major=2, rh_minor=1, OUTER/slow)
    //                 -> H[X=0][0]=kVGroup (rh_major=1, rh_minor=0, INNER/fast)
    //     lane_id = d_pack_local * kVGroup + load_row
    //   Y[0]          -> H[X=1][2]=kVec (rh_major=2, rh_minor=2)
    //
    // Ps2RHssMajor = tuple<sequence<2>, sequence<2, 1>>
    //   P[0] major = {2}    (one factor: H[X=1][0]=kVWarps)
    //   P[1] major = {2, 1} (two factors, listed outer-to-inner:
    //                         H[X=1][1]=kDPacksPerWarp slow, H[X=0][0]=kVGroup fast)
    // Ps2RHssMinor = tuple<sequence<0>, sequence<1, 0>>
    //   P[0] minor = {0}
    //   P[1] minor = {1, 0}  (H[X=1] minor=1 = kDPacksPerWarp slow;
    //                          H[X=0] minor=0 = kVGroup fast)
    // Ys2RHsMajor = sequence<2>, Ys2RHsMinor = sequence<2>
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLoadDstr()
    {
        constexpr index_t kVec           = 8;
        constexpr index_t kDPacks        = kVHdimTile / kVec;
        constexpr index_t kVWarps        = kBlockSize / 64;
        constexpr index_t kDPacksPerWarp = kDPacks / kVWarps;

        static_assert(kVHdimTile % kVec == 0, "kVHdimTile must be divisible by kVec=8");
        static_assert(kDPacks % kVWarps == 0, "kDPacks must be divisible by kVWarps");
        static_assert(kBlockSize == kVGroup * kDPacks,
                      "kBlockSize must equal kVGroup * kDPacks for full V tile coverage");

        // P[1]=lane_id decomposition (outer-to-inner in listing order):
        //   Outer (slow): H[X=1][1]=kDPacksPerWarp -> major=2, minor=1
        //   Inner (fast): H[X=0][0]=kVGroup        -> major=1, minor=0
        //   lane_id = d_pack_local * kVGroup + load_row
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kVGroup>, sequence<kVWarps, kDPacksPerWarp, kVec>>,
                tuple<sequence<2>, sequence<2, 1>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2>,
                sequence<2>>{});
    }

    // -------------------------------------------------------------------------
    // Phase 4 tile distribution: write-back V hat [kVHdimTile, kVGroupsPerBlock]
    // from LDS smem_fp4 to global v_hat.
    //
    // Each thread copies kHatElemsPerThread = kVGroup/2 = 16 bytes.
    // Thread layout:
    //   write_d_idx = tid / kVGroupsPerBlock = warp_id*kDPerWarp + lane_id/kVGroupsPerBlock
    //   write_g_idx = tid % kVGroupsPerBlock = lane_id % kVGroupsPerBlock
    //
    // Distribution encoding for [kVHdimTile=32, kVGroupsPerBlock=4] with Y=[16]:
    //   kVWarps = kBlockSize / 64 = 2
    //   kDPerWarp = kVHdimTile / kVWarps = 16
    //   kHatElemsPerThread = kVGroup / 2 = 16
    //
    //   H[X=0] = [kVWarps, kDPerWarp]            (d_idx dimension)
    //   H[X=1] = [kVGroupsPerBlock, kHatElemsPerThread]  (g_idx dimension with Y)
    //
    //   P[0]=warp_id  -> H[X=0][0]=kVWarps                    (rh_major=1, rh_minor=0)
    //   P[1]=lane_id  -> H[X=0][1]=kDPerWarp    (slow)        (rh_major=1, rh_minor=1)
    //                 -> H[X=1][0]=kVGroupsPerBlock (fast)    (rh_major=2, rh_minor=0)
    //     lane_id = d_local * kVGroupsPerBlock + write_g_idx
    //   Y[0]          -> H[X=1][1]=kHatElemsPerThread         (rh_major=2, rh_minor=1)
    //
    // Ps2RHssMajor = tuple<sequence<1>, sequence<1, 2>>
    // Ps2RHssMinor = tuple<sequence<0>, sequence<1, 0>>
    // Ys2RHsMajor = sequence<2>, Ys2RHsMinor = sequence<1>
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeHatDstr()
    {
        constexpr index_t kVWarps            = kBlockSize / 64;
        constexpr index_t kDPerWarp          = kVHdimTile / kVWarps;
        constexpr index_t kHatElemsPerThread = kVGroup / 2;

        static_assert(kVHdimTile % kVWarps == 0, "kVHdimTile must be divisible by kVWarps");

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kVWarps, kDPerWarp>,
                      sequence<kVGroupsPerBlock, kHatElemsPerThread>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2>,
                sequence<1>>{});
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t d_idx     = blockIdx.y;
        const index_t bh_idx    = blockIdx.z;
        const index_t head_idx  = bh_idx % kargs.nhead;
        const index_t batch_idx = bh_idx / kargs.nhead;

        // row_local: element index within a V-group (0..kVGroup-1), used by Phase 2.
        // g_abs_base: base V-group index (in units of kVGroup Sk-elements) for this CTA.
        const index_t tid        = get_thread_id();
        const index_t row_local  = tid % kVGroup;
        const index_t g_abs_base = blockIdx.x * kVGroupsPerBlock;

        const index_t col_start = d_idx * kVHdimTile;

        // Smem layout: [smem_v (1 group, reused) | smem_fp4 (all groups) | smem_scale (all)]
        // smem_v:     kVGroup * (kVHdimTile + kLDSPad) * 4 = 4224 bytes
        // smem_fp4:   kVHdimTile * kVGroupsPerBlock * (kVGroup/2) = 2048 bytes
        // smem_scale: kVHdimTile * kVGroupsPerBlock = 128 bytes
        // Total: 6400 bytes -> 16 CTAs/CU (wavefront-limited) vs 3 CTAs/CU (19072 bytes) before.
        __shared__ char smem_raw[GetSmemSize()];
        auto* smem_v =
            reinterpret_cast<float (*)[kVGroup][kVHdimTile + kLDSPad]>(smem_raw);
        uint8_t* smem_fp4   = reinterpret_cast<uint8_t*>(smem_raw + kSmemVBytes);
        uint8_t* smem_scale = smem_fp4 + kSmemFp4Bytes;

        const InputT* v_base =
            kargs.v_ptr + batch_idx * kargs.batch_stride_v + head_idx * kargs.nhead_stride_v;

        constexpr float rcp_dst_max  = 1.0f / 6.0f;
        constexpr index_t kVec       = 8;
        static_assert(kVHdimTile % kVec == 0, "kVHdimTile must be divisible by 8");

        // Phase 1 distribution for [kVGroup, kVHdimTile] V tile load.
        // Thread layout (matches existing load_row / load_d_pack):
        //   load_row    = tid % kVGroup  = lane_id % kVGroup  (row in Sk, fast)
        //   load_d_pack = tid / kVGroup  = warp_id*kDPacksPerWarp + lane_id/kVGroup
        constexpr auto v_load_dstr = MakeVLoadDstr();

        // Phase 4 distribution for hat write-back [kVHdimTile, kVGroupsPerBlock*kHatElems].
        constexpr index_t kHatElemsPerThread = kVGroup / 2;
        constexpr auto hat_dstr              = MakeHatDstr();

        // Loop over groups: smem_v (4224 bytes) reused each iteration.
        // => 10 CTAs/CU vs. 3 previously (3.3x more occupancy).
        for(index_t grp_iter = 0; grp_iter < kVGroupsPerBlock; grp_iter++)
        {
            // ---- Phase 1: All 128 threads load one group's V tile -> smem_v ----
            // Uses load_tile with v_load_dstr for [kVGroup, kVHdimTile] tile.
            // Each thread loads kVec=8 InputT elements, converts to float, stores to LDS.
            //
            // Global V view for this group: [kVGroup, kVHdimTile] InputT
            // Row stride = kargs.hdim (full head dimension).
            const index_t sk_base = (g_abs_base + grp_iter) * kVGroup;
            const InputT* v_grp_ptr = v_base + sk_base * kargs.hdim + col_start;

            const auto v_global_view = make_naive_tensor_view<address_space_enum::global>(
                v_grp_ptr,
                make_tuple(number<kVGroup>{}, number<kVHdimTile>{}),
                make_tuple(kargs.hdim, number<1>{}),
                number<kVec>{},
                number<1>{});
            auto v_global_win = make_tile_window(
                v_global_view,
                make_tuple(number<kVGroup>{}, number<kVHdimTile>{}),
                {0, 0},
                v_load_dstr);

            // Load InputT tile from global.
            const auto v_tile = load_tile(v_global_win);

            // Convert InputT -> float.
            auto v_float_tile = tile_elementwise_in(type_convert<float, InputT>, v_tile);

            // Zero out this thread's elements if its sk_row is out of bounds.
            // load_row = lane_id % kVGroup (same as tid % kVGroup).
            // (All kVec elements for this thread share the same load_row.)
            const index_t load_row = tid % kVGroup;
            if(sk_base + load_row >= kargs.seqlen_k_real)
            {
                tile_elementwise_inout([](float& v) { v = 0.0f; }, v_float_tile);
            }

            // LDS smem_v view: [kVGroup, kVHdimTile] float with padded row stride.
            // Row stride = kVHdimTile + kLDSPad = 33 floats (bank-conflict-free).
            auto smem_v_view = make_naive_tensor_view<address_space_enum::lds>(
                reinterpret_cast<float*>(smem_raw),
                make_tuple(number<kVGroup>{}, number<kVHdimTile>{}),
                make_tuple(number<kVHdimTile + kLDSPad>{}, number<1>{}),
                number<kVec>{},
                number<1>{});
            auto smem_v_win = make_tile_window(
                smem_v_view,
                make_tuple(number<kVGroup>{}, number<kVHdimTile>{}),
                {0, 0},
                v_load_dstr);

            // Store float tile into LDS smem_v.
            store_tile(smem_v_win, v_float_tile);
            block_sync_lds();

            // ---- Phase 2: Quantize smem_v -> stage into smem_fp4 / smem_scale ----
            // row_local iterates over D-columns owned by this thread.
            // kVHdimTile == kVGroup (= 32): loop runs exactly once per thread.
            // Each thread with row_local < kVHdimTile owns one D-column.
            // With kBlockSize > kVHdimTile, threads with row_local >= kVHdimTile idle
            // (only possible when kVGroupsPerBlock > 1 and kVHdimTile < kVGroup; here
            // kVHdimTile == kVGroup so every thread has exactly one d_local to process).
            for(index_t d_local = row_local; d_local < kVHdimTile; d_local += kVGroup)
            {
                float group_data[kVGroup];
                float max_abs = 0.0f;
                for(index_t j = 0; j < kVGroup; j++)
                {
                    group_data[j] = smem_v[0][j][d_local];
                    max_abs       = max(max_abs, abs(group_data[j]));
                }

                const float scale = bit_cast<float>(
                    (bit_cast<uint32_t>(max_abs * rcp_dst_max) +
                     numeric_traits<float>::mant_mask) &
                    numeric_traits<float>::head_mask);

                const index_t stage_idx = d_local * kVGroupsPerBlock + grp_iter;
                smem_scale[stage_idx]   = static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);
                PackFP4Group<kVGroup>(group_data, smem_fp4 + stage_idx * (kVGroup / 2), scale);
            }
            block_sync_lds();
        }

        // ---- Phase 3: barrier (all groups staged) ----
        // (block_sync_lds at end of last grp_iter already serves as this barrier)

        // ---- Phase 4: Coalesced write-back from LDS staging to global ----
        // All staging done.  Remap threads for coalesced writes:
        //   tid -> (write_d_idx = tid/kVGroupsPerBlock, write_g_idx = tid%kVGroupsPerBlock)
        // Consecutive write_g_idx -> consecutive 16-byte hat chunks -> coalesced.
        const index_t write_d_idx    = tid / kVGroupsPerBlock;
        const index_t write_g_abs    = g_abs_base + tid % kVGroupsPerBlock;
        const index_t write_d_global = col_start + write_d_idx;

        uint8_t* v_hat_base = kargs.v_hat_ptr + batch_idx * kargs.batch_stride_v_hat +
                              head_idx * kargs.nhead_stride_v_hat;
        uint8_t* v_scale_base = kargs.v_scale_ptr + batch_idx * kargs.batch_stride_v_scale +
                                head_idx * kargs.nhead_stride_v_scale;

        // Scale write (single byte per thread -- scalar, no benefit from tiling).
        const index_t stage_idx = write_d_idx * kVGroupsPerBlock + (tid % kVGroupsPerBlock);
        v_scale_base[write_d_global * kargs.stride_v_scale + write_g_abs] =
            smem_scale[stage_idx];

        // Hat write: load kHatElemsPerThread=16 uint8 from LDS smem_fp4, store to global.
        // smem_fp4 layout: [kVHdimTile * kVGroupsPerBlock, kHatElemsPerThread] uint8
        //   = [kVHdimTile, kVGroupsPerBlock, kHatElemsPerThread] linearized.
        // Global v_hat layout: [hdim, seqlen_k/kVGroup * kHatElemsPerThread] uint8
        //   stride_v_hat: stride between hdim rows.
        //
        // LDS smem_fp4 view: [kVHdimTile, kVGroupsPerBlock * kHatElemsPerThread] uint8
        // (linearized to 2D for the hat distribution)
        constexpr index_t kHatColsPerRow = kVGroupsPerBlock * kHatElemsPerThread;
        auto fp4_lds_view = make_naive_tensor_view<address_space_enum::lds>(
            smem_fp4,
            make_tuple(number<kVHdimTile>{}, number<kHatColsPerRow>{}),
            make_tuple(number<kHatColsPerRow>{}, number<1>{}),
            number<kHatElemsPerThread>{},
            number<1>{});
        auto fp4_lds_win = make_tile_window(
            fp4_lds_view,
            make_tuple(number<kVHdimTile>{}, number<kHatColsPerRow>{}),
            {0, 0},
            hat_dstr);
        const auto hat_tile = load_tile(fp4_lds_win);

        // Global v_hat view: [kVHdimTile, kVGroupsPerBlock * kHatElemsPerThread] uint8.
        // Advance the pointer to this CTA's tile start so the buffer resource covers
        // exactly the tile -- ensuring all writes are within the AMD buffer bounds.
        // Row stride = kargs.stride_v_hat (runtime, stride between consecutive hdim rows).
        uint8_t* v_hat_tile_ptr = v_hat_base + col_start * kargs.stride_v_hat +
                                  g_abs_base * kHatElemsPerThread;
        auto fp4_global_view = make_naive_tensor_view<address_space_enum::global>(
            v_hat_tile_ptr,
            make_tuple(number<kVHdimTile>{}, number<kHatColsPerRow>{}),
            make_tuple(kargs.stride_v_hat, number<1>{}),
            number<kHatElemsPerThread>{},
            number<1>{});
        auto fp4_global_win = make_tile_window(
            fp4_global_view,
            make_tuple(number<kVHdimTile>{}, number<kHatColsPerRow>{}),
            {0, 0},
            hat_dstr);
        store_tile(fp4_global_win, hat_tile);
    }
};

} // namespace ck_tile
