// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
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
//   2. Thread tile distribution: one thread per (row, MXFP4-group) pair.
//      kBlockSize = kRows * (kCols / kScaleGranularity) threads.
//      Each thread owns one contiguous kG=kScaleGranularity-element group in its row.
//      The Y dimension has kG elements; thread buffer slots 0..kG-1 give compile-time
//      indices for the MXFP4 quantize inner loop.
//
//   3. Global I/O via CK_TILE tile APIs: make_naive_tensor_view + make_tile_window +
//      load_tile / store_tile for all global and LDS memory operations.
//
//   4. RunKSmoothAndQuantize: vectorized global load/store via load_tile / store_tile.
//
// Thread tile distribution for Q/K tiles [kRows, kCols] (MakeQKTileDstr):
//   kGroups = kCols / kScaleGranularity  (number of MXFP4 groups per row)
//   kG = kScaleGranularity = 32          (elements per MXFP4 group; Y dimension)
//   kWarps = kBlockSize / 64             (number of warps per CTA)
//   kRowsPerWarp = 64 / kGroups          (row slots per warp)
//   Thread assignment: tid = warp_id * 64 + lane_id
//     row_idx = warp_id * kRowsPerWarp + lane_id / kGroups
//     grp_idx = lane_id % kGroups
//   Each thread owns Q[row_idx, grp_idx*kG .. (grp_idx+1)*kG - 1] (kG elements in Y)
//
//   tile_distribution_encoding<                         // NDimP=2
//       sequence<>,                                     // no R dims
//       tuple<sequence<kWarps, kRowsPerWarp>,           // H[X=0]: row dim (2 factors)
//             sequence<kGroups, kG>>,                   // H[X=1]: col dim (2 factors)
//       tuple<sequence<1>, sequence<1, 2>>,             // P[0]=warp_id->H[X=0][0];
//                                                       // P[1]=lane_id->H[X=0][1]+H[X=1][0]
//       tuple<sequence<0>, sequence<1, 0>>,
//       sequence<2>,                                    // Y[0] -> H[X=1][1]=kG (registers/thread)
//       sequence<1>>
//
// Thread tile distribution for column-mean reduction (MakeMeanReduceTileDstr):
//   1D tile of shape [kCols].  Each column is handled by kThreadsPerCol threads
//   all within the SAME warp, enabling pure warp-shuffle reduction (no cross-warp smem).
//   kColsPerWarp = kCols / kWarps  (columns per warp)
//   Thread assignment: warp_id = tid / 64,  lane_id = tid % 64
//     col_idx = warp_id * kColsPerWarp + lane_id / kThreadsPerCol
//     r_id    = lane_id % kThreadsPerCol   (row group; 0 = leader)
//   Invariant: kColsPerWarp * kThreadsPerCol == 64  (always holds for
//              kBlockSize = kRows * kCols / kScaleGranularity).
//   After block_tile_reduce_xor_sync, all kThreadsPerCol threads in a column
//   group hold the column sum; the leader (r_id==0) writes the mean.
//
//   tile_distribution_encoding<
//       sequence<kThreadsPerCol>,              // R[0]: intra-warp replication
//       tuple<sequence<kWarps, kColsPerWarp>>, // H[X=0]: column dim
//       tuple<sequence<1>, sequence<1, 0>>,    // P[0]=warp_id->H[X=0][0];
//                                              // P[1]=lane_id->H[X=0][1](slow)+R[0](fast)
//       tuple<sequence<0>, sequence<1, 0>>,
//       sequence<>,                            // no Y
//       sequence<>>
//
// Smem layout when kUseLdsQ=true (bytes):
//   [0 .. kQTileBytes):               Q tile as InputT (kRows x kCols)
//   [kQTileBytes .. +kSmemMeanBytes): column means as float32 (kCols)
//
// Smem layout when kUseLdsQ=false (bytes):
//   [0 .. kSmemMeanBytes):            column means as float32 (kCols)
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
    static_assert(kBlockSize == kRows * (kCols / kScaleGranularity),
                  "kBlockSize must equal kRows * kGroups for full QK tile coverage");

    // kGroups: MXFP4 groups per row; kG: elements per group (= kScaleGranularity).
    static constexpr index_t kGroups = kCols / kScaleGranularity;
    static constexpr index_t kG      = kScaleGranularity;

    // kThreadsPerCol: threads collaborating on a single column mean (for mean reduction).
    // With MakeMeanReduceTileDstr all kThreadsPerCol threads for a column fall within
    // ONE warp, so block_tile_reduce_xor_sync handles the reduction without smem.
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
    // Reserve headroom for k_mean cache (kCols*4) and mean buffer (kCols*4).
    static constexpr index_t kMaxSmemForLdsQ = 49152; // 48 KB
    static constexpr bool kUseLdsQ           = (kQTileBytes <= kMaxSmemForLdsQ);

    // Slot: column means (float32, kCols).
    // No smem_partial needed: block_tile_reduce_xor_sync uses warp shuffles only.
    static constexpr index_t kSmemMeanWords = kCols;

    // Byte offsets within smem.
    // When kUseLdsQ: Q tile occupies smem[0..kQTileBytes), then mean follows.
    // When !kUseLdsQ: mean starts at 0.
    static constexpr index_t kSmemMeanOffset = kUseLdsQ ? kQTileBytes : 0;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kSmemMeanOffset + kSmemMeanWords * 4;
    }

    // -------------------------------------------------------------------------
    // MakeQKTileDstr: tile distribution for Q/K tiles [kRows, kCols].
    //
    // Thread mapping: tid = warp_id * 64 + lane_id
    //   row_idx = warp_id * kRowsPerWarp + lane_id / kGroups
    //   grp_idx = lane_id % kGroups
    // Thread buffer: kG=32 consecutive elements at (row_idx, grp_idx*kG).
    //
    // NDimP=2: P[0]=warp_id, P[1]=lane_id (required for kBlockSize > 64).
    //
    // H[X=0] = [kWarps, kRowsPerWarp]   (row dimension decomposed)
    // H[X=1] = [kGroups, kG]            (col dimension decomposed)
    //
    // P[0]=warp_id  -> H[X=0][0]=kWarps         (rh_major=1, rh_minor=0)
    // P[1]=lane_id  -> H[X=0][1]=kRowsPerWarp   (rh_major=1, rh_minor=1)
    //               -> H[X=1][0]=kGroups         (rh_major=2, rh_minor=0)
    //               (lane_id = row_in_warp * kGroups + grp_idx, grp is fast axis)
    // Y[0]          -> H[X=1][1]=kG             (rh_major=2, rh_minor=1)
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeQKTileDstr()
    {
        constexpr index_t kWarps       = kBlockSize / 64;
        constexpr index_t kRowsPerWarp = 64 / kGroups;
        static_assert(kBlockSize % 64 == 0, "kBlockSize must be a multiple of 64 (warp size)");
        static_assert(64 % kGroups == 0, "kGroups must divide 64 (warp size)");
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kWarps, kRowsPerWarp>, sequence<kGroups, kG>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<0>, sequence<1, 0>>,
                sequence<2>,
                sequence<1>>{});
    }

    // -------------------------------------------------------------------------
    // MakeMeanReduceTileDstr: 1D tile distribution for column-mean reduction.
    //
    // Assigns columns to warps so that all kThreadsPerCol threads for one column
    // are within the SAME warp. Enables block_tile_reduce_xor_sync without smem.
    //
    // kColsPerWarp = kCols / kWarps = kScaleGranularity * 64 / kBlockSize
    // Invariant: kColsPerWarp * kThreadsPerCol == 64 (warp size). This holds for
    // the standard kBlockSize = kRows * (kCols / kScaleGranularity).
    //
    // Thread assignment: warp_id = tid / 64, lane_id = tid % 64
    //   col_idx = warp_id * kColsPerWarp + lane_id / kThreadsPerCol
    //   r_id    = lane_id % kThreadsPerCol  (0 = leader)
    //
    // H[X=0] = [kWarps, kColsPerWarp, 1]  (size-1 tail factor hosts Y[0])
    // R[0]   = kThreadsPerCol (intra-warp replication; reduces via XOR shuffle)
    // P[0]=warp_id  -> H[X=0][0]=kWarps
    // P[1]=lane_id  -> H[X=0][1]=kColsPerWarp (slower) + R[0]=kThreadsPerCol (faster)
    //   lid = col_in_warp * kThreadsPerCol + r_id
    //   ps_over_rs_derivative_[1][0] = 1  (r_id is the fast axis of lane_id)
    // Y[0]          -> H[X=0][2]=1          (size-1; gives thread_buffer_size=1)
    // -------------------------------------------------------------------------
    CK_TILE_HOST_DEVICE static constexpr auto MakeMeanReduceTileDstr()
    {
        constexpr index_t kWarps_      = kBlockSize / 64;
        constexpr index_t kColsPerWarp = kCols / kWarps_;
        static_assert(kBlockSize % 64 == 0, "kBlockSize must be a multiple of 64");
        static_assert(kCols % kWarps_ == 0, "kCols must be divisible by kWarps");
        // All kThreadsPerCol threads for one column fall within a single warp.
        static_assert(kColsPerWarp * kThreadsPerCol == 64,
                      "kColsPerWarp * kThreadsPerCol must equal warp size (64)");
        // H[X=0] has 3 factors: [kWarps, kColsPerWarp, 1].
        // The size-1 tail factor is assigned to Y[0] so that the thread buffer has 1 slot,
        // which is required by make_static_tile_distribution (NDimY >= 1).
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<kThreadsPerCol>,                   // R[0]: intra-warp replication
                tuple<sequence<kWarps_, kColsPerWarp, 1>>,  // H[X=0]: [kWarps,kColsPerWarp,1]
                tuple<sequence<1>, sequence<1, 0>>,         // P[0]->H[X=0][0]; P[1]->H[X=0][1]+R[0]
                tuple<sequence<0>, sequence<1, 0>>,         // minor indices
                sequence<1>,                                // Y[0] -> H[X=0][2]=1
                sequence<2>>{});                            // Y[0] minor index = 2
    }

    // -------------------------------------------------------------------------
    // Step 0 (kUseLdsQ only): Load Q tile from global into LDS.
    //   Uses make_naive_tensor_view + load_tile / store_tile with the Q tile
    //   distribution. Padded rows (row >= n_rows_valid) are zeroed before storing.
    //   Caller must issue block_sync_lds() before RunQMean reads LDS.
    // -------------------------------------------------------------------------
    CK_TILE_DEVICE void RunLoadQTile(const InputT* __restrict__ src_ptr,
                                     void* smem,
                                     index_t n_rows_valid) const
    {
        static_assert(kUseLdsQ, "RunLoadQTile requires kUseLdsQ=true");
        InputT* smem_q = reinterpret_cast<InputT*>(smem);

        constexpr auto dstr = MakeQKTileDstr();

        // Global view: [kRows, kCols] row-major, vectorized with vec=8 (safe for all types).
        // Use runtime kRows to keep BufferSizeType = index_t, avoiding constant<N> issues.
        const auto src_view = make_naive_tensor_view<address_space_enum::global>(
            src_ptr,
            make_tuple(static_cast<index_t>(kRows), number<kCols>{}),
            make_tuple(number<kCols>{}, number<1>{}),
            number<8>{},
            number<1>{});

        // LDS view: [kRows, kCols] row-major (packed), vectorized with vec=8.
        auto dst_view = make_naive_tensor_view<address_space_enum::lds>(
            smem_q,
            make_tuple(static_cast<index_t>(kRows), number<kCols>{}),
            make_tuple(number<kCols>{}, number<1>{}),
            number<8>{},
            number<1>{});

        auto src_win = make_tile_window(
            src_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        auto dst_win = make_tile_window(
            dst_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);

        // Load Q tile from global into registers.
        auto q_tile = load_tile(src_win);

        // Zero out this thread's elements if its row is out of bounds.
        const index_t tid     = get_thread_id();
        const index_t row_idx = tid / kGroups;
        if(row_idx >= n_rows_valid)
        {
            tile_elementwise_inout([](auto& v) { v = InputT{0}; }, q_tile);
        }

        // Store (padded) Q tile from registers into LDS.
        store_tile(dst_win, q_tile);
    }

    // -------------------------------------------------------------------------
    // Step 1a: Compute column mean from LDS Q tile (kUseLdsQ=true path).
    //   Uses MakeMeanReduceTileDstr: each column is assigned to kThreadsPerCol
    //   threads all within one warp. Accumulates LDS rows in a scalar acc, stores
    //   into a 1D reduce tile, then block_tile_reduce_xor_sync reduces across the
    //   kThreadsPerCol R-dimension via warp XOR butterfly (no smem needed).
    //   Leader (r_id==0) stores mean to smem_mean and q_mean_ptr.
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

        // Column assignment via MakeMeanReduceTileDstr:
        //   col_idx = warp_id * kColsPerWarp + lane_id / kThreadsPerCol
        //   r_id    = lane_id % kThreadsPerCol   (row group; 0 = leader)
        constexpr index_t kWarps_      = kBlockSize / 64;
        constexpr index_t kColsPerWarp = kCols / kWarps_;
        const index_t tid              = get_thread_id();
        const index_t warp_id          = tid / 64;
        const index_t lane_id          = tid % 64;
        const index_t col_idx          = warp_id * kColsPerWarp + lane_id / kThreadsPerCol;
        const index_t r_id             = lane_id % kThreadsPerCol;

        // Each thread accumulates LDS rows r_id, r_id+kThreadsPerCol, ... for col_idx.
        float acc = 0.0f;
        for(index_t r = r_id; r < n_rows_valid; r += kThreadsPerCol)
            acc += static_cast<float>(smem_q[r * kCols + col_idx]);

        // Reduce across kThreadsPerCol within the same warp using XOR butterfly.
        // block_tile_reduce_xor_sync uses warp_shuffle (does_p_own_r_[1][0]=true,
        // lid_over_rid_derivative=1). No cross-warp smem is needed because all
        // kThreadsPerCol threads for col_idx are in the same warp.
        auto acc_tile = make_static_distributed_tensor<float>(MakeMeanReduceTileDstr());
        acc_tile.get_thread_buffer()(number<0>{}) = acc;
        block_tile_reduce_xor_sync(acc_tile, [](float a, float b) { return a + b; });

        // After XOR reduce, all kThreadsPerCol threads hold the column sum.
        // Leader (r_id==0) writes the mean.
        if(r_id == 0)
        {
            const float mean    = acc_tile.get_thread_buffer()[number<0>{}] /
                                  static_cast<float>(n_rows_valid);
            smem_mean[col_idx]  = mean;
            q_mean_ptr[col_idx] = static_cast<InputT>(mean);
        }
    }

    // -------------------------------------------------------------------------
    // Step 1b: Compute column mean from global Q tile (kUseLdsQ=false path).
    //   Reads Q from global memory. Stores mean to smem_mean and q_mean_ptr.
    //   Uses the same MakeMeanReduceTileDstr + block_tile_reduce_xor_sync approach.
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

        constexpr index_t kWarps_      = kBlockSize / 64;
        constexpr index_t kColsPerWarp = kCols / kWarps_;
        const index_t tid              = get_thread_id();
        const index_t warp_id          = tid / 64;
        const index_t lane_id          = tid % 64;
        const index_t col_idx          = warp_id * kColsPerWarp + lane_id / kThreadsPerCol;
        const index_t r_id             = lane_id % kThreadsPerCol;

        float acc = 0.0f;
        for(index_t r = r_id; r < n_rows_valid; r += kThreadsPerCol)
            acc += static_cast<float>(q_ptr[r * kCols + col_idx]);

        auto acc_tile = make_static_distributed_tensor<float>(MakeMeanReduceTileDstr());
        acc_tile.get_thread_buffer()(number<0>{}) = acc;
        block_tile_reduce_xor_sync(acc_tile, [](float a, float b) { return a + b; });

        if(r_id == 0)
        {
            const float mean    = acc_tile.get_thread_buffer()[number<0>{}] /
                                  static_cast<float>(n_rows_valid);
            smem_mean[col_idx]  = mean;
            q_mean_ptr[col_idx] = static_cast<InputT>(mean);
        }
    }

    // -------------------------------------------------------------------------
    // Step 2a: Quantize Q from LDS (kUseLdsQ=true path).
    //   Uses load_tile with the Q tile distribution (one thread per (row, group)).
    //   Thread buffer slot j = Q element at column (d_start + j).
    //   Mean is read from smem_mean[d_start + j] using compile-time j.
    //   Produces MXFP4 hat and scale outputs.
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
        constexpr index_t kNumGroups = kGroups;
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

        // Load this thread's kG elements from LDS via a tile window with the Q/K distribution.
        // Each thread gets kG elements (Y dimension) at (row_idx, grp_idx*kG).
        // Use vector size 8 (safe for all InputT: 8*sizeof(InputT) <= 16 bytes).
        constexpr auto dstr = MakeQKTileDstr();
        const auto q_lds_view = make_naive_tensor_view<address_space_enum::lds>(
            smem_q,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(number<kCols>{}, number<1>{}),
            number<8>{},
            number<1>{});
        auto q_lds_win = make_tile_window(
            q_lds_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        const auto q_tile = load_tile(q_lds_win);

        // Iterate over the kG thread-buffer slots via sweep_tile_span on the col span
        // (compile-time index j in 0..kG-1). For MakeQKTileDstr, X=1 carries the Y[0]
        // dimension (kG elements); each j_idx = tile_distributed_index<j>, and j is
        // extracted as a compile-time number<j> via impl_.at(number<0>{}).
        constexpr auto q_spans = remove_cvref_t<decltype(q_tile)>::get_distributed_spans();
        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        sweep_tile_span(q_spans[number<1>{}], [&](auto j_idx) {
            constexpr auto j     = decltype(j_idx)::impl_.at(number<0>{}); // number<j>
            const float mean_val = smem_mean[d_start + j];
            const float val      = static_cast<float>(q_tile.get_thread_buffer()[j]) - mean_val;
            group_data[j]        = val;
            max_abs              = max(max_abs, abs(val));
        });

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
    //   Q data loaded from global using load_tile; mean read from smem_mean.
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
        constexpr index_t kNumGroups = kGroups;
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

        // Load this thread's kG elements from global using load_tile with Q/K distribution.
        // Use vector size 8 (safe for all InputT: 8*sizeof(InputT) <= 16 bytes for fp16).
        // Use runtime row stride to avoid buffer_view<constant<N>> instantiation issues.
        constexpr auto dstr = MakeQKTileDstr();
        const auto q_global_view = make_naive_tensor_view<address_space_enum::global>(
            q_ptr,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(static_cast<index_t>(kCols), number<1>{}),
            number<8>{},
            number<1>{});
        auto q_global_win = make_tile_window(
            q_global_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        const auto q_tile = load_tile(q_global_win);

        constexpr auto qg_spans = remove_cvref_t<decltype(q_tile)>::get_distributed_spans();
        float group_data[kScaleGranularity];
        float max_abs = 0.0f;

        sweep_tile_span(qg_spans[number<1>{}], [&](auto j_idx) {
            constexpr auto j     = decltype(j_idx)::impl_.at(number<0>{});
            const float mean_val = smem_mean[d_start + j];
            const float val      = static_cast<float>(q_tile.get_thread_buffer()[j]) - mean_val;
            group_data[j]        = val;
            max_abs              = max(max_abs, abs(val));
        });

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
    // Step 3: K' = K_tile - k_mean -> k_prime_ptr, then quantize K' -> MXFP4.
    //   k_mean is cached into smem[0..kCols*4] (reuses Q tile LDS space or
    //   the beginning of smem when kUseLdsQ=false).
    //   Uses load_tile for K from global; sweep_tile over thread buffer for compute;
    //   store_tile for K' to global.
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
        // All kBlockSize threads cooperatively load k_mean[0..kCols) into smem as float32.
        // Each thread strides over kCols with step kBlockSize (scalar, bounds-safe).
        float* smem_f     = reinterpret_cast<float*>(smem);
        const index_t tid = get_thread_id();
        for(index_t d = tid; d < kCols; d += kBlockSize)
            smem_f[d] = static_cast<float>(k_mean_ptr[d]);
        block_sync_lds();

        // Thread layout for K quantize: tid = row_idx * kGroups + grp_idx.
        constexpr index_t kNumGroups = kGroups;
        const index_t row_idx        = tid / kNumGroups;
        const index_t grp_idx        = tid % kNumGroups;
        const index_t d_start        = grp_idx * kScaleGranularity;

        if(row_idx >= n_rows_valid)
        {
            // Zero K' for out-of-bounds rows and clear scale/hat.
            InputT* dst_row = k_prime_ptr + row_idx * k_prime_stride + d_start;
            for(index_t j = 0; j < kScaleGranularity; j++)
                dst_row[j] = InputT{0};
            dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
            uint8_t* hat_dst =
                dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2);
            for(index_t j = 0; j < kScaleGranularity / 2; j++)
                hat_dst[j] = 0;
            return;
        }

        constexpr float rcp_dst_max = 1.0f / 6.0f;

        // Load K tile for this thread's (row, group) from global using load_tile.
        // Use vector size 8 (safe for all InputT: 8*sizeof(float) = 32 bytes <= AMD max).
        // Use runtime row stride to avoid buffer_view<constant<N>> issues.
        constexpr auto dstr = MakeQKTileDstr();
        const auto k_src_view = make_naive_tensor_view<address_space_enum::global>(
            src_ptr,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(static_cast<index_t>(kCols), number<1>{}),
            number<8>{},
            number<1>{});
        auto k_src_win = make_tile_window(
            k_src_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        const auto k_tile = load_tile(k_src_win);

        // Compute K' = K - k_mean using sweep_tile_span over the kG thread-buffer slots.
        constexpr auto k_spans = remove_cvref_t<decltype(k_tile)>::get_distributed_spans();
        float group_data[kScaleGranularity];
        float max_abs = 0.0f;
        auto kprime_tile = make_static_distributed_tensor<InputT>(dstr);

        sweep_tile_span(k_spans[number<1>{}], [&](auto j_idx) {
            constexpr auto j                   = decltype(j_idx)::impl_.at(number<0>{});
            const float k_val                  = static_cast<float>(k_tile.get_thread_buffer()[j]);
            const float centered               = k_val - smem_f[d_start + j];
            group_data[j]                      = centered;
            max_abs                            = max(max_abs, abs(centered));
            kprime_tile.get_thread_buffer()(j) = static_cast<InputT>(centered);
        });

        // Store K' to global via store_tile.
        // k_prime has a potentially non-unit row stride (k_prime_stride).
        // Use vector size 8 to stay within AMD buffer instruction limits for float.
        const auto k_dst_view = make_naive_tensor_view<address_space_enum::global>(
            k_prime_ptr,
            make_tuple(number<kRows>{}, number<kCols>{}),
            make_tuple(k_prime_stride, number<1>{}),
            number<8>{},
            number<1>{});
        auto k_dst_win = make_tile_window(
            k_dst_view, make_tuple(number<kRows>{}, number<kCols>{}), {0, 0}, dstr);
        store_tile(k_dst_win, kprime_tile);

        // MXFP4 quantize K'.
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

};

} // namespace ck_tile
