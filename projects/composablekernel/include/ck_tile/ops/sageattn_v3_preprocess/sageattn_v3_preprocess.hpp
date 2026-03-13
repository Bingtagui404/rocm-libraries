// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/sageattn_v3_preprocess/pipeline/sageattn_v3_preprocess_pipeline.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/kernel/sageattn_v3_preprocess_kernel.hpp"

#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

// User-facing args for SageAttnV3Preprocess::run().
// Fill only input strides and output pointers; all output strides and
// padded dimensions are derived internally from (seqlen_q, seqlen_k, hdim).
template <typename InputT>
struct SageAttnV3PreprocessArgs
{
    // --- Q: [batch, nhead, seqlen_q, hdim] InputT ---
    const InputT* q_ptr;
    index_t seqlen_q;
    index_t hdim;
    index_t stride_q;         // = hdim (row-major Q)
    index_t nhead_stride_q;   // = seqlen_q * hdim
    index_t batch_stride_q;   // = nhead * seqlen_q * hdim

    // Q outputs (pointers only; strides derived from padded dims)
    uint8_t* q_hat_ptr;   // [batch, nhead, seqlen_q_padded, hdim/2] uint8
    uint8_t* q_scale_ptr; // [batch, nhead, seqlen_q_padded, hdim/32] uint8
    InputT*  q_mean_ptr;  // [batch, nhead, num_q_tiles, hdim] InputT

    // --- K: [batch, nhead, seqlen_k, hdim] InputT ---
    const InputT* k_ptr;
    index_t seqlen_k;
    index_t stride_k;         // = hdim (row-major K)
    index_t nhead_stride_k;   // = seqlen_k * hdim
    index_t batch_stride_k;   // = nhead * seqlen_k * hdim

    // K outputs (pointers only)
    uint8_t* k_hat_ptr;   // [batch, nhead, seqlen_k_padded, hdim/2] uint8
    uint8_t* k_scale_ptr; // [batch, nhead, seqlen_k_padded, hdim/32] uint8

    // --- V: [batch, nhead, seqlen_k, hdim] InputT ---
    const InputT* v_ptr;
    index_t nhead_stride_v;  // = seqlen_k * hdim
    index_t batch_stride_v;  // = nhead * seqlen_k * hdim

    // V outputs (pointers only; transposed layout [batch, nhead, hdim, seqlen_k_padded/...])
    uint8_t* v_hat_ptr;   // [batch, nhead, hdim, seqlen_k_padded/2] uint8
    uint8_t* v_scale_ptr; // [batch, nhead, hdim, seqlen_k_padded/32] uint8

    // --- Common dimensions ---
    index_t batch;
    index_t nhead;
};

// Stage bitmask for sageattn_v3_preprocess_run.
// Each bit enables one kernel launch; default (kSA3StageAll) runs all four.
static constexpr uint32_t kSA3StageKMean       = 1u << 0; // Launch 0: k_mean kernel
static constexpr uint32_t kSA3StagePreprocess  = 1u << 1; // Launch 1: Q/K preprocess
static constexpr uint32_t kSA3StageVPreprocess = 1u << 2; // Launch 1b: V preprocess
static constexpr uint32_t kSA3StageDeltaS      = 1u << 3; // Launch 2: delta_s GEMM
static constexpr uint32_t kSA3StageAll         = 0xFu;    // All stages

// ============================================================================
// sageattn_v3_preprocess_run
//
// Host-side wrapper that launches the four kernels required for SA3
// preprocessing on a single HIP stream:
//
//   Launch 0: SageAttnV3KMeanKernel
//     Computes k_mean[b, h, d] = mean_n(K[b,h,n,d]), stored as InputT.
//     Grid: (1, nhead, batch) -- single-pass, no atomics.
//
//   Launch 1: SageAttnV3PreprocessKernel
//     Q: mean → quantize (MXFP4, stores q_mean as InputT)
//     K: smooth (K' = K - k_mean) → k_prime_buf; quantize K' → MXFP4
//     Grid: (max(num_q_tiles, num_k_tiles), nhead, batch).
//
//   Launch 1b: SageAttnV3VPreprocessKernel
//     V: LDS-based 2-D tile transpose + quantize → MXFP4.
//     Grid: (seqlen_k/32, hdim/32, batch*nhead), BlockSize: 32.
//     Each CTA loads a [32, 32] tile of V coalesced into float32 LDS with
//     1 float32 padding per row (bank-conflict-free for ds_read_b32 on
//     gfx950 where effective bank count = 32), then quantizes column-wise.
//
//   Launch 2: BatchedGemmKernel
//     delta_s = q_mean @ K'^T
//     A: q_mean  [batch*nhead, num_q_tiles, hdim] InputT  RowMajor
//     B: K'      [batch*nhead, seqlen_k,    hdim] InputT  ColMajor (= K'^T)
//     C: delta_s [batch*nhead, num_q_tiles, seqlen_k] float RowMajor
//
// Template parameters:
//   InputT:  fp16_t or bf16_t (matches Q / K / V element type)
//   kRows:   tile rows for Q and K (kM0 == kN0)
//   kCols:   hdim
//
// delta_s strides are derived from M=num_q_tiles, N=seqlen_k (no stride params needed).
//
// Caller-allocated device buffers (all on the same device as Q/K/V):
//   k_mean_buf:  InputT [batch, nhead, hdim]
//   k_prime_buf: InputT [batch, nhead, seqlen_k, hdim]
// ============================================================================

// Buffer size descriptor returned by get_buffer_sizes.
// All *_bytes fields are byte counts for device allocation.
struct SageAttnV3PreprocessBufferSizes
{
    // ---- outputs produced by sageattn_v3_preprocess_run ----
    size_t delta_s_bytes; // float  [B, H, num_q_tiles, seqlen_k_padded]
    size_t k_prime_bytes; // InputT [B, H, seqlen_k_padded, hdim]
    size_t q_hat_bytes;   // uint8  [B, H, seqlen_q_padded, hdim/2]
    size_t q_scale_bytes; // uint8  [B, H, seqlen_q_padded, hdim/32]
    size_t q_mean_bytes;  // InputT [B, H, num_q_tiles, hdim]
    size_t k_hat_bytes;   // uint8  [B, H, seqlen_k_padded, hdim/2]
    size_t k_scale_bytes; // uint8  [B, H, seqlen_k_padded, hdim/32]
    size_t v_hat_bytes;   // uint8  [B, H, hdim, seqlen_k_padded/2]
    size_t v_scale_bytes; // uint8  [B, H, hdim, seqlen_k_padded/32]
    // ---- caller-allocated scratch (unchanged by padding) ----
    size_t k_mean_bytes; // InputT [B, H, hdim]
    // ---- padded dims ----
    index_t seqlen_q_padded;
    index_t seqlen_k_padded;
    index_t num_q_tiles;
    index_t num_k_tiles;
};

// GEMM tile configuration for delta_s = q_mean @ K'^T.
//   M = num_q_tiles (small, typically 1–64)
//   N = seqlen_k    (large)
//   K = hdim        (128 or 256)
//
// BlockTile<32,64,32>, BlockWarps<1,2,1>: MWarp=1, NWarp=2 → 128 threads.
// WarpTile K dimension differs by input type:
//   fp16/bf16: WarpTile<32,32,16> → mfma_f32_32x32x8f16 × 2 iters  (kK=16)
//   float:     WarpTile<32,32, 8> → mfma_f32_32x32x2f32 × 4 iters  (kK= 8)
// kPadM=true to handle num_q_tiles not divisible by 32.
template <typename InputT>
using DeltaSV3GemmShape = std::conditional_t<
    std::is_same_v<InputT, float>,
    TileGemmShape<sequence<32, 64, 32>, sequence<1, 2, 1>, sequence<32, 32, 8>>,
    TileGemmShape<sequence<32, 64, 32>, sequence<1, 2, 1>, sequence<32, 32, 16>>>;

// ============================================================================
// SageAttnV3Preprocess<InputT, kRows, kCols>
//
// Thin struct wrapper around get_buffer_sizes() and sageattn_v3_preprocess_run()
// that groups the two free functions under a single named type, preventing callers
// from accidentally mixing buffer sizes from one template instantiation with a
// run() call from another.
//
// Example usage:
//   using SA3 = ck_tile::SageAttnV3Preprocess<fp16_t, 128, 128>;
//   auto bsz  = SA3::get_buffer_sizes(batch, nhead, seqlen_q, seqlen_k, hdim);
//   SA3::run(prep_args, delta_s_ptr, k_mean_buf, k_prime_buf, stream);
// ============================================================================
template <typename InputT, index_t kRows, index_t kCols>
struct SageAttnV3Preprocess
{
    using Args        = SageAttnV3PreprocessArgs<InputT>;
    using BufferSizes = SageAttnV3PreprocessBufferSizes;

    static BufferSizes
    get_buffer_sizes(index_t batch, index_t nhead, index_t seqlen_q, index_t seqlen_k, index_t hdim)
    {
        constexpr index_t kG = 32; // MXFP4 scale granularity

        // V preprocess kernel requires seqlen_k_padded divisible by kVGroup * kVGroupsPerBlock.
        // With kVGroup=32 and kVGroupsPerBlock=4, that is 128. Take the larger of kRows and 128.
        constexpr index_t kVPad = 128; // = kVGroup (32) * kVGroupsPerBlock (4)
        constexpr index_t kQPad = kRows;
        constexpr index_t kKPad = (kVPad > kRows) ? kVPad : kRows; // lcm when both are powers of 2

        const index_t sq_pad = ((seqlen_q + kQPad - 1) / kQPad) * kQPad;
        const index_t sk_pad = ((seqlen_k + kKPad - 1) / kKPad) * kKPad;
        const index_t nqt    = sq_pad / kRows;
        const index_t nkt    = sk_pad / kRows;

        SageAttnV3PreprocessBufferSizes s{};
        s.seqlen_q_padded = sq_pad;
        s.seqlen_k_padded = sk_pad;
        s.num_q_tiles     = nqt;
        s.num_k_tiles     = nkt;

        s.delta_s_bytes = static_cast<size_t>(batch * nhead * nqt * sk_pad) * sizeof(float);
        s.k_prime_bytes = static_cast<size_t>(batch * nhead * sk_pad * hdim) * sizeof(InputT);
        s.q_hat_bytes   = static_cast<size_t>(batch * nhead * sq_pad * (hdim / 2));
        s.q_scale_bytes = static_cast<size_t>(batch * nhead * sq_pad * (hdim / kG));
        s.q_mean_bytes  = static_cast<size_t>(batch * nhead * nqt * hdim) * sizeof(InputT);
        s.k_hat_bytes   = static_cast<size_t>(batch * nhead * sk_pad * (hdim / 2));
        s.k_scale_bytes = static_cast<size_t>(batch * nhead * sk_pad * (hdim / kG));
        s.v_hat_bytes   = static_cast<size_t>(batch * nhead * hdim * (sk_pad / 2));
        s.v_scale_bytes = static_cast<size_t>(batch * nhead * hdim * (sk_pad / kG));
        s.k_mean_bytes = static_cast<size_t>(batch * nhead * hdim) * sizeof(InputT);
        return s;
    }

    static void run( // InputT: fp16_t, bf16_t, or float.
                     // ---- preprocess args ----
        const SageAttnV3PreprocessArgs<InputT>& args,
        // ---- delta_s output ----
        float* delta_s_ptr, // [batch, nhead, num_q_tiles, seqlen_k]
        // ---- caller-allocated scratch / output buffers ----
        InputT* k_mean_buf,  // [batch, nhead, hdim]          InputT
        InputT* k_prime_buf, // [batch, nhead, seqlen_k, hdim] InputT
        // ---- stream ----
        hipStream_t stream,
        // ---- optional: select which kernel stages to launch ----
        uint32_t stages = kSA3StageAll)
    {
        const index_t batch    = args.batch;
        const index_t nhead    = args.nhead;
        const index_t hdim     = args.hdim;
        const index_t seqlen_q = args.seqlen_q;
        const index_t seqlen_k = args.seqlen_k;
        constexpr index_t kG   = 32; // MXFP4 scale granularity

        const auto bsz            = get_buffer_sizes(batch, nhead, seqlen_q, seqlen_k, hdim);
        const index_t sq_pad      = bsz.seqlen_q_padded;
        const index_t sk_pad      = bsz.seqlen_k_padded;
        const index_t num_q_tiles = bsz.num_q_tiles;
        const index_t num_k_tiles = bsz.num_k_tiles;

        // ------------------------------------------------------------------ //
        // Launch 0: k_mean kernel
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageKMean)
        {
            using KMeanKernel = SageAttnV3KMeanKernel<InputT, kRows, kCols>;
            using KMeanKargs  = typename KMeanKernel::Kargs;

            KMeanKargs kargs{};
            kargs.k_ptr              = args.k_ptr;
            kargs.seqlen_k           = seqlen_k;
            kargs.hdim               = hdim;
            kargs.stride_k           = args.stride_k;
            kargs.nhead_stride_k     = args.nhead_stride_k;
            kargs.batch_stride_k     = args.batch_stride_k;
            kargs.k_mean_ptr         = k_mean_buf;
            kargs.nhead_stride_kmean = hdim;
            kargs.batch_stride_kmean = nhead * hdim;
            kargs.nhead              = nhead;
            kargs.batch              = batch;

            const dim3 grids   = KMeanKernel::GridSize(kargs);
            const dim3 blocks  = KMeanKernel::BlockSize();
            const index_t smem = KMeanKernel::GetSmemSize();

            stream_config sc{stream};
            launch_and_check(sc, make_kernel(KMeanKernel{}, grids, blocks, smem, kargs));
        } // end if(stages & kSA3StageKMean)

        // ------------------------------------------------------------------ //
        // Launch 1: preprocess kernel (Q + K')
        // ------------------------------------------------------------------ //
        if(stages & kSA3StagePreprocess)
        {
            using PrepKernel = SageAttnV3PreprocessKernel<InputT, kRows, kCols>;
            using PrepKargs  = typename PrepKernel::Kargs;

            PrepKargs kargs{};
            kargs.q_ptr                = args.q_ptr;
            kargs.seqlen_q             = seqlen_q;
            kargs.hdim                 = hdim;
            kargs.stride_q             = args.stride_q;
            kargs.nhead_stride_q       = args.nhead_stride_q;
            kargs.batch_stride_q       = args.batch_stride_q;
            kargs.q_hat_ptr            = args.q_hat_ptr;
            kargs.stride_q_hat         = hdim / 2;
            kargs.nhead_stride_q_hat   = sq_pad * (hdim / 2);
            kargs.batch_stride_q_hat   = nhead * sq_pad * (hdim / 2);
            kargs.q_scale_ptr          = args.q_scale_ptr;
            kargs.stride_q_scale       = hdim / kG;
            kargs.nhead_stride_q_scale = sq_pad * (hdim / kG);
            kargs.batch_stride_q_scale = nhead * sq_pad * (hdim / kG);
            kargs.q_mean_ptr           = args.q_mean_ptr;
            kargs.q_tile_size          = kRows;
            kargs.stride_q_mean        = hdim;
            kargs.nhead_stride_q_mean  = num_q_tiles * hdim;
            kargs.batch_stride_q_mean  = nhead * num_q_tiles * hdim;
            kargs.k_ptr                = args.k_ptr;
            kargs.seqlen_k             = seqlen_k;
            kargs.stride_k             = args.stride_k;
            kargs.nhead_stride_k       = args.nhead_stride_k;
            kargs.batch_stride_k       = args.batch_stride_k;
            kargs.k_hat_ptr            = args.k_hat_ptr;
            kargs.stride_k_hat         = hdim / 2;
            kargs.nhead_stride_k_hat   = sk_pad * (hdim / 2);
            kargs.batch_stride_k_hat   = nhead * sk_pad * (hdim / 2);
            kargs.k_scale_ptr          = args.k_scale_ptr;
            kargs.stride_k_scale       = hdim / kG;
            kargs.nhead_stride_k_scale = sk_pad * (hdim / kG);
            kargs.batch_stride_k_scale = nhead * sk_pad * (hdim / kG);
            kargs.k_mean_ptr           = k_mean_buf;
            kargs.nhead_stride_k_mean  = hdim;
            kargs.batch_stride_k_mean  = nhead * hdim;
            kargs.k_prime_ptr          = k_prime_buf;
            kargs.stride_k_prime       = hdim;
            kargs.nhead_stride_k_prime = sk_pad * hdim;
            kargs.batch_stride_k_prime = nhead * sk_pad * hdim;
            kargs.batch                = batch;
            kargs.nhead                = nhead;
            kargs.num_q_tiles          = num_q_tiles;
            kargs.num_k_tiles          = num_k_tiles;

            const dim3 grids   = PrepKernel::GridSize(kargs);
            const dim3 blocks  = PrepKernel::BlockSize();
            const index_t smem = PrepKernel::GetSmemSize();

            stream_config sc{stream};
            launch_and_check(sc, make_kernel(PrepKernel{}, grids, blocks, smem, kargs));
        } // end if(stages & kSA3StagePreprocess)

        // ------------------------------------------------------------------ //
        // Launch 1b: V preprocess -- LDS-based tile transpose + MXFP4 quantize
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageVPreprocess)
        {
            using VKernel = SageAttnV3VPreprocessKernel<InputT>;
            using VKargs  = typename VKernel::Kargs;

            VKargs kargs{};
            kargs.v_ptr                = args.v_ptr;
            kargs.seqlen_k             = sk_pad;   // used for GridSize (must be padded)
            kargs.seqlen_k_real        = seqlen_k; // used for input bounds check
            kargs.hdim                 = hdim;
            kargs.nhead_stride_v       = args.nhead_stride_v;
            kargs.batch_stride_v       = args.batch_stride_v;
            kargs.v_hat_ptr            = args.v_hat_ptr;
            kargs.stride_v_hat         = sk_pad / 2;
            kargs.nhead_stride_v_hat   = hdim * (sk_pad / 2);
            kargs.batch_stride_v_hat   = nhead * hdim * (sk_pad / 2);
            kargs.v_scale_ptr          = args.v_scale_ptr;
            kargs.stride_v_scale       = sk_pad / kG;
            kargs.nhead_stride_v_scale = hdim * (sk_pad / kG);
            kargs.batch_stride_v_scale = nhead * hdim * (sk_pad / kG);
            kargs.nhead                = nhead;
            kargs.batch                = batch;

            const dim3 grids   = VKernel::GridSize(kargs);
            const dim3 blocks  = VKernel::BlockSize();
            const index_t smem = VKernel::GetSmemSize();

            stream_config sc{stream};
            launch_and_check(sc, make_kernel(VKernel{}, grids, blocks, smem, kargs));
        } // end if(stages & kSA3StageVPreprocess)

        // ------------------------------------------------------------------ //
        // Launch 2: batched GEMM  delta_s = q_mean @ K'^T
        // ------------------------------------------------------------------ //
        if(stages & kSA3StageDeltaS)
        {
            using ALayout = tensor_layout::gemm::RowMajor;
            // K' stored naturally as [seqlen_k=N, hdim=K] row-major -> ColMajor B[K, N] -> K'^T
            using BLayout = tensor_layout::gemm::ColumnMajor;
            using CLayout = tensor_layout::gemm::RowMajor;

            constexpr bool kPadM         = true;  // num_q_tiles may not be multiple of M_Tile
            constexpr bool kPadN         = true;  // seqlen_k may not be multiple of N_Tile
            constexpr bool kPadK         = false; // hdim is always multiple of K_Tile
            constexpr bool DoubleSmemBuf = false;
            constexpr bool TransposeC    = false;

            using GemmTraits = TileGemmUniversalTraits<kPadM,
                                                       kPadN,
                                                       kPadK,
                                                       DoubleSmemBuf,
                                                       ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       TransposeC>;

            using Shape = DeltaSV3GemmShape<InputT>;

            using GemmProblem = UniversalGemmPipelineProblem<InputT, // ADataType
                                                             InputT, // BDataType
                                                             float,  // AccDataType
                                                             Shape,
                                                             GemmTraits,
                                                             GemmPipelineScheduler::Intrawave>;

            using GemmPipeline = GemmPipelineAgBgCrCompV3<GemmProblem>;

            using GemmEpilogue = CShuffleEpilogue<CShuffleEpilogueProblem<
                InputT,  // ADataType
                InputT,  // BDataType
                tuple<>, // DsDataType (no D)
                float,   // AccDataType
                float,   // CDataType (delta_s is float)
                tuple<>, // DsLayout
                CLayout,
                element_wise::PassThrough,
                Shape::kM,
                Shape::kN,
                /*MWarp=*/1,
                /*NWarp=*/2,
                /*MWarpTile=*/32,
                /*NWarpTile=*/32,
                Shape::WarpTile::at(number<2>{}), // 16 for fp16/bf16, 8 for float
                TransposeC>>;

            using GemmTilePartitioner = GemmSpatiallyLocalTilePartitioner<Shape,
                                                                          /*GroupNum=*/8,
                                                                          /*M01=*/4>;
            using GemmKernel = BatchedGemmKernel<GemmTilePartitioner, GemmPipeline, GemmEpilogue>;

            // M = num_q_tiles, N = sk_pad, K = hdim, batch_count = batch * nhead
            const index_t M           = num_q_tiles;
            const index_t N           = sk_pad;
            const index_t K           = hdim;
            const index_t batch_count = batch * nhead;

            // A: q_mean  [batch*nhead, num_q_tiles, hdim]  row-major, stride=hdim
            // B: K'      [batch*nhead, sk_pad, hdim]  col-major (leading dim = hdim = K)
            //   ColMajor B[K,N] where K'[n,k] = B[k,n] -> GEMM computes q_mean @ K'^T
            // C: delta_s [batch*nhead, num_q_tiles, sk_pad]  row-major
            BatchedGemmHostArgs gemm_hargs(args.q_mean_ptr, // A ptr
                                           k_prime_buf,     // B ptr (K')
                                           delta_s_ptr,     // C ptr
                                           /*k_batch=*/1,
                                           M,
                                           N,
                                           K,
                                           /*stride_A=*/K, // q_mean row stride = hdim
                                           /*stride_B=*/K, // ColMajor leading dim = hdim (= K)
                                           /*stride_C=*/N, // delta_s row stride = sk_pad
                                           /*batch_stride_A=*/M * K, // num_q_tiles * hdim
                                           /*batch_stride_B=*/N * K, // sk_pad * hdim
                                           /*batch_stride_C=*/M * N, // num_q_tiles * sk_pad
                                           batch_count);

            auto kargs        = GemmKernel::MakeKernelArgs(gemm_hargs);
            const dim3 grids  = GemmKernel::GridSize(M, N, /*k_batch=*/1, batch_count);
            const dim3 blocks = GemmKernel::BlockSize();
            // BatchedGemmKernel uses statically-declared __shared__ arrays; dynamic smem = 0.
            stream_config sc{stream};
            launch_and_check(sc, make_kernel(GemmKernel{}, grids, blocks, 0, kargs));
        }
    }
};

} // namespace ck_tile
