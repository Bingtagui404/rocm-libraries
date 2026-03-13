// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

template <typename TileLoadThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmLoadWave;

// 1-stage prefetch
template <typename TileLoadThreadGroup>
struct GridwiseGemmLoadWave<TileLoadThreadGroup, 1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */)
    {
        // TODO: improve applicability
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep>
    static __device__ void RunLoadWavePipeline(const AGridDesc& a_grid_desc,
                                               const ABlockDesc& a_block_desc,
                                               ABlockTransfer& a_blockwise_copy,
                                               const AGridBuffer& a_grid_buf,
                                               ABlockBuffer& a_block_buf,
                                               const ABlockTransferStep& a_block_copy_step,
                                               const BGridDesc& b_grid_desc,
                                               const BBlockDesc& b_block_desc,
                                               BBlockTransfer& b_blockwise_copy,
                                               const BGridBuffer& b_grid_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BBlockTransferStep& b_block_copy_step,
                                               index_t num_loop)
    {
        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // LDS write 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // sync for Load threads()
                block_sync_lds();
                // global read i + 1
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // sync with math threads()
                block_sync_lds();

                // LDS write i+1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            // GEMM num_loop - 1
        }
    }

    // Offset round-trip variant: same barrier structure as RunLoadWavePipeline,
    // but uses offset-compute + offset-load transfer classes instead of v4r1.
    // Each load thread computes offsets → writes to offset LDS → reads back → issues loads.
    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename AOffsetCompute,
              typename AOffsetLoad,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BOffsetCompute,
              typename BOffsetLoad,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep>
    static __device__ void
    RunLoadWavePipelineWithOffsets(const AGridDesc& a_grid_desc,
                                   const ABlockDesc& a_block_desc,
                                   AOffsetCompute& a_offset_compute,
                                   AOffsetLoad& a_offset_load,
                                   const AGridBuffer& a_grid_buf,
                                   ABlockBuffer& a_block_buf,
                                   const ABlockTransferStep& a_block_copy_step,
                                   int32_t* a_offset_lds,
                                   const BGridDesc& b_grid_desc,
                                   const BBlockDesc& b_block_desc,
                                   BOffsetCompute& b_offset_compute,
                                   BOffsetLoad& b_offset_load,
                                   const BGridBuffer& b_grid_buf,
                                   BBlockBuffer& b_block_buf,
                                   const BBlockTransferStep& b_block_copy_step,
                                   int32_t* b_offset_lds,
                                   index_t num_loop)
    {
        // Prologue: compute offsets, load via offsets, write to data LDS (step 0)
        a_offset_compute.RunComputeOffsets(a_grid_desc, a_offset_lds);
        b_offset_compute.RunComputeOffsets(b_grid_desc, b_offset_lds);

        a_offset_load.RunReadFromOffsetBuffer(a_offset_lds, a_grid_buf);
        b_offset_load.RunReadFromOffsetBuffer(b_offset_lds, b_grid_buf);

        // move to 1
        a_offset_compute.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_offset_compute.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // LDS write 0
        a_offset_load.RunWrite(a_block_desc, a_block_buf);
        b_offset_load.RunWrite(b_block_desc, b_block_buf);

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // sync for Load threads
                block_sync_lds();

                // compute offsets i + 1
                a_offset_compute.RunComputeOffsets(a_grid_desc, a_offset_lds);
                b_offset_compute.RunComputeOffsets(b_grid_desc, b_offset_lds);

                // global read i + 1
                a_offset_load.RunReadFromOffsetBuffer(a_offset_lds, a_grid_buf);
                b_offset_load.RunReadFromOffsetBuffer(b_offset_lds, b_grid_buf);

                // move to i + 2
                a_offset_compute.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_offset_compute.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // sync with math threads
                block_sync_lds();

                // LDS write i+1
                a_offset_load.RunWrite(a_block_desc, a_block_buf);
                b_offset_load.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            // GEMM num_loop - 1
        }
    }
};

template <typename TileMathThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmMathWave;
// 1- stage prefetch
template <typename TileMathThreadGroup>
struct GridwiseGemmMathWave<TileMathThreadGroup, 1>
{

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void RunMathWavePipeline(ABlockBuffer& a_block_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BlockwiseGemm& block_gemm,
                                               CThreadBuffer& c_thread_buf,
                                               index_t num_loop)
    {
        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds();

                // GEMM i
                block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// =============================================================================
// 3-way wave split pipeline (index + load + math)
//
// Barrier protocol (all three roles must execute the same block_sync_lds() calls):
//
// Prologue:
//   P1: after index computes offsets for step 0
//   P2: after load reads offsets and issues buffer_loads for step 0
//   P3: after load writes data to LDS for step 0
//
// Main loop (i = 0..num_loop-2):
//   B1: data LDS for step i ready (math reads, load issues step i+1, index computes step i+2)
//   B2: GEMM done, loads done, offsets written (load writes step i+1 to LDS)
//
// Tail:
//   T1: data LDS for last step ready (math does final GEMM)
//
// Offset ping-pong: index writes offset_buf[step % 2], load reads offset_buf[(step+1) % 2].
// At any given time, index and load access different halves of the offset buffer.
//
// HasMainLoop means num_loop > 2 (3-way needs at least 3 steps: prologue fills 2, tail does 1).
// =============================================================================

// TODO(human): Implement the three 3-way pipeline functions below.
//
// Each function represents one wave role's view of the synchronized pipeline.
// All three MUST execute the same number of block_sync_lds() calls in the same order.
//
// Use the barrier protocol above as your guide. The 2-way pipelines (RunLoadWavePipeline,
// RunMathWavePipeline) and the offset round-trip variant (RunLoadWavePipelineWithOffsets)
// are good references for the coding pattern.
//
// Template parameters for each function should match what the gridwise kernel can provide:
//   Index: AGridDesc, AOffsetCompute, ABlockTransferStep (+ same for B), offset LDS pointers
//   Load:  ABlockDesc, AOffsetLoad, AGridBuffer, ABlockBuffer (+ same for B), offset LDS pointers
//   Math:  ABlockBuffer, BBlockBuffer, BlockwiseGemm, CThreadBuffer
//
// Key details:
//   - num_loop is the total number of K-loop iterations (same semantics as 2-way)
//   - HasMainLoop = (num_loop > 2) for 3-way (prologue consumes 2 steps)
//   - Index waves advance the source window after each ComputeOffsets call
//   - Load waves do NOT have a source coordinate — they only read offsets and issue loads
//   - For the tail: index and load are idle, only math does the final GEMM

template <typename TileIndexThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmIndexWave;

template <typename TileIndexThreadGroup>
struct GridwiseGemmIndexWave<TileIndexThreadGroup, 1>
{
    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename AOffsetCompute,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BOffsetCompute,
              typename BBlockTransferStep>
    static __device__ void RunIndexWavePipeline(const AGridDesc& a_grid_desc,
                                                AOffsetCompute& a_offset_compute,
                                                const ABlockTransferStep& a_block_copy_step,
                                                int32_t* a_offset_lds_0,
                                                int32_t* a_offset_lds_1,
                                                const BGridDesc& b_grid_desc,
                                                BOffsetCompute& b_offset_compute,
                                                const BBlockTransferStep& b_block_copy_step,
                                                int32_t* b_offset_lds_0,
                                                int32_t* b_offset_lds_1,
                                                index_t num_loop)
    {
        // Prologue: compute offsets (step 0) → buf[0]
        a_offset_compute.RunComputeOffsets(a_grid_desc, a_offset_lds_0);
        b_offset_compute.RunComputeOffsets(b_grid_desc, b_offset_lds_0);

        a_offset_compute.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_offset_compute.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // P1: buf[0] ready for Load to read
        block_sync_lds();

        // Compute offsets (step 1) → buf[1] (Load reads buf[0] concurrently)
        a_offset_compute.RunComputeOffsets(a_grid_desc, a_offset_lds_1);
        b_offset_compute.RunComputeOffsets(b_grid_desc, b_offset_lds_1);

        a_offset_compute.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_offset_compute.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // P2: buf[1] ready, Load done reading buf[0]
        block_sync_lds();

        // P3: Load done writing data LDS
        block_sync_lds();

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // B1: data LDS ready for Math
                block_sync_lds();

                // compute offsets (step i+2) → buf[i%2]
                // (Load reads buf[(i+1)%2] concurrently)
                int32_t* a_lds = (i % 2 == 0) ? a_offset_lds_0 : a_offset_lds_1;
                int32_t* b_lds = (i % 2 == 0) ? b_offset_lds_0 : b_offset_lds_1;

                a_offset_compute.RunComputeOffsets(a_grid_desc, a_lds);
                b_offset_compute.RunComputeOffsets(b_grid_desc, b_lds);

                a_offset_compute.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_offset_compute.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // B2: GEMM done, offsets written
                block_sync_lds();

                ++i;
            } while(i < (num_loop - 2));
        }

        // Tail: Index is done computing — match remaining Load/Math barriers.
        // Load runs (num_loop-2) extra main iterations, each with B1+B2, then T1.
        {
            // B1 (Load i=num_loop-2): Load reads last offsets
            block_sync_lds();
            // B2 (Load i=num_loop-2): Load writes last data to LDS
            block_sync_lds();

            // T1: data LDS for last step ready, Math does final GEMM
            block_sync_lds();
        }
    }
};

template <typename TileLoadThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmLoadWave3Way;

template <typename TileLoadThreadGroup>
struct GridwiseGemmLoadWave3Way<TileLoadThreadGroup, 1>
{
    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
    }

    template <bool HasMainLoop,
              typename ABlockDesc,
              typename AOffsetLoad,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename BBlockDesc,
              typename BOffsetLoad,
              typename BGridBuffer,
              typename BBlockBuffer>
    static __device__ void RunLoadWavePipeline3Way(const ABlockDesc& a_block_desc,
                                                   AOffsetLoad& a_offset_load,
                                                   const AGridBuffer& a_grid_buf,
                                                   ABlockBuffer& a_block_buf,
                                                   int32_t* a_offset_lds_0,
                                                   int32_t* a_offset_lds_1,
                                                   const BBlockDesc& b_block_desc,
                                                   BOffsetLoad& b_offset_load,
                                                   const BGridBuffer& b_grid_buf,
                                                   BBlockBuffer& b_block_buf,
                                                   int32_t* b_offset_lds_0,
                                                   int32_t* b_offset_lds_1,
                                                   index_t num_loop)
    {
        // P1: buf[0] ready (Index wrote step 0)
        block_sync_lds();

        // Read step 0 offsets from buf[0] (Index writes buf[1] concurrently)
        a_offset_load.RunReadFromOffsetBuffer(a_offset_lds_0, a_grid_buf);
        b_offset_load.RunReadFromOffsetBuffer(b_offset_lds_0, b_grid_buf);

        // P2: Load done reading buf[0], Index done writing buf[1]
        block_sync_lds();

        // Write step 0 data to data LDS
        a_offset_load.RunWrite(a_block_desc, a_block_buf);
        b_offset_load.RunWrite(b_block_desc, b_block_buf);

        // P3: data LDS has step 0
        block_sync_lds();

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // B1: data LDS step i ready for Math
                block_sync_lds();

                // Read offsets for step i+1 from buf[(i+1)%2]
                // (Index writes buf[i%2] concurrently)
                int32_t* a_lds = (i % 2 == 0) ? a_offset_lds_1 : a_offset_lds_0;
                int32_t* b_lds = (i % 2 == 0) ? b_offset_lds_1 : b_offset_lds_0;

                a_offset_load.RunReadFromOffsetBuffer(a_lds, a_grid_buf);
                b_offset_load.RunReadFromOffsetBuffer(b_lds, b_grid_buf);

                // B2: Math done reading, offsets in VGPRs
                block_sync_lds();

                // Write step i+1 data to LDS
                a_offset_load.RunWrite(a_block_desc, a_block_buf);
                b_offset_load.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            // T1: data LDS for last step ready
            block_sync_lds();
        }
    }
};

template <typename TileMathThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmMathWave3Way;

template <typename TileMathThreadGroup>
struct GridwiseGemmMathWave3Way<TileMathThreadGroup, 1>
{
    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
    }

    template <bool HasMainLoop,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void RunMathWavePipeline3Way(ABlockBuffer& a_block_buf,
                                                   BBlockBuffer& b_block_buf,
                                                   const BlockwiseGemm& block_gemm,
                                                   CThreadBuffer& c_thread_buf,
                                                   index_t num_loop)
    {
        // Initialize C accumulator
        c_thread_buf.Clear();

        // P1: Index done writing offsets for step 0 (Math idle)
        block_sync_lds();

        // P2: Index done writing offsets for step 1, Load done reading step 0 offsets (Math idle)
        block_sync_lds();

        // P3: Load done writing step 0 data to LDS — data LDS now has step 0
        block_sync_lds();

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // B1: data LDS for step i ready — safe to read
                block_sync_lds();

                // GEMM step i (reads A+B from data LDS)
                block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // B2: GEMM done reading LDS — Load can now overwrite with step i+1
                block_sync_lds();

                ++i;
            } while(i < (num_loop - 1));
        }

        // Tail
        {
            // T1: data LDS for last step ready
            block_sync_lds();

            // Final GEMM (last K-block)
            block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

} // namespace ck
