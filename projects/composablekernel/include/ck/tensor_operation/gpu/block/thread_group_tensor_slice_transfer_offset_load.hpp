// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_offset_load.hpp"

namespace ck {

// Block-level wrapper around ThreadwiseTensorSliceTransfer_OffsetLoad.
// Distributes offset-consuming loads across a thread group using a cluster descriptor.
//
// This mirrors ThreadGroupTensorSliceTransfer_v4r1's structure but:
// - Has no source descriptor or source coordinate — offsets come from an LDS buffer
// - RunReadFromOffsetBuffer replaces RunRead
// - RunWrite is identical (delegates to per-thread)
template <typename ThreadGroup,
          typename SrcElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcData,
          typename DstData,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool ThreadTransferDstResetCoordinateAfterRun,
          index_t NumThreadScratch = 1>
struct ThreadGroupTensorSliceTransfer_OffsetLoad
{
    static constexpr index_t nDim = remove_reference_t<DstDesc>::GetNumOfDimension();

    static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    using ThreadwiseTransfer =
        ThreadwiseTensorSliceTransfer_OffsetLoad<decltype(thread_slice_lengths),
                                                 SrcElementwiseOperation,
                                                 DstElementwiseOperation,
                                                 DstInMemOp,
                                                 SrcData,
                                                 DstData,
                                                 DstDesc,
                                                 SrcDimAccessOrder,
                                                 DstDimAccessOrder,
                                                 SrcVectorDim,
                                                 DstVectorDim,
                                                 SrcScalarPerVector,
                                                 DstScalarPerVector,
                                                 SrcScalarStrideInVector,
                                                 DstScalarStrideInVector,
                                                 ThreadTransferDstResetCoordinateAfterRun,
                                                 NumThreadScratch>;

    static constexpr index_t NumAccessPointsPerThread = ThreadwiseTransfer::NumAccessPoints;

    __device__ constexpr ThreadGroupTensorSliceTransfer_OffsetLoad(
        const SrcElementwiseOperation& src_element_op,
        const DstDesc& dst_desc,
        const Index& dst_block_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : threadwise_transfer_(
              src_element_op, dst_desc, make_zero_multi_index<nDim>(), dst_element_op)
    {
        static_assert(
            is_same<BlockSliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
            "wrong! threads should be mapped to cover entire slicing window");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "wrong! ThreadGroup::GetNumOfThread() too small");

        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(ThreadGroup::GetThreadId()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            threadwise_transfer_.SetDstSliceOrigin(dst_desc,
                                                   dst_block_slice_origin + thread_data_idx_begin);
        }
    }

    // Read from global memory using pre-computed offsets stored in an LDS buffer.
    // LDS layout: [thread_id_in_cluster][access_point_index] of int32_t (packed offset|validity)
    template <typename SrcBuffer, index_t ThreadScratchId = 0>
    __device__ void
    RunReadFromOffsetBuffer(const int32_t* offset_lds_buf,
                            const SrcBuffer& src_grid_buf,
                            Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            // Read this thread's packed offsets from LDS
            const index_t thread_id = ThreadGroup::GetThreadId();
            const index_t lds_base  = thread_id * NumAccessPointsPerThread;

            int32_t packed_offsets[NumAccessPointsPerThread];
            for(index_t i = 0; i < NumAccessPointsPerThread; ++i)
            {
                packed_offsets[i] = offset_lds_buf[lds_base + i];
            }

            // Issue loads using precomputed offsets
            threadwise_transfer_.RunReadFromOffsets(
                src_grid_buf, packed_offsets, thread_scratch_id);
        }
    }

    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.RunWrite(dst_desc, dst_buf, thread_scratch_id);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_transfer_.MoveDstSliceWindow(dst_desc, step);
        }
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    ThreadwiseTransfer threadwise_transfer_;
};

} // namespace ck
