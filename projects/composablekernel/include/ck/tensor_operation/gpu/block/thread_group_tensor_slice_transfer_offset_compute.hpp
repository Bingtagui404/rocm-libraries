// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_offset_compute.hpp"

namespace ck {

// Block-level wrapper around ThreadwiseTensorSliceTransfer_OffsetCompute.
// Distributes offset computation across a thread group using a cluster descriptor,
// and writes packed offsets to an LDS buffer.
//
// This mirrors ThreadGroupTensorSliceTransfer_v4r1's structure, but only computes
// offsets (no actual data loads).
template <typename ThreadGroup,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDesc,
          typename SrcDimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          bool ThreadTransferSrcResetCoordinateAfterRun>
struct ThreadGroupTensorSliceTransfer_OffsetCompute
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();

    static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};

    using Index = MultiIndex<nDim>;

    using ThreadwiseOffsetCompute =
        ThreadwiseTensorSliceTransfer_OffsetCompute<decltype(thread_slice_lengths),
                                                    SrcDesc,
                                                    SrcDimAccessOrder,
                                                    SrcVectorDim,
                                                    SrcScalarPerVector,
                                                    ThreadTransferSrcResetCoordinateAfterRun>;

    // Number of access points per thread
    static constexpr index_t NumAccessPointsPerThread = ThreadwiseOffsetCompute::NumAccessPoints;

    // Total access points across all threads in the cluster
    static constexpr index_t TotalAccessPoints =
        NumAccessPointsPerThread * ThreadClusterLengths::template Reduce<ck::math::multiplies>();

    __device__ constexpr ThreadGroupTensorSliceTransfer_OffsetCompute(
        const SrcDesc& src_desc, const Index& src_block_slice_origin)
        : threadwise_offset_compute_(src_desc, make_zero_multi_index<nDim>())
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

            threadwise_offset_compute_.SetSrcSliceOrigin(
                src_desc, src_block_slice_origin + thread_data_idx_begin);
        }
    }

    // Compute packed offsets and write to an LDS buffer.
    // Each thread writes NumAccessPointsPerThread entries at its designated LDS offset.
    // LDS layout: [thread_id_in_cluster][access_point_index] of int32_t
    template <typename OffsetLdsBuffer>
    __device__ void RunComputeOffsets(const SrcDesc& src_desc, OffsetLdsBuffer& offset_lds_buf)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            // Thread-local packed offset array (VGPR)
            int32_t packed_offsets[NumAccessPointsPerThread];

            threadwise_offset_compute_.RunComputePackedOffsets(src_desc, packed_offsets);

            // Write to LDS at this thread's slot
            const index_t thread_id = ThreadGroup::GetThreadId();
            const index_t lds_base  = thread_id * NumAccessPointsPerThread;

            for(index_t i = 0; i < NumAccessPointsPerThread; ++i)
            {
                offset_lds_buf[lds_base + i] = packed_offsets[i];
            }
        }
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            threadwise_offset_compute_.MoveSrcSliceWindow(src_desc, step);
        }
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_block_slice_origin)
    {
        if(ThreadGroup::GetNumOfThread() == thread_cluster_desc_.GetElementSize() or
           ThreadGroup::GetThreadId() < thread_cluster_desc_.GetElementSize())
        {
            const auto thread_cluster_idx = thread_cluster_desc_.CalculateBottomIndex(
                make_multi_index(ThreadGroup::GetThreadId()));

            const auto thread_data_idx_begin = thread_cluster_idx * thread_slice_lengths;

            threadwise_offset_compute_.SetSrcSliceOrigin(
                src_desc, src_block_slice_origin + thread_data_idx_begin);
        }
    }

    // Unpack helpers (delegates to threadwise)
    __device__ static index_t UnpackOffset(int32_t packed)
    {
        return ThreadwiseOffsetCompute::UnpackOffset(packed);
    }

    __device__ static bool UnpackValidity(int32_t packed)
    {
        return ThreadwiseOffsetCompute::UnpackValidity(packed);
    }

    private:
    static constexpr auto thread_cluster_desc_ =
        make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});

    ThreadwiseOffsetCompute threadwise_offset_compute_;
};

} // namespace ck
