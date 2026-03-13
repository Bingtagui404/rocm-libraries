// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"

namespace ck {

// Standalone offset-computing transfer class for the 3-way wavelet split.
//
// Takes a source descriptor and produces an array of (offset, validity) pairs —
// the same offsets that v3r1's RunRead() would use, but without issuing any buffer_load.
// This allows index waves (pure VALU) to precompute offsets and pass them to load waves
// (pure VMEM) via an LDS offset buffer.
//
// Template parameters mirror the source-side of ThreadwiseTensorSliceTransfer_v3r1.
template <typename SliceLengths,
          typename SrcDesc,
          typename SrcDimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector_,
          bool SrcResetCoordinateAfterRun>
struct ThreadwiseTensorSliceTransfer_OffsetCompute
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    // Compute access geometry as constexpr functions to avoid class-scope auto issues
    __device__ static constexpr auto GetSrcScalarPerAccess()
    {
        return generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector_>{}, Number<nDim>{});
    }

    __device__ static constexpr auto GetSrcAccessLengths()
    {
        return SliceLengths{} / GetSrcScalarPerAccess();
    }

    __device__ static constexpr auto GetOrderedSrcAccessLengths()
    {
        return container_reorder_given_new2old(GetSrcAccessLengths(), SrcDimAccessOrder{});
    }

    // Total number of access points per RunComputeOffsets call
    __host__ __device__ static constexpr index_t GetNumAccessPoints()
    {
        constexpr auto access_lengths = GetSrcAccessLengths();
        index_t n                     = 1;
        static_for<0, nDim, 1>{}([&](auto i) { n *= access_lengths[i]; });
        return n;
    }

    static constexpr index_t NumAccessPoints = GetNumAccessPoints();

    __device__ constexpr ThreadwiseTensorSliceTransfer_OffsetCompute(const SrcDesc& src_desc,
                                                                     const Index& src_slice_origin)
        : src_coord_(make_tensor_coordinate(src_desc, src_slice_origin))
    {
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_ = make_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    // Compute offsets into a packed int32_t array.
    // Format: offset in bits [30:0], validity in bit [31] (1 = valid).
    // The iteration order matches v3r1's RunRead exactly (same SFC pattern).
    template <typename PackedOffsetArray>
    __device__ void RunComputePackedOffsets(const SrcDesc& src_desc,
                                            PackedOffsetArray& packed_offset_array)
    {
        constexpr auto src_scalar_per_access  = GetSrcScalarPerAccess();
        constexpr auto ordered_access_lengths = GetOrderedSrcAccessLengths();
        constexpr auto src_dim_access_order   = SrcDimAccessOrder{};

        const auto src_forward_steps  = ComputeForwardSteps(src_desc, src_scalar_per_access);
        const auto src_backward_steps = ComputeBackwardSteps(src_desc, src_scalar_per_access);

        index_t access_idx = 0;

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_src_access_idx) {
            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            // Pack: offset in bits [30:0], validity in bit [31]
            index_t offset = src_coord_.GetOffset();
            int32_t packed = static_cast<int32_t>(offset);
            packed         = is_src_valid ? (packed | static_cast<int32_t>(0x80000000u)) : packed;
            packed_offset_array[access_idx] = packed;

            ++access_idx;

            // Move src coord (same pattern as v3r1)
            constexpr auto forward_sweep =
                ComputeForwardSweep(ordered_src_access_idx, ordered_access_lengths);

            constexpr auto move_on_dim =
                ComputeMoveOnDim(ordered_src_access_idx, ordered_access_lengths);

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_forward_steps[src_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            src_desc, src_coord_, src_backward_steps[src_dim_access_order[i]]);
                    }
                }
            });
        });

        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_step =
                make_tensor_coordinate_step(src_desc, GetSrcCoordinateResetStep());
            move_tensor_coordinate(src_desc, src_coord_, src_reset_step);
        }
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        const auto adjusted_step = make_tensor_coordinate_step(src_desc, adjusted_step_idx);
        move_tensor_coordinate(src_desc, src_coord_, adjusted_step);
    }

    // Unpack helpers (static, can be called by load waves too)
    __device__ static index_t UnpackOffset(int32_t packed)
    {
        return static_cast<index_t>(packed & static_cast<int32_t>(0x7FFFFFFFu));
    }

    __device__ static bool UnpackValidity(int32_t packed)
    {
        return (packed & static_cast<int32_t>(0x80000000u)) != 0;
    }

    private:
    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto src_scalar_per_access  = GetSrcScalarPerAccess();
        constexpr auto ordered_access_lengths = GetOrderedSrcAccessLengths();

        constexpr auto ordered_access_lengths_minus_1 = generate_tuple(
            [&](auto i) { return Number<ordered_access_lengths.At(i) - 1>{}; }, Number<nDim>{});
        constexpr auto forward_sweep =
            ComputeForwardSweep(ordered_access_lengths_minus_1, ordered_access_lengths);

        constexpr auto data_idx = [&]() {
            Index ordered_idx;
            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });
            return container_reorder_given_old2new(ordered_idx, SrcDimAccessOrder{}) *
                   src_scalar_per_access;
        }();

        constexpr auto reset_data_step = [&]() {
            Index reset_data_step_;
            static_for<0, nDim, 1>{}([&](auto i) { reset_data_step_(i) = -data_idx[i]; });
            return reset_data_step_;
        }();

        return reset_data_step;
    }

    template <typename OrderedAccessIdx, typename OrderedAccessLengths>
    __device__ static constexpr auto
    ComputeForwardSweep(const OrderedAccessIdx& ordered_access_idx,
                        const OrderedAccessLengths& ordered_access_lengths)
    {
        StaticallyIndexedArray<bool, nDim> forward_sweep_;
        forward_sweep_(I0) = true;
        static_for<1, nDim, 1>{}([&](auto i) {
            index_t tmp = ordered_access_idx[I0];
            static_for<1, i, 1>{}(
                [&](auto j) { tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j]; });
            forward_sweep_(i) = tmp % 2 == 0;
        });
        return forward_sweep_;
    }

    template <typename OrderedAccessIdx, typename OrderedAccessLengths>
    __device__ static constexpr auto
    ComputeMoveOnDim(const OrderedAccessIdx& ordered_access_idx,
                     const OrderedAccessLengths& ordered_access_lengths)
    {
        StaticallyIndexedArray<bool, nDim> move_on_dim_;
        static_for<0, nDim, 1>{}([&](auto i) {
            move_on_dim_(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;
            static_for<i + 1, nDim, 1>{}([&](auto j) {
                move_on_dim_(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
            });
        });
        return move_on_dim_;
    }

    template <typename Desc, typename ScalarPerAccess>
    __device__ static auto ComputeForwardSteps(const Desc& desc,
                                               const ScalarPerAccess& scalar_per_access)
    {
        return generate_tuple(
            [&](auto i) {
                Index forward_step_idx;
                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step_idx(j) = (i.value == j.value) ? scalar_per_access[i] : 0;
                });
                return make_tensor_coordinate_step(desc, forward_step_idx);
            },
            Number<nDim>{});
    }

    template <typename Desc, typename ScalarPerAccess>
    __device__ static auto ComputeBackwardSteps(const Desc& desc,
                                                const ScalarPerAccess& scalar_per_access)
    {
        return generate_tuple(
            [&](auto i) {
                Index backward_step_idx;
                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step_idx(j) = (i.value == j.value) ? -scalar_per_access[i] : 0;
                });
                return make_tensor_coordinate_step(desc, backward_step_idx);
            },
            Number<nDim>{});
    }

    SrcCoord src_coord_;
};

} // namespace ck
