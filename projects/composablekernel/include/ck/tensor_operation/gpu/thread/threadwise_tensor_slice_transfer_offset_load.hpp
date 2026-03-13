// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/utility/is_detected.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_util.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_offset_compute.hpp"

namespace ck {

// Offset-consuming load transfer class for the 3-way wavelet split.
//
// Reads pre-computed packed (offset|validity) entries from an array, issues buffer_load
// instructions, and stores results in VGPR thread scratch buffers compatible with v3r1's
// RunWrite(). This allows load waves to operate without any source coordinate state —
// eliminating the conv-to-GEMM descriptor transform VALU overhead entirely.
//
// Template parameters mirror v3r1's, minus SrcDesc (no source coordinate needed).
// SrcDimAccessOrder and SrcVectorDim are still required for the SFC iteration that
// maps access points to correct scratch buffer positions.
template <typename SliceLengths,
          typename SrcElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename SrcData,
          typename DstData,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector_,
          index_t DstScalarPerVector_,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          index_t NumThreadScratch = 1>
struct ThreadwiseTensorSliceTransfer_OffsetLoad
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    static constexpr auto I0  = Number<0>{};
    static constexpr auto I1  = Number<1>{};
    static constexpr auto I2  = Number<2>{};
    static constexpr auto I3  = Number<3>{};
    static constexpr auto I4  = Number<4>{};
    static constexpr auto I5  = Number<5>{};
    static constexpr auto I6  = Number<6>{};
    static constexpr auto I7  = Number<7>{};
    static constexpr auto I8  = Number<8>{};
    static constexpr auto I10 = Number<10>{};
    static constexpr auto I12 = Number<12>{};
    static constexpr auto I13 = Number<13>{};
    static constexpr auto I14 = Number<14>{};
    static constexpr auto I16 = Number<16>{};

    static constexpr index_t PackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<SrcData>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr auto SrcScalarPerVector = Number<SrcScalarPerVector_ / PackedSize>{};
    static constexpr auto DstScalarPerVector = Number<DstScalarPerVector_ / PackedSize>{};

    // Access geometry for SFC iteration (same as v3r1)
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

    __host__ __device__ static constexpr index_t GetNumAccessPoints()
    {
        constexpr auto access_lengths = GetSrcAccessLengths();
        index_t n                     = 1;
        static_for<0, nDim, 1>{}([&](auto i) { n *= access_lengths[i]; });
        return n;
    }

    static constexpr index_t NumAccessPoints = GetNumAccessPoints();

    __device__ constexpr ThreadwiseTensorSliceTransfer_OffsetLoad(
        const SrcElementwiseOperation& src_element_op,
        const DstDesc& dst_desc,
        const Index& dst_slice_origin,
        const DstElementwiseOperation& dst_element_op)
        : dst_coord_(make_tensor_coordinate(dst_desc, dst_slice_origin)),
          src_element_op_(src_element_op),
          dst_element_op_(dst_element_op)
    {
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    // Read from global memory using pre-computed packed offsets.
    // packed_offsets: array of int32_t, one per access point, packed as offset|validity.
    // The SFC iteration order matches v3r1's RunRead exactly.
    template <typename SrcBuffer, index_t ThreadScratchId = 0>
    __device__ void
    RunReadFromOffsets(const SrcBuffer& src_buf,
                       const int32_t* packed_offsets,
                       Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>::value,
            "wrong! SrcBuffer and SrcData data type are inconsistent");

        constexpr auto src_scalar_per_access  = GetSrcScalarPerAccess();
        constexpr auto ordered_access_lengths = GetOrderedSrcAccessLengths();
        constexpr auto src_dim_access_order   = SrcDimAccessOrder{};

        index_t access_idx = 0;

        // Same SFC iteration as v3r1's RunRead — computes data_idx for scratch placement
        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_src_access_idx) {
            constexpr auto forward_sweep =
                ComputeForwardSweep(ordered_src_access_idx, ordered_access_lengths);

            constexpr auto src_data_idx = ComputeDataIndex(ordered_src_access_idx,
                                                           ordered_access_lengths,
                                                           forward_sweep,
                                                           src_dim_access_order,
                                                           src_scalar_per_access);

            constexpr auto src_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<src_data_idx[i]>{}; }, Number<src_data_idx.Size()>{});

            // Unpack offset and validity from pre-computed array
            const int32_t packed = packed_offsets[access_idx];
            const index_t src_offset =
                ThreadwiseTensorSliceTransfer_OffsetCompute<SliceLengths,
                                                            DstDesc, // dummy, not used
                                                            SrcDimAccessOrder,
                                                            SrcVectorDim,
                                                            SrcScalarPerVector_,
                                                            true>::UnpackOffset(packed);
            const bool is_src_valid =
                ThreadwiseTensorSliceTransfer_OffsetCompute<SliceLengths,
                                                            DstDesc,
                                                            SrcDimAccessOrder,
                                                            SrcVectorDim,
                                                            SrcScalarPerVector_,
                                                            true>::UnpackValidity(packed);

            // Store validity for RunWrite's OOB handling
            src_oob_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<bool>(src_data_idx_seq, is_src_valid);

            // Issue vector loads using the precomputed offset (same split-vector logic as v3r1)
            using dst_vector_type = vector_type_maker_t<DstData, SrcScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;
            dst_vector_type op_r_v;

            constexpr auto get_elem_op_vec_len = []() {
                if constexpr(is_detected<is_pack8_invocable_t, SrcElementwiseOperation>::value)
                {
                    if constexpr(SrcElementwiseOperation::is_pack8_invocable)
                        return math::min(8, SrcScalarPerVector);
                }
                else if constexpr(is_detected<is_pack4_invocable_t, SrcElementwiseOperation>::value)
                {
                    if constexpr(SrcElementwiseOperation::is_pack4_invocable)
                        return math::min(4, SrcScalarPerVector);
                }
                else if constexpr(is_detected<is_pack2_invocable_t, SrcElementwiseOperation>::value)
                {
                    if constexpr(SrcElementwiseOperation::is_pack2_invocable)
                        return math::min(2, SrcScalarPerVector);
                }
                else
                {
                    return 1;
                }
            };

            constexpr index_t elem_op_vec_len = get_elem_op_vec_len();

            using src_elem_op_vec_t = typename vector_type<SrcData, elem_op_vec_len>::type;
            using dst_elem_op_vec_t = typename vector_type<DstData, elem_op_vec_len>::type;

            using VectorSizeLookupTable    = Tuple<Sequence<>,
                                                   Sequence<I1>,
                                                   Sequence<I2>,
                                                   Sequence<I2, I1>,
                                                   Sequence<I4>,
                                                   Sequence<I4, I1>,
                                                   Sequence<I4, I2>,
                                                   Sequence<I4, I2, I1>,
                                                   Sequence<I8>,
                                                   Sequence<I8, I1>,
                                                   Sequence<I8, I2>,
                                                   Sequence<I8, I2, I1>,
                                                   Sequence<I8, I4>,
                                                   Sequence<I8, I4, I1>,
                                                   Sequence<I8, I4, I2>,
                                                   Sequence<I8, I4, I2, I1>,
                                                   Sequence<I16>>;
            using VectorOffsetsLookupTable = Tuple<Sequence<>,
                                                   Sequence<I0>,
                                                   Sequence<I0>,
                                                   Sequence<I0, I2>,
                                                   Sequence<I0>,
                                                   Sequence<I0, I4>,
                                                   Sequence<I0, I4>,
                                                   Sequence<I0, I4, I6>,
                                                   Sequence<I0>,
                                                   Sequence<I0, I8>,
                                                   Sequence<I0, I8>,
                                                   Sequence<I0, I8, I10>,
                                                   Sequence<I0, I8>,
                                                   Sequence<I0, I8, I12>,
                                                   Sequence<I0, I8, I12>,
                                                   Sequence<I0, I8, I12, I14>,
                                                   Sequence<I0>>;

            static_for<0, tuple_element_t<SrcScalarPerVector, VectorSizeLookupTable>::Size(), 1>{}(
                [&](auto v_idx) {
                    constexpr auto VectorLoadSize =
                        tuple_element_t<SrcScalarPerVector, VectorSizeLookupTable>::At(v_idx);
                    constexpr auto LoadOffset =
                        tuple_element_t<SrcScalarPerVector, VectorOffsetsLookupTable>::At(v_idx);

                    using src_vector_container   = vector_type_maker_t<SrcData, VectorLoadSize>;
                    using src_vector_container_t = typename src_vector_container::type;

                    // Use precomputed offset instead of src_coord_.GetOffset()
                    src_vector_container src_vector =
                        src_vector_container{src_buf.template Get<src_vector_container_t>(
                            src_offset / PackedSize + LoadOffset, true)};

                    static_for<0, VectorLoadSize / elem_op_vec_len, 1>{}([&](auto idx) {
                        src_element_op_(
                            op_r_v.template AsType<dst_elem_op_vec_t>()(idx + LoadOffset),
                            src_vector.template AsType<src_elem_op_vec_t>()[idx]);
                    });
                });

            // Store to scratch at the correct SFC-mapped position
            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<dst_vector_t>(src_data_idx_seq,
                                                  op_r_v.template AsType<dst_vector_t>()[I0]);

            ++access_idx;
            // No coordinate movement — offsets are precomputed
        });
    }

    // RunWrite: identical to v3r1's RunWrite
    template <typename DstBuffer, index_t ThreadScratchId = 0>
    __device__ void RunWrite(const DstDesc& dst_desc,
                             DstBuffer& dst_buf,
                             Number<ThreadScratchId> thread_scratch_id = Number<ThreadScratchId>{})
    {
        TransferDataFromSrcThreadScratchToDstThreadScratch(thread_scratch_id);

        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Global or
                          DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "wrong!");

        static_assert(
            is_same<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>::value,
            "wrong! DstBuffer data type is wrong");

        constexpr auto dst_scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector_>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        const auto dst_forward_steps  = ComputeForwardSteps(dst_desc, dst_scalar_per_access);
        const auto dst_backward_steps = ComputeBackwardSteps(dst_desc, dst_scalar_per_access);

        static_ford<decltype(ordered_dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            constexpr auto forward_sweep =
                ComputeForwardSweep(ordered_dst_access_idx, ordered_dst_access_lengths);

            constexpr auto dst_data_idx = ComputeDataIndex(ordered_dst_access_idx,
                                                           ordered_dst_access_lengths,
                                                           forward_sweep,
                                                           dst_dim_access_order,
                                                           dst_scalar_per_access);

            constexpr auto dst_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<dst_data_idx[i]>{}; }, Number<dst_data_idx.Size()>{});

            const bool is_dst_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(dst_desc, dst_coord_);

            using dst_vector_type = vector_type_maker_t<DstData, DstScalarPerVector>;
            using dst_vector_t    = typename dst_vector_type::type;

            auto dst_vector_container = dst_vector_type{
                dst_thread_scratch_.template GetAsType<dst_vector_t>(dst_data_idx_seq)};

            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                DstData dst_v;
                dst_element_op_(dst_v, dst_vector_container.template AsType<DstData>()[i]);
            });

            dst_buf.template Set<dst_vector_t>(
                dst_coord_.GetOffset() / PackedSize,
                is_dst_valid,
                dst_vector_container.template AsType<dst_vector_t>()[I0]);

            constexpr auto move_on_dim =
                ComputeMoveOnDim(ordered_dst_access_idx, ordered_dst_access_lengths);

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_forward_steps[dst_dim_access_order[i]]);
                    }
                    else
                    {
                        move_tensor_coordinate(
                            dst_desc, dst_coord_, dst_backward_steps[dst_dim_access_order[i]]);
                    }
                }
            });
        });

        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_step =
                make_tensor_coordinate_step(dst_desc, GetDstCoordinateResetStep());
            move_tensor_coordinate(dst_desc, dst_coord_, dst_reset_step);
        }
    }

    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        const auto adjusted_step = make_tensor_coordinate_step(dst_desc, adjusted_step_idx);
        move_tensor_coordinate(dst_desc, dst_coord_, adjusted_step);
    }

    private:
    // TransferData: OOB zeroing + optional sub-dword transpose (same as v3r1)
    template <index_t ThreadScratchId>
    __device__ void
    TransferDataFromSrcThreadScratchToDstThreadScratch(Number<ThreadScratchId> thread_scratch_id)
    {
#if !CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE
        static_ford<SliceLengths>{}([&](auto idx) {
            dst_thread_scratch_(idx) = src_thread_scratch_tuple_[thread_scratch_id][idx];
        });
#else
        // OOB zeroing
        constexpr auto src_scalar_per_access  = GetSrcScalarPerAccess();
        constexpr auto ordered_access_lengths = GetOrderedSrcAccessLengths();
        constexpr auto src_dim_access_order   = SrcDimAccessOrder{};

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_src_access_idx) {
            constexpr auto forward_sweep =
                ComputeForwardSweep(ordered_src_access_idx, ordered_access_lengths);

            constexpr auto src_data_idx = ComputeDataIndex(ordered_src_access_idx,
                                                           ordered_access_lengths,
                                                           forward_sweep,
                                                           src_dim_access_order,
                                                           src_scalar_per_access);

            constexpr auto src_data_idx_seq = generate_sequence_v2(
                [&](auto i) { return Number<src_data_idx[i]>{}; }, Number<src_data_idx.Size()>{});

            using vector_t = typename vector_type_maker<DstData, SrcScalarPerVector>::type::type;

            auto op_r = src_thread_scratch_tuple_(thread_scratch_id)
                            .template GetAsType<vector_t>(src_data_idx_seq);

            const bool is_src_valid = src_oob_thread_scratch_tuple_(thread_scratch_id)
                                          .template GetAsType<bool>(src_data_idx_seq);

            auto op_r_v = is_src_valid ? op_r : vector_t(0);

            src_thread_scratch_tuple_(thread_scratch_id)
                .template SetAsType<vector_t>(src_data_idx_seq, op_r_v);
        });

        // Sub-dword transpose between src and dst scratch buffers (same logic as v3r1)
        if constexpr(SrcVectorDim != DstVectorDim &&
                     ((is_same<half_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 2 == 0 && DstScalarPerVector % 2 == 0) ||
                      (is_same<int8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0) ||
                      (is_same<f8_t, remove_cvref_t<DstData>>::value &&
                       SrcScalarPerVector % 4 == 0 && DstScalarPerVector % 4 == 0)))
        {
            constexpr index_t num_src_vector = Number<DstScalarPerVector>{};
            constexpr index_t num_dst_vector = Number<SrcScalarPerVector>{};

            constexpr auto src_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

            constexpr auto dst_scalar_step_in_vector = generate_sequence(
                detail::lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

            constexpr auto scalar_per_access = generate_sequence(
                detail::lambda_scalar_per_access_for_src_and_dst<SrcVectorDim,
                                                                 SrcScalarPerVector,
                                                                 DstVectorDim,
                                                                 DstScalarPerVector>{},
                Number<nDim>{});

            constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

            static_ford<decltype(access_lengths)>{}([&](auto access_idx) {
                constexpr auto data_idx = access_idx * scalar_per_access;

                constexpr auto data_idx_seq = generate_sequence_v2(
                    [&](auto i) { return Number<data_idx[i]>{}; }, Number<nDim>{});

                using src_vector_t = vector_type_maker_t<DstData, SrcScalarPerVector>;
                using dst_vector_t = vector_type_maker_t<DstData, DstScalarPerVector>;

                const auto src_vector_refs = generate_tie(
                    [&](auto i) -> const src_vector_t& {
                        return src_thread_scratch_tuple_[thread_scratch_id].GetVectorTypeReference(
                            data_idx_seq + i * dst_scalar_step_in_vector);
                    },
                    Number<num_src_vector>{});

                auto dst_vector_refs = generate_tie(
                    [&](auto i) -> dst_vector_t& {
                        return dst_thread_scratch_.GetVectorTypeReference(
                            data_idx_seq + i * src_scalar_step_in_vector);
                    },
                    Number<num_dst_vector>{});

                transpose_vectors<DstData, DstScalarPerVector, SrcScalarPerVector>{}(
                    src_vector_refs, dst_vector_refs);
            });
        }
        else
        {
            constexpr auto packed_per_access = generate_sequence(
                detail::lambda_scalar_per_access<SrcVectorDim, PackedSize>{}, Number<nDim>{});

            constexpr auto packed_access_lengths = SliceLengths{} / packed_per_access;

            static_ford<decltype(packed_access_lengths)>{}([&](auto idx) {
                dst_thread_scratch_(idx) = src_thread_scratch_tuple_[thread_scratch_id][idx];
            });
        }
#endif
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        return ComputeCoordinateResetStep<DstVectorDim, DstScalarPerVector_, DstDimAccessOrder>();
    }

    // --- SFC helpers (same as v3r1) ---

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

    template <typename OrderedAccessIdx,
              typename OrderedAccessLengths,
              typename ForwardSweep,
              typename DimAccessOrder,
              typename ScalarPerAccess>
    __device__ static constexpr auto
    ComputeDataIndex(const OrderedAccessIdx& ordered_access_idx,
                     const OrderedAccessLengths& ordered_access_lengths,
                     const ForwardSweep& forward_sweep,
                     const DimAccessOrder& dim_access_order,
                     const ScalarPerAccess& scalar_per_access)
    {
        Index ordered_idx;
        static_for<0, nDim, 1>{}([&](auto i) {
            ordered_idx(i) = forward_sweep[i]
                                 ? ordered_access_idx[i]
                                 : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
        });
        return container_reorder_given_old2new(ordered_idx, dim_access_order) * scalar_per_access;
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

    template <index_t VectorDim, index_t ScalarPerVector, typename DimAccessOrder>
    __device__ static constexpr auto ComputeCoordinateResetStep()
    {
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector>{}, Number<nDim>{});

        constexpr auto access_lengths   = SliceLengths{} / scalar_per_access;
        constexpr auto dim_access_order = DimAccessOrder{};
        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        constexpr auto ordered_access_lengths_minus_1 = generate_tuple(
            [&](auto i) { return Number<ordered_access_lengths.At(i) - 1>{}; }, Number<nDim>{});
        constexpr auto forward_sweep =
            ComputeForwardSweep(ordered_access_lengths_minus_1, ordered_access_lengths);

        constexpr auto data_idx = [&]() {
            Index ordered_idx;
            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });
            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                   scalar_per_access;
        }();

        constexpr auto reset_data_step = [&]() {
            Index reset_data_step_;
            static_for<0, nDim, 1>{}([&](auto i) { reset_data_step_(i) = -data_idx[i]; });
            return reset_data_step_;
        }();

        return reset_data_step;
    }

    // --- Scratch buffer descriptors and types (same as v3r1) ---

    __device__ static constexpr auto GetSrcThreadScratchDescriptor()
    {
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector_>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        constexpr auto access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(access_lengths), Number<SrcScalarPerVector_>{});

        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(access_lengths_and_vector_length);

        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(access_lengths_and_vector_length[i],
                                   access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == SrcVectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    __device__ static constexpr auto GetDstThreadScratchDescriptor()
    {
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<DstVectorDim, DstScalarPerVector_>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        constexpr auto access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(access_lengths), Number<DstScalarPerVector_>{});

        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(access_lengths_and_vector_length);

        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(access_lengths_and_vector_length[i],
                                   access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == DstVectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss =
            generate_tuple([&](auto i) { return Sequence<i.value>{}; }, Number<nDim>{});

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    static constexpr auto src_thread_scratch_desc_ = decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto src_oob_thread_scratch_desc_ =
        decltype(GetSrcThreadScratchDescriptor()){};
    static constexpr auto dst_thread_scratch_desc_ = decltype(GetDstThreadScratchDescriptor()){};

    using SrcThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             DstData,
                                                             SrcScalarPerVector,
                                                             decltype(src_thread_scratch_desc_),
                                                             true>;

    using SrcOOBThreadScratch =
        StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                        bool,
                                        1,
                                        decltype(src_oob_thread_scratch_desc_),
                                        true>;

    using DstThreadScratch = StaticTensorTupleOfVectorBuffer<AddressSpaceEnum::Vgpr,
                                                             DstData,
                                                             DstScalarPerVector,
                                                             decltype(dst_thread_scratch_desc_),
                                                             true>;

    StaticallyIndexedArray<SrcThreadScratch, NumThreadScratch> src_thread_scratch_tuple_;
    StaticallyIndexedArray<SrcOOBThreadScratch, NumThreadScratch> src_oob_thread_scratch_tuple_;

    DstThreadScratch dst_thread_scratch_;

    DstCoord dst_coord_;
    const SrcElementwiseOperation src_element_op_;
    const DstElementwiseOperation dst_element_op_;
};

} // namespace ck
