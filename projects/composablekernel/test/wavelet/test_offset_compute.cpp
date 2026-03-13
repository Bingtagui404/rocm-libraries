// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test: verifies that ThreadwiseTensorSliceTransfer_OffsetCompute produces
// the same offsets as the v3r1 SFC iteration pattern for various descriptor shapes.

#include <iostream>
#include <vector>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_offset_compute.hpp"

#include "hip/hip_runtime.h"

#define HIP_CHECK(call)                                                                      \
    do                                                                                       \
    {                                                                                        \
        hipError_t err = (call);                                                             \
        if(err != hipSuccess)                                                                \
        {                                                                                    \
            std::cerr << "HIP error " << hipGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                              \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while(0)

using namespace ck;

// ===================================================================
// SFC helpers (mirrors v3r1 internals for the reference path)
// ===================================================================

template <index_t nDim>
struct SfcHelpers
{
    template <typename OrderedAccessIdx, typename OrderedAccessLengths>
    __device__ static constexpr auto
    ComputeForwardSweep(const OrderedAccessIdx& ordered_access_idx,
                        const OrderedAccessLengths& ordered_access_lengths)
    {
        StaticallyIndexedArray<bool, nDim> forward_sweep_;
        forward_sweep_(Number<0>{}) = true;
        static_for<1, nDim, 1>{}([&](auto i) {
            index_t tmp = ordered_access_idx[Number<0>{}];
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
};

// ===================================================================
// GPU kernels: test offset compute for specific descriptor configs
// ===================================================================

// Test 1: 3D descriptor [K0=4, M=128(padded from 64), K1=8]
// SliceLengths=<4,32,8>, AccessOrder=<1,0,2>, VectorDim=2, ScalarPerVec=8, Reset=true
__global__ void kernel_test_3d_padded(int32_t* d_ref, int32_t* d_oc, int32_t* d_mismatches)
{
    constexpr index_t K0       = 4;
    constexpr index_t M        = 64;
    constexpr index_t M_Padded = 128;
    constexpr index_t K1       = 8;

    using SliceLengths                   = Sequence<4, 32, 8>;
    using SrcDimAccessOrder              = Sequence<1, 0, 2>;
    constexpr index_t SrcVectorDim       = 2;
    constexpr index_t SrcScalarPerVector = 8;
    constexpr bool ResetAfterRun         = true;
    constexpr index_t NumSteps           = 2;
    constexpr index_t nDim               = 3;
    using Index                          = MultiIndex<nDim>;
    using Helpers                        = SfcHelpers<nDim>;

    // Build descriptor
    const auto desc0 = make_naive_tensor_descriptor(
        make_tuple(Number<K0>{}, Number<M_Padded>{}, Number<K1>{}),
        make_tuple(Number<M_Padded * K1>{}, Number<K1>{}, Number<1>{}));

    const auto src_desc = transform_tensor_descriptor(
        desc0,
        make_tuple(make_pass_through_transform(Number<K0>{}),
                   make_right_pad_transform(Number<M>{}, Number<M_Padded - M>{}),
                   make_pass_through_transform(Number<K1>{})),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

    const auto slice_origin = make_multi_index(0, 0, 0);
    const auto step_index   = make_multi_index(4, 0, 0);

    // --- OffsetCompute path ---
    using OffsetComputeType = ThreadwiseTensorSliceTransfer_OffsetCompute<SliceLengths,
                                                                          decltype(src_desc),
                                                                          SrcDimAccessOrder,
                                                                          SrcVectorDim,
                                                                          SrcScalarPerVector,
                                                                          ResetAfterRun>;

    auto oc                         = OffsetComputeType(src_desc, slice_origin);
    constexpr index_t num_access    = OffsetComputeType::NumAccessPoints;
    constexpr index_t total_offsets = num_access * (NumSteps + 1);

    int32_t oc_packed[num_access];

    for(index_t step = 0; step <= NumSteps; ++step)
    {
        oc.RunComputePackedOffsets(src_desc, oc_packed);
        for(index_t i = 0; i < num_access; ++i)
            d_oc[step * num_access + i] = oc_packed[i];
        if(step < NumSteps)
            oc.MoveSrcSliceWindow(src_desc, step_index);
    }

    // --- Reference SFC path ---
    auto ref_coord = make_tensor_coordinate(src_desc, slice_origin);

    constexpr auto scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});
    constexpr auto access_lengths   = SliceLengths{} / scalar_per_access;
    constexpr auto dim_access_order = SrcDimAccessOrder{};
    constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    for(index_t step = 0; step <= NumSteps; ++step)
    {
        const auto fwd_steps = generate_tuple(
            [&](auto d) {
                Index idx;
                static_for<0, nDim, 1>{}(
                    [&](auto j) { idx(j) = (d.value == j.value) ? scalar_per_access[d] : 0; });
                return make_tensor_coordinate_step(src_desc, idx);
            },
            Number<nDim>{});

        const auto bwd_steps = generate_tuple(
            [&](auto d) {
                Index idx;
                static_for<0, nDim, 1>{}(
                    [&](auto j) { idx(j) = (d.value == j.value) ? -scalar_per_access[d] : 0; });
                return make_tensor_coordinate_step(src_desc, idx);
            },
            Number<nDim>{});

        index_t ref_idx = 0;

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_idx) {
            const bool valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, ref_coord);
            index_t off    = ref_coord.GetOffset();
            int32_t packed = static_cast<int32_t>(off);
            packed         = valid ? (packed | static_cast<int32_t>(0x80000000u)) : packed;
            d_ref[step * num_access + ref_idx] = packed;
            ++ref_idx;

            constexpr auto fw = Helpers::ComputeForwardSweep(ordered_idx, ordered_access_lengths);
            constexpr auto md = Helpers::ComputeMoveOnDim(ordered_idx, ordered_access_lengths);

            static_for<0, nDim, 1>{}([&](auto ii) {
                if constexpr(md[ii])
                {
                    if constexpr(fw[ii])
                        move_tensor_coordinate(
                            src_desc, ref_coord, fwd_steps[dim_access_order[ii]]);
                    else
                        move_tensor_coordinate(
                            src_desc, ref_coord, bwd_steps[dim_access_order[ii]]);
                }
            });
        });

        if(step < NumSteps)
        {
            // v3r1 behavior: ResetAfterRun=true -> reset coord, then apply step
            const auto reset_step = make_tensor_coordinate_step(src_desc, [&]() {
                constexpr auto olm1 = generate_tuple(
                    [&](auto i) { return Number<ordered_access_lengths.At(i) - 1>{}; },
                    Number<nDim>{});
                constexpr auto lfw = Helpers::ComputeForwardSweep(olm1, ordered_access_lengths);
                Index reset_idx;
                static_for<0, nDim, 1>{}([&](auto ii) {
                    index_t last_pos = lfw[ii] ? (ordered_access_lengths[ii] - 1) : index_t(0);
                    Index ordered;
                    static_for<0, nDim, 1>{}([&](auto jj) { ordered(jj) = 0; });
                    ordered(ii) = last_pos;
                    // This simplified approach won't work — need full computation
                    ignore = ordered;
                    ignore = last_pos;
                });
                // Just use the same approach as OffsetCompute's reset
                constexpr auto data_idx_inner = [&]() {
                    Index oi;
                    static_for<0, nDim, 1>{}(
                        [&](auto ii) { oi(ii) = lfw[ii] ? ordered_access_lengths[ii] - 1 : 0; });
                    return container_reorder_given_old2new(oi, dim_access_order) *
                           scalar_per_access;
                }();
                static_for<0, nDim, 1>{}([&](auto ii) { reset_idx(ii) = -data_idx_inner[ii]; });
                return reset_idx;
            }());
            move_tensor_coordinate(src_desc, ref_coord, reset_step);

            const auto window_step = make_tensor_coordinate_step(src_desc, step_index);
            move_tensor_coordinate(src_desc, ref_coord, window_step);
        }
    }

    // Compare
    int32_t mismatches = 0;
    for(index_t i = 0; i < total_offsets; ++i)
    {
        if(d_ref[i] != d_oc[i])
            ++mismatches;
    }
    *d_mismatches = mismatches;
}

// Test 2: 2D descriptor [K0=8, N=128], no padding
// SliceLengths=<8,16>, AccessOrder=<0,1>, VectorDim=1, ScalarPerVec=4, Reset=false
__global__ void kernel_test_2d_simple(int32_t* d_ref, int32_t* d_oc, int32_t* d_mismatches)
{
    constexpr index_t K0 = 8;
    constexpr index_t N  = 128;

    using SliceLengths                   = Sequence<8, 16>;
    using SrcDimAccessOrder              = Sequence<0, 1>;
    constexpr index_t SrcVectorDim       = 1;
    constexpr index_t SrcScalarPerVector = 4;
    constexpr bool ResetAfterRun         = false;
    constexpr index_t NumSteps           = 1;
    constexpr index_t nDim               = 2;
    using Index                          = MultiIndex<nDim>;
    using Helpers                        = SfcHelpers<nDim>;

    const auto src_desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<K0>{}, Number<N>{}));

    const auto slice_origin = make_multi_index(0, 0);
    const auto step_index   = make_multi_index(8, 0);

    // --- OffsetCompute ---
    using OffsetComputeType = ThreadwiseTensorSliceTransfer_OffsetCompute<SliceLengths,
                                                                          decltype(src_desc),
                                                                          SrcDimAccessOrder,
                                                                          SrcVectorDim,
                                                                          SrcScalarPerVector,
                                                                          ResetAfterRun>;

    auto oc                         = OffsetComputeType(src_desc, slice_origin);
    constexpr index_t num_access    = OffsetComputeType::NumAccessPoints;
    constexpr index_t total_offsets = num_access * (NumSteps + 1);

    int32_t oc_packed[num_access];

    for(index_t step = 0; step <= NumSteps; ++step)
    {
        oc.RunComputePackedOffsets(src_desc, oc_packed);
        for(index_t i = 0; i < num_access; ++i)
            d_oc[step * num_access + i] = oc_packed[i];
        if(step < NumSteps)
            oc.MoveSrcSliceWindow(src_desc, step_index);
    }

    // --- Reference SFC ---
    auto ref_coord = make_tensor_coordinate(src_desc, slice_origin);

    constexpr auto scalar_per_access = generate_sequence(
        detail::lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});
    constexpr auto access_lengths   = SliceLengths{} / scalar_per_access;
    constexpr auto dim_access_order = SrcDimAccessOrder{};
    constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    // Precompute reset step
    constexpr auto olm1 = generate_tuple(
        [&](auto i) { return Number<ordered_access_lengths.At(i) - 1>{}; }, Number<nDim>{});
    constexpr auto last_fw       = Helpers::ComputeForwardSweep(olm1, ordered_access_lengths);
    constexpr auto last_data_idx = [&]() {
        Index oi;
        static_for<0, nDim, 1>{}(
            [&](auto ii) { oi(ii) = last_fw[ii] ? ordered_access_lengths[ii] - 1 : 0; });
        return container_reorder_given_old2new(oi, dim_access_order) * scalar_per_access;
    }();
    constexpr auto reset_step_idx = [&]() {
        Index rs;
        static_for<0, nDim, 1>{}([&](auto ii) { rs(ii) = -last_data_idx[ii]; });
        return rs;
    }();

    for(index_t step = 0; step <= NumSteps; ++step)
    {
        const auto fwd_steps = generate_tuple(
            [&](auto d) {
                Index idx;
                static_for<0, nDim, 1>{}(
                    [&](auto j) { idx(j) = (d.value == j.value) ? scalar_per_access[d] : 0; });
                return make_tensor_coordinate_step(src_desc, idx);
            },
            Number<nDim>{});

        const auto bwd_steps = generate_tuple(
            [&](auto d) {
                Index idx;
                static_for<0, nDim, 1>{}(
                    [&](auto j) { idx(j) = (d.value == j.value) ? -scalar_per_access[d] : 0; });
                return make_tensor_coordinate_step(src_desc, idx);
            },
            Number<nDim>{});

        index_t ref_idx = 0;

        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_idx) {
            const bool valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, ref_coord);
            index_t off    = ref_coord.GetOffset();
            int32_t packed = static_cast<int32_t>(off);
            packed         = valid ? (packed | static_cast<int32_t>(0x80000000u)) : packed;
            d_ref[step * num_access + ref_idx] = packed;
            ++ref_idx;

            constexpr auto fw = Helpers::ComputeForwardSweep(ordered_idx, ordered_access_lengths);
            constexpr auto md = Helpers::ComputeMoveOnDim(ordered_idx, ordered_access_lengths);

            static_for<0, nDim, 1>{}([&](auto ii) {
                if constexpr(md[ii])
                {
                    if constexpr(fw[ii])
                        move_tensor_coordinate(
                            src_desc, ref_coord, fwd_steps[dim_access_order[ii]]);
                    else
                        move_tensor_coordinate(
                            src_desc, ref_coord, bwd_steps[dim_access_order[ii]]);
                }
            });
        });

        if(step < NumSteps)
        {
            // ResetAfterRun=false: fuse reset + step
            const auto fused_step_idx = step_index + reset_step_idx;
            const auto fused_step     = make_tensor_coordinate_step(src_desc, fused_step_idx);
            move_tensor_coordinate(src_desc, ref_coord, fused_step);
        }
    }

    int32_t mismatches = 0;
    for(index_t i = 0; i < total_offsets; ++i)
    {
        if(d_ref[i] != d_oc[i])
            ++mismatches;
    }
    *d_mismatches = mismatches;
}

// ===================================================================
// Host runner
// ===================================================================

bool run_kernel(const char* name,
                void (*kernel)(int32_t*, int32_t*, int32_t*),
                index_t total_offsets)
{
    std::cout << "Test: " << name << " (TotalOffsets=" << total_offsets << ")" << std::endl;

    int32_t *d_ref, *d_oc, *d_mis;
    HIP_CHECK(hipMalloc(&d_ref, total_offsets * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_oc, total_offsets * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_mis, sizeof(int32_t)));
    HIP_CHECK(hipMemset(d_ref, 0, total_offsets * sizeof(int32_t)));
    HIP_CHECK(hipMemset(d_oc, 0, total_offsets * sizeof(int32_t)));
    HIP_CHECK(hipMemset(d_mis, 0, sizeof(int32_t)));

    kernel<<<1, 1>>>(d_ref, d_oc, d_mis);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int32_t> h_ref(total_offsets), h_oc(total_offsets);
    int32_t h_mis = 0;
    HIP_CHECK(
        hipMemcpy(h_ref.data(), d_ref, total_offsets * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_oc.data(), d_oc, total_offsets * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_mis, d_mis, sizeof(int32_t), hipMemcpyDeviceToHost));

    if(h_mis > 0)
    {
        std::cerr << "  FAIL: " << h_mis << " mismatches" << std::endl;
        int printed = 0;
        for(index_t i = 0; i < total_offsets && printed < 10; ++i)
        {
            if(h_ref[i] != h_oc[i])
            {
                std::cerr << "    [" << i << "] ref=" << (h_ref[i] & 0x7FFFFFFF)
                          << (h_ref[i] < 0 ? " valid" : " OOB") << "  oc=" << (h_oc[i] & 0x7FFFFFFF)
                          << (h_oc[i] < 0 ? " valid" : " OOB") << std::endl;
                ++printed;
            }
        }
    }
    else
    {
        std::cout << "  PASS: all " << total_offsets << " offsets match" << std::endl;
    }

    HIP_CHECK(hipFree(d_ref));
    HIP_CHECK(hipFree(d_oc));
    HIP_CHECK(hipFree(d_mis));

    return h_mis == 0;
}

int main()
{
    bool all_pass = true;

    // Test 1: 3D padded, 128 access points, 3 steps => 384 offsets
    all_pass &= run_kernel("3D_Padded [4x128x8, valid=64]", kernel_test_3d_padded, 384);

    // Test 2: 2D simple, 32 access points, 2 steps => 64 offsets
    all_pass &= run_kernel("2D_Simple [8x128]", kernel_test_2d_simple, 64);

    if(all_pass)
    {
        std::cout << "\nAll offset compute tests PASSED" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "\nSome offset compute tests FAILED" << std::endl;
        return 1;
    }
}
