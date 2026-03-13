// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test: verifies that the offset-compute → offset-load round-trip produces
// bit-identical results to v3r1's RunRead → RunWrite path.
//
// Strategy: single-thread GPU kernel that:
// 1. Fills global memory with sequential values
// 2. Creates a 3D descriptor [K0, M, K1] (with padding on M)
// 3. v3r1 path: RunRead from global → RunWrite to output A
// 4. Offset path: OffsetCompute → packed offsets → OffsetLoad RunReadFromOffsets → RunWrite to
// output B
// 5. Compare outputs A vs B on host

#include <iostream>
#include <vector>
#include <cstdlib>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v3r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_offset_compute.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_offset_load.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

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

using EwPassThrough = ck::tensor_operation::element_wise::PassThrough;
using F16           = half_t;

// ===================================================================
// Test kernel: 3D descriptor [K0=4, M_padded=128(valid=64), K1=8]
// SliceLengths = <4, 32, 8>
// SrcAccessOrder = <1, 0, 2>, SrcVectorDim = 2, SrcScalarPerVector = 8
// DstAccessOrder = <0, 1, 2>, DstVectorDim = 2, DstScalarPerVector = 8
// ===================================================================

// Source descriptor builder (global memory with padding)
__device__ auto make_src_desc_3d()
{
    constexpr index_t K0       = 4;
    constexpr index_t M        = 64;
    constexpr index_t M_Padded = 128;
    constexpr index_t K1       = 8;

    const auto desc0 = make_naive_tensor_descriptor(
        make_tuple(Number<K0>{}, Number<M_Padded>{}, Number<K1>{}),
        make_tuple(Number<M_Padded * K1>{}, Number<K1>{}, Number<1>{}));

    return transform_tensor_descriptor(
        desc0,
        make_tuple(make_pass_through_transform(Number<K0>{}),
                   make_right_pad_transform(Number<M>{}, Number<M_Padded - M>{}),
                   make_pass_through_transform(Number<K1>{})),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
}

// Destination descriptor builder (simple packed for output comparison)
__device__ auto make_dst_desc_3d()
{
    constexpr index_t K0 = 4;
    constexpr index_t M  = 32; // same as SliceLengths M dim
    constexpr index_t K1 = 8;

    return make_naive_tensor_descriptor_packed(make_tuple(Number<K0>{}, Number<M>{}, Number<K1>{}));
}

using SrcDescType = decltype(make_src_desc_3d());
using DstDescType = decltype(make_dst_desc_3d());

__global__ void kernel_test_offset_load(const F16* d_src,
                                        F16* d_out_v3r1,
                                        F16* d_out_offset,
                                        int32_t* d_num_mismatches)
{
    using SliceLengths                   = Sequence<4, 32, 8>;
    using SrcDimAccessOrder              = Sequence<1, 0, 2>;
    using DstDimAccessOrder              = Sequence<0, 1, 2>;
    constexpr index_t SrcVectorDim       = 2;
    constexpr index_t DstVectorDim       = 2;
    constexpr index_t SrcScalarPerVector = 8;
    constexpr index_t DstScalarPerVector = 8;

    const auto src_desc = make_src_desc_3d();
    const auto dst_desc = make_dst_desc_3d();

    const auto src_buf =
        make_dynamic_buffer<AddressSpaceEnum::Global>(d_src, src_desc.GetElementSpaceSize());

    const auto src_origin = make_multi_index(0, 0, 0);
    const auto dst_origin = make_multi_index(0, 0, 0);

    // ---------------------------------------------------------------
    // Path 1: v3r1 RunRead → RunWrite
    // ---------------------------------------------------------------
    {
        auto out_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            d_out_v3r1, dst_desc.GetElementSpaceSize());

        auto v3r1_transfer = ThreadwiseTensorSliceTransfer_v3r1<SliceLengths,
                                                                EwPassThrough,
                                                                EwPassThrough,
                                                                InMemoryDataOperationEnum::Set,
                                                                F16,
                                                                F16,
                                                                decltype(src_desc),
                                                                decltype(dst_desc),
                                                                SrcDimAccessOrder,
                                                                DstDimAccessOrder,
                                                                SrcVectorDim,
                                                                DstVectorDim,
                                                                SrcScalarPerVector,
                                                                DstScalarPerVector,
                                                                1,
                                                                1,
                                                                true, // SrcResetCoordinateAfterRun
                                                                true, // DstResetCoordinateAfterRun
                                                                1>(
            src_desc, src_origin, EwPassThrough{}, dst_desc, dst_origin, EwPassThrough{});

        v3r1_transfer.RunRead(src_desc, src_buf);
        v3r1_transfer.RunWrite(dst_desc, out_buf);
    }

    // ---------------------------------------------------------------
    // Path 2: OffsetCompute → packed offsets → OffsetLoad → RunWrite
    // ---------------------------------------------------------------
    {
        auto out_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            d_out_offset, dst_desc.GetElementSpaceSize());

        // Step 2a: Compute offsets
        using OffsetComputeType = ThreadwiseTensorSliceTransfer_OffsetCompute<SliceLengths,
                                                                              decltype(src_desc),
                                                                              SrcDimAccessOrder,
                                                                              SrcVectorDim,
                                                                              SrcScalarPerVector,
                                                                              true>;

        auto offset_compute          = OffsetComputeType(src_desc, src_origin);
        constexpr index_t num_access = OffsetComputeType::NumAccessPoints;

        int32_t packed_offsets[num_access];
        offset_compute.RunComputePackedOffsets(src_desc, packed_offsets);

        // Step 2b: Load from offsets and write to output
        using OffsetLoadType =
            ThreadwiseTensorSliceTransfer_OffsetLoad<SliceLengths,
                                                     EwPassThrough,
                                                     EwPassThrough,
                                                     InMemoryDataOperationEnum::Set,
                                                     F16,
                                                     F16,
                                                     decltype(dst_desc),
                                                     SrcDimAccessOrder,
                                                     DstDimAccessOrder,
                                                     SrcVectorDim,
                                                     DstVectorDim,
                                                     SrcScalarPerVector,
                                                     DstScalarPerVector,
                                                     1,
                                                     1,
                                                     true, // DstResetCoordinateAfterRun
                                                     1>;

        auto offset_load = OffsetLoadType(EwPassThrough{}, dst_desc, dst_origin, EwPassThrough{});

        offset_load.RunReadFromOffsets(src_buf, packed_offsets);
        offset_load.RunWrite(dst_desc, out_buf);
    }

    // ---------------------------------------------------------------
    // Compare (on-device, quick check)
    // ---------------------------------------------------------------
    constexpr index_t output_size = 4 * 32 * 8; // K0 * M_slice * K1
    int32_t mismatches            = 0;
    for(index_t i = 0; i < output_size; ++i)
    {
        if(d_out_v3r1[i] != d_out_offset[i])
            ++mismatches;
    }
    *d_num_mismatches = mismatches;
}

// ===================================================================
// Test kernel 2: 2D descriptor [K0=8, N=128], no padding
// SliceLengths = <8, 16>, SrcAccessOrder=<0,1>, VectorDim=1, ScalarPerVec=4
// ===================================================================

__device__ auto make_src_desc_2d()
{
    return make_naive_tensor_descriptor_packed(make_tuple(Number<8>{}, Number<128>{}));
}

__device__ auto make_dst_desc_2d()
{
    return make_naive_tensor_descriptor_packed(make_tuple(Number<8>{}, Number<16>{}));
}

__global__ void kernel_test_offset_load_2d(const F16* d_src,
                                           F16* d_out_v3r1,
                                           F16* d_out_offset,
                                           int32_t* d_num_mismatches)
{
    using SliceLengths                   = Sequence<8, 16>;
    using SrcDimAccessOrder              = Sequence<0, 1>;
    using DstDimAccessOrder              = Sequence<0, 1>;
    constexpr index_t SrcVectorDim       = 1;
    constexpr index_t DstVectorDim       = 1;
    constexpr index_t SrcScalarPerVector = 4;
    constexpr index_t DstScalarPerVector = 4;

    const auto src_desc = make_src_desc_2d();
    const auto dst_desc = make_dst_desc_2d();

    const auto src_buf =
        make_dynamic_buffer<AddressSpaceEnum::Global>(d_src, src_desc.GetElementSpaceSize());

    const auto src_origin = make_multi_index(0, 0);
    const auto dst_origin = make_multi_index(0, 0);

    // Path 1: v3r1
    {
        auto out_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            d_out_v3r1, dst_desc.GetElementSpaceSize());

        auto v3r1_transfer = ThreadwiseTensorSliceTransfer_v3r1<SliceLengths,
                                                                EwPassThrough,
                                                                EwPassThrough,
                                                                InMemoryDataOperationEnum::Set,
                                                                F16,
                                                                F16,
                                                                decltype(src_desc),
                                                                decltype(dst_desc),
                                                                SrcDimAccessOrder,
                                                                DstDimAccessOrder,
                                                                SrcVectorDim,
                                                                DstVectorDim,
                                                                SrcScalarPerVector,
                                                                DstScalarPerVector,
                                                                1,
                                                                1,
                                                                true,
                                                                true,
                                                                1>(
            src_desc, src_origin, EwPassThrough{}, dst_desc, dst_origin, EwPassThrough{});

        v3r1_transfer.RunRead(src_desc, src_buf);
        v3r1_transfer.RunWrite(dst_desc, out_buf);
    }

    // Path 2: OffsetCompute → OffsetLoad
    {
        auto out_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            d_out_offset, dst_desc.GetElementSpaceSize());

        using OffsetComputeType = ThreadwiseTensorSliceTransfer_OffsetCompute<SliceLengths,
                                                                              decltype(src_desc),
                                                                              SrcDimAccessOrder,
                                                                              SrcVectorDim,
                                                                              SrcScalarPerVector,
                                                                              true>;

        auto oc                      = OffsetComputeType(src_desc, src_origin);
        constexpr index_t num_access = OffsetComputeType::NumAccessPoints;

        int32_t packed_offsets[num_access];
        oc.RunComputePackedOffsets(src_desc, packed_offsets);

        using OffsetLoadType =
            ThreadwiseTensorSliceTransfer_OffsetLoad<SliceLengths,
                                                     EwPassThrough,
                                                     EwPassThrough,
                                                     InMemoryDataOperationEnum::Set,
                                                     F16,
                                                     F16,
                                                     decltype(dst_desc),
                                                     SrcDimAccessOrder,
                                                     DstDimAccessOrder,
                                                     SrcVectorDim,
                                                     DstVectorDim,
                                                     SrcScalarPerVector,
                                                     DstScalarPerVector,
                                                     1,
                                                     1,
                                                     true,
                                                     1>;

        auto ol = OffsetLoadType(EwPassThrough{}, dst_desc, dst_origin, EwPassThrough{});
        ol.RunReadFromOffsets(src_buf, packed_offsets);
        ol.RunWrite(dst_desc, out_buf);
    }

    // Compare
    constexpr index_t output_size = 8 * 16;
    int32_t mismatches            = 0;
    for(index_t i = 0; i < output_size; ++i)
    {
        if(d_out_v3r1[i] != d_out_offset[i])
            ++mismatches;
    }
    *d_num_mismatches = mismatches;
}

// ===================================================================
// Host runner
// ===================================================================

template <typename KernelFunc>
bool run_test(const char* name, KernelFunc kernel, index_t src_size, index_t output_size)
{
    std::cout << "Test: " << name << std::endl;

    // Prepare source data: sequential half_t values
    std::vector<F16> h_src(src_size);
    for(index_t i = 0; i < src_size; ++i)
        h_src[i] = static_cast<F16>(static_cast<float>(i % 256));

    F16 *d_src, *d_out_v3r1, *d_out_offset;
    int32_t* d_mis;

    HIP_CHECK(hipMalloc(&d_src, src_size * sizeof(F16)));
    HIP_CHECK(hipMalloc(&d_out_v3r1, output_size * sizeof(F16)));
    HIP_CHECK(hipMalloc(&d_out_offset, output_size * sizeof(F16)));
    HIP_CHECK(hipMalloc(&d_mis, sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(d_src, h_src.data(), src_size * sizeof(F16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out_v3r1, 0, output_size * sizeof(F16)));
    HIP_CHECK(hipMemset(d_out_offset, 0, output_size * sizeof(F16)));
    HIP_CHECK(hipMemset(d_mis, 0, sizeof(int32_t)));

    kernel<<<1, 1>>>(d_src, d_out_v3r1, d_out_offset, d_mis);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    int32_t h_mis = 0;
    HIP_CHECK(hipMemcpy(&h_mis, d_mis, sizeof(int32_t), hipMemcpyDeviceToHost));

    if(h_mis > 0)
    {
        std::cerr << "  FAIL: " << h_mis << " mismatches out of " << output_size << std::endl;

        // Print first few mismatches
        std::vector<F16> h_v3r1(output_size), h_offset(output_size);
        HIP_CHECK(
            hipMemcpy(h_v3r1.data(), d_out_v3r1, output_size * sizeof(F16), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(
            h_offset.data(), d_out_offset, output_size * sizeof(F16), hipMemcpyDeviceToHost));

        int printed = 0;
        for(index_t i = 0; i < output_size && printed < 10; ++i)
        {
            if(h_v3r1[i] != h_offset[i])
            {
                std::cerr << "    [" << i << "] v3r1=" << static_cast<float>(h_v3r1[i])
                          << " offset=" << static_cast<float>(h_offset[i]) << std::endl;
                ++printed;
            }
        }
    }
    else
    {
        std::cout << "  PASS: all " << output_size << " elements match" << std::endl;
    }

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out_v3r1));
    HIP_CHECK(hipFree(d_out_offset));
    HIP_CHECK(hipFree(d_mis));

    return h_mis == 0;
}

int main()
{
    bool all_pass = true;

    // Test 1: 3D padded descriptor, src_size = K0*M_padded*K1 = 4*128*8 = 4096
    //         output_size = K0*M_slice*K1 = 4*32*8 = 1024
    all_pass &=
        run_test("3D_Padded [4x128x8, valid=64, slice=32]", kernel_test_offset_load, 4096, 1024);

    // Test 2: 2D simple, src_size = 8*128 = 1024, output_size = 8*16 = 128
    all_pass &= run_test("2D_Simple [8x128, slice=16]", kernel_test_offset_load_2d, 1024, 128);

    if(all_pass)
    {
        std::cout << "\nAll offset load tests PASSED" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "\nSome offset load tests FAILED" << std::endl;
        return 1;
    }
}
