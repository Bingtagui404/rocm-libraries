// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"

#include "get_wave_size_helper.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

template <typename AType,
          typename BType,
          typename CType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK>
__global__ void test_pipeline(void* a, void* b, void* c)
{
    using CompilerTarget = decltype(get_compiler_target());

    using Pipeline = WaveWiseMma<AType,
                                 BType,
                                 CType,
                                 FragM,
                                 FragN,
                                 FragK,
                                 MmaOpFamily::DENSE,
                                 MmaAccumPolicy::ROW_MAJOR,
                                 CompilerTarget>;

    using MmaOpTraits             = typename Pipeline::BlockWiseMmaOpTraits;
    using AVecType                = typename MmaOpTraits::AVecType;
    using BVecType                = typename MmaOpTraits::BVecType;
    using CVecType                = typename MmaOpTraits::CVecType;
    constexpr static auto BlocksM = Pipeline::BlocksM;
    constexpr static auto BlocksN = Pipeline::BlocksN;
    constexpr static auto BlocksK = Pipeline::BlocksK;
    Pipeline::exec(*reinterpret_cast<AVecType(*)[BlocksM][BlocksK]>(a),
                   *reinterpret_cast<BVecType(*)[BlocksN][BlocksK]>(b),
                   *reinterpret_cast<CVecType(*)[BlocksM][BlocksN]>(c));
}

TEST(WaveWiseMmaPipeline, testKIter)
{
    int devCount;
    hipDevice_t dev;
    HIP_CHECK_ERROR(hipGetDevice(&dev));
    HIP_CHECK_ERROR(hipGetDeviceCount(&devCount));

    hipDeviceProp_t devProp;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&devProp, dev));

    auto currentArchId = hip_device_prop_gcn_arch_name_to_amdgcn_target_id(devProp.gcnArchName);
    bool hasDevice     = static_cast<bool>(devCount > 0);
    int deviceWarpSize = devProp.warpSize;

    bool isSupportedWmma = false;
    bool isSupportedMfma =
        (currentArchId >= amdgcn_target_id::GFX942) && (currentArchId <= amdgcn_target_id::GFX950);
    // TODO: c++20 add check for arch id
    if(!hasDevice || (currentArchId == amdgcn_target_id::HOST) ||
       !(isSupportedWmma || isSupportedMfma))
    {
        GTEST_SKIP() << "No HIP device found. Skipping test.";
    }

    using AType = fp16_t;
    using BType = fp16_t;
    using CType = fp32_t;

    // Fragment size, also the expected block size from the selector.
    // Note: Actual blockK might be slightly different due to hardware implementation, but the
    // test_accum_over_k kernel will loop over the K dimension to ensure that the total K is
    // correct.
    static constexpr uint32_t FragM  = 16;
    static constexpr uint32_t FragN  = 16;
    static constexpr uint32_t FragK  = 32;
    static constexpr uint32_t BlockM = FragM;
    static constexpr uint32_t BlockN = FragN;
    static constexpr uint32_t BlockK = FragK;

    // The number of elements per thread
    uint32_t AElements = BlockM * BlockK / deviceWarpSize;
    uint32_t BElements = BlockN * BlockK / deviceWarpSize;
    uint32_t CElements = BlockM * BlockN / deviceWarpSize;

    uint32_t ASize = AElements * sizeof(AType);
    uint32_t BSize = BElements * sizeof(BType);
    uint32_t CSize = CElements * sizeof(CType);

    // Initialize A and B to all 1's, C to all 0's
    std::vector<AType> h_a(AElements, static_cast<AType>(1));
    std::vector<BType> h_b(BElements, static_cast<BType>(1));
    std::vector<CType> h_c(CElements, static_cast<CType>(0));
    std::vector<CType> h_out(CElements, static_cast<CType>(0));

    AType* d_a;
    BType* d_b;
    CType* d_c;
    CType* d_out;

    HIP_CHECK_ERROR(hipMalloc(&d_a, ASize));
    HIP_CHECK_ERROR(hipMalloc(&d_b, BSize));
    HIP_CHECK_ERROR(hipMalloc(&d_c, CSize));
    HIP_CHECK_ERROR(hipMalloc(&d_out, CSize));

    // Copy inputs to device
    HIP_CHECK_ERROR(hipMemcpy(d_a, h_a.data(), ASize, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_b, h_b.data(), BSize, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_c, h_c.data(), CSize, hipMemcpyHostToDevice));

    const auto wave_size = getDeviceWaveSize();
    test_pipeline<AType, BType, CType, FragM, FragN, FragK><<<1, wave_size>>>(d_a, d_b, d_c);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    HIP_CHECK_ERROR(hipMemcpy(h_out.data(), d_out, CSize, hipMemcpyDeviceToHost));

    // Output should be FragK for all elements, because the inputs are all 1's
    for(size_t i = 0; i < CElements; ++i)
    {
        CType expected = static_cast<CType>(FragK);

        EXPECT_NEAR(h_out[i], expected, 1e-3);
    }

    HIP_CHECK_ERROR(hipFree(d_a));
    HIP_CHECK_ERROR(hipFree(d_b));
    HIP_CHECK_ERROR(hipFree(d_c));
    HIP_CHECK_ERROR(hipFree(d_out));
}
