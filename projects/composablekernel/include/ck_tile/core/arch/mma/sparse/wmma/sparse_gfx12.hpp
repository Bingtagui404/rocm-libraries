// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

namespace ck_tile::core::arch::mma {

// TODO: c++20 template <CtrlFlagsSparseWmmaI CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<fp16_t,
                  fp16_t,
                  fp32_t,
                  16u,
                  16u,
                  32u,
                  CtrlFlags,
                  CompilerTarget,
                  MmaOpFamily::SPARSE,
                  enable_if_target_family_gfx12_t<CompilerTarget>>
{
    using OpType                          = WmmaOp;
    static constexpr MmaOpFamily OpFamily = MmaOpFamily::SPARSE;

    // Data types
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = fp32_t;

    // Fragment sizes
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 32;

    // Layout constants
    static constexpr index_t kABKPerLane  = 16;
    static constexpr index_t kAKNumAccess = 1;
    static constexpr index_t kARepeat     = 1;
    static constexpr index_t kBKNumAccess = 1;
    static constexpr index_t kBRepeat     = 1;
    static constexpr index_t kCMPerLane   = 8;
    static constexpr index_t kCMNumAccess = 1;

    // Register types (derived)
    static constexpr index_t waveSize = static_cast<index_t>(CompilerTarget::WAVE_SIZE_ID);
    static_assert((kM * kK * kARepeat) % waveSize == 0);
    static_assert((kN * kK * kBRepeat) % waveSize == 0);
    static_assert((kM * kN) % waveSize == 0);

    using AVecType = ext_vector_t<ADataType, kM * kK * kARepeat / waveSize>;
    using BVecType = ext_vector_t<BDataType, kN * kK * kBRepeat / waveSize>;
    using CVecType = ext_vector_t<CDataType, kM * kN / waveSize>;

    CK_TILE_DEVICE static auto
    exec(AVecType& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        static constexpr index_t AVecN             = vector_traits<AVecType>::vector_size;
        static constexpr index_t kCompressionRatio = 2;
        static constexpr index_t CompressedSize    = AVecN / kCompressionRatio;
        using AVecCompressed                       = ext_vector_t<fp16_t, CompressedSize>;

        static_assert(CompressedSize == 8);
        // TODO: Compressing A on-the-fly should be OK for now, but we need to validate
        // and evaluate changing this to a transform at a higher level.
        // aVec not being const can cause problems when running multiple intrinsics.
        const int32_t idx = ck_tile::compress_a_impl<fp16_t, CompressedSize>(aVec);

        const AVecCompressed a_vec_pruned = {
            aVec[0], aVec[1], aVec[2], aVec[3], aVec[4], aVec[5], aVec[6], aVec[7]};

        return {__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a_vec_pruned, bVec, cVec, idx)};
    }
};

} // namespace ck_tile::core::arch::mma
