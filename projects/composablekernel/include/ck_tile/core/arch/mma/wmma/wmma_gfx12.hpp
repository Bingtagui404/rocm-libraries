// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "wmma_traits.hpp"

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_wmma for fp16_t, fp16_t, fp32_t MMA operation on GFX12
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx12I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<fp16_t,
                  fp16_t,
                  fp32_t,
                  16u,
                  16u,
                  16u,
                  CtrlFlags,
                  CompilerTarget,
                  MmaOpFamily::DENSE,
                  enable_if_target_family_gfx12_t<CompilerTarget>>
{
    using OpType                          = WmmaOp;
    static constexpr MmaOpFamily OpFamily = MmaOpFamily::DENSE;

    // Data types
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = fp32_t;

    // Fragment sizes
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // Layout constants
    static constexpr index_t kABKPerLane  = 8;
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
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aVec, bVec, cVec)};
    }
};

} // namespace ck_tile::core::arch::mma
