// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_op_family.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/ignore.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck_tile::core::arch::mma {

// TODO: Describe layout params.
/**
 *  @class  amdgcn_mma_base
 *  @brief  Helper base class for amdgcn_mma structs to avoid a lot of code duplication. Also puts
 *          all generic parameter derivations and static asserts in one place. Houses all of the
 *          amdgcn struct types and variables, except for the exec() function.
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          uint32_t WaveSize_,
          index_t kABKPerLane_,
          index_t kAKNumAccess_,
          index_t kARepeat_,
          index_t kBKNumAccess_,
          index_t kBRepeat_,
          index_t kCMPerLane_,
          index_t kCMNumAccess_,
          typename OpType_,
          MmaOpFamily OpFamily_>
struct amdgcn_mma_base
{
    using OpType                          = OpType_;
    static constexpr MmaOpFamily OpFamily = OpFamily_;

    // Data types
    using ADataType = ADataType_;
    using BDataType = BDataType_;
    using CDataType = CDataType_;

    // Fragment sizes
    static constexpr index_t kM = BlockM;
    static constexpr index_t kN = BlockN;
    static constexpr index_t kK = BlockK;

    // Layout constants
    static constexpr index_t kABKPerLane  = kABKPerLane_;
    static constexpr index_t kAKNumAccess = kAKNumAccess_;
    static constexpr index_t kARepeat     = kARepeat_;
    static constexpr index_t kBKNumAccess = kBKNumAccess_;
    static constexpr index_t kBRepeat     = kBRepeat_;
    static constexpr index_t kCMPerLane   = kCMPerLane_;
    static constexpr index_t kCMNumAccess = kCMNumAccess_;

    // Register types (derived)
    static constexpr index_t WaveSize = WaveSize_;
    static_assert((kM * kK * kARepeat) % WaveSize == 0);
    static_assert((kN * kK * kBRepeat) % WaveSize == 0);
    static_assert((kM * kN) % WaveSize == 0);

    using AVecType = ext_vector_t<ADataType, kM * kK * kARepeat / WaveSize>;
    using BVecType = ext_vector_t<BDataType, kN * kK * kBRepeat / WaveSize>;
    using CVecType = ext_vector_t<CDataType, kM * kN / WaveSize>;
};

/**
 * @struct Unsupported
 * @brief  Meta-tag to indicate unsupported amdgcn_mma instance.
 */
struct Unsupported;

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

#include <concepts>
/**
 * @concept MmaOpI
 * @brief  Expresses the meta-data interface required for each MmaOp policy.
 */
template <typename MmaOp>
concept MmaOpI = requires(MmaOp op) {
    // Requires an op context
    typename MmaOp::OpType;
    typename MmaOp::OpFamily;

    // Captures types for inputs / outputs to mma function
    typename MmaOp::ADataType;
    typename MmaOp::BDataType;
    typename MmaOp::CDataType;
    typename MmaOp::AVecType;
    typename MmaOp::BVecType;
    typename MmaOp::CVecType;

    // Captures CK-specific layout properties
    { MmaOp::kABKPerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kAKNumAccess } -> std::convertible_to<unsigned int>;
    { MmaOp::kARepeat } -> std::convertible_to<unsigned int>;
    { MmaOp::kBKNumAccess } -> std::convertible_to<unsigned int>;
    { MmaOp::kBRepeat } -> std::convertible_to<unsigned int>;
    { MmaOp::kCMPerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCMNumAccess } -> std::convertible_to<unsigned int>;

    // Static exec function
    {
        MmaOp::exec(
            typename MmaOp::AVecType{}, typename MmaOp::BVecType{}, typename MmaOp::CVecType{})
    } -> std::convertible_to<typename MmaOp::CVecType>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 *  @class  amdgcn_mma
 *  @brief  This is the default MmaOp policy.
 *          Instances of this class are to be used as MmaOp policies.
 *          Light builtin wrapper for mfma / wmma instructions. This class's job is to
 *          provide a uniform interface to invoke the appropriate instruction
 *          based on the template parameters provided. This interface is to bridge
 *          the gap between the ck_tile API types and the native __builtin types.
 *  @tparam ADataType Datatype of input A
 *  @tparam BDataType Datatype of input B
 *  @tparam CDataType Datatype of accumulator
 *  @tparam BlockM M-dimension of mma block
 *  @tparam BlockN N-dimension of mma block
 *  @tparam BlockK K-dimension of mma block
 *  @tparam CtrlFlags Control flags for mma operation
 *  @tparam CompilerTarget The current compiler target
 *  @tparam Enabler SFINAE enabler
 */
// clang-format off
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename CtrlFlags,
          typename CompilerTarget,
          MmaOpFamily OpFamily_,
          typename Enabler = void>
struct amdgcn_mma : amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 1u, 1u, 1u, 1u, 1, 1, 1, 1, 1, 1, 1, Unsupported, MmaOpFamily::UNDEFINED>
{
    // This is a default pass-through implementation that doesn't do anything practical.
    CK_TILE_DEVICE static CVecType const&
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        ignore(regsA, regsB);
        return regsC; // No-op, just return C
    }
};
// clang-format on

} // namespace ck_tile::core::arch::mma
#pragma clang diagnostic pop

// Include the implementations
#include "wmma/wmma.hpp"
#include "mfma/mfma.hpp"
#include "sparse/sparse.hpp"
