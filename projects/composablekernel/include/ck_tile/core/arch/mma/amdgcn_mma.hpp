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
struct amdgcn_mma
{
    // The base instance is unsupported because there is no __builtin to wrap.
    using OpType                          = Unsupported;
    static constexpr MmaOpFamily OpFamily = MmaOpFamily::UNDEFINED;

    // Fragment sizes - default to 1
    static constexpr index_t kM = 1;
    static constexpr index_t kN = 1;
    static constexpr index_t kK = 1;

    // Layout constants - default to 1
    static constexpr index_t kABKPerLane  = 1;
    static constexpr index_t kAKNumAccess = 1;
    static constexpr index_t kARepeat     = 1;
    static constexpr index_t kBKNumAccess = 1;
    static constexpr index_t kBRepeat     = 1;
    static constexpr index_t kCMPerLane   = 1;
    static constexpr index_t kCMNumAccess = 1;

    // Register types for A, B, C vectors
    using AVecType = ext_vector_t<ADataType, 1>;
    using BVecType = ext_vector_t<BDataType, 1>;
    using CVecType = ext_vector_t<CDataType, 1>;

    // This is a default pass-through implementation that doesn't do anything practical.
    CK_TILE_DEVICE static CVecType const&
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        ignore(regsA, regsB);
        return regsC; // No-op, just return C
    }
};

} // namespace ck_tile::core::arch::mma
#pragma clang diagnostic pop

// Include the implementations
#include "wmma/wmma.hpp"
#include "mfma/mfma.hpp"
#include "sparse/sparse.hpp"
