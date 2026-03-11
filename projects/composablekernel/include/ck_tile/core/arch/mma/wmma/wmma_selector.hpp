// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class WmmaDefaultSelector
 * @brief Implements a default WMMA selector strategy for gfx11/12 target architectures.
 * This implements the K dimension search strategy to find the largest supported WMMA
 * instruction for the given M/N chunk sizes and datatypes.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam ChunkM Size of the M dimension
 * @tparam ChunkN Size of the N dimension
 * @tparam ChunkKTest Size of the K dimension
 * @tparam CompilerTarget The compiler target
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t ChunkM,
          uint32_t ChunkN,
          uint32_t ChunkKTest,
          typename CompilerTarget>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires(is_rdna_arch_id(CompilerTarget) && is_power_of_two_integer(ChunkKTest))
struct WmmaDefaultSelector
{
    private:
    // By default, let's assume no special flags for WMMA
    using CtrlFlags = DefaultWmmaCtrlFlags<ADataType, BDataType, CDataType>;

    // Define our candidate WMMA implementation for the current parameters
    using CandidateOp = amdgcn_mma<ADataType,
                                   BDataType,
                                   CDataType,
                                   ChunkM,
                                   ChunkN,
                                   ChunkKTest,
                                   CtrlFlags,
                                   CompilerTarget,
                                   MmaOpFamily::DENSE>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, test another smaller ChunkK. If no existing implementations, we will get ChunkK=0u
    // and fall back to the unsupported pass-through implementation.
    using SelectedOp = std::conditional_t<MmaOpTraits<CandidateOp>::IsSupported,
                                          CandidateOp,
                                          typename WmmaDefaultSelector<ADataType,
                                                                       BDataType,
                                                                       CDataType,
                                                                       ChunkM,
                                                                       ChunkN,
                                                                       ChunkKTest / 2u,
                                                                       CompilerTarget>::SelectedOp>;
};

/**
 * @struct WmmaDefaultSelector
 * @brief Implements a default WMMA selector strategy for gfx11/12 target architectures.
 * This implements the K dimension == 1, which is the base case for the recursive K dimension
 * search. If no supported instruction is found, falls back to an unsupported pass-through
 * implementation.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam ChunkM Size of the M dimension
 * @tparam ChunkN Size of the N dimension
 * @tparam CompilerTarget The compiler target
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t ChunkM,
          uint32_t ChunkN,
          typename CompilerTarget>
// TODO: c++20 amdgcn_target_arch_id GfxTargetId>
struct WmmaDefaultSelector<ADataType, BDataType, CDataType, ChunkM, ChunkN, 1u, CompilerTarget>
{
    // By default, let's assume no special flags for WMMA
    using CtrlFlags = DefaultWmmaCtrlFlags<ADataType, BDataType, CDataType>;

    // Default unsupported pass-through if no instruction is found
    using SelectedOp = amdgcn_mma<ADataType,
                                  BDataType,
                                  CDataType,
                                  ChunkM,
                                  ChunkN,
                                  1u,
                                  CtrlFlags,
                                  CompilerTarget,
                                  MmaOpFamily::DENSE>;
};

/**
 * @struct MmaDefaultSelector
 * @brief Implements the rdna default MMA selector strategy for wave-wise MMA decomposition.
 * This implements the M/N chunk size search strategy to find the largest supported WMMA
 * instruction for the given datatypes.
 * If no supported instruction is found, falls back to an unsupported pass-through implementation.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam ChunkM Size of the M dimension of the chunk to decompose
 * @tparam ChunkN Size of the N dimension of the chunk to decompose
 * @tparam ChunkK Size of the K dimension of the chunk to decompose
 * @tparam CompilerTarget The compiler target
 * @tparam OpFamily The MMA operation family
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t ChunkM,
          uint32_t ChunkN,
          uint32_t ChunkK,
          typename CompilerTarget,
          MmaOpFamily OpFamily>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          ChunkM,
                          ChunkN,
                          ChunkK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<enable_if_target_arch_rdna_t<CompilerTarget>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::DENSE>>>
{
    private:
    // Provide the default depth-K search strategy for each class of common WMMA shapes.
    // Start searching from the largest K dimension MFMA shape down to the smallest.
    using CandidateOp16x16 = typename WmmaDefaultSelector<ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          16u,
                                                          16u,
                                                          128u,
                                                          CompilerTarget>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp =
        typename WmmaDefaultSelector<ADataType, BDataType, CDataType, 1u, 1u, 1u, CompilerTarget>::
            SelectedOp;

    // Check if each candidate is supported for the given chunk sizes
    // For this case, we require the chunk sizes to be multiples of the WMMA shape
    static constexpr bool IsSupported16x16 =
        MmaOpTraits<CandidateOp16x16>::IsSupported && (ChunkM % CandidateOp16x16::kM == 0u) &&
        (ChunkN % CandidateOp16x16::kN == 0u) && (ChunkK % CandidateOp16x16::kK == 0u);

    public:
    // Select the largest supported WMMA operation for the given chunk shape
    using SelectedOp = std::conditional_t<IsSupported16x16, CandidateOp16x16, DefaultOp>;
};

} // namespace ck_tile::core::arch::mma
