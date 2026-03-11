// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "mfma_traits.hpp"
#include "mfma_gfx9.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class MfmaDefaultSelector
 * @brief Implements a default MFMA selector strategy for gfx9 target architectures.
 * This implements the K dimension search strategy to find the largest supported MFMA
 * instruction for the given M/N chunk sizes and datatypes.
 * If no supported instruction is found, falls back to an unsupported pass-through
 implementation.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam ChunkM Chunk M dimension size
 * @tparam ChunkN Chunk N dimension size
 * @tparam ChunkKTest Current Chunk K dimension size to test
 * @tparam CompilerTarget The compiler target
 * @note Here we assume that ChunkKTest is always a power-of-two integer.
 *       The search strategy starts from a maximum ChunkKTest size down to 1u by halving
 *       each time.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t ChunkM,
          uint32_t ChunkN,
          uint32_t ChunkKTest,
          typename CompilerTarget> // TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires(is_gfx9_arch_id(CompilerTarget) && is_power_of_two_integer(ChunkKTest))
struct MfmaDefaultSelector
{
    private:
    // Define our candidate MFMA implementation for the current parameters
    using CandidateOp =
        amdgcn_mma<ADataType,
                   BDataType,
                   CDataType,
                   ChunkM,
                   ChunkN,
                   ChunkKTest,
                   DefaultMfmaCtrlFlags, // By default, let's assume no special flags for MFMA
                   CompilerTarget,
                   MmaOpFamily::DENSE>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, test another smaller ChunkK. If no existing implementations, we will get ChunkK=0u
    // and fall back to the unsupported pass-through implementation.
    using SelectedOp = std::conditional_t<MmaOpTraits<CandidateOp>::IsSupported,
                                          CandidateOp,
                                          typename MfmaDefaultSelector<ADataType,
                                                                       BDataType,
                                                                       CDataType,
                                                                       ChunkM,
                                                                       ChunkN,
                                                                       ChunkKTest / 2u,
                                                                       CompilerTarget>::SelectedOp>;
};

/**
 * @struct MfmaDefaultSelector
 * @brief Implements the base case for the default MFMA selector when no supported instruction is
 * found.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam ChunkM Chunk M dimension size
 * @tparam ChunkN Chunk N dimension size
 * @tparam CompilerTarget The compiler target
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t ChunkM,
          uint32_t ChunkN,
          typename CompilerTarget> // TODO: c++20 amdgcn_target_arch_id CompilerTarget>
struct MfmaDefaultSelector<ADataType, BDataType, CDataType, ChunkM, ChunkN, 1u, CompilerTarget>
{
    // Default unsupported pass-through if no instruction is found
    using SelectedOp =
        amdgcn_mma<ADataType,
                   BDataType,
                   CDataType,
                   ChunkM,
                   ChunkN,
                   1u,
                   DefaultMfmaCtrlFlags, // By default, let's assume no special flags for MFMA
                   CompilerTarget,
                   MmaOpFamily::DENSE>;
};

/**
 * @struct MmaDefaultSelector
 * @brief Implements the gfx9 default MMA selector strategy for wave-wise MMA decomposition.
 * This implements the M/N chunk size search strategy to find the largest supported MFMA
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
          MmaOpFamily OpFamily> // TODO: c++20 amdgcn_target_arch_id CompilerTarget>
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          ChunkM,
                          ChunkN,
                          ChunkK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<enable_if_target_family_gfx9_t<CompilerTarget>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::DENSE>>>
{
    private:
    // Provide the default depth-K search strategy for each class of common MFMA shapes.
    // Start searching from the largest K dimension MFMA shape down to the smallest.
    using CandidateOp4x4 =
        typename MfmaDefaultSelector<ADataType, BDataType, CDataType, 4u, 4u, 4u, CompilerTarget>::
            SelectedOp;
    using CandidateOp16x16 = typename MfmaDefaultSelector<ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          16u,
                                                          16u,
                                                          128u,
                                                          CompilerTarget>::SelectedOp;
    using CandidateOp32x32 = typename MfmaDefaultSelector<ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          32u,
                                                          32u,
                                                          64u,
                                                          CompilerTarget>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp =
        typename MfmaDefaultSelector<ADataType, BDataType, CDataType, 1u, 1u, 1u, CompilerTarget>::
            SelectedOp;

    // Check if each candidate is supported for the given chunk sizes
    // For this case, we require the chunk sizes to be multiples of the MFMA shape
    static constexpr bool IsSupported4x4 =
        MmaOpTraits<CandidateOp4x4>::IsSupported && (ChunkM % CandidateOp4x4::kM == 0u) &&
        (ChunkN % CandidateOp4x4::kN == 0u) && (ChunkK % CandidateOp4x4::kK == 0u);
    static constexpr bool IsSupported16x16 =
        MmaOpTraits<CandidateOp16x16>::IsSupported && (ChunkM % CandidateOp16x16::kM == 0u) &&
        (ChunkN % CandidateOp16x16::kN == 0u) && (ChunkK % CandidateOp16x16::kK == 0u);
    static constexpr bool IsSupported32x32 =
        MmaOpTraits<CandidateOp32x32>::IsSupported && (ChunkM % CandidateOp32x32::kM == 0u) &&
        (ChunkN % CandidateOp32x32::kN == 0u) && (ChunkK % CandidateOp32x32::kK == 0u);

    public:
    // Select the largest supported MFMA operation for the given chunk shape
    using SelectedOp = std::conditional_t<
        IsSupported32x32,
        CandidateOp32x32,
        std::conditional_t<IsSupported16x16,
                           CandidateOp16x16,
                           std::conditional_t<IsSupported4x4, CandidateOp4x4, DefaultOp>>>;
};

} // namespace ck_tile::core::arch::mma
