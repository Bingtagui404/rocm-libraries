// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_traits.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class SparseMfmaDefaultSelector
 * @brief Implements a default sparse MFMA selector strategy. The SelectedOp can be unsupported.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam FragM Size of the M dimension
 * @tparam FragN Size of the N dimension
 * @tparam FragKTest Size of the K dimension
 * @tparam CompilerTarget The compiler target
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragKTest,
          typename CompilerTarget>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires(is_target_arch_cdna(CompilerTarget) && is_power_of_two_integer(FragKTest))
struct SparseMfmaDefaultSelector
{
    private:
    // Define our candidate MFMA implementation for the current parameters
    using CandidateOp = amdgcn_mma<ADataType,
                                   BDataType,
                                   CDataType,
                                   FragM,
                                   FragN,
                                   FragKTest,
                                   DefaultSparseMfmaCtrlFlags,
                                   CompilerTarget,
                                   MmaOpFamily::SPARSE>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, fall back to the unsupported pass-through implementation.
    using SelectedOp = std::conditional_t<MmaOpTraits<CandidateOp>::IsSupported,
                                          CandidateOp,
                                          amdgcn_mma<ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     FragM,
                                                     FragN,
                                                     FragKTest,
                                                     void,
                                                     amdgcn_target<>,
                                                     MmaOpFamily::UNDEFINED>>;
};

/**
 * @struct MmaDefaultSelector
 * @brief Implements the CDNA default MMA selector strategy for sparse MFMA.
 * If no supported instruction is found, falls back to an unsupported pass-through implementation.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 * @tparam FragM Size of the M dimension of the fragment to decompose
 * @tparam FragN Size of the N dimension of the fragment to decompose
 * @tparam FragK Size of the K dimension of the fragment to decompose
 * @tparam CompilerTarget The compiler target
 * @tparam OpFamily The MMA operation family
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          typename CompilerTarget,
          MmaOpFamily OpFamily>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          FragM,
                          FragN,
                          FragK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<std::enable_if_t<is_any_value_of(CompilerTarget::TARGET_ID,
                                                                         amdgcn_target_id::GFX942,
                                                                         amdgcn_target_id::GFX950)>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::SPARSE>>>
{
    private:
    // Provide the default depth-K search strategy for each class of common MFMA shapes.
    // Start searching from the largest K dimension MFMA shape down to the smallest.
    using CandidateOp16x16 = typename SparseMfmaDefaultSelector<ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                16u,
                                                                16u,
                                                                32u,
                                                                CompilerTarget>::SelectedOp;
    using CandidateOp32x32 = typename SparseMfmaDefaultSelector<ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                32u,
                                                                32u,
                                                                64u,
                                                                CompilerTarget>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp = typename SparseMfmaDefaultSelector<ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         1u,
                                                         1u,
                                                         1u,
                                                         CompilerTarget>::SelectedOp;

    // Check if each candidate is supported for the given fragment sizes
    // For this case, we require the fragment sizes to be multiples of the MFMA shape
    static constexpr bool IsSupported16x16 = MmaOpTraits<CandidateOp16x16>::IsSupported && 
                                            (FragM % CandidateOp16x16::kM == 0u) &&
                                            (FragN % CandidateOp16x16::kN == 0u) && 
                                            (FragK % CandidateOp16x16::kK == 0u);
    static constexpr bool IsSupported32x32 = MmaOpTraits<CandidateOp32x32>::IsSupported && 
                                            (FragM % CandidateOp32x32::kM == 0u) &&
                                            (FragN % CandidateOp32x32::kN == 0u) && 
                                            (FragK % CandidateOp32x32::kK == 0u);

    public:
    // Select the largest supported MFMA operation for the given fragment shape
    using SelectedOp =
        std::conditional_t<IsSupported32x32,
                           CandidateOp32x32,
                           std::conditional_t<IsSupported16x16, CandidateOp16x16, DefaultOp>>;
};

} // namespace ck_tile::core::arch::mma
