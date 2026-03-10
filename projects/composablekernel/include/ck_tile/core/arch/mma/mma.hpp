// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

#include "amdgcn_mma.hpp"
#include "mma_selector.hpp"
#include "mma_transforms.hpp"

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"

namespace ck_tile::core::arch::mma {

/*! @enum MmaAccumPolicy
 * @brief Accumulation order for Mma decomposition
 */
enum struct MmaAccumPolicy
{
    // Decomposition and accumulation in row-major block order
    ROW_MAJOR,
    // Decomposition and accumulation in col-major block order
    COL_MAJOR
};

/**
 * @class Mma
 * @brief Driver for the wave-tile Mma operation. Given a backend block-wise MmaOp implementation
 * (e.g., mfma or wmma), this class performs block-wise decomposition to matrix-multiply input
 * chunks of (A: ChunkM x ChunkK) x (B: ChunkK x ChunkN) and accumulates results into output chunk
 * (C: ChunkM x ChunkN).
 * @tparam ADataType Data type of input chunk A
 * @tparam BDataType Data type of input chunk B
 * @tparam CDataType Data type of input/output chunk C (accumulator)
 * @tparam ChunkM Mma chunk M dimension
 * @tparam ChunkN Mma chunk K dimension
 * @tparam ChunkK Mma chunk M dimension
 * @tparam AccumPolicy The block order of the accumulation registers (row major or col major block
 * order)
 * @tparam CompilerTarget The compiler target
 * @tparam MmaOp The backend wrapper class that will perform block-wise mma op (e.g., mfma or
 * wmma)
 * @tparam MmaTransforms The set of transforms to be applied to input/output chunks
 * @par This is an example of an Mma decomposition driver class that can be used in a wave-tile
 * context. Given a chunk size, we can decompose the chunk into smaller block-wise mma ops
 * that are natively supported by the hardware (e.g., mfma or wmma). The class also supports
 * applying transforms to the input/output chunks as needed (e.g., layout conversions, data type
 * conversions, etc.). We may also specify the accumulation order (row-major or col-major) for the
 * output chunk. This is a powerful example of how to build a flexible and reusable mma driver
 * that can adapt to different hardware capabilities and requirements.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t ChunkM,
          uint32_t ChunkN,
          uint32_t ChunkK,
          MmaOpFamily OpFamily,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          ChunkM,
                                          ChunkN,
                                          ChunkK,
                                          CompilerTarget,
                                          OpFamily>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp, CompilerTarget>::SelectedTransforms>
struct WaveWiseMma
{
    using BlockWiseMmaOp = MmaOp;

    // Block dimensions
    constexpr static uint32_t FragM = MmaOp::kM;
    constexpr static uint32_t FragN = MmaOp::kN;
    constexpr static uint32_t FragK = MmaOp::kK;

    // Block counts for decomposition
    constexpr static uint32_t BlocksM = ChunkM / FragM;
    constexpr static uint32_t BlocksN = ChunkN / FragN;
    constexpr static uint32_t BlocksK = ChunkK / FragK;
    constexpr static uint32_t BlocksC = BlocksM * BlocksN;

    // Vector types for packed registers in each block
    using AVecType = typename MmaOp::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    // Buffer types for chunks
    using ABufferType = AVecType[BlocksM][BlocksK];
    using BBufferType = BVecType[BlocksN][BlocksK];
    using CBufferType = CVecType[BlocksM][BlocksN];

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(ChunkM >= FragM, "ChunkM must be larger than FragM");
    static_assert(ChunkN >= FragN, "ChunkN must be larger than FragN");
    static_assert(ChunkK >= FragK, "ChunkK must be larger than FragK");
    static_assert(ChunkM % FragM == 0u, "ChunkM must be a multiple of FragM");
    static_assert(ChunkN % FragN == 0u, "ChunkN must be a multiple of FragN");
    static_assert(ChunkK % FragK == 0u, "ChunkK must be a multiple of FragK");

    private:
    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT const& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input chunks to the native vector types
        // required by the BlockWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(SrcT), "Size mismatch in formatBuffer");
        return reinterpret_cast<DstT const&>(inputBuffer);
    }

    template <typename DstT, typename SrcT>
    CK_TILE_DEVICE static auto formatBuffer(SrcT& inputBuffer)
    {
        // TODO: Implement formatting logic as needed.
        // This is intended to convert input chunks to the native vector types
        // required by the BlockWiseMma operation for iteration
        static_assert(sizeof(DstT) == sizeof(SrcT), "Size mismatch in formatBuffer");
        return reinterpret_cast<DstT&>(inputBuffer);
    }

    /*! @brief Execute Mma in row-major accumulation order.
     * @tparam VecTA The input chunk A vector type
     * @tparam VecTB The input chunk B vector type
     * @tparam VecTC The input/output chunk C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec_col_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // We implement an example wave-tile pipeline here.
        // First, we apply the necessary transforms to the input chunks,
        // then we convert the result into buffers of native vector formats
        // that we can easily index. Native vector formats are necessary inputs
        // to the given MmaOp exec function.
        auto a_chunk = formatBuffer<ABufferType>(ATransform::exec(a));
        auto b_chunk = formatBuffer<BBufferType>(BTransform::exec(b));
        auto c_chunk = formatBuffer<CBufferType>(CTransform::exec(accum));

        // "Col-major" accumulation over the M-dimension blocks first.
        // Pseudo code here, but we would basically iterate over the blocks in col-major order
        for(uint32_t bn = 0u; bn < BlocksN; ++bn)
        {
            for(uint32_t bm = 0u; bm < BlocksM; ++bm)
            {
                for(uint32_t bk = 0u; bk < BlocksK; ++bk)
                {
                    c_chunk[bm][bn] =
                        BlockWiseMmaOp::exec(a_chunk[bm][bk], b_chunk[bn][bk], c_chunk[bm][bn]);
                }
            }
        }

        // Convert native vector results back to the output chunk format
        // and then return after we apply the final output transform.
        return DTransform::exec(formatBuffer<std::decay_t<VecTC>>(c_chunk));
    }

    /*! @brief Execute Mma in row-major accumulation order.
     * @tparam VecTA The input chunk A vector type
     * @tparam VecTB The input chunk B vector type
     * @tparam VecTC The input/output chunk C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec_row_major(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        // We implement an example wave-tile pipeline here.
        // First, we apply the necessary transforms to the input chunks,
        // then we convert the result into buffers of native vector formats
        // that we can easily index. Native vector formats are necessary inputs
        // to the given MmaOp exec function.
        auto a_chunk = formatBuffer<ABufferType>(ATransform::exec(a));
        auto b_chunk = formatBuffer<BBufferType>(BTransform::exec(b));
        auto c_chunk = formatBuffer<CBufferType>(CTransform::exec(accum));

        // "Row-major" accumulation over the N-dimension blocks first.
        // Pseudo code here, but we would basically iterate over the blocks in row-major order.
        // We also have to ensure that the incoming vector chunks are converted to native vector
        // types before passing to the BlockWiseMma exec function.
        for(uint32_t bm = 0u; bm < BlocksM; ++bm)
        {
            for(uint32_t bn = 0u; bn < BlocksN; ++bn)
            {
                for(uint32_t bk = 0u; bk < BlocksK; ++bk)
                {
                    c_chunk[bm][bn] =
                        BlockWiseMmaOp::exec(a_chunk[bm][bk], b_chunk[bn][bk], c_chunk[bm][bn]);
                }
            }
        }

        // Convert native vector results back to the output chunk format
        // and then return after we apply the final output transform.
        return DTransform::exec(formatBuffer<std::decay_t<VecTC>>(c_chunk));
    }

    public:
    /*! @brief Forward to Mma operation with specified accumulation order.
     * @tparam VecTA The input chunk A vector type
     * @tparam VecTB The input chunk B vector type
     * @tparam VecTC The input/output chunk C vector type
     */
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
        {
            return exec_row_major(
                std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        }
        else // if constexpr(AccumPolicy == MmaAccumPolicy::COL_MAJOR)
        {
            return exec_col_major(
                std::forward<VecTA>(a), std::forward<VecTB>(b), std::forward<VecTC>(accum));
        }
    }
};

} // namespace ck_tile::core::arch::mma
