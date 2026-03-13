// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "mfma_traits.hpp"

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for MFMA on GFX9 targets
 *
 * This specialization implements the MFMA instruction for fp16_t A and B
 * matrices, and fp32_t accumulator matrix, with 16x16x16 fragment sizes.
 *
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for MFMA on GFX950 targets
 *
 * This specialization implements the MFMA instruction for fp16_t A and B
 * matrices, and fp32_t accumulator matrix, with 16x16x32 fragment sizes.
 *
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_f16(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x1f32
// signature: V32fffV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 32u, 64u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 32u, 64u, 1u, 64u, 1, 1, 2, 1, 1, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x1f32(aVec[0], // FINDME manual fix
                                                     bVec[0], // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x1f32
// signature: V32fffV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 32u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 32u, 1u, 64u, 1, 1, 1, 1, 2, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x1f32(aVec[0], // FINDME manual fix
                                                     bVec[0], // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x1f32
// signature: V16fffV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 64u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 64u, 1u, 64u, 1, 1, 4, 1, 1, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x1f32(aVec[0], // FINDME manual fix
                                                     bVec[0], // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x1f32
// signature: V16fffV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 16u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 16u, 1u, 64u, 1, 1, 1, 1, 4, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x1f32(aVec[0], // FINDME manual fix
                                                     bVec[0], // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x1f32
// signature: V4fffV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 4u, 64u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 4u, 64u, 1u, 64u, 1, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x1f32(aVec[0], // FINDME manual fix
                                                   bVec[0], // FINDME manual fix
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x1f32
// signature: V4fffV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 64u, 4u, 1u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 64u, 4u, 1u, 64u, 1, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x1f32(aVec[0], // FINDME manual fix
                                                   bVec[0], // FINDME manual fix
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x2f32
// signature: V16fffV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{KM} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 32u, 32u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 32u, 32u, 2u, 64u, 1, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x2f32(aVec[0], // FINDME manual fix
                                                     bVec[0], // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x4f32
// signature: V4fffV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{KM} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp32_t, fp32_t, fp32_t, 16u, 16u, 4u, 64u, 1, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4f32(aVec[0], // FINDME manual fix
                                                     bVec[0], // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x4f16
// signature: V32fV4hV4hV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x4f16
// signature: V32fV4hV4hV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x4f16
// signature: V16fV4hV4hV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x4f16
// signature: V16fV4hV4hV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x4f16
// signature: V4fV4hV4hV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4f16(aVec,
                                                   bVec,
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x4f16
// signature: V4fV4hV4hV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4f16(aVec,
                                                   bVec,
                                                   cVec,
                                                   static_cast<int>(CtrlFlags::Cbsz),
                                                   static_cast<int>(CtrlFlags::Abid),
                                                   static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x8f16
// signature: V16fV4hV4hV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{4} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 32u, 32u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<fp16_t, fp16_t, fp32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x8f16(aVec,
                                                     bVec,
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_32x32x4i8
// signature: V32iiiV32iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_32x32x4i8(bit_cast<int>(aVec), // FINDME manual fix
                                                    bit_cast<int>(bVec), // FINDME manual fix
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_32x32x4i8
// signature: V32iiiV32iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_32x32x4i8(bit_cast<int>(aVec), // FINDME manual fix
                                                    bit_cast<int>(bVec), // FINDME manual fix
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_16x16x4i8
// signature: V16iiiV16iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_16x16x4i8(bit_cast<int>(aVec), // FINDME manual fix
                                                    bit_cast<int>(bVec), // FINDME manual fix
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_16x16x4i8
// signature: V16iiiV16iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_16x16x4i8(bit_cast<int>(aVec), // FINDME manual fix
                                                    bit_cast<int>(bVec), // FINDME manual fix
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_4x4x4i8
// signature: V4iiiV4iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_4x4x4i8(bit_cast<int>(aVec), // FINDME manual fix
                                                  bit_cast<int>(bVec), // FINDME manual fix
                                                  cVec,
                                                  static_cast<int>(CtrlFlags::Cbsz),
                                                  static_cast<int>(CtrlFlags::Abid),
                                                  static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_4x4x4i8
// signature: V4iiiV4iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_4x4x4i8(bit_cast<int>(aVec), // FINDME manual fix
                                                  bit_cast<int>(bVec), // FINDME manual fix
                                                  cVec,
                                                  static_cast<int>(CtrlFlags::Cbsz),
                                                  static_cast<int>(CtrlFlags::Abid),
                                                  static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_32x32x8i8
// signature: V16iiiV16iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{4} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_32x32x8i8(bit_cast<int>(aVec), // FINDME manual fix
                                                    bit_cast<int>(bVec), // FINDME manual fix
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_16x16x16i8
// signature: V4iiiV4iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{4} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_i32_16x16x16i8(bit_cast<int>(aVec), // FINDME manual fix
                                                     bit_cast<int>(bVec), // FINDME manual fix
                                                     cVec,
                                                     static_cast<int>(CtrlFlags::Cbsz),
                                                     static_cast<int>(CtrlFlags::Abid),
                                                     static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x2bf16
// signature: V32fV2sV2sV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 64u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 64u, 2u, 64u, 2, 1, 2, 1, 1, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x2bf16
// signature: V32fV2sV2sV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 32u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 32u, 2u, 64u, 2, 1, 1, 1, 2, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x2bf16
// signature: V16fV2sV2sV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 64u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 64u, 2u, 64u, 2, 1, 4, 1, 1, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x2bf16
// signature: V16fV2sV2sV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 16u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 16u, 2u, 64u, 2, 1, 1, 1, 4, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x2bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x2bf16
// signature: V4fV2sV2sV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 4u, 64u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 4u, 64u, 2u, 64u, 2, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x2bf16(aVec,
                                                    bVec,
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x2bf16
// signature: V4fV2sV2sV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 4u, 2u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 4u, 2u, 64u, 2, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x2bf16(aVec,
                                                    bVec,
                                                    cVec,
                                                    static_cast<int>(CtrlFlags::Cbsz),
                                                    static_cast<int>(CtrlFlags::Abid),
                                                    static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x4bf16
// signature: V16fV2sV2sV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{2} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 4u, 64u, 2, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x8bf16
// signature: V4fV2sV2sV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{2} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 8u, 64u, 2, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x8bf16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x4bf16_1k
// signature: V32fV4sV4sV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 64u, 4u, 64u, 4, 1, 2, 1, 1, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x4bf16_1k
// signature: V32fV4sV4sV32fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{2, 4} L{M1N} V{BM2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 32u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 32u, 4u, 64u, 4, 1, 1, 1, 2, 32, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x4bf16_1k
// signature: V16fV4sV4sV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 64u, 4u, 64u, 4, 1, 4, 1, 1, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x4bf16_1k
// signature: V16fV4sV4sV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=M{4} L{M1N} V{BM0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 16u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 16u, 4u, 64u, 4, 1, 1, 1, 4, 16, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x4bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x4bf16_1k
// signature: V4fV4sV4sV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 4u, 64u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 4u, 64u, 4u, 64u, 4, 1, 16, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_4x4x4bf16_1k
// signature: V4fV4sV4sV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=L{BM} V{K} B=same C/D=L{BN} V{M}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 64u, 4u, 4u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 64u, 4u, 4u, 64u, 4, 1, 1, 1, 16, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x8bf16_1k
// signature: V16fV4sV4sV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{4} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 32u, 32u, 8u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 32u, 32u, 8u, 64u, 4, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(aVec,
                                                         bVec,
                                                         cVec,
                                                         static_cast<int>(CtrlFlags::Cbsz),
                                                         static_cast<int>(CtrlFlags::Abid),
                                                         static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x16bf16_1k
// signature: V4fV4sV4sV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{4} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna2_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf16_t, bf16_t, fp32_t, 16u, 16u, 16u, 64u, 4, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aVec,
                                                          bVec,
                                                          cVec,
                                                          static_cast<int>(CtrlFlags::Cbsz),
                                                          static_cast<int>(CtrlFlags::Abid),
                                                          static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_16x16x32_i8
// signature: V4iWiWiV4iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {
            __builtin_amdgcn_mfma_i32_16x16x32_i8(bit_cast<long long>(aVec), // FINDME manual fix
                                                  bit_cast<long long>(bVec), // FINDME manual fix
                                                  cVec,
                                                  static_cast<int>(CtrlFlags::Cbsz),
                                                  static_cast<int>(CtrlFlags::Abid),
                                                  static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_i32_32x32x16_i8
// signature: V16iWiWiV16iIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<int8_t, int8_t, int32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<int8_t, int8_t, int32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {
            __builtin_amdgcn_mfma_i32_32x32x16_i8(bit_cast<long long>(aVec), // FINDME manual fix
                                                  bit_cast<long long>(bVec), // FINDME manual fix
                                                  cVec,
                                                  static_cast<int>(CtrlFlags::Cbsz),
                                                  static_cast<int>(CtrlFlags::Abid),
                                                  static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8
// signature: V4fWiWiV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8
// signature: V4fWiWiV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8
// signature: V4fWiWiV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8
// signature: V4fWiWiV4fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{4} L{M1N} V{M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 16u, 16u, 32u, 64u, 8, 1, 1, 1, 1, 4, 1, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8
// signature: V16fWiWiV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8
// signature: V16fWiWiV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<bf8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8
// signature: V16fWiWiV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, bf8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

// __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8
// signature: V16fWiWiV16fIiIiIi
// flags: CBSZ, ABID, BLGP
// layouts: A=K{8} L{K1M} V{K0} B=same C/D=M{2, 4} L{M1N} V{M2M0}
template <typename CtrlFlags, typename CompilerTarget>
// clang-format off
//               | A B C DataTypes      | MNK + WaveSize    |AParams |BPar |CPar |
struct amdgcn_mma<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, CtrlFlags, CompilerTarget, MmaOpFamily::DENSE, enable_if_target_cdna3_or_higher_t<CompilerTarget>>
: amdgcn_mma_base<fp8_t, fp8_t, fp32_t, 32u, 32u, 16u, 64u, 8, 1, 1, 1, 1, 16, 4, MfmaOp, MmaOpFamily::DENSE>
// clang-format on
{
    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
            bit_cast<long long>(aVec), // FINDME manual fix
            bit_cast<long long>(bVec), // FINDME manual fix
            cVec,
            static_cast<int>(CtrlFlags::Cbsz),
            static_cast<int>(CtrlFlags::Abid),
            static_cast<int>(CtrlFlags::Blgp))};
    }
};

} // namespace ck_tile::core::arch::mma
