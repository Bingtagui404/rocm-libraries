// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

// Maximum Global Memory throughput pipeline with >=32KB data in fly
// GlobalPrefetchStages: >=2
// LocalPreFillStages: 1
// LocalPreFetchStages: 0
// LocalSharedMemoryBuffer: 1
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompAsyncDefaultPolicy>
struct GemmPipelineAgBgCrMemAsync : public BaseGemmPipelineAgBgCrMem<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrMem<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");
    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr bool Async = true;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "MEMORY";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrMe",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    using Base::PrefetchStages;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // // A/B tiles in LDS
            // ADataType* __restrict__ p_a_lds = static_cast<ADataType*>(p_smem);
            // constexpr auto a_lds_block_desc =
            //     Policy::template MakeALdsBlockDescriptor<Problem, ADataType>();
            // auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds,
            // a_lds_block_desc);

            // // TODO: LDS alignment should come from Policy!
            // constexpr index_t a_lds_block_space_size_aligned = integer_least_multiple(
            //     sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16);

            // // B tile in LDS
            // BDataType* __restrict__ p_b_lds = static_cast<BDataType*>(
            //     static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
            // constexpr auto b_lds_block_desc = Policy::template
            // MakeBLdsBlockDescriptor<Problem>(); auto b_lds_block =
            // make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

            // using ALDSBlock = decltype(a_lds_block);
            // using BLDSBlock = decltype(b_lds_block);
            // tuple_array<ALDSBlock, PrefetchStages> a_lds_blocks;
            // tuple_array<BLDSBlock, PrefetchStages> b_lds_blocks;
            // a_lds_blocks.at(I0{}) = a_lds_block;
            // b_lds_blocks.at(I0{}) = b_lds_block;
            // static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
            //     auto&& [a_lds_block_idx, b_lds_block_idx] =
            //         Base::GetABLdsTensorViews(static_cast<char*>(p_smem) +
            //                                   prefetch_idx * Policy::template
            //                                   GetSmemSize<Problem>());
            //     a_lds_blocks.at(prefetch_idx) = a_lds_block_idx;
            //     b_lds_blocks.at(prefetch_idx) = b_lds_block_idx;
            // });

            // A DRAM tile window(s) for load
            auto a_tile_windows =
                make_tile_window(a_dram_block_window_tmp[I0{}].get_bottom_tensor_view(),
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 a_dram_block_window_tmp[I0{}].get_window_origin(),
                                 Policy::template MakeADramTileDistribution<Problem>());
            // B DRAM window(s) for load
            auto b_tile_windows =
                make_tile_window(b_dram_block_window_tmp[I0{}].get_bottom_tensor_view(),
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 b_dram_block_window_tmp[I0{}].get_window_origin(),
                                 Policy::template MakeBDramTileDistribution<Problem>());

            // set up LDS tile shapes
            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
                else
                    return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
            }();

            constexpr auto b_lds_shape = []() {
                if constexpr(is_b_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
                else
                    return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
            }();

            // LDS tile windows for storing, one per LDS buffer
            auto a_copy_lds_window = make_tile_window(a_lds_block, a_lds_shape, {0, 0});
            auto b_copy_lds_window = make_tile_window(b_lds_block, b_lds_shape, {0, 0});

            // using ALDSWindow = decltype(make_tile_window(a_lds_block, a_lds_shape, {0, 0}));
            // using BLDSWindow = decltype(make_tile_window(b_lds_block, b_lds_shape, {0, 0}));

            // tuple_array<ALDSWindow, PrefetchStages> a_copy_lds_windows;
            // tuple_array<BLDSWindow, PrefetchStages> b_copy_lds_windows;
            // static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
            //     a_copy_lds_windows.at(prefetch_idx) =
            //         make_tile_window(a_lds_blocks.at(prefetch_idx), a_lds_shape, {0, 0});
            //     b_copy_lds_windows.at(prefetch_idx) =
            //         make_tile_window(b_lds_blocks.at(prefetch_idx), b_lds_shape, {0, 0});
            // });

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // tile distribution for the register tiles
            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            // register tiles
            tuple_array<ALdsTile, PrefetchStages> a_block_tiles;
            tuple_array<BLdsTile, PrefetchStages> b_block_tiles;

            // ALdsTile a_block_tile;
            // BLdsTile b_block_tile;

            constexpr auto a_lds_input_tile_distr = [ALdsTileDistr]() {
                if constexpr(is_a_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(ALdsTileDistr)::DstrEncode,
                            typename Problem::ADataType>::TransposedDstrEncode{});
                else
                    return ALdsTileDistr;
            }();
            constexpr auto b_lds_input_tile_distr = [BLdsTileDistr]() {
                if constexpr(is_b_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(BLdsTileDistr)::DstrEncode,
                            typename Problem::BDataType>::TransposedDstrEncode{});
                else
                    return BLdsTileDistr;
            }();

            // LDS tile windows for reading;
            // they share the data pointer with the LDS windows for storing
            // but also associate with a distribution to produce a register tile when reading
            auto a_lds_ld_window =
                make_tile_window(a_lds_block, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto b_lds_ld_window =
                make_tile_window(b_lds_block, b_lds_shape, {0, 0}, b_lds_input_tile_distr);

            // using ALDSLoadWindow = decltype(make_tile_window(a_lds_block, a_lds_shape, {0, 0},
            // a_lds_input_tile_distr)); using BLDSLoadWindow =
            // decltype(make_tile_window(b_lds_block, b_lds_shape, {0, 0}, b_lds_input_tile_distr));
            // tuple_array<ALDSLoadWindow, PrefetchStages> a_lds_ld_windows;
            // tuple_array<BLDSLoadWindow, PrefetchStages> b_lds_ld_windows;
            // static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
            //     a_lds_ld_windows.at(prefetch_idx) =
            //         make_tile_window(a_lds_blocks.at(prefetch_idx), a_lds_shape, {0, 0},
            //         a_lds_input_tile_distr);
            //     b_lds_ld_windows.at(prefetch_idx) =
            //         make_tile_window(b_lds_blocks.at(prefetch_idx), b_lds_shape, {0, 0},
            //         b_lds_input_tile_distr);
            // });

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            ignore = a_element_func;
            ignore = b_element_func;

            // constexpr index_t BlockSize   = Problem::kBlockSize;
            // constexpr index_t A_LOAD_INST  = MPerBlock * KPerBlock / BlockSize /
            // GetVectorSizeA(); constexpr index_t B_LOAD_INST  = NPerBlock * KPerBlock / BlockSize
            // / GetVectorSizeB(); constexpr index_t TOT_LOAD_INST = A_LOAD_INST + B_LOAD_INST;

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                Base::GlobalPrefetchAsync(
                    a_copy_lds_window, a_tile_windows, a_dram_tile_window_step);
                Base::GlobalPrefetchAsync(
                    b_copy_lds_window, b_tile_windows, b_dram_tile_window_step);

                block_sync_lds_direct_load();

                Base::LocalPrefetch(
                    a_block_tiles.at(prefetch_idx), a_lds_ld_window, is_a_load_tr_v);
                Base::LocalPrefetch(
                    b_block_tiles.at(prefetch_idx), b_lds_ld_window, is_b_load_tr_v);

                block_sync_lds();
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile(";; HotLoop Start ;;");
                    __builtin_amdgcn_sched_barrier(0);

                    // ----------------
                    // Multi LDS buffer
                    // ----------------
                    // static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                    //     block_sync_lds_direct_load<TOT_LOAD_INST * (PrefetchStages - 1)>();

                    //     Base::LocalPrefetch(a_block_tile, a_lds_ld_windows.at(prefetch_idx),
                    //     is_a_load_tr_v); Base::LocalPrefetch(b_block_tile,
                    //     b_lds_ld_windows.at(prefetch_idx), is_b_load_tr_v);

                    //     __builtin_amdgcn_sched_barrier(0);
                    //     block_sync_lds();

                    //     Base::GlobalPrefetchAsync(a_copy_lds_windows.at(prefetch_idx),
                    //     a_tile_windows, a_dram_tile_window_step);
                    //     Base::GlobalPrefetchAsync(b_copy_lds_windows.at(prefetch_idx),
                    //     b_tile_windows, b_dram_tile_window_step);

                    //     __builtin_amdgcn_sched_barrier(0);
                    //     block_gemm(c_block_tile, a_block_tile, b_block_tile);
                    //     __builtin_amdgcn_sched_barrier(0);
                    // });

                    // ---------------------
                    // Multi register buffer
                    // ---------------------
                    Base::GlobalPrefetchAsync(
                        a_copy_lds_window, a_tile_windows, a_dram_tile_window_step);
                    Base::GlobalPrefetchAsync(
                        b_copy_lds_window, b_tile_windows, b_dram_tile_window_step);

                    __builtin_amdgcn_sched_barrier(0);

                    block_gemm(c_block_tile, a_block_tiles.at(I0{}), b_block_tiles.at(I0{}));

                    __builtin_amdgcn_sched_barrier(0);

                    static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds_direct_load();

                        Base::LocalPrefetch(a_block_tiles.at(number<prefetch_idx - 1>{}),
                                            a_lds_ld_window,
                                            is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tiles.at(number<prefetch_idx - 1>{}),
                                            b_lds_ld_window,
                                            is_b_load_tr_v);

                        block_sync_lds();

                        Base::GlobalPrefetchAsync(
                            a_copy_lds_window, a_tile_windows, a_dram_tile_window_step);
                        Base::GlobalPrefetchAsync(
                            b_copy_lds_window, b_tile_windows, b_dram_tile_window_step);

                        __builtin_amdgcn_sched_barrier(0);
                        block_gemm(c_block_tile,
                                   a_block_tiles.at(prefetch_idx),
                                   b_block_tiles.at(prefetch_idx));
                        __builtin_amdgcn_sched_barrier(0);
                    });

                    block_sync_lds_direct_load();

                    Base::LocalPrefetch(a_block_tiles.at(number<PrefetchStages - 1>{}),
                                        a_lds_ld_window,
                                        is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tiles.at(number<PrefetchStages - 1>{}),
                                        b_lds_ld_window,
                                        is_b_load_tr_v);

                    block_sync_lds();

                    i += PrefetchStages;
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile(";; HotLoop End ;;");
                    __builtin_amdgcn_sched_barrier(0);
                } while(i < (num_loop - PrefetchStages));
            }

            // ----------------
            // Multi LDS buffer
            // ----------------
            // static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
            //     block_sync_lds_direct_load<TOT_LOAD_INST * (PrefetchStages - 1 -
            //     prefetch_idx)>(); Base::LocalPrefetch(a_block_tile,
            //     a_lds_ld_windows.at(prefetch_idx), is_a_load_tr_v);
            //     Base::LocalPrefetch(b_block_tile, b_lds_ld_windows.at(prefetch_idx),
            //     is_b_load_tr_v);

            //     block_sync_lds();

            //     __builtin_amdgcn_sched_barrier(0);
            //     block_gemm(c_block_tile, a_block_tile, b_block_tile);
            //     __builtin_amdgcn_sched_barrier(0);
            // });

            // ---------------------
            // Multi register buffer
            // ---------------------
            static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                block_gemm(
                    c_block_tile, a_block_tiles.at(prefetch_idx), b_block_tiles.at(prefetch_idx));
            });

            // auto HotLoopTail = [&](auto tail_num) {
            //     static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
            //         block_sync_lds();

            //         block_gemm.LocalPrefetch(
            //             a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
            //         block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            //         block_sync_lds();

            //         if constexpr(is_a_col_major && !is_a_load_tr_v())
            //         {
            //             auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
            //                 Policy::template MakeShuffledARegTileDistribution<Problem>());
            //             transpose_tile2d(a_shuffle_tmp,
            //             a_block_tiles.get(number<prefetch_idx>{}));
            //             Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
            //         }
            //         else
            //         {
            //             Base::LocalPrefill(a_copy_lds_window,
            //                                a_block_tiles.get(number<prefetch_idx>{}));
            //         }
            //         if constexpr(is_b_row_major && !is_b_load_tr_v())
            //         {
            //             auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
            //                 Policy::template MakeShuffledBRegTileDistribution<Problem>());
            //             transpose_tile2d(b_shuffle_tmp,
            //             b_block_tiles.get(number<prefetch_idx>{}));
            //             Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
            //         }
            //         else
            //         {
            //             Base::LocalPrefill(b_copy_lds_window,
            //                                b_block_tiles.get(number<prefetch_idx>{}));
            //         }
            //     });

            //     block_sync_lds();
            //     block_gemm.LocalPrefetch(
            //         a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
            //     block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            // };

            // if constexpr(TailNum == TailNumber::One)
            // {
            //     block_sync_lds();
            //     block_gemm.LocalPrefetch(
            //         a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
            //     block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            // }
            // else if constexpr(TailNum == TailNumber::Two)
            // {
            //     HotLoopTail(number<2>{});
            // }
            // else if constexpr(TailNum == TailNumber::Three)
            // {
            //     HotLoopTail(number<3>{});
            // }
            // else if constexpr(TailNum == TailNumber::Four)
            // {
            //     HotLoopTail(number<4>{});
            // }
            // else if constexpr(TailNum == TailNumber::Five)
            // {
            //     HotLoopTail(number<5>{});
            // }
            // else if constexpr(TailNum == TailNumber::Six)
            // {
            //     HotLoopTail(number<6>{});
            // }
            // else if constexpr(TailNum == TailNumber::Seven)
            // {
            //     HotLoopTail(number<7>{});
            // }
            // else if constexpr(TailNum == TailNumber::Full)
            // {
            //     HotLoopTail(number<PrefetchStages>{});
            // }

            return c_block_tile;
        }
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Interwave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A/B tiles in LDS
            // With c++20 could simplify to below line.
            // Currently get error: captured structured bindings are a C++20 extension
            // auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);
            auto ab_lds_blocks = Base::GetABLdsTensorViews(p_smem);
            auto& a_lds_block  = ab_lds_blocks.at(I0{});
            auto& b_lds_block  = ab_lds_blocks.at(I1{});

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto a_windows =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto& a_copy_dram_window = a_windows.at(I0{});
            auto& a_copy_lds_window  = a_windows.at(I1{});
            auto& a_lds_gemm_window  = a_windows.at(I2{});

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto b_windows =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);
            auto& b_copy_dram_window = b_windows.at(I0{});
            auto& b_copy_lds_window  = b_windows.at(I1{});
            auto& b_lds_gemm_window  = b_windows.at(I2{});

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr =
                decltype(a_copy_dram_window[number<0>{}].get_tile_distribution());
            using BBlockTileDistr =
                decltype(b_copy_dram_window[number<0>{}].get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0

            // Load tile — during value loading, an elementwise function is executed for each A0,
            // A1, … AN. The values A0, A1, … AN are read by the same thread.
            a_block_tiles.at(I0{}) = load_tile_with_elementwise(a_copy_dram_window, a_element_func);

            // Move each A — the enhanced function move_tile_window is executed, which takes a tuple
            // as input.
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

            // Load tile — during value loading, an elementwise function is executed for each B0,
            // B1, … BN. The values B0, B1, … BN are read by the same thread.
            b_block_tiles.at(I0{}) = load_tile_with_elementwise(b_copy_dram_window, b_element_func);

            // Move each B — the enhanced function move_tile_window is executed, which takes a tuple
            // as input.
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(I0{}));
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}));
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tiles.get(I0{}));
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, b_block_tiles.get(I0{}));
            }

            // Global prefetch [1, PrefetchStages]
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                a_block_tiles.at(number<prefetch_idx>{}) =
                    load_tile_with_elementwise(a_copy_dram_window, a_element_func);

                move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                b_block_tiles.at(number<prefetch_idx>{}) =
                    load_tile_with_elementwise(b_copy_dram_window, b_element_func);

                move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        block_gemm(c_block_tile,
                                   a_lds_gemm_window,
                                   b_lds_gemm_window,
                                   is_a_load_tr_v,
                                   is_b_load_tr_v);
                        // no second block_sync_lds because it's interwave

                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(
                                a_shuffle_tmp,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                            Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(
                                b_shuffle_tmp,
                                b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                            Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                b_copy_lds_window,
                                b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                        }

                        a_block_tiles.at(number<prefetch_idx>{}) =
                            load_tile_with_elementwise(a_copy_dram_window, a_element_func);

                        move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

                        b_block_tiles.at(number<prefetch_idx>{}) =
                            load_tile_with_elementwise(b_copy_dram_window, b_element_func);

                        move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            auto HotLoopTail = [&](auto tail_num) {
                static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();
                    block_gemm(c_block_tile,
                               a_lds_gemm_window,
                               b_lds_gemm_window,
                               is_a_load_tr_v,
                               is_b_load_tr_v);
                    // no second block_sync_lds because it's interwave

                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(number<prefetch_idx>{}));
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window,
                                           a_block_tiles.get(number<prefetch_idx>{}));
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tiles.get(number<prefetch_idx>{}));
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window,
                                           b_block_tiles.get(number<prefetch_idx>{}));
                    }
                });

                block_sync_lds();
                block_gemm(c_block_tile,
                           a_lds_gemm_window,
                           b_lds_gemm_window,
                           is_a_load_tr_v,
                           is_b_load_tr_v);
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm(c_block_tile,
                           a_lds_gemm_window,
                           b_lds_gemm_window,
                           is_a_load_tr_v,
                           is_b_load_tr_v);
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
            }

            return c_block_tile;
        }
    };

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            constexpr bool hot_loop    = hot_loop_.value;
            constexpr auto tail_num    = tail_num_.value;
            constexpr auto PassThrough = [](auto& e, const auto& x) { e = x; };
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop, tail_num>(
                a_dram_block_window_tmp,
                PassThrough,
                b_dram_block_window_tmp,
                PassThrough,
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                [](auto& e, const ADataType& a) { e = a; },
                b_dram_block_window_tmp,
                [](auto& e, const BDataType& b) { e = b; },
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          a_element_func,
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          b_element_func,
                          num_loop,
                          p_smem);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          num_loop,
                          has_hot_loop,
                          tail_number,
                          p_smem);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          num_loop,
                          p_smem);
    }
};

} // namespace ck_tile
