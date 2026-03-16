# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import itertools
import logging
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Import the base GemmKernelBuilder from the gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(gemm_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


def _import_validation_utils():
    """Import validation utilities from the gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "validation_utils",
        os.path.join(gemm_dir, "gemm_validation_utils.py"),
    )
    validation_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_utils)

    return validation_utils


GemmKernelBuilder = _import_gemm_kernel_builder()
_validation_utils = _import_validation_utils()
is_tile_config_valid = _validation_utils.is_tile_config_valid
get_abc_layouts = _validation_utils.get_abc_layouts
get_dtype_string = _validation_utils.get_dtype_string

# ABQuant only supports compv3 pipeline with intrawave scheduler
ABQUANT_TRAIT_UNSUPPORTED_COMBINATIONS = {
    ("mem", "default", "intrawave"),
    ("mem", "default", "interwave"),
    ("compv3", "default", "interwave"),
}


def is_abquant_trait_combination_valid(
    pipeline, epilogue, scheduler, b_preshuffle_quant, preshuffle_b
):
    """Check if an ABQuant trait combination is valid.

    Rules:
    - Only compv3 pipeline with intrawave scheduler is supported
    - preshuffle_b and b_preshuffle_quant are independent flags
    """
    if (pipeline, epilogue, scheduler) in ABQUANT_TRAIT_UNSUPPORTED_COMBINATIONS:
        return False

    return True


class GemmABQuantKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json=None,
    ):
        super().__init__(
            kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
        )
        self.group_size_k = self.config.get("group_size_k", 128)

    def _get_tile_configs(self):
        """Get tile configurations for the current datatype and layout."""
        tile_config = self.config["tile_config"]

        # Generate values in the config if default range is given
        for dim in ["tile_m", "tile_n", "tile_k"]:
            if tile_config.get(dim).get("values") is None:
                tile_config.get(dim)["values"] = self._generate_values(
                    tile_config.get(dim).get("min"),
                    tile_config.get(dim).get("max"),
                    tile_config.get(dim).get("step"),
                )

        tile_m_values = tile_config.get("tile_m").get("values")
        tile_n_values = tile_config.get("tile_n").get("values")
        tile_k_values = tile_config.get("tile_k").get("values")
        warp_m_values = tile_config.get("warp_m").get("values")
        warp_n_values = tile_config.get("warp_n").get("values")
        warp_k_values = tile_config.get("warp_k").get("values")
        warp_tile_m_values = tile_config.get("warp_tile_m").get("values")
        warp_tile_n_values = tile_config.get("warp_tile_n").get("values")
        warp_tile_k_values = tile_config.get("warp_tile_k").get("values")

        # For ABQuant, pipeline is compv3
        default_pipeline = "compv3"

        configs = []
        for combo in itertools.product(
            tile_m_values,
            tile_n_values,
            tile_k_values,
            warp_m_values,
            warp_n_values,
            warp_k_values,
            warp_tile_m_values,
            warp_tile_n_values,
            warp_tile_k_values,
        ):
            (
                tile_m,
                tile_n,
                tile_k,
                warp_m,
                warp_n,
                warp_k,
                warp_tile_m,
                warp_tile_n,
                warp_tile_k,
            ) = combo

            if self._validate_tile_config(
                tile_m,
                tile_n,
                tile_k,
                warp_m,
                warp_n,
                warp_k,
                warp_tile_m,
                warp_tile_n,
                warp_tile_k,
                default_pipeline,
            ):
                configs.append(
                    {
                        "tile_m": tile_m,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "warp_m": warp_m,
                        "warp_n": warp_n,
                        "warp_k": warp_k,
                        "warp_tile_m": warp_tile_m,
                        "warp_tile_n": warp_tile_n,
                        "warp_tile_k": warp_tile_k,
                    }
                )
        return configs

    def _validate_tile_config(
        self,
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        pipeline,
    ):
        """Validate tile configuration for ABQuant GEMM."""
        a_datatype = self.datatype
        b_datatype = self.datatype
        c_datatype = "fp16" if self.datatype in ["fp8", "bf8"] else self.datatype

        # ABQuant-specific: tile_k must be a multiple of group_size_k
        if tile_k % self.group_size_k != 0 or tile_k < self.group_size_k:
            return False

        # ABQuant-specific: group_size_k must be divisible by warp_tile_k
        if self.group_size_k % warp_tile_k != 0:
            return False

        # ABQuant-specific: enforce warp_tile_m == warp_tile_n (MFMA requirement)
        if warp_tile_m != warp_tile_n:
            return False

        # ABQuant-specific: enforce warp_tile_m <-> warp_tile_k coupling
        if self.datatype in ["fp8", "bf8"]:
            if self.gpu_target == "gfx950":
                expected_k = 64 if warp_tile_m == 32 else 128
            else:
                expected_k = 32 if warp_tile_m == 32 else 64
            if warp_tile_k != expected_k:
                return False

        if not is_tile_config_valid(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            a_datatype,
            b_datatype,
            c_datatype,
            pipeline,
            self.layout,
            self.gpu_target,
        ):
            return False

        return True

    def _generate_trait_combinations(self):
        """Generate valid ABQuant trait combinations.

        Trait tuple: (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k,
                      b_preshuffle_quant, preshuffle_b, bquant_group_size)
        """
        trait_config = self.config["trait_config"]

        pipelines = trait_config.get("pipeline").get("values")
        epilogues = trait_config.get("epilogue").get("values")
        schedulers = trait_config.get("scheduler").get("values")
        pad_m_values = trait_config.get("pad_m").get("values")
        pad_n_values = trait_config.get("pad_n").get("values")
        pad_k_values = trait_config.get("pad_k").get("values")
        b_preshuffle_values = trait_config.get("b_preshuffle_quant").get("values")
        preshuffle_b_values = trait_config.get("preshuffle_b").get("values")
        bquant_group_size_values = trait_config.get("b_quant_group_size").get("values")

        all_combinations = list(
            itertools.product(
                pipelines,
                epilogues,
                schedulers,
                pad_m_values,
                pad_n_values,
                pad_k_values,
                b_preshuffle_values,
                preshuffle_b_values,
                bquant_group_size_values,
            )
        )

        combinations = []
        for combo in all_combinations:
            pipeline, epilogue, scheduler = combo[:3]
            b_preshuffle_quant = combo[6]
            preshuffle_b = combo[7]
            if not is_abquant_trait_combination_valid(
                pipeline, epilogue, scheduler, b_preshuffle_quant, preshuffle_b
            ):
                logging.debug(
                    f"Skipping unsupported ABQuant trait: "
                    f"{pipeline}-{epilogue}-{scheduler}-bpreshuffle={b_preshuffle_quant}"
                    f"-preshuffleb={preshuffle_b}"
                )
                continue

            # PreshuffleQuant with BQuantGrouped only supports ColumnMajor BQ layout.
            # BQ layout follows B layout, so skip when B is RowMajor (layout[1] == 'r').
            layout_code = self.layout.lower()
            if b_preshuffle_quant in [True, "true"] and layout_code[1] == "r":
                logging.debug(
                    f"Skipping b_preshuffle_quant=True for layout {layout_code}: "
                    f"BQ layout would be RowMajor, but PreshuffleQuant requires ColumnMajor"
                )
                continue

            combinations.append(combo)
        return combinations

    def _list_kernels(self):
        """Write kernel list to file for CMake to read."""
        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                (
                    pipeline,
                    epilogue,
                    scheduler,
                    pad_m,
                    pad_n,
                    pad_k,
                    b_preshuffle_quant,
                    preshuffle_b,
                    bquant_group_size,
                ) = trait_combo

                kernel_name = (
                    f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}"
                    f"_{pipeline}_{epilogue}_{scheduler}"
                    f"_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}"
                    f"_{str(pad_k).capitalize()}"
                    f"_{str(b_preshuffle_quant).capitalize()}"
                    f"_{str(preshuffle_b).capitalize()}"
                    f"_{bquant_group_size}"
                )

                tile_str = (
                    f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                    f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                    f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}"
                    f"x{tile_config['warp_tile_k']}"
                )
                kernel_name += f"_{tile_str}"

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                    }
                )

        # Write kernel count
        count_file = self.working_path / f"{self.kernel_name_prefix}_kernel_count.txt"
        with open(count_file, "w") as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        list_file = self.working_path / f"{self.kernel_name_prefix}_kernel_list.txt"
        with open(list_file, "w") as f:
            for kernel in kernel_list:
                tile_config = kernel["tile_config"]
                trait_combo = kernel["trait_combo"]

                tile_str = (
                    f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                    f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                    f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}"
                    f"x{tile_config['warp_tile_k']}"
                )
                trait_str = (
                    f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_"
                    + "_".join(str(x) for x in trait_combo[3:])
                )
                f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _generate_kernel_instance(self, tile_config, trait_combo):
        """Generate a single ABQuant kernel instance header."""
        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            b_preshuffle_quant,
            preshuffle_b,
            bquant_group_size,
        ) = trait_combo

        # Build kernel name
        kernel_name = (
            f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}"
            f"_{pipeline}_{epilogue}_{scheduler}"
            f"_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}"
            f"_{str(pad_k).capitalize()}"
            f"_{str(b_preshuffle_quant).capitalize()}"
            f"_{str(preshuffle_b).capitalize()}"
            f"_{bquant_group_size}"
        )
        tile_str = (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
            f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}"
            f"x{tile_config['warp_tile_k']}"
        )
        kernel_name += f"_{tile_str}"

        # Determine output type
        c_type = "fp16" if self.datatype in ["fp8", "bf8"] else self.datatype

        # Layouts
        a_layout, b_layout, c_layout = get_abc_layouts(self.layout)

        # AQ layout: follows ALayout (RowMajor for RCR/RRR, ColumnMajor for CCR/CRR)
        aq_layout = a_layout
        # BQ layout: follows BLayout (ColumnMajor for RCR, RowMajor for RRR, etc.)
        bq_layout = b_layout

        # Parse bquant_group_size string (e.g., "1x1x128" or "1x128x128")
        bq_gs_parts = bquant_group_size.split("x")
        bq_gs_m = int(bq_gs_parts[0])
        bq_gs_n = int(bq_gs_parts[1])
        bq_gs_k = int(bq_gs_parts[2])

        # Determine pipeline based on preshuffle_b flag
        is_preshuffle_b_active = preshuffle_b in [True, "true"]

        if is_preshuffle_b_active:
            pipeline_impl = "ck_tile::WPABQuantBPipelineAgBgCrV2"
        else:
            pipeline_impl = "ck_tile::ABQuantGemmPipelineAgBgCrCompV3"

        base_pipeline_map = {
            "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
        }
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
        }

        is_pad_m = "true" if pad_m in [True, "true"] else "false"
        is_pad_n = "true" if pad_n in [True, "true"] else "false"
        is_pad_k = "true" if pad_k in [True, "true"] else "false"
        is_b_preshuffle = "true" if b_preshuffle_quant in [True, "true"] else "false"
        is_preshuffle_b = "true" if is_preshuffle_b_active else "false"
        double_smem_buffer = "true" if is_preshuffle_b_active else "false"

        instance_code = f"""// Generated ABQuant kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
using AQDataType = float;
using BQDataType = float;
using AccDataType = float;
using CDataType = {get_dtype_string(c_type)};

using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};
using AQLayout = {aq_layout};
using BQLayout = {bq_layout};

// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";

// Wrapper for simplified launch interface
struct SelectedKernel {{
    // Tile configuration
    static constexpr ck_tile::index_t BlockSize = 256;
    static constexpr ck_tile::index_t TileM = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t TileN = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t TileK = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t WarpTileM = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t WarpTileK = {tile_config["warp_tile_k"]};

    // ABQuant traits
    static constexpr bool kPadM = {is_pad_m};
    static constexpr bool kPadN = {is_pad_n};
    static constexpr bool kPadK = {is_pad_k};
    static constexpr bool TransposeC = false;
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = {is_b_preshuffle};
    static constexpr bool PreshuffleB = {is_preshuffle_b};
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};
    static constexpr ck_tile::index_t GroupSizeK = {self.group_size_k};

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    // Quantization group sizes
    using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, GroupSizeK>>;
    using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<{bq_gs_m}, {bq_gs_n}, {bq_gs_k}>>;

    // Traits
    using Traits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant,
        BPreshuffleQuant,
        PreshuffleB,
        ALayout, BLayout, CLayout,
        ck_tile::QuantType::ABQuantGrouped,
        AQLayout,
        BQLayout>;

    // Pipeline problem
    using GemmPipelineProblem = ck_tile::GemmABQuantPipelineProblem<
        ADataType,
        BDataType,
        AQDataType,
        BQDataType,
        AccDataType,
        TileShape,
        Traits,
        AQuantGroupSize,
        BQuantGroupSize,
        BDataType,
        {scheduler_type_map.get(scheduler)}>;

    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline)}<GemmPipelineProblem>;

    // Launch function
    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& stream) {{

        using _GemmPipelineProblem = SelectedKernel::GemmPipelineProblem;
        using GemmPipeline = {pipeline_impl}<_GemmPipelineProblem>;

        // Epilogue
        using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
            ADataType,
            BDataType,
            ck_tile::tuple<>,
            AccDataType,
            CDataType,
            ck_tile::tuple<>,
            CLayout,
            ck_tile::element_wise::PassThrough,
            SelectedKernel::TileM,
            SelectedKernel::TileN,
            SelectedKernel::kPadM,
            SelectedKernel::kPadN,
            SelectedKernel::WarpTileM,
            SelectedKernel::WarpTileN,
            SelectedKernel::WarpTileK,
            SelectedKernel::TransposeC>;

        using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;

        // Kernel type
        using Kernel = ck_tile::QuantGemmKernel<
            SelectedKernel::TilePartitioner, GemmPipeline, GemmEpilogue,
            ck_tile::QuantType::ABQuantGrouped>;

        // Kernel arguments
        auto kargs = Kernel::MakeKernelArgs(args);

        if (!Kernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Unsupported kernel arguments; skipping GEMM launch.");
        }}

        // Get grid and block sizes
        const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(stream.log_level_ > 0) {{
            std::cout << "Launching kernel: " << Kernel::GetName() << '\\n'
                      << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
        }}

        // Launch kernel
        constexpr int kBlockPerCu = {k_block_per_cu};
        float ave_time = ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));

        return ave_time;
    }}
}};
"""

        # Write the header file
        simplified_name = kernel_name
        if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
            simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

        header_file = (
            self.working_path
            / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")
        return kernel_name, instance_code

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files with parallel processing."""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (
                        tile_config,
                        trait_combo,
                        self.kernel_name_prefix,
                        self.working_path,
                        self.gpu_target,
                        self.datatype,
                        self.layout,
                        self.config_json,
                    )
                )

        print(
            f"Generating {len(work_items)} individual kernel files "
            f"using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")

        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 50 == 0 or completed == len(work_items):
                    print(
                        f"  Progress: {completed}/{len(work_items)} kernels generated"
                    )
                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Kernel generation failed for {item}: {exc}")

        kernel_list.sort(key=lambda x: x[0])
        self._generate_cmake_individual_targets(kernel_list)
        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual ABQuant kernel file."""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ) = work_item

    builder = GemmABQuantKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        # _generate_kernel_instance() already writes the header file to disk,
        # so no need to write again here.
        kernel_name, _ = builder._generate_kernel_instance(tile_config, trait_combo)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="GEMM ABQuant kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp8", "bf8"],
        help="Data type (fp8 or bf8)",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single",
        action="store_true",
        help="Generate a single kernel file",
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string")
    parser.add_argument("--trait_combo", help="Trait combination string")
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )

    args = parser.parse_args()

    kernel_name_prefix = "gemm_abquant"
    builder = GemmABQuantKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        # Parse tile config
        tile_parts = args.tile_config.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")

        tile_config = {
            "tile_m": int(tile_dims[0]),
            "tile_n": int(tile_dims[1]),
            "tile_k": int(tile_dims[2]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_k": int(warp_dims[2]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "warp_tile_k": int(warp_tile_dims[2]),
        }

        # Parse trait combo:
        # pipeline_epilogue_scheduler_padM_padN_padK_bPreshuffleQuant_preshuffleB_bquantGroupSize
        trait_parts = args.trait_combo.split("_")
        # bquant_group_size is the last part and contains 'x' chars (e.g., 1x1x128)
        # Find where the bquant group size starts by looking for the AxBxC pattern
        # The trait string is: compv3_default_intrawave_False_False_True_False_False_1x1x128
        # Parts: [compv3, default, intrawave, False, False, True, False, False, 1x1x128]
        trait_combo = (
            trait_parts[0],  # pipeline
            trait_parts[1],  # epilogue
            trait_parts[2],  # scheduler
            trait_parts[3] == "True",  # pad_m
            trait_parts[4] == "True",  # pad_n
            trait_parts[5] == "True",  # pad_k
            trait_parts[6] == "True",  # b_preshuffle_quant
            trait_parts[7] == "True",  # preshuffle_b
            trait_parts[8],  # bquant_group_size (e.g., "1x1x128")
        )

        builder._generate_kernel_instance(tile_config, trait_combo)
    elif args.gen_all_individual:
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
