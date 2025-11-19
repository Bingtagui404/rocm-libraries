# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import origami
import math
import yaml
from pathlib import Path

KERNEL_COLUMNS = ['MT_M', 'MT_N', 'MT_K', 'MI_M', 'MI_N', 'MI_K', 'Occupancy', 'WGM', 'NTA', 'NTB']

def parseArguments():
    parser = argparse.ArgumentParser(description="Test Solution Ranking.")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--print", action="store_true", help="Print hardware info")
    parser.add_argument(
        "--yaml", type=str, required=True, help="Yaml file containing sizes to benchmark")
    
    args = parser.parse_args()
    return args

def build_tile_list(library_name):
    tilelist = []
    with open("./tiles_library/" + library_name, "rt") as libinput:
        solutions = yaml.load(libinput, Loader=yaml.SafeLoader)
    for item in solutions:
        tilelist.append(tuple([item[col] for col in KERNEL_COLUMNS]))
    return tilelist
        
def main():
    args = parseArguments()
    
    benchyaml = args.yaml
    if not(Path(benchyaml).is_file()):
        print(f"Yaml file {args.yaml} not found.")
        exit(1)
    resultsfile = Path(benchyaml).stem + ".out"

    hardware = origami.get_hardware_for_device(args.device)

    if args.print:
        hardware.print()

    with open(args.yaml, "rt") as finput:
        benchmarks = yaml.load(finput, Loader=yaml.SafeLoader)
    
    print(f"Loaded yaml with {len(benchmarks)} size(s).")

    prefix = 'gfx950_Cijk_'
    transA_coord = {'N': 'Ailk', 'T': 'Alik'}
    transB_coord = {'N': 'Bljk', 'T': 'Bjlk'}
    datadict = {"bf16": "BBS", "f32": "SB", "xf32": "S_MX"}
    hblt_to_origami = {"bf16_r": "bf16", "f32_r": "f32"}

    sizes_list = []
    for bench_item in benchmarks:
        solutionlib = prefix + transA_coord[bench_item['transA']]
        solutionlib += "_" + transB_coord[bench_item['transB']]
    
        if bench_item['a_type'] == 'bf16_r':
            solutionlib += "_" + datadict['bf16']
        elif bench_item['compute_type'] == 'c_xf32_r':
            solutionlib += "_" + datadict['xf32']
        else:
            solutionlib += "_" + datadict['f32']
        solutionlib += "_tiles.yaml"
    
        tile_list = build_tile_list(solutionlib)
        transA = bench_item['transA']=='T'
        transB = bench_item['transB']=='T'
        type_a = hblt_to_origami[bench_item['a_type']]
        type_b = hblt_to_origami[bench_item['b_type']]
        type_d = hblt_to_origami[bench_item['d_type']]
        result_tuple = origami.select_best_macro_tile_size(
                        bench_item['M'],
                        bench_item['N'],
                        bench_item['K'],
                        bench_item['batch_count'],
                        transA,
                        transB,
                        hardware,
                        tile_list,
                        origami.datatype_to_bits(origami.string_to_datatype(type_a)),
                        origami.datatype_to_bits(origami.string_to_datatype(type_b)),
                        origami.datatype_to_bits(origami.string_to_datatype(type_d)),
                        origami.string_to_datatype(type_a),
                        0,
                        0.8,
                        args.print,
                        6,
                    )
        results_list = []
        for rank_item in result_tuple:
            if rank_item[0] < 1e12:
                rv = {key:round(value) for key, value in zip(['Latency'] + KERNEL_COLUMNS, rank_item)}
                results_list.append(rv)
        sizes_list.append(results_list)

    yaml_args = {"default_flow_style": None, "sort_keys": False}
    with open("./bench_sizes/results/" + resultsfile, "wt") as foutput:
        yaml.dump(sizes_list, foutput, **yaml_args)

if __name__ == "__main__":
    exit(main())