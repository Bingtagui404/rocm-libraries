################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Print error message along with the last N lines of a log file
# Args: error_message log_file [num_lines]
print_error_with_log() {
    local error_message=$1
    local log_file=$2
    local num_lines=${3:-30}  # Default to 30 lines
    
    echo "Error: $error_message" >&2
    if [[ -f "$log_file" ]]; then
        echo "" >&2
        echo "Last $num_lines lines of log:" >&2
        echo "----------------------------------------" >&2
        tail -n $num_lines "$log_file" >&2
        echo "----------------------------------------" >&2
        echo "Full log available at: $log_file" >&2
    else
        echo "Log file not found: $log_file" >&2
    fi
}

# Parse command line arguments
usage() {
    echo "Usage: $0 -m <mode> -y <yaml_file> -o <out_dir>" >&2
    echo "  -m, --mode      Mode: 'baseline','cms', or 'cms-fast'" >&2
    echo "  -y, --yaml      Path to YAML file" >&2
    echo "  -o, --out       Output directory" >&2
    exit 1
}

export mode=""
export yaml_file=""
export out_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            mode="$2"
            shift 2
            ;;
        -y|--yaml)
            yaml_file="$2"
            shift 2
            ;;
        -o|--out)
            out_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$mode" ]]; then
    echo "Error: -m/--mode is required" >&2
    usage
fi
if [[ "$mode" != "baseline" && "$mode" != "cms" && "$mode" != "cms-fast" ]]; then
    echo "Error: mode must be either 'baseline', 'cms', or 'cms-fast', but got '$mode'" >&2
    exit 1
fi

if [[ -z "$yaml_file" ]]; then
    echo "Error: -y/--yaml is required" >&2
    usage
fi
if [[ ! -f "$yaml_file" ]]; then
    echo "Error: yaml_file '$yaml_file' does not exist." >&2
    exit 1
fi

if [[ -z "$out_dir" ]]; then
    echo "Error: -o/--out is required" >&2
    usage
fi
mkdir -p "$out_dir"

# Get a value from a YAML file. Use bracket notation: "['GlobalParameters']['NumWarmups']"
get_yaml_value() {
    python3 -c "import yaml; d=yaml.safe_load(open('$1')); print(d$2)"
}

# Given the line number of the header line, and the column name, return the value of the column from the provided file.
get_entry() {
    local file=$1
    local header_line_num=$2
    local col_name=$3
    awk -F',' -v line="$header_line_num" 'NR==(line) {for(i=1;i<=NF;i++) if($i=="'$col_name'") col=i} NR==(line+1) {gsub(/^[[:space:]]+/, ""); print $col}' $file
}

# Given the file and the solution index, return the line number of the header line in the file.
get_header_line_for_solution_index() {
    local file=$1
    local solution_index=$2
    # Track the most recent header line, and when we find the matching solution index, return it
    awk -v sol_idx="$solution_index" '
        # Track header lines (lines starting with "[number]:")
        /^[[:space:]]*\[[0-9]+\]:/ {
            header_line = NR
        }
        /Solution index:/ {
            # Extract the solution index number
            split($0, parts, ":")
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", parts[2])
            if (parts[2] == sol_idx) {
                # Found matching solution index, return the most recent header line
                print header_line
                exit
            }
        }
    ' "$file"
}

# Given the file and the kernel shape, return the solution indices of all kernels matching the shape.
# kernel_pattern: Pattern to match (e.g., "128x256x32" for MT shape only, or "128x256x32_MI32x32" for full match)
find_indices_for_kernel_shape() {
    local file=$1
    local kernel_pattern=$2
    # Search for "Solution name:" lines and check if next line contains MT${kernel_pattern}
    # Pattern: MT followed by kernel dimensions (e.g., 192x256x32), optionally with _MI shape
    # Return the solution index from the line before "Solution name:"
    awk -v kernel="$kernel_pattern" '
        /Solution index:/ {
            # Extract the solution index number (format: "--Solution index: X")
            # Split by colon and take the last field, then trim whitespace
            split($0, parts, ":")
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", parts[2])
            solution_index = parts[2]
        }
        /Solution name:/ {
            # Read the next line (kernel name line)
            getline next_line
            if (next_line ~ "MT" kernel) {
                print solution_index
            }
        }
    ' "$file"
}

# Given the file, return the winner's solution index from the "Winner:" section at the end.
find_winner_index() {
    local file=$1
    grep -A5 "^Winner:" "$file" | grep -oP '(?<=--Solution index: )\d+' | head -1
}

# Given the file and the solution index, return the kernel name.
find_kernel_name_for_solution_index() {
    local file=$1
    local solution_index=$2
    local header_line_num=$(get_header_line_for_solution_index $file $solution_index)
    # Find the kernel name line (starts with "--kernel name:") that comes after the header line
    # Extract everything after "--kernel name:" and trim leading whitespace
    awk -v header_line="$header_line_num" '
        NR >= header_line && /kernel name:/ {
            # Extract kernel name (everything after "--kernel name:")
            sub(/^[[:space:]]*--kernel name:[[:space:]]*/, "")
            print $0
            exit
        }
    ' "$file"
}

# Create a non-CMS version of a yaml file by setting UseCustomMainLoopSchedule to [0]
# - Validates there is exactly one instance of UseCustomMainLoopSchedule
# - Errors if UseCustomMainLoopSchedule is not [1] (CMS must be enabled in source)
# - Creates output file with UseCustomMainLoopSchedule set to [0]
# Args: input_yaml_file output_yaml_file
# Returns 0 on success, 1 on error (prints error message to stderr)
create_non_cms_yaml() {
    local input_file=$1
    local output_file=$2
    
    # Count occurrences of UseCustomMainLoopSchedule (accounting for whitespace)
    local count=$(grep -cE '^\s*-?\s*UseCustomMainLoopSchedule\s*:' "$input_file" 2>/dev/null || echo "0")
    
    if [[ "$count" -eq 0 ]]; then
        echo "Error: No UseCustomMainLoopSchedule found in $input_file" >&2
        return 1
    elif [[ "$count" -gt 1 ]]; then
        echo "Error: Multiple UseCustomMainLoopSchedule entries ($count) found in $input_file" >&2
        return 1
    fi
    
    # Extract the current value (handle whitespace around colons/brackets)
    local current_value=$(grep -oE 'UseCustomMainLoopSchedule\s*:\s*\[?\s*[^\]]*\s*\]?' "$input_file" | head -1)
    
    # Normalize True/1/true -> 1, False/0/false -> 0
    local cms_value=""
    if echo "$current_value" | grep -qiE '\[\s*(1|true)\s*\]|\[\s*(1|true)\s*$|:\s*(1|true)\s*$'; then
        cms_value="1"
    elif echo "$current_value" | grep -qiE '\[\s*(0|false)\s*\]|\[\s*(0|false)\s*$|:\s*(0|false)\s*$'; then
        cms_value="0"
    else
        echo "Error: Unexpected value format for UseCustomMainLoopSchedule: $current_value" >&2
        return 1
    fi
    
    # Error if CMS is disabled in the original yaml
    if [[ "$cms_value" != "1" ]]; then
        echo "Error: $input_file has UseCustomMainLoopSchedule: [0]. Expected CMS to be enabled ([1])." >&2
        return 1
    fi
    
    # Create copy with UseCustomMainLoopSchedule set to [0]
    # Handle various formats: "- UseCustomMainLoopSchedule: [1]", "UseCustomMainLoopSchedule: [True]", etc.
    sed -E 's/(^\s*-?\s*UseCustomMainLoopSchedule\s*:\s*)\[?\s*[^\]]*\s*\]?/\1[0]/' "$input_file" > "$output_file"
    
    echo "Created non-CMS yaml: $output_file (original CMS value was: [$cms_value])"
}

# Look through Tensile output and print out just the gflops from the last run.
print_gflops_from_tensile_log() {
    local log_file="$1"
    # Data lines start with a number (run index), gflops is field 15
    # (field 12 in CSV header, but problem-sizes contains commas in quotes adding 3 extra fields)
    local gflops=$(grep -E "^[0-9]+," "$log_file" | tail -1 | cut -d',' -f15)
    if [[ -n "$gflops" ]]; then
        echo "GFLOPS: $gflops"
    else
        echo "Warning: Could not extract GFLOPS from tensile log" >&2
    fi
}

# Get M, N, K from "- Exact: [M, N, _, K]" line
export M N K; read M N _ K <<< $(grep -oE 'Exact:\s*\[\s*[0-9]+\s*,\s*[0-9]+\s*,\s*[0-9]+\s*,\s*[0-9]+\s*\]' "$yaml_file" | grep -oE '[0-9]+' | tr '\n' ' ')

# Calculate kernel shape from MatrixInstruction and DepthU
mi_array=$(grep -A1 'MatrixInstruction:' "$yaml_file" | grep -oE '\[.*\]' | tr -d '[]' | tr ',' ' ')
depthu=$(grep -oE 'DepthU:\s*\[[0-9]+\]' "$yaml_file" | grep -oE '[0-9]+')
read mi0 mi1 _ _ _ mi5 mi6 mi7 mi8 <<< "$mi_array"
mt0=$((mi0 * mi5 * mi7))
mt1=$((mi1 * mi6 * mi8))
export kernel=${mt0}x${mt1}x${depthu}

# Get transA and transB from the yaml file (supports 0/1 and True/False)
transpose_a=$(grep -iE 'TransposeA:\s*(0|1|true|false)' "$yaml_file" | grep -oE '(0|1|true|false)' | tr '[:upper:]' '[:lower:]')
export transA="N"; if [[ "$transpose_a" != "0" && "$transpose_a" != "false" ]]; then export transA="T"; fi

transpose_b=$(grep -iE 'TransposeB:\s*(0|1|true|false)' "$yaml_file" | grep -oE '(0|1|true|false)' | tr '[:upper:]' '[:lower:]')
export transB="N"; if [[ "$transpose_b" != "0" && "$transpose_b" != "false" ]]; then export transB="T"; fi

# Get MFMA shape from MatrixInstruction (first 2 elements)
export mfma_shape=${mi0}x${mi1}

# String to search for in hipblaslt-bench results.
export kernel_str="MT${kernel}_MI${mfma_shape}"

# TODO: Append hipblaslt-bench command to top of output file.
# TODO: Also measure frequencies.
# Mode Baseline: Find baseline performance for non-CMS kernel
if [[ "$mode" == "baseline" ]]; then
    export baseline_dir=${out_dir}_non_cms
    mkdir -p $baseline_dir
    
    export non_cms_yaml_file=$baseline_dir/non_cms_${yaml_file}
    # Validate and create non-CMS yaml file
    create_non_cms_yaml "$yaml_file" "$non_cms_yaml_file"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    # 1. Run non-cms through Tensile to get baseline performance (BT).
    export tensile_log_file=$baseline_dir/traces/tensile.log
    CU=256 Tensile $non_cms_yaml_file $baseline_dir &>> $tensile_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "Tensile failed to run for non-CMS kernel" "$tensile_log_file"
        exit 1
    fi
    export run_script=$(find $baseline_dir -name "run.sh")

    # TODO: Hardcoded iteration range
    mkdir -p $baseline_dir/traces
    export rocprofv3_log_file=$baseline_dir/traces/rocprofv3.log
    rocprofv3 --att \
        --att-activity 10 \
        --att-target-cu 0 \
        --kernel-include-regex Cijk \
        -d $baseline_dir/traces \
        --kernel-iteration-range 20-120 \
        --output-format csv \
        -- $run_script &>> $rocprofv3_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "rocprofv3 failed to run for non-CMS kernel" "$rocprofv3_log_file"
        exit 1
    fi

    print_gflops_from_tensile_log $tensile_log_file
    # TODO: 2. Pull traces, PMC, and efficiency for BT.
    # TODO: Get mainloop efficiency script from https://github.com/ROCm/hipblaslt-tools/
    exit 0

    # 3. Run non-cms through hipblaslt-bench to get baseline performance (BH).
    export non_cms_tmp_hipblaslt_results=$baseline_dir/non_cms_hipblaslt-bench_all.txt
    export hipblaslt_log_file=$baseline_dir/hipblaslt-bench.log
    # Run a very fast iteration to find all kernels of the right shape.
    # TODO: Turn these hipblaslt-bench calls into a function that can be fed to rocprofv3
    hipblaslt-bench --function matmul \
        --sizem $M --sizen $N --sizek $K  \
        --transA $transA --transB $transB \
        --algo_method all --api_method cpp \
        --iters 1 --cold_iters 1 \
        --alpha 1 --beta 0  --initialization trig_float --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type xf32_r --use_gpu_timer --scaleA 0 --scaleB 0 --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --scale_type f32_r --bias_type f32_r --print_kernel_info -v \
        > $non_cms_tmp_hipblaslt_results 2> $hipblaslt_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "hipblaslt-bench failed to run for non-CMS kernel" "$hipblaslt_log_file"
        exit 1
    fi
    
    # Search for kernels matching both kernel shape AND MFMA shape (full kernel_str pattern)
    # If not found, fall back to the winner kernel
    kernel_indices=$(find_indices_for_kernel_shape $non_cms_tmp_hipblaslt_results "${kernel}_MI${mfma_shape}")
    if [[ -n "$kernel_indices" ]]; then
        kernel_count=$(echo $kernel_indices | wc -w)
        if [[ "$kernel_count" -eq 1 ]]; then
            # Only one kernel found, use it automatically
            chosen_index=$kernel_indices
            kernel_name=$(find_kernel_name_for_solution_index $non_cms_tmp_hipblaslt_results $chosen_index)
            echo "Found single matching kernel, using it automatically:"
            echo "$chosen_index: $kernel_name"
        else
            # Print all matching kernels and ask user to specify which one to use.
            for kernel_index in $kernel_indices; do            
                kernel_name=$(find_kernel_name_for_solution_index $non_cms_tmp_hipblaslt_results $kernel_index)
                echo "$kernel_index: $kernel_name"
            done
            echo "--------------------------------"

            # Ask the user to specify solution index they want to go with.
            while true; do
                read -p "Enter the solution index to use: " chosen_index
                # Check if the chosen index is in the list of valid kernel indices
                valid=false
                for idx in $kernel_indices; do
                    if [[ "$chosen_index" == "$idx" ]]; then
                        valid=true
                        break
                    fi
                done
                if [[ "$valid" == "true" ]]; then
                    echo "Selected solution index: $chosen_index"
                    break
                else
                    echo "Error: '$chosen_index' is not a valid solution index. Please choose from: $kernel_indices" >&2
                fi
            done
        fi
    else
        # TODO: If no kernels found matching full pattern, fall back to winner. Winner must be run wiht more iterations to get something meaningfull..
        chosen_index=$(find_winner_index $non_cms_tmp_hipblaslt_results)
        if [[ -z "$chosen_index" ]]; then
            echo "Error: No kernels found matching full pattern ${kernel_str}, and no winner kernel found in $non_cms_tmp_hipblaslt_results" >&2
            exit 1
        fi
        echo "No kernels found matching full pattern ${kernel_str}, falling back to winner. with solution index."
        echo "Winner kernel $chosen_index: $(find_kernel_name_for_solution_index $non_cms_tmp_hipblaslt_results $chosen_index)"
    fi
    # For each kernel, rerun hipblaslt-bench with real iters and cold iters to get the actual performance.
    output_file=$baseline_dir/non_cms_hipblaslt-bench_${kernel_index}.txt
    # TODO: 4. Trace, PMC, and efficiency for hipblaslt-bench for chosen kernel.

    # hipblaslt-bench --function matmul \
    #     --sizem $M --sizen $N --sizek $K  \
    #     --transA $transA --transB $transB \
    #     --algo_method index --solution_index $kernel_index \
    #     --iters 5000 --cold_iters 5000 \
    #     --alpha 1 --beta 0  --initialization trig_float --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type xf32_r --use_gpu_timer --scaleA 0 --scaleB 0 --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --scale_type f32_r --bias_type f32_r --print_kernel_info -v \
    #     | tee $output_file
    # TODO: 
    exit 0
elif [[ "$mode" == "cms-fast" ]]; then
    export trace_dir=$out_dir/traces
    mkdir -p $trace_dir
    # 1. Run through Tensile to generate kernel
    export tensile_log_file=$trace_dir/tensile.log
    if [[ -f "$tensile_log_file" ]]; then rm "$tensile_log_file"; fi

    echo "Running Tensile..."
    CU=256 Tensile $yaml_file $out_dir &>> $tensile_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "Tensile failed to run for CMS kernel" "$tensile_log_file"
        exit 1
    fi

    # 2. Benchmark
    export rocprofv3_log_file=$trace_dir/rocprofv3.log
    if [[ -f "$rocprofv3_log_file" ]]; then rm "$rocprofv3_log_file"; fi

    export run_script=$(find $out_dir -name "run.sh")
    # TODO: Hardcoded iteration range. Need cold iters and iters from the yaml file.
    echo "Running rocprofv3..."
    rocprofv3 --att \
        --att-activity 10 \
        --att-target-cu 0 \
        --kernel-include-regex Cijk \
        -d $trace_dir \
        --kernel-iteration-range 20-49 \
        --output-format csv \
        -- $run_script &>> $rocprofv3_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "rocprofv3 failed to run for CMS kernel" "$rocprofv3_log_file"
        exit 1
    fi

    # 3. Print and generate stats
    echo "Analyzing Tensile log..."
    print_gflops_from_tensile_log "$tensile_log_file"

    # Extract run_id from rocprofv3 log (number before _shader_engine_)
    run_id=$(grep -m1 "_shader_engine_" "$rocprofv3_log_file" | sed 's/.*_\([0-9]\+\)_shader_engine_.*/\1/')
    if [[ -z "$run_id" ]]; then
        echo "Error: Could not extract run_id from $rocprofv3_log_file" >&2
        exit 1
    fi

    # 5. Run analyze.py on the csv files in the traces directory
    export figures_dir=$out_dir/figures
    mkdir -p $figures_dir
    python "$SCRIPT_DIR/analyze.py" \
        --csv-dir $trace_dir \
        --pattern ${run_id} \
        --yaml $yaml_file \
        --tensile-dir $out_dir \
        --output-dir $figures_dir \
        --output-latency-json

    # TODO: PMC
    exit 0
elif [[ "$mode" == "cms" ]]; then
    # Run through Tensile.
    export tensile_log_file=$out_dir/tensile.log
    mkdir -p $out_dir
    CU=256 Tensile $yaml_file $out_dir &>> $tensile_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "Tensile failed to run for CMS kernel" "$tensile_log_file"
        exit 1
    fi
    # TODO: Trace, PMC, efficiency, and gflops through Tensile.
    
    # TODO: Run through hipblaslt-bench.
    export tensile_create_lib_log_file=$out_dir/tensile_create_library.log
    ../Tensile/bin/TensileCreateLibrary --code-object-version=5 --cxx-compiler=amdclang++ \
      --library-format=msgpack --architecture gfx950 \
      $out_dir/3_LibraryLogic \
      $out_dir/lib_out \
      HIP &>> $tensile_create_lib_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "TensileCreateLibrary failed to run for CMS kernel" "$tensile_create_lib_log_file"
        exit 1
    fi
    
    output_file=$out_dir/cms_hipblaslt-bench.txt
    export hipblaslt_log_file=$out_dir/hipblaslt-bench.log
    HIPBLASLT_TENSILE_LIBPATH=$out_dir/lib_out/library hipblaslt-bench --function matmul \
        --sizem $M --sizen $N --sizek $K  \
        --transA N --transB T \
        --algo_method index --solution_index 0 \
        --iters 5000 --cold_iters 5000 \
        --alpha 1 --beta 0  --initialization trig_float  --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type xf32_r --use_gpu_timer --scaleA 0 --scaleB 0 --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --scale_type f32_r --bias_type f32_r --print_kernel_info -v \
        > $output_file 2> $hipblaslt_log_file
    if [[ $? -ne 0 ]]; then
        print_error_with_log "hipblaslt-bench failed to run for CMS kernel" "$hipblaslt_log_file"
        exit 1
    fi
    # TODO: Pull traces, PMC, and efficiency for hipblaslt-bench.
    exit 0
fi

# Should never get here since we check for valid mode above.
echo "Error: Invalid mode: $mode" >&2
exit 1


# Mode CMS: Get baseline for CMS kernel

# Performance Iteration
# 1. Export CMS and run through Tensile.
# CU=256 Tensile 128x256x32NT.yaml $out_dir
# 2. Trace the kernel
# 3. Get PMC values
# 4. Calculate mainloop efficiency.

# Final steps.
# 1. Run non-CMS kernel through Tensile to get baseline
# 2. Run non-CMS kernel through hipblaslt-bench to get baseline
# 3. Run CMS through Tensile to get new performance.
# 4. Use LibraryLogic to get new performance from hipblaslt-bench.


# TODO: When done, have mode to zip up and collect all of the files needed for submission.
# TODO: Also merge Library results into the correct files.