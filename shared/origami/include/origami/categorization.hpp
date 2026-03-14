/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "origami/types.hpp"

namespace origami {

// ============================================================================
// M/N dimension ranges
// ============================================================================

/**
 * @brief Size range labels for M and N dimensions.
 *
 * M and N define the output tile grid:
 *   num_tiles = ceil(M/MT_M) * ceil(N/MT_N)
 *
 * This determines parallelism and tile utilization.  Five ranges
 * capture distinct occupancy regimes.  Boundaries are derived from
 * the macro-tile sizes used in practice (MT: 32–256) and the CU
 * counts of target hardware (256–304 CUs):
 *
 *   tiny   [1, 64]:     Smaller than most tiles.  At the largest tile
 *                        (256), a single tile covers the dimension with
 *                        heavy padding waste. Only vector-like kernels or
 *                        one-tile solutions are competitive.
 *
 *   small  [65, 256]:    Up to one maximum tile (MT_max = 256).  Exactly
 *                        1 tile row/column — no inter-tile parallelism
 *                        along this dimension. Edge effects dominate.
 *
 *   medium [257, 1024]:  2–4 max-tiles.  Partial-wave occupancy: the GPU
 *                        is not fully utilized.  Tile choice strongly
 *                        affects utilization (e.g. M=768 → 3 tiles of
 *                        256 vs 6 tiles of 128).
 *
 *   large  [1025, 4096]: 4–16 max-tiles.  For square problems at 256x256
 *                        tiles, num_tiles = 16*16 = 256, which matches
 *                        a 256-CU GPU (gfx950). This is the transition
 *                        from partial-GPU to full-GPU utilization.
 *
 *   xlarge [4097, inf):  16+ max-tiles per dimension.  Multiple waves
 *                        are needed, scheduling hides latency, L2/MALL
 *                        capacity effects dominate.
 *
 * These roles are transpose-invariant: the output C is always M x N,
 * so the tile grid is always ceil(M/MT_M) * ceil(N/MT_N) regardless of
 * which matrix is transposed.
 */
enum class mn_range_t : std::uint8_t {
  tiny   = 0,  ///< [1, 64]
  small  = 1,  ///< [65, 256]
  medium = 2,  ///< [257, 1024]
  large  = 3,  ///< [1025, 4096]
  xlarge = 4,  ///< [4097, inf)

  count  = 5
};

/**
 * @brief Size range labels for the K (reduction) dimension.
 *
 * K defines the reduction loop depth: iterations = ceil(K/MT_K).
 * K monotonically raises arithmetic intensity:
 *
 *   AI = 2MNK / ((MK + KN + MN) * bpe)
 *   1/AI = (bpe/2) * (1/M + 1/N + 1/K)
 *
 * There is one structural transition: memory-bound → compute-bound.
 * The K=2048 boundary places moderate-sized square problems (M=N=1024)
 * near the roofline crossover for BF16 on current hardware:
 *   AI(1024, 1024, 2048, bpe=2) = 2*1024*1024*2048 / ((2*1024*2048 + 1024^2)*2)
 *                                ≈ 585 ops/byte
 * which is above the MI300X roofline (~247), confirming that long_k
 * problems of moderate M/N size are indeed compute-bound.
 *
 * Layout does not change K's role: K is always the reduction
 * dimension.  Transpose affects which address pattern is used to
 * load K-slices, but not the loop iteration count or AI.
 */
enum class k_range_t : std::uint8_t {
  short_k = 0,  ///< [1, 2048]    — typically memory-bound
  long_k  = 1,  ///< [2049, inf)  — typically compute-bound

  count   = 2
};

// ============================================================================
// Range boundaries
// ============================================================================

inline constexpr std::array<std::size_t, 5> MN_RANGE_UPPER_BOUNDS = {
    64, 256, 1024, 4096, SIZE_MAX};

inline constexpr std::array<std::size_t, 2> K_RANGE_UPPER_BOUNDS = {
    2048, SIZE_MAX};

/// Total categories: 5(M) * 5(N) * 2(K) = 50.
inline constexpr std::size_t NUM_GEMM_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count);

static_assert(NUM_GEMM_CATEGORIES == 50, "Category count must be 50");

// ============================================================================
// Category type
// ============================================================================

/**
 * @brief GEMM category — size-based classification for heuristic lookup.
 *
 * Categorizes GEMM problems by (M, N, K) into one of 50 buckets.
 * Batch is tracked as a flag but does not affect the category ID,
 * keeping the category space at 50.
 *
 * The intended heuristic lookup combines this category with layout
 * and dtype as separate axes:
 *
 *   heuristic_params = lookup(category.id(), layout, dtype)
 *
 * Insights from NVIDIA CUTLASS / nvMatmulHeuristics / cuBLAS:
 *
 *   - NVIDIA uses an analytical model (not fixed size buckets) to
 *     predict runtime, L2 hit rate, and memory bandwidth for each
 *     candidate tile.  Origami already does this via
 *     compute_total_latency().  The categorization here is for
 *     applying heuristic WEIGHT ADJUSTMENTS to that model.
 *
 *   - NVIDIA's MatmulTile enum lists tiles (8x8 through 256x192).
 *     Our M/N boundaries at {64, 256} align with the tile range:
 *     64 is the smallest practical tile, 256 is the largest.
 *
 *   - NVIDIA's ClusterShape (1x1x1 through 16x1x1) groups thread
 *     blocks for L2 sharing on Hopper.  This is analogous to
 *     origami's workgroup_mapping (WGM) — a parameter selected
 *     WITHIN a category, not a categorization axis.
 *
 *   - NVIDIA's split-K strategies (none, stream-K, segment-K)
 *     are selected based on K-vs-MN ratio.  Our short_k/long_k
 *     split captures this regime transition.
 */
struct gemm_category_t {
  mn_range_t m_range;
  mn_range_t n_range;
  k_range_t  k_range;
  bool       batched = false;  ///< batch > 1 (not part of id())

  /// Unique category id in [0, NUM_GEMM_CATEGORIES), based on (M, N, K) only.
  std::size_t id() const noexcept;

  std::size_t m_lower() const noexcept;
  std::size_t m_upper() const noexcept;
  std::size_t n_lower() const noexcept;
  std::size_t n_upper() const noexcept;
  std::size_t k_lower() const noexcept;
  std::size_t k_upper() const noexcept;

  /**
   * @brief AI at the geometric center of this category's ranges.
   *
   * @param bytes_per_element Element size in bytes (default 2.0 for BF16)
   */
  double representative_arithmetic_intensity(double bytes_per_element = 2.0) const noexcept;

  /// e.g. "cat042_M[257-1024]_N[65-256]_K[1-2048]_single"
  std::string to_string() const;

  bool operator==(const gemm_category_t& o) const noexcept {
    return m_range == o.m_range && n_range == o.n_range &&
           k_range == o.k_range && batched == o.batched;
  }
  bool operator!=(const gemm_category_t& o) const noexcept { return !(*this == o); }
};

// ============================================================================
// Classification functions
// ============================================================================

mn_range_t classify_mn(std::size_t dim) noexcept;
k_range_t classify_k(std::size_t dim) noexcept;

/**
 * @brief Categorize a GEMM problem by its dimensions.
 *
 * Uses problem.size.{m,n,k} for the category ID and problem.batch
 * for the batched flag.  Layout and dtype do NOT affect the category.
 */
gemm_category_t categorize(const problem_t& problem) noexcept;

/**
 * @brief Categorize from raw M, N, K (batch defaults to single).
 */
gemm_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept;

/**
 * @brief Reconstruct a category from its integer id.
 *
 * @param id Category id in [0, NUM_GEMM_CATEGORIES)
 * @throws std::out_of_range if id >= NUM_GEMM_CATEGORIES
 */
gemm_category_t category_from_id(std::size_t id);

/**
 * @brief Compute GEMM arithmetic intensity (layout-independent).
 *
 *   AI = 2*M*N*K / ((M*K + K*N + M*N) * bytes_per_element)
 */
double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element = 2.0) noexcept;

}  // namespace origami
