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
 * The boundaries are derived from the origami latency model and
 * TensileLite's solution-matching behavior:
 *
 * 1. ORIGAMI'S LATENCY MODEL decomposes total latency as:
 *
 *      L_total = L_timestep * num_timesteps
 *
 *    where num_timesteps = ceil(num_tiles / N_CU), and
 *    num_tiles = ceil(M/MT_M) * ceil(N/MT_N).
 *
 *    The key regime transitions occur when num_tiles crosses
 *    multiples of N_CU (256 for gfx950).  For the maximum tile
 *    size MT_max = 256, this gives:
 *
 *      num_tiles_per_dim = M / MT_max
 *
 *      M <=  256:  1 tile,    num_tiles_per_dim <= 1   (sub-tile)
 *      M <= 1024:  2-4 tiles, partial-wave occupancy
 *      M <= 4096:  4-16 tiles, approaching full-GPU
 *      M >  4096:  16+ tiles, multi-wave
 *
 * 2. TENSILELITE'S RATIO DISTANCE operates in log-space:
 *
 *      d(p1, p2) = |log(M1/M2)| + |log(N1/N2)| + |log(K1/K2)|
 *
 *    This means solutions are matched by multiplicative ratio,
 *    not absolute difference.  The natural binning in log-space
 *    is geometric: each boundary should be a constant multiple
 *    of the previous one.
 *
 *    The ratio between consecutive boundaries is 4x:
 *      1 -> 64 -> 256 -> 1024 -> 4096 -> inf
 *           (x4)   (x4)    (x4)
 *
 *    In log2-space, the boundaries are at {6, 8, 10, 12},
 *    i.e. evenly spaced at intervals of 2 (factor of 4).
 *
 * 3. TILE SIZE ALIGNMENT:
 *
 *    The boundaries coincide with the macro-tile range used in
 *    origami heuristics (MT: 32-256, from heuristics.cpp CMS configs):
 *
 *      64  = smallest practical tile (MI_M=16/32, 2-4 MI blocks)
 *      256 = largest practical tile  (MT_max in CMS kernels)
 *
 *    The work_utilization formula from gemm.cpp:
 *
 *      utilization = (M * N * K) / (ceil(M/MT_M)*MT_M * ceil(N/MT_N)*MT_N * ceil(K/MT_K)*MT_K)
 *
 *    has discontinuities at M = k*MT_M for integer k.  The range
 *    boundaries at {64, 256, 1024, 4096} bound the number of tiles
 *    (1, 1-4, 4-16, 16+) which determines how much utilization loss
 *    varies within each range.
 */
enum class mn_range_t : std::uint8_t {
  tiny   = 0,  ///< [1, 64]     — sub-tile: M < smallest practical tile
  small  = 1,  ///< [65, 256]   — single-tile: M <= MT_max (256)
  medium = 2,  ///< [257, 1024] — few-tile: 2-4 max-tiles, partial-wave
  large  = 3,  ///< [1025, 4096]— multi-tile: 4-16 max-tiles, near full-GPU
  xlarge = 4,  ///< [4097, inf) — many-tile: 16+ max-tiles, multi-wave

  count  = 5
};

/**
 * @brief Size range labels for the K (reduction) dimension.
 *
 * K controls the inner loop iteration count and arithmetic intensity.
 * From the origami latency model (gemm.cpp line 861-862):
 *
 *   L_tile = max(L_compute * w_compute, L_mem * w_memory) * num_k_iter
 *
 * The regime transition is when L_compute overtakes L_mem, i.e.
 * when the problem crosses the roofline.  This happens when:
 *
 *   AI = 2MNK / ((MK + KN + MN) * bpe) > peak_compute / peak_bandwidth
 *
 * Solving for K at the roofline crossover (for square M=N=D, bpe=2):
 *
 *   AI = DK / (2K + D)  =>  K_cross = AI_roof * D / (D - 2*AI_roof)
 *
 * For MI300X (gfx942): AI_roof ~ 247, D=1024 => K_cross ~ 477
 * For MI350X (gfx950): AI_roof ~ 288, D=1024 => K_cross ~ 658
 *
 * K=2048 is well above these crossover points for D>=1024, confirming
 * that long_k problems of moderate M/N are compute-bound on both
 * architectures.  For smaller problems (D=256), K_cross is negative
 * (always memory-bound regardless of K), which is correctly captured
 * since even long_k with tiny M/N stays in a memory-bound category.
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
 * The intended heuristic lookup combines this with layout and dtype:
 *
 *   heuristic_params = lookup(category.id(), layout, dtype)
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

  /// e.g. "cat42_M[257-1024]_N[65-256]_K[1-2048]_single"
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
