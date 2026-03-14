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
 * This determines parallelism and tile utilization. Five ranges capture
 * the distinct occupancy regimes regardless of layout:
 *
 *   tiny:   fits in a single tile (limited parallelism)
 *   small:  1-4 tiles per dim (partial-wave effects)
 *   medium: moderate parallelism, L2 reuse emerges
 *   large:  fully parallel, bandwidth-limited
 *   xlarge: deeply parallel, MALL/L2 capacity effects
 *
 * These roles are transpose-invariant: the output is always M x N,
 * the tile grid is always ceil(M/MT_M) * ceil(N/MT_N), regardless of
 * whether A or B is transposed.
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
 * Increasing K monotonically raises arithmetic intensity:
 *
 *   AI = 2MNK / ((MK + KN + MN) * bpe)
 *   1/AI = (bpe/2) * (1/M + 1/N + 1/K)
 *
 * There is essentially one structural transition: from memory-bound
 * (short K) to compute-bound (long K). Two ranges suffice.
 *
 * This role is also transpose-invariant: K is always the reduction
 * dimension regardless of whether A or B stores K contiguously.
 * Layout affects HOW K-slices are loaded (coalescing pattern) but
 * not the number of loop iterations or the AI.
 */
enum class k_range_t : std::uint8_t {
  short_k = 0,  ///< [1, 2048]    — typically memory-bound
  long_k  = 1,  ///< [2049, inf)  — typically compute-bound

  count   = 2
};

// ============================================================================
// Batch regime
// ============================================================================

/**
 * @brief Batch regime.
 *
 * Batched GEMMs (batch > 1) use fundamentally different workgroup
 * mapping strategies (staggerU disabled, WGM changed, XCC mapping
 * adjusted) compared to single GEMMs.
 */
enum class batch_class_t : std::uint8_t {
  single  = 0,  ///< batch == 1
  batched = 1,  ///< batch > 1

  count   = 2
};

// ============================================================================
// Range boundaries
// ============================================================================

inline constexpr std::array<std::size_t, 5> MN_RANGE_UPPER_BOUNDS = {
    64, 256, 1024, 4096, SIZE_MAX};

inline constexpr std::array<std::size_t, 2> K_RANGE_UPPER_BOUNDS = {
    2048, SIZE_MAX};

/// Total categories: 5(M) * 5(N) * 2(K) * 2(batch) = 100.
inline constexpr std::size_t NUM_GEMM_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count) *
    static_cast<std::size_t>(batch_class_t::count);

static_assert(NUM_GEMM_CATEGORIES == 100, "Category count must be 100");

// ============================================================================
// Category type
// ============================================================================

/**
 * @brief GEMM category — size-based classification for heuristic lookup.
 *
 * Categorizes GEMM problems by (M, N, K, batch) into one of 100 buckets.
 * Problems in the same category are expected to pick similar solutions.
 *
 * The heuristic lookup key combines this category with layout and dtype:
 *   heuristic_params = lookup(category.id(), layout, dtype)
 *
 * Design rationale for why M/N/K ranges are layout-invariant:
 *
 *   1. M and N always define the output tile grid and parallelism.
 *      num_tiles = ceil(M/MT_M) * ceil(N/MT_N) — no transpose in this.
 *
 *   2. K always defines the reduction loop depth.
 *      AI = 2MNK / ((MK+KN+MN)*bpe) — layout-independent.
 *
 *   3. Layout changes which dimension is contiguous in memory (affecting
 *      coalescing and cache behavior), but the regime boundaries for
 *      tile utilization and compute-vs-memory balance are the same.
 *
 *   4. The heuristic RULE applied within each category should differ
 *      per layout and dtype, but the category BOUNDARIES do not.
 */
struct gemm_category_t {
  mn_range_t    m_range;
  mn_range_t    n_range;
  k_range_t     k_range;
  batch_class_t batch;

  /// Unique category id in [0, NUM_GEMM_CATEGORIES).
  std::size_t id() const noexcept;

  std::size_t m_lower() const noexcept;
  std::size_t m_upper() const noexcept;
  std::size_t n_lower() const noexcept;
  std::size_t n_upper() const noexcept;
  std::size_t k_lower() const noexcept;
  std::size_t k_upper() const noexcept;

  /**
   * @brief Arithmetic intensity at the geometric center of this category.
   *
   * @param bytes_per_element Element size in bytes (default 2.0 for BF16)
   * @return double AI in ops/byte
   */
  double representative_arithmetic_intensity(double bytes_per_element = 2.0) const noexcept;

  /// e.g. "cat042_M[257-1024]_N[65-256]_K[1-2048]_single"
  std::string to_string() const;

  bool operator==(const gemm_category_t& o) const noexcept {
    return m_range == o.m_range && n_range == o.n_range &&
           k_range == o.k_range && batch == o.batch;
  }
  bool operator!=(const gemm_category_t& o) const noexcept { return !(*this == o); }
};

// ============================================================================
// Classification functions
// ============================================================================

mn_range_t classify_mn(std::size_t dim) noexcept;
k_range_t classify_k(std::size_t dim) noexcept;
batch_class_t classify_batch(std::size_t batch) noexcept;

/**
 * @brief Categorize a GEMM problem by its size dimensions and batch.
 *
 * Uses problem.size.{m,n,k} and problem.batch.
 * Layout and dtype do NOT affect the category; they are separate
 * heuristic lookup axes.
 *
 * @param problem GEMM problem description
 * @return gemm_category_t Category descriptor
 */
gemm_category_t categorize(const problem_t& problem) noexcept;

/**
 * @brief Categorize from raw dimensions (batch defaults to 1).
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
 * @brief Compute GEMM arithmetic intensity.
 *
 *   AI = 2*M*N*K / ((M*K + K*N + M*N) * bytes_per_element)
 *
 * Layout-independent: only depends on dimensions and element size.
 */
double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element = 2.0) noexcept;

}  // namespace origami
