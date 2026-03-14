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

/**
 * @brief Size range labels for M and N dimensions.
 *
 * Five logarithmic ranges chosen at power-of-2 boundaries that align
 * with typical macro-tile sizes (64, 128, 256) and workgroup parallelism
 * regime transitions.
 *
 * The M and N dimensions determine tile shape and the degree of
 * parallelism (numWGs = ceil(M/MT_M) * ceil(N/MT_N)).  Five ranges
 * are needed to distinguish problems where tile utilization differs
 * significantly:
 *   - tiny:   fits in a single tile row/column (limited parallelism)
 *   - small:  1-4 tiles per dimension (partial-wave effects dominate)
 *   - medium: moderate parallelism, L2 reuse patterns emerge
 *   - large:  fully parallel, memory bandwidth limited
 *   - xlarge: deeply parallel, MALL/L2 capacity effects
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
 * Only two ranges are needed for K because K affects solution
 * selection through a single monotonic mechanism: arithmetic
 * intensity (AI).
 *
 * Arithmetic intensity for GEMM:
 *   AI = 2*M*N*K / ((M*K + K*N + M*N) * bytes_per_element)
 *
 * Equivalently:  1/AI = (bpe/2) * (1/M + 1/N + 1/K)
 *
 * Increasing K monotonically increases AI (the 1/K term shrinks).
 * At the K=2048 boundary, square problems of moderate size
 * (M=N >= 1024) cross the roofline from memory-bound to
 * compute-bound on current hardware (MI300X: ~245 ops/byte,
 * MI250X: ~120 ops/byte for BF16).
 */
enum class k_range_t : std::uint8_t {
  short_k = 0,  ///< [1, 2048]    — typically memory-bound
  long_k  = 1,  ///< [2049, inf)  — typically compute-bound

  count   = 2
};

/**
 * @brief Upper bounds (inclusive) for M/N dimension ranges.
 *
 * Index i gives the upper bound for mn_range_t(i).
 * The last entry uses SIZE_MAX to represent infinity.
 */
inline constexpr std::array<std::size_t, 5> MN_RANGE_UPPER_BOUNDS = {
    64, 256, 1024, 4096, SIZE_MAX};

/**
 * @brief Upper bounds (inclusive) for K dimension ranges.
 *
 * Index i gives the upper bound for k_range_t(i).
 * The last entry uses SIZE_MAX to represent infinity.
 */
inline constexpr std::array<std::size_t, 2> K_RANGE_UPPER_BOUNDS = {
    2048, SIZE_MAX};

/// Total number of GEMM categories: |M ranges| * |N ranges| * |K ranges|.
inline constexpr std::size_t NUM_GEMM_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count);

static_assert(NUM_GEMM_CATEGORIES == 50, "Category count must be 50");

/**
 * @brief Describes a GEMM category as a tuple of dimension ranges.
 *
 * Each category represents a region of the (M, N, K) problem space
 * where GEMMs are expected to pick similar solutions.  The category
 * ID is a unique integer in [0, 50) computed as:
 *
 *   id = m_idx * |N_ranges| * |K_ranges| + n_idx * |K_ranges| + k_idx
 *      = m_idx * 10 + n_idx * 2 + k_idx
 */
struct gemm_category_t {
  mn_range_t m_range;
  mn_range_t n_range;
  k_range_t  k_range;

  /// Unique category identifier in [0, NUM_GEMM_CATEGORIES).
  std::size_t id() const noexcept;

  /// Lower bound (inclusive) for the M dimension in this category.
  std::size_t m_lower() const noexcept;

  /// Upper bound (inclusive) for the M dimension in this category.
  std::size_t m_upper() const noexcept;

  /// Lower bound (inclusive) for the N dimension in this category.
  std::size_t n_lower() const noexcept;

  /// Upper bound (inclusive) for the N dimension in this category.
  std::size_t n_upper() const noexcept;

  /// Lower bound (inclusive) for the K dimension in this category.
  std::size_t k_lower() const noexcept;

  /// Upper bound (inclusive) for the K dimension in this category.
  std::size_t k_upper() const noexcept;

  /**
   * @brief Compute the arithmetic intensity at the geometric center of
   *        this category's (M, N, K) range.
   *
   * AI = 2*M*N*K / ((M*K + K*N + M*N) * bytes_per_element)
   *
   * Uses clamped geometric means of each range as representative values.
   * For the unbounded xlarge ranges, a representative cap of 16384 is used.
   *
   * @param bytes_per_element Element size in bytes (e.g. 2.0 for BF16)
   * @return double Arithmetic intensity (ops/byte)
   */
  double representative_arithmetic_intensity(double bytes_per_element = 2.0) const noexcept;

  /// Human-readable string, e.g. "cat21_M[257-1024]_N[65-256]_K[1-2048]".
  std::string to_string() const;

  bool operator==(const gemm_category_t& o) const noexcept {
    return m_range == o.m_range && n_range == o.n_range && k_range == o.k_range;
  }

  bool operator!=(const gemm_category_t& o) const noexcept { return !(*this == o); }
};

/**
 * @brief Classify an M or N dimension value into its range bucket.
 *
 * @param dim Dimension value (must be >= 1)
 * @return mn_range_t The range bucket for this dimension value
 */
mn_range_t classify_mn(std::size_t dim) noexcept;

/**
 * @brief Classify a K dimension value into its range bucket.
 *
 * @param dim Dimension value (must be >= 1)
 * @return k_range_t The range bucket for this dimension value
 */
k_range_t classify_k(std::size_t dim) noexcept;

/**
 * @brief Categorize a GEMM problem into one of 50 categories.
 *
 * Maps any (M, N, K) triple into a category based on logarithmic
 * ranges of each dimension.  Problems within the same category are
 * expected to select similar GEMM solutions (tile sizes, configs).
 *
 * Mathematical basis:
 *   The GEMM arithmetic intensity AI = 2MNK / ((MK+KN+MN)*bpe)
 *   determines whether a problem is compute-bound or memory-bound.
 *   1/AI = (bpe/2) * (1/M + 1/N + 1/K) shows AI is governed by
 *   the harmonic relationship of M, N, K — the smallest dimension
 *   dominates.  The 5 M/N ranges capture tile-utilization and
 *   parallelism regimes, while the 2 K ranges separate memory-bound
 *   from compute-bound problems.
 *
 * @param problem GEMM problem description
 * @return gemm_category_t Category descriptor
 */
gemm_category_t categorize(const problem_t& problem) noexcept;

/**
 * @brief Categorize by raw M, N, K dimensions.
 *
 * @param m M dimension
 * @param n N dimension
 * @param k K dimension
 * @return gemm_category_t Category descriptor
 */
gemm_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept;

/**
 * @brief Reconstruct a category descriptor from its integer id.
 *
 * @param id Category id in [0, NUM_GEMM_CATEGORIES)
 * @return gemm_category_t Category descriptor
 * @throws std::out_of_range if id >= NUM_GEMM_CATEGORIES
 */
gemm_category_t category_from_id(std::size_t id);

/**
 * @brief Compute GEMM arithmetic intensity.
 *
 *   AI = 2*M*N*K / ((M*K + K*N + M*N) * bytes_per_element)
 *
 * This is the ratio of floating-point operations to bytes transferred,
 * and determines the roofline regime:
 *   - AI < hardware_peak_ops/memory_bw  →  memory-bound
 *   - AI > hardware_peak_ops/memory_bw  →  compute-bound
 *
 * @param m M dimension
 * @param n N dimension
 * @param k K dimension
 * @param bytes_per_element Element size in bytes (e.g. 2.0 for BF16)
 * @return double Arithmetic intensity (ops/byte)
 */
double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element = 2.0) noexcept;

}  // namespace origami
