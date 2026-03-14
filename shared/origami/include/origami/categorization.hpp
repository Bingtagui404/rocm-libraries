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
 * Logarithmic ranges chosen at power-of-2 boundaries that align with
 * typical macro-tile sizes and GEMM regime transitions.
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
 * Fewer ranges than M/N because K primarily affects loop iteration
 * count and memory-vs-compute balance rather than tile shape.
 */
enum class k_range_t : std::uint8_t {
  small  = 0,  ///< [1, 256]
  medium = 1,  ///< [257, 2048]
  large  = 2,  ///< [2049, 8192]
  xlarge = 3,  ///< [8193, inf)

  count  = 4
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
inline constexpr std::array<std::size_t, 4> K_RANGE_UPPER_BOUNDS = {
    256, 2048, 8192, SIZE_MAX};

/// Total number of GEMM categories: |M ranges| * |N ranges| * |K ranges|.
inline constexpr std::size_t NUM_GEMM_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count);

static_assert(NUM_GEMM_CATEGORIES == 100, "Category count must be 100");

/**
 * @brief Describes a GEMM category as a tuple of dimension ranges.
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

  /// Human-readable string, e.g. "cat042_M[257-1024]_N[65-256]_K[257-2048]".
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
 * @brief Categorize a GEMM problem into one of 100 categories.
 *
 * Maps any (M, N, K) triple into a category based on logarithmic
 * ranges of each dimension. Problems within the same category are
 * expected to select similar GEMM solutions (tile sizes, configs).
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

}  // namespace origami
