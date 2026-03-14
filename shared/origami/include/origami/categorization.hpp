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
// Axis 1: Size ranges (M, N, K) — the geometric shape of the GEMM
// ============================================================================

/**
 * @brief Size range labels for M and N dimensions.
 *
 * Five logarithmic ranges at power-of-2 boundaries aligned with
 * typical macro-tile sizes (64, 128, 256) and parallelism regimes.
 *
 * The number of workgroups is ceil(M/MT_M) * ceil(N/MT_N), so M and
 * N determine tile utilization and CU occupancy.  Five ranges
 * distinguish the key parallelism regimes:
 *   tiny:   single-tile row/column, limited parallelism
 *   small:  1-4 tiles, partial-wave effects dominate
 *   medium: moderate parallelism, L2 reuse emerges
 *   large:  fully parallel, bandwidth-limited
 *   xlarge: deeply parallel, MALL/L2 capacity effects
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
 * K affects solution selection monotonically through arithmetic
 * intensity (AI):
 *
 *   AI = 2MNK / ((MK + KN + MN) * bpe)
 *   1/AI = (bpe/2) * (1/M + 1/N + 1/K)
 *
 * Increasing K shrinks the 1/K term, raising AI.  A single boundary
 * at K=2048 separates the memory-bound regime (short K, low AI) from
 * the compute-bound regime (long K, high AI) for typical hardware
 * and BF16 precision.
 */
enum class k_range_t : std::uint8_t {
  short_k = 0,  ///< [1, 2048]    — typically memory-bound
  long_k  = 1,  ///< [2049, inf)  — typically compute-bound

  count   = 2
};

inline constexpr std::array<std::size_t, 5> MN_RANGE_UPPER_BOUNDS = {
    64, 256, 1024, 4096, SIZE_MAX};

inline constexpr std::array<std::size_t, 2> K_RANGE_UPPER_BOUNDS = {
    2048, SIZE_MAX};

/// Number of size-only categories: 5(M) * 5(N) * 2(K) = 50.
inline constexpr std::size_t NUM_SIZE_CATEGORIES =
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(mn_range_t::count) *
    static_cast<std::size_t>(k_range_t::count);

static_assert(NUM_SIZE_CATEGORIES == 50, "Size category count must be 50");

// ============================================================================
// Axis 2: Layout — matrix transpose combination
// ============================================================================

/**
 * @brief Matrix layout (transpose pair) for op(A) * op(B).
 *
 * Layout determines which dimension is contiguous in memory for each
 * operand, critically affecting vectorized load width, L2 cache line
 * utilization, and coalescing efficiency:
 *
 *   Layout | A contiguous dim | B contiguous dim
 *   -------+------------------+-----------------
 *     NN   |       M          |       K
 *     NT   |       M          |       N
 *     TN   |       K          |       K
 *     TT   |       K          |       N
 *
 * (Column-major / BLAS convention: op(A) is M x K.
 *  transA=N → A is M x K stored col-major, M contiguous.
 *  transA=T → A is K x M stored col-major, K contiguous.)
 *
 * TN is the most common layout in deep learning (both operands
 * have K contiguous, enabling the widest vector loads in the
 * inner loop).
 */
enum class layout_t : std::uint8_t {
  NN = 0,
  NT = 1,
  TN = 2,
  TT = 3,

  count = 4
};

// ============================================================================
// Axis 3: Data type class — grouped by compute precision
// ============================================================================

/**
 * @brief Data type class grouping types with similar compute behavior.
 *
 * Types within a class share:
 *   - Similar bytes-per-element (bpe) → same AI at given (M,N,K)
 *   - Same or similar matrix instruction dimensions (MI_M, MI_N, MI_K)
 *   - Similar valid macro-tile sizes
 *
 * Grouping prevents category explosion while preserving the key
 * distinctions that affect solution selection.
 */
enum class dtype_class_t : std::uint8_t {
  f64      = 0,  ///< Double, ComplexDouble          (8 bytes)
  f32      = 1,  ///< Float, XFloat32                (4 bytes)
  f16      = 2,  ///< Half, BFloat16                 (2 bytes)
  f8       = 3,  ///< Float8*, BFloat8* variants     (1 byte)
  i8       = 4,  ///< Int8                           (1 byte, integer path)
  sub_byte = 5,  ///< Int4, Float4, Float6, BFloat6  (< 1 byte)

  count    = 6
};

// ============================================================================
// Axis 4: Batch regime
// ============================================================================

/**
 * @brief Batch regime.
 *
 * Batched GEMMs (batch > 1) use fundamentally different workgroup
 * mapping strategies (staggerU disabled, WGM changed) and can
 * sometimes fuse across the batch dimension for better occupancy.
 */
enum class batch_class_t : std::uint8_t {
  single  = 0,  ///< batch == 1
  batched = 1,  ///< batch > 1

  count   = 2
};

// ============================================================================
// Composite category types
// ============================================================================

/**
 * @brief Size-only category (M/N/K ranges).
 *
 * Captures the geometric shape of the GEMM independent of layout,
 * data type, or batch count.  50 categories.
 */
struct gemm_size_category_t {
  mn_range_t m_range;
  mn_range_t n_range;
  k_range_t  k_range;

  /// Unique id in [0, NUM_SIZE_CATEGORIES).
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

  std::string to_string() const;

  bool operator==(const gemm_size_category_t& o) const noexcept {
    return m_range == o.m_range && n_range == o.n_range && k_range == o.k_range;
  }
  bool operator!=(const gemm_size_category_t& o) const noexcept { return !(*this == o); }
};

/// Total heuristic key space: 50 sizes * 4 layouts * 6 dtypes * 2 batch = 2400.
inline constexpr std::size_t NUM_FULL_CATEGORIES =
    NUM_SIZE_CATEGORIES *
    static_cast<std::size_t>(layout_t::count) *
    static_cast<std::size_t>(dtype_class_t::count) *
    static_cast<std::size_t>(batch_class_t::count);

static_assert(NUM_FULL_CATEGORIES == 2400, "Full category space must be 2400");

/**
 * @brief Full GEMM problem category — the complete heuristic lookup key.
 *
 * Combines all four classification axes extracted from problem_t:
 *   1. Size regime   (M/N/K ranges)   — 50 values
 *   2. Layout        (transpose pair)  — 4 values
 *   3. Data type     (precision class)  — 6 values
 *   4. Batch regime  (single/batched)  — 2 values
 *
 * Design rationale (following NVIDIA CUTLASS/nvMatmulHeuristics approach):
 *
 *   NVIDIA's cublasLt classifies problems by
 *   (op_type, A_type, B_type, compute_type, layout_A, layout_B)
 *   then selects tile sizes (MatmulTile), cluster shapes (ClusterShape),
 *   and split-K strategies per classification.
 *
 *   Similarly, this categorization serves as a heuristic key:
 *     heuristic_params = lookup(category.full_id())
 *
 * Memory layout semantics (BLAS column-major convention):
 *
 *   For op(A)=M x K, op(B)=K x N:
 *
 *   | Layout | A stored as | A contiguous dim | B stored as | B contiguous dim |
 *   |--------|-------------|------------------|-------------|------------------|
 *   |   NN   |  M x K      |       M          |  K x N      |       K          |
 *   |   NT   |  M x K      |       M          |  N x K      |       N          |
 *   |   TN   |  K x M      |       K          |  K x N      |       K          |
 *   |   TT   |  K x M      |       K          |  N x K      |       N          |
 *
 *   When the contiguous dimension is small, vectorized loads are narrow
 *   and cache line utilization drops.  This affects the optimal tile
 *   shape and stagger strategy within each size category.
 */
struct gemm_category_t {
  gemm_size_category_t size;
  layout_t             layout;
  dtype_class_t        dtype;
  batch_class_t        batch;

  /// Unique id in [0, NUM_FULL_CATEGORIES).
  std::size_t full_id() const noexcept;

  /// Size-only category id in [0, NUM_SIZE_CATEGORIES) for backward compat.
  std::size_t size_id() const noexcept { return size.id(); }

  /**
   * @brief The contiguous (fastest-varying) dimension for operand A.
   *
   * Returns 'm' if M is contiguous in memory (transA=N),
   *         'k' if K is contiguous in memory (transA=T).
   */
  char contiguous_dim_a() const noexcept;

  /**
   * @brief The contiguous (fastest-varying) dimension for operand B.
   *
   * Returns 'k' if K is contiguous (transB=N),
   *         'n' if N is contiguous (transB=T).
   */
  char contiguous_dim_b() const noexcept;

  /**
   * @brief Bytes per element for the classified data type.
   */
  double bytes_per_element() const noexcept;

  /**
   * @brief Arithmetic intensity using this category's dtype bpe.
   */
  double representative_arithmetic_intensity() const noexcept;

  std::string to_string() const;

  bool operator==(const gemm_category_t& o) const noexcept {
    return size == o.size && layout == o.layout && dtype == o.dtype && batch == o.batch;
  }
  bool operator!=(const gemm_category_t& o) const noexcept { return !(*this == o); }
};

// ============================================================================
// Classification functions
// ============================================================================

mn_range_t classify_mn(std::size_t dim) noexcept;
k_range_t classify_k(std::size_t dim) noexcept;
layout_t classify_layout(transpose_t a_transpose, transpose_t b_transpose) noexcept;
dtype_class_t classify_dtype(data_type_t mi_dtype) noexcept;
batch_class_t classify_batch(std::size_t batch) noexcept;

/**
 * @brief Categorize a GEMM problem using all fields of problem_t.
 *
 * Extracts:
 *   - size.m, size.n, size.k → size category
 *   - a_transpose, b_transpose → layout
 *   - mi_dtype → dtype class
 *   - batch → batch regime
 *
 * @param problem GEMM problem description
 * @return gemm_category_t Full category descriptor
 */
gemm_category_t categorize(const problem_t& problem) noexcept;

/**
 * @brief Categorize from raw dimensions only (size category, defaults for rest).
 *
 * Uses TN layout, f16 dtype, single batch as defaults.
 */
gemm_size_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept;

/**
 * @brief Reconstruct a size category from its integer id.
 *
 * @param id Category id in [0, NUM_SIZE_CATEGORIES)
 * @throws std::out_of_range if id >= NUM_SIZE_CATEGORIES
 */
gemm_size_category_t size_category_from_id(std::size_t id);

/**
 * @brief Reconstruct a full category from its integer id.
 *
 * @param id Category id in [0, NUM_FULL_CATEGORIES)
 * @throws std::out_of_range if id >= NUM_FULL_CATEGORIES
 */
gemm_category_t category_from_full_id(std::size_t id);

/**
 * @brief Compute GEMM arithmetic intensity.
 *
 *   AI = 2*M*N*K / ((M*K + K*N + M*N) * bytes_per_element)
 *
 * @param m M dimension
 * @param n N dimension
 * @param k K dimension
 * @param bytes_per_element Element size in bytes (default 2.0 for BF16)
 * @return double Arithmetic intensity (ops/byte)
 */
double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element = 2.0) noexcept;

// ============================================================================
// String conversion helpers
// ============================================================================

const char* layout_to_string(layout_t layout) noexcept;
const char* dtype_class_to_string(dtype_class_t dtype) noexcept;
const char* batch_class_to_string(batch_class_t batch) noexcept;

}  // namespace origami
