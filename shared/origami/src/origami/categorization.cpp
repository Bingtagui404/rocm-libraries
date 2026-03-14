// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/categorization.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace origami {

// ============================================================================
// Range classification helpers
// ============================================================================

mn_range_t classify_mn(std::size_t dim) noexcept {
  for (std::size_t i = 0; i < MN_RANGE_UPPER_BOUNDS.size(); ++i) {
    if (dim <= MN_RANGE_UPPER_BOUNDS[i]) {
      return static_cast<mn_range_t>(i);
    }
  }
  return mn_range_t::xlarge;
}

k_range_t classify_k(std::size_t dim) noexcept {
  for (std::size_t i = 0; i < K_RANGE_UPPER_BOUNDS.size(); ++i) {
    if (dim <= K_RANGE_UPPER_BOUNDS[i]) {
      return static_cast<k_range_t>(i);
    }
  }
  return k_range_t::long_k;
}

// ============================================================================
// Categorization
// ============================================================================

gemm_category_t categorize(const problem_t& problem) noexcept {
  return categorize_mnk(problem.size.m, problem.size.n, problem.size.k);
}

gemm_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept {
  return {classify_mn(m), classify_mn(n), classify_k(k)};
}

gemm_category_t category_from_id(std::size_t id) {
  if (id >= NUM_GEMM_CATEGORIES) {
    throw std::out_of_range("Category id " + std::to_string(id) +
                            " is out of range [0, " +
                            std::to_string(NUM_GEMM_CATEGORIES) + ")");
  }

  const auto k_count = static_cast<std::size_t>(k_range_t::count);
  const auto n_count = static_cast<std::size_t>(mn_range_t::count);

  auto k_idx = id % k_count;
  auto n_idx = (id / k_count) % n_count;
  auto m_idx = id / (k_count * n_count);

  return {static_cast<mn_range_t>(m_idx),
          static_cast<mn_range_t>(n_idx),
          static_cast<k_range_t>(k_idx)};
}

// ============================================================================
// gemm_category_t members
// ============================================================================

std::size_t gemm_category_t::id() const noexcept {
  const auto k_count = static_cast<std::size_t>(k_range_t::count);
  const auto n_count = static_cast<std::size_t>(mn_range_t::count);

  return static_cast<std::size_t>(m_range) * n_count * k_count +
         static_cast<std::size_t>(n_range) * k_count +
         static_cast<std::size_t>(k_range);
}

static std::size_t mn_lower_bound(mn_range_t r) noexcept {
  auto idx = static_cast<std::size_t>(r);
  return idx == 0 ? 1 : MN_RANGE_UPPER_BOUNDS[idx - 1] + 1;
}

static std::size_t k_lower_bound(k_range_t r) noexcept {
  auto idx = static_cast<std::size_t>(r);
  return idx == 0 ? 1 : K_RANGE_UPPER_BOUNDS[idx - 1] + 1;
}

std::size_t gemm_category_t::m_lower() const noexcept { return mn_lower_bound(m_range); }
std::size_t gemm_category_t::m_upper() const noexcept { return MN_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(m_range)]; }
std::size_t gemm_category_t::n_lower() const noexcept { return mn_lower_bound(n_range); }
std::size_t gemm_category_t::n_upper() const noexcept { return MN_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(n_range)]; }
std::size_t gemm_category_t::k_lower() const noexcept { return k_lower_bound(k_range); }
std::size_t gemm_category_t::k_upper() const noexcept { return K_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(k_range)]; }

double gemm_category_t::representative_arithmetic_intensity(double bytes_per_element) const noexcept {
  constexpr double XLARGE_CAP = 16384.0;

  auto geom_mean = [&](double lo, double hi) -> double {
    double effective_hi = (hi == static_cast<double>(SIZE_MAX)) ? XLARGE_CAP : hi;
    return std::sqrt(lo * effective_hi);
  };

  double m = geom_mean(static_cast<double>(m_lower()), static_cast<double>(m_upper()));
  double n = geom_mean(static_cast<double>(n_lower()), static_cast<double>(n_upper()));
  double k = geom_mean(static_cast<double>(k_lower()), static_cast<double>(k_upper()));

  return compute_arithmetic_intensity(m, n, k, bytes_per_element);
}

std::string gemm_category_t::to_string() const {
  auto format_bound = [](std::size_t v) -> std::string {
    return v == SIZE_MAX ? "inf" : std::to_string(v);
  };

  return "cat" + std::string(id() < 10 ? "0" : "") +
         std::to_string(id()) +
         "_M[" + std::to_string(m_lower()) + "-" + format_bound(m_upper()) + "]" +
         "_N[" + std::to_string(n_lower()) + "-" + format_bound(n_upper()) + "]" +
         "_K[" + std::to_string(k_lower()) + "-" + format_bound(k_upper()) + "]";
}

// ============================================================================
// Arithmetic Intensity
// ============================================================================

double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element) noexcept {
  double flops  = 2.0 * m * n * k;
  double bytes  = (m * k + k * n + m * n) * bytes_per_element;
  if (bytes <= 0.0) return 0.0;
  return flops / bytes;
}

}  // namespace origami
