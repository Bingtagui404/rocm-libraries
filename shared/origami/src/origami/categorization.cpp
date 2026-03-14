// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/categorization.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace origami {

// ============================================================================
// Classification
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

batch_class_t classify_batch(std::size_t batch) noexcept {
  return batch <= 1 ? batch_class_t::single : batch_class_t::batched;
}

// ============================================================================
// Categorization
// ============================================================================

gemm_category_t categorize(const problem_t& problem) noexcept {
  return {classify_mn(problem.size.m),
          classify_mn(problem.size.n),
          classify_k(problem.size.k),
          classify_batch(problem.batch)};
}

gemm_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept {
  return {classify_mn(m), classify_mn(n), classify_k(k), batch_class_t::single};
}

gemm_category_t category_from_id(std::size_t id) {
  if (id >= NUM_GEMM_CATEGORIES) {
    throw std::out_of_range("Category id " + std::to_string(id) +
                            " out of range [0, " +
                            std::to_string(NUM_GEMM_CATEGORIES) + ")");
  }

  const auto batch_count = static_cast<std::size_t>(batch_class_t::count);
  const auto k_count     = static_cast<std::size_t>(k_range_t::count);
  const auto n_count     = static_cast<std::size_t>(mn_range_t::count);

  auto batch_idx = id % batch_count;
  auto k_idx     = (id / batch_count) % k_count;
  auto n_idx     = (id / (batch_count * k_count)) % n_count;
  auto m_idx     = id / (batch_count * k_count * n_count);

  return {static_cast<mn_range_t>(m_idx),
          static_cast<mn_range_t>(n_idx),
          static_cast<k_range_t>(k_idx),
          static_cast<batch_class_t>(batch_idx)};
}

// ============================================================================
// gemm_category_t members
// ============================================================================

std::size_t gemm_category_t::id() const noexcept {
  const auto batch_count = static_cast<std::size_t>(batch_class_t::count);
  const auto k_count     = static_cast<std::size_t>(k_range_t::count);
  const auto n_count     = static_cast<std::size_t>(mn_range_t::count);

  return static_cast<std::size_t>(m_range) * n_count * k_count * batch_count +
         static_cast<std::size_t>(n_range) * k_count * batch_count +
         static_cast<std::size_t>(k_range) * batch_count +
         static_cast<std::size_t>(batch);
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
  auto geom_mean = [](double lo, double hi) -> double {
    double effective_hi = (hi == static_cast<double>(SIZE_MAX)) ? XLARGE_CAP : hi;
    return std::sqrt(lo * effective_hi);
  };
  double m = geom_mean(static_cast<double>(m_lower()), static_cast<double>(m_upper()));
  double n = geom_mean(static_cast<double>(n_lower()), static_cast<double>(n_upper()));
  double k = geom_mean(static_cast<double>(k_lower()), static_cast<double>(k_upper()));
  return compute_arithmetic_intensity(m, n, k, bytes_per_element);
}

std::string gemm_category_t::to_string() const {
  auto fmt = [](std::size_t v) -> std::string {
    return v == SIZE_MAX ? "inf" : std::to_string(v);
  };
  auto pad = [](std::size_t id) -> std::string {
    if (id < 10)  return "00" + std::to_string(id);
    if (id < 100) return "0"  + std::to_string(id);
    return std::to_string(id);
  };

  return "cat" + pad(id()) +
         "_M[" + std::to_string(m_lower()) + "-" + fmt(m_upper()) + "]" +
         "_N[" + std::to_string(n_lower()) + "-" + fmt(n_upper()) + "]" +
         "_K[" + std::to_string(k_lower()) + "-" + fmt(k_upper()) + "]" +
         "_" + (batch == batch_class_t::batched ? "batched" : "single");
}

// ============================================================================
// Arithmetic intensity
// ============================================================================

double compute_arithmetic_intensity(double m, double n, double k,
                                    double bytes_per_element) noexcept {
  double flops = 2.0 * m * n * k;
  double bytes = (m * k + k * n + m * n) * bytes_per_element;
  if (bytes <= 0.0) return 0.0;
  return flops / bytes;
}

}  // namespace origami
