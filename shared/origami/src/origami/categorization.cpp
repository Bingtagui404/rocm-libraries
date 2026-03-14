// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/categorization.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace origami {

// ============================================================================
// Range classification
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

layout_t classify_layout(transpose_t a_transpose, transpose_t b_transpose) noexcept {
  if (a_transpose == transpose_t::N && b_transpose == transpose_t::N) return layout_t::NN;
  if (a_transpose == transpose_t::N && b_transpose == transpose_t::T) return layout_t::NT;
  if (a_transpose == transpose_t::T && b_transpose == transpose_t::N) return layout_t::TN;
  return layout_t::TT;
}

dtype_class_t classify_dtype(data_type_t mi_dtype) noexcept {
  switch (mi_dtype) {
    case data_type_t::Double:
    case data_type_t::ComplexDouble:
    case data_type_t::ComplexFloat:
      return dtype_class_t::f64;

    case data_type_t::Float:
    case data_type_t::XFloat32:
      return dtype_class_t::f32;

    case data_type_t::Half:
    case data_type_t::BFloat16:
      return dtype_class_t::f16;

    case data_type_t::Float8:
    case data_type_t::BFloat8:
    case data_type_t::Float8BFloat8:
    case data_type_t::BFloat8Float8:
    case data_type_t::Float8_fnuz:
    case data_type_t::BFloat8_fnuz:
    case data_type_t::Float8BFloat8_fnuz:
    case data_type_t::BFloat8Float8_fnuz:
      return dtype_class_t::f8;

    case data_type_t::Int8:
    case data_type_t::Int8x4:
    case data_type_t::Int32:
    case data_type_t::Int64:
      return dtype_class_t::i8;

    case data_type_t::Int4:
    case data_type_t::Float4:
    case data_type_t::Float6:
    case data_type_t::BFloat6:
      return dtype_class_t::sub_byte;

    default:
      return dtype_class_t::f16;
  }
}

batch_class_t classify_batch(std::size_t batch) noexcept {
  return batch <= 1 ? batch_class_t::single : batch_class_t::batched;
}

// ============================================================================
// Categorization
// ============================================================================

gemm_category_t categorize(const problem_t& problem) noexcept {
  return {
      {classify_mn(problem.size.m), classify_mn(problem.size.n), classify_k(problem.size.k)},
      classify_layout(problem.a_transpose, problem.b_transpose),
      classify_dtype(problem.mi_dtype),
      classify_batch(problem.batch)};
}

gemm_size_category_t categorize_mnk(std::size_t m, std::size_t n, std::size_t k) noexcept {
  return {classify_mn(m), classify_mn(n), classify_k(k)};
}

// ============================================================================
// gemm_size_category_t
// ============================================================================

std::size_t gemm_size_category_t::id() const noexcept {
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

std::size_t gemm_size_category_t::m_lower() const noexcept { return mn_lower_bound(m_range); }
std::size_t gemm_size_category_t::m_upper() const noexcept { return MN_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(m_range)]; }
std::size_t gemm_size_category_t::n_lower() const noexcept { return mn_lower_bound(n_range); }
std::size_t gemm_size_category_t::n_upper() const noexcept { return MN_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(n_range)]; }
std::size_t gemm_size_category_t::k_lower() const noexcept { return k_lower_bound(k_range); }
std::size_t gemm_size_category_t::k_upper() const noexcept { return K_RANGE_UPPER_BOUNDS[static_cast<std::size_t>(k_range)]; }

double gemm_size_category_t::representative_arithmetic_intensity(double bytes_per_element) const noexcept {
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

std::string gemm_size_category_t::to_string() const {
  auto fmt = [](std::size_t v) -> std::string {
    return v == SIZE_MAX ? "inf" : std::to_string(v);
  };
  return "sz" + std::string(id() < 10 ? "0" : "") + std::to_string(id()) +
         "_M[" + std::to_string(m_lower()) + "-" + fmt(m_upper()) + "]" +
         "_N[" + std::to_string(n_lower()) + "-" + fmt(n_upper()) + "]" +
         "_K[" + std::to_string(k_lower()) + "-" + fmt(k_upper()) + "]";
}

gemm_size_category_t size_category_from_id(std::size_t id) {
  if (id >= NUM_SIZE_CATEGORIES) {
    throw std::out_of_range("Size category id " + std::to_string(id) +
                            " out of range [0, " +
                            std::to_string(NUM_SIZE_CATEGORIES) + ")");
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
// gemm_category_t
// ============================================================================

std::size_t gemm_category_t::full_id() const noexcept {
  const auto batch_count = static_cast<std::size_t>(batch_class_t::count);
  const auto dtype_count = static_cast<std::size_t>(dtype_class_t::count);
  const auto layout_count = static_cast<std::size_t>(layout_t::count);

  return size.id() * layout_count * dtype_count * batch_count +
         static_cast<std::size_t>(layout) * dtype_count * batch_count +
         static_cast<std::size_t>(dtype) * batch_count +
         static_cast<std::size_t>(batch);
}

char gemm_category_t::contiguous_dim_a() const noexcept {
  return (layout == layout_t::TN || layout == layout_t::TT) ? 'k' : 'm';
}

char gemm_category_t::contiguous_dim_b() const noexcept {
  return (layout == layout_t::NT || layout == layout_t::TT) ? 'n' : 'k';
}

double gemm_category_t::bytes_per_element() const noexcept {
  switch (dtype) {
    case dtype_class_t::f64:      return 8.0;
    case dtype_class_t::f32:      return 4.0;
    case dtype_class_t::f16:      return 2.0;
    case dtype_class_t::f8:       return 1.0;
    case dtype_class_t::i8:       return 1.0;
    case dtype_class_t::sub_byte: return 0.5;
    default:                      return 2.0;
  }
}

double gemm_category_t::representative_arithmetic_intensity() const noexcept {
  return size.representative_arithmetic_intensity(bytes_per_element());
}

std::string gemm_category_t::to_string() const {
  return size.to_string() +
         "_" + layout_to_string(layout) +
         "_" + dtype_class_to_string(dtype) +
         "_" + batch_class_to_string(batch);
}

gemm_category_t category_from_full_id(std::size_t id) {
  if (id >= NUM_FULL_CATEGORIES) {
    throw std::out_of_range("Full category id " + std::to_string(id) +
                            " out of range [0, " +
                            std::to_string(NUM_FULL_CATEGORIES) + ")");
  }

  const auto batch_count  = static_cast<std::size_t>(batch_class_t::count);
  const auto dtype_count  = static_cast<std::size_t>(dtype_class_t::count);
  const auto layout_count = static_cast<std::size_t>(layout_t::count);

  auto batch_idx  = id % batch_count;
  auto dtype_idx  = (id / batch_count) % dtype_count;
  auto layout_idx = (id / (batch_count * dtype_count)) % layout_count;
  auto size_idx   = id / (batch_count * dtype_count * layout_count);

  return {size_category_from_id(size_idx),
          static_cast<layout_t>(layout_idx),
          static_cast<dtype_class_t>(dtype_idx),
          static_cast<batch_class_t>(batch_idx)};
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

// ============================================================================
// String helpers
// ============================================================================

const char* layout_to_string(layout_t layout) noexcept {
  switch (layout) {
    case layout_t::NN: return "NN";
    case layout_t::NT: return "NT";
    case layout_t::TN: return "TN";
    case layout_t::TT: return "TT";
    default:           return "??";
  }
}

const char* dtype_class_to_string(dtype_class_t dtype) noexcept {
  switch (dtype) {
    case dtype_class_t::f64:      return "f64";
    case dtype_class_t::f32:      return "f32";
    case dtype_class_t::f16:      return "f16";
    case dtype_class_t::f8:       return "f8";
    case dtype_class_t::i8:       return "i8";
    case dtype_class_t::sub_byte: return "sub_byte";
    default:                      return "??";
  }
}

const char* batch_class_to_string(batch_class_t batch) noexcept {
  switch (batch) {
    case batch_class_t::single:  return "single";
    case batch_class_t::batched: return "batched";
    default:                     return "??";
  }
}

}  // namespace origami
