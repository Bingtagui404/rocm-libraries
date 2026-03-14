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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <set>
#include "common.hpp"
#include "origami/categorization.hpp"

using Catch::Approx;

// ========================================================================
// Size category tests (50 categories)
// ========================================================================

TEST_CASE("Categorization: NUM_SIZE_CATEGORIES is 50", "[categorization]") {
  REQUIRE(origami::NUM_SIZE_CATEGORIES == 50);
}

TEST_CASE("Categorization: NUM_FULL_CATEGORIES is 2400", "[categorization]") {
  REQUIRE(origami::NUM_FULL_CATEGORIES == 2400);
}

TEST_CASE("Categorization: classify_mn boundaries", "[categorization]") {
  REQUIRE(origami::classify_mn(1) == origami::mn_range_t::tiny);
  REQUIRE(origami::classify_mn(64) == origami::mn_range_t::tiny);
  REQUIRE(origami::classify_mn(65) == origami::mn_range_t::small);
  REQUIRE(origami::classify_mn(256) == origami::mn_range_t::small);
  REQUIRE(origami::classify_mn(257) == origami::mn_range_t::medium);
  REQUIRE(origami::classify_mn(1024) == origami::mn_range_t::medium);
  REQUIRE(origami::classify_mn(1025) == origami::mn_range_t::large);
  REQUIRE(origami::classify_mn(4096) == origami::mn_range_t::large);
  REQUIRE(origami::classify_mn(4097) == origami::mn_range_t::xlarge);
  REQUIRE(origami::classify_mn(100000) == origami::mn_range_t::xlarge);
}

TEST_CASE("Categorization: classify_k boundaries", "[categorization]") {
  REQUIRE(origami::classify_k(1) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2048) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2049) == origami::k_range_t::long_k);
  REQUIRE(origami::classify_k(65536) == origami::k_range_t::long_k);
}

TEST_CASE("Categorization: size category id uniqueness", "[categorization]") {
  std::set<std::size_t> ids;
  for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi) {
    for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni) {
      for (int ki = 0; ki < static_cast<int>(origami::k_range_t::count); ++ki) {
        origami::gemm_size_category_t cat{
            static_cast<origami::mn_range_t>(mi),
            static_cast<origami::mn_range_t>(ni),
            static_cast<origami::k_range_t>(ki)};
        REQUIRE(cat.id() < origami::NUM_SIZE_CATEGORIES);
        ids.insert(cat.id());
      }
    }
  }
  REQUIRE(ids.size() == origami::NUM_SIZE_CATEGORIES);
}

TEST_CASE("Categorization: size_category_from_id round-trip", "[categorization]") {
  for (std::size_t id = 0; id < origami::NUM_SIZE_CATEGORIES; ++id) {
    auto cat = origami::size_category_from_id(id);
    REQUIRE(cat.id() == id);
  }
}

TEST_CASE("Categorization: size_category_from_id out-of-range", "[categorization]") {
  REQUIRE_THROWS_AS(origami::size_category_from_id(50), std::out_of_range);
  REQUIRE_THROWS_AS(origami::size_category_from_id(999), std::out_of_range);
}

TEST_CASE("Categorization: full problem space coverage", "[categorization]") {
  std::vector<std::size_t> test_dims = {1, 32, 64, 65, 128, 256, 257, 512, 1024,
                                        1025, 2048, 4096, 4097, 8192, 16384};
  std::vector<std::size_t> test_k_dims = {1, 128, 512, 1024, 2048,
                                          2049, 4096, 8192, 16384, 65536};

  for (auto m : test_dims) {
    for (auto n : test_dims) {
      for (auto k : test_k_dims) {
        auto cat = origami::categorize_mnk(m, n, k);
        REQUIRE(cat.id() < origami::NUM_SIZE_CATEGORIES);
        REQUIRE(m >= cat.m_lower());
        REQUIRE(n >= cat.n_lower());
        REQUIRE(k >= cat.k_lower());
        if (cat.m_upper() != SIZE_MAX) REQUIRE(m <= cat.m_upper());
        if (cat.n_upper() != SIZE_MAX) REQUIRE(n <= cat.n_upper());
        if (cat.k_upper() != SIZE_MAX) REQUIRE(k <= cat.k_upper());
      }
    }
  }
}

// ========================================================================
// Layout classification
// ========================================================================

TEST_CASE("Categorization: classify_layout", "[categorization]") {
  REQUIRE(origami::classify_layout(origami::transpose_t::N, origami::transpose_t::N) == origami::layout_t::NN);
  REQUIRE(origami::classify_layout(origami::transpose_t::N, origami::transpose_t::T) == origami::layout_t::NT);
  REQUIRE(origami::classify_layout(origami::transpose_t::T, origami::transpose_t::N) == origami::layout_t::TN);
  REQUIRE(origami::classify_layout(origami::transpose_t::T, origami::transpose_t::T) == origami::layout_t::TT);
}

TEST_CASE("Categorization: contiguous dimensions", "[categorization]") {
  SECTION("NN: A contiguous=M, B contiguous=K") {
    origami::problem_t problem;
    problem.size = {1024, 1024, 1024};
    problem.a_transpose = origami::transpose_t::N;
    problem.b_transpose = origami::transpose_t::N;
    problem.mi_dtype = origami::data_type_t::BFloat16;
    auto cat = origami::categorize(problem);
    REQUIRE(cat.contiguous_dim_a() == 'm');
    REQUIRE(cat.contiguous_dim_b() == 'k');
  }

  SECTION("NT: A contiguous=M, B contiguous=N") {
    origami::problem_t problem;
    problem.size = {1024, 1024, 1024};
    problem.a_transpose = origami::transpose_t::N;
    problem.b_transpose = origami::transpose_t::T;
    problem.mi_dtype = origami::data_type_t::BFloat16;
    auto cat = origami::categorize(problem);
    REQUIRE(cat.contiguous_dim_a() == 'm');
    REQUIRE(cat.contiguous_dim_b() == 'n');
  }

  SECTION("TN: A contiguous=K, B contiguous=K") {
    origami::problem_t problem;
    problem.size = {1024, 1024, 1024};
    problem.a_transpose = origami::transpose_t::T;
    problem.b_transpose = origami::transpose_t::N;
    problem.mi_dtype = origami::data_type_t::BFloat16;
    auto cat = origami::categorize(problem);
    REQUIRE(cat.contiguous_dim_a() == 'k');
    REQUIRE(cat.contiguous_dim_b() == 'k');
  }

  SECTION("TT: A contiguous=K, B contiguous=N") {
    origami::problem_t problem;
    problem.size = {1024, 1024, 1024};
    problem.a_transpose = origami::transpose_t::T;
    problem.b_transpose = origami::transpose_t::T;
    problem.mi_dtype = origami::data_type_t::BFloat16;
    auto cat = origami::categorize(problem);
    REQUIRE(cat.contiguous_dim_a() == 'k');
    REQUIRE(cat.contiguous_dim_b() == 'n');
  }
}

// ========================================================================
// Data type classification
// ========================================================================

TEST_CASE("Categorization: classify_dtype", "[categorization]") {
  REQUIRE(origami::classify_dtype(origami::data_type_t::Double) == origami::dtype_class_t::f64);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Float) == origami::dtype_class_t::f32);
  REQUIRE(origami::classify_dtype(origami::data_type_t::XFloat32) == origami::dtype_class_t::f32);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Half) == origami::dtype_class_t::f16);
  REQUIRE(origami::classify_dtype(origami::data_type_t::BFloat16) == origami::dtype_class_t::f16);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Float8) == origami::dtype_class_t::f8);
  REQUIRE(origami::classify_dtype(origami::data_type_t::BFloat8) == origami::dtype_class_t::f8);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Float8BFloat8) == origami::dtype_class_t::f8);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Int8) == origami::dtype_class_t::i8);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Int4) == origami::dtype_class_t::sub_byte);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Float4) == origami::dtype_class_t::sub_byte);
  REQUIRE(origami::classify_dtype(origami::data_type_t::Float6) == origami::dtype_class_t::sub_byte);
}

TEST_CASE("Categorization: bytes_per_element by dtype class", "[categorization]") {
  origami::problem_t problem;
  problem.size = {1024, 1024, 1024};
  problem.a_transpose = origami::transpose_t::T;
  problem.b_transpose = origami::transpose_t::N;

  problem.mi_dtype = origami::data_type_t::Double;
  REQUIRE(origami::categorize(problem).bytes_per_element() == 8.0);

  problem.mi_dtype = origami::data_type_t::Float;
  REQUIRE(origami::categorize(problem).bytes_per_element() == 4.0);

  problem.mi_dtype = origami::data_type_t::BFloat16;
  REQUIRE(origami::categorize(problem).bytes_per_element() == 2.0);

  problem.mi_dtype = origami::data_type_t::Float8;
  REQUIRE(origami::categorize(problem).bytes_per_element() == 1.0);

  problem.mi_dtype = origami::data_type_t::Float4;
  REQUIRE(origami::categorize(problem).bytes_per_element() == 0.5);
}

// ========================================================================
// Batch classification
// ========================================================================

TEST_CASE("Categorization: classify_batch", "[categorization]") {
  REQUIRE(origami::classify_batch(1) == origami::batch_class_t::single);
  REQUIRE(origami::classify_batch(0) == origami::batch_class_t::single);
  REQUIRE(origami::classify_batch(2) == origami::batch_class_t::batched);
  REQUIRE(origami::classify_batch(128) == origami::batch_class_t::batched);
}

// ========================================================================
// Full categorization from problem_t
// ========================================================================

TEST_CASE("Categorization: full categorize from problem_t", "[categorization]") {
  origami::problem_t problem;
  problem.size = {2048, 4096, 1024};
  problem.batch = 1;
  problem.a_transpose = origami::transpose_t::T;
  problem.b_transpose = origami::transpose_t::N;
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;
  problem.mi_dtype = origami::data_type_t::BFloat16;

  auto cat = origami::categorize(problem);

  REQUIRE(cat.size.m_range == origami::mn_range_t::large);
  REQUIRE(cat.size.n_range == origami::mn_range_t::large);
  REQUIRE(cat.size.k_range == origami::k_range_t::short_k);
  REQUIRE(cat.layout == origami::layout_t::TN);
  REQUIRE(cat.dtype == origami::dtype_class_t::f16);
  REQUIRE(cat.batch == origami::batch_class_t::single);
}

TEST_CASE("Categorization: different layouts produce different categories", "[categorization]") {
  auto make = [](origami::transpose_t a, origami::transpose_t b) {
    origami::problem_t p;
    p.size = {1024, 1024, 4096};
    p.a_transpose = a;
    p.b_transpose = b;
    p.mi_dtype = origami::data_type_t::BFloat16;
    return origami::categorize(p);
  };

  auto tn = make(origami::transpose_t::T, origami::transpose_t::N);
  auto nt = make(origami::transpose_t::N, origami::transpose_t::T);
  auto nn = make(origami::transpose_t::N, origami::transpose_t::N);

  REQUIRE(tn.size == nt.size);
  REQUIRE(tn.layout != nt.layout);
  REQUIRE(tn.layout != nn.layout);
  REQUIRE(tn.full_id() != nt.full_id());
  REQUIRE(tn.full_id() != nn.full_id());
}

TEST_CASE("Categorization: different dtypes produce different categories", "[categorization]") {
  auto make = [](origami::data_type_t dtype) {
    origami::problem_t p;
    p.size = {1024, 1024, 4096};
    p.a_transpose = origami::transpose_t::T;
    p.b_transpose = origami::transpose_t::N;
    p.mi_dtype = dtype;
    return origami::categorize(p);
  };

  auto bf16 = make(origami::data_type_t::BFloat16);
  auto fp32 = make(origami::data_type_t::Float);
  auto fp8  = make(origami::data_type_t::Float8);

  REQUIRE(bf16.size == fp32.size);
  REQUIRE(bf16.dtype != fp32.dtype);
  REQUIRE(bf16.dtype != fp8.dtype);
  REQUIRE(bf16.full_id() != fp32.full_id());
}

TEST_CASE("Categorization: batched vs single produce different categories", "[categorization]") {
  origami::problem_t single_problem;
  single_problem.size = {1024, 1024, 1024};
  single_problem.batch = 1;
  single_problem.a_transpose = origami::transpose_t::T;
  single_problem.b_transpose = origami::transpose_t::N;
  single_problem.mi_dtype = origami::data_type_t::BFloat16;

  origami::problem_t batch_problem = single_problem;
  batch_problem.batch = 16;

  auto single_cat = origami::categorize(single_problem);
  auto batch_cat  = origami::categorize(batch_problem);

  REQUIRE(single_cat.size == batch_cat.size);
  REQUIRE(single_cat.batch != batch_cat.batch);
  REQUIRE(single_cat.full_id() != batch_cat.full_id());
}

// ========================================================================
// Full ID round-trip
// ========================================================================

TEST_CASE("Categorization: category_from_full_id round-trip", "[categorization]") {
  for (std::size_t sz = 0; sz < origami::NUM_SIZE_CATEGORIES; ++sz) {
    for (int li = 0; li < static_cast<int>(origami::layout_t::count); ++li) {
      for (int di = 0; di < static_cast<int>(origami::dtype_class_t::count); ++di) {
        for (int bi = 0; bi < static_cast<int>(origami::batch_class_t::count); ++bi) {
          origami::gemm_category_t cat{
              origami::size_category_from_id(sz),
              static_cast<origami::layout_t>(li),
              static_cast<origami::dtype_class_t>(di),
              static_cast<origami::batch_class_t>(bi)};
          auto id = cat.full_id();
          REQUIRE(id < origami::NUM_FULL_CATEGORIES);
          auto recovered = origami::category_from_full_id(id);
          REQUIRE(recovered == cat);
        }
      }
    }
  }
}

TEST_CASE("Categorization: category_from_full_id out-of-range", "[categorization]") {
  REQUIRE_THROWS_AS(origami::category_from_full_id(2400), std::out_of_range);
}

// ========================================================================
// Arithmetic intensity
// ========================================================================

TEST_CASE("Categorization: AI varies with dtype bpe", "[categorization]") {
  origami::problem_t p;
  p.size = {2048, 2048, 4096};
  p.a_transpose = origami::transpose_t::T;
  p.b_transpose = origami::transpose_t::N;

  p.mi_dtype = origami::data_type_t::Float8;
  auto cat_f8 = origami::categorize(p);

  p.mi_dtype = origami::data_type_t::BFloat16;
  auto cat_f16 = origami::categorize(p);

  p.mi_dtype = origami::data_type_t::Float;
  auto cat_f32 = origami::categorize(p);

  REQUIRE(cat_f8.representative_arithmetic_intensity() >
          cat_f16.representative_arithmetic_intensity());
  REQUIRE(cat_f16.representative_arithmetic_intensity() >
          cat_f32.representative_arithmetic_intensity());
}

TEST_CASE("Categorization: compute_arithmetic_intensity formula", "[categorization]") {
  double m = 1024, n = 1024, k = 1024, bpe = 2.0;
  double expected = 2.0 * m * n * k / ((m * k + k * n + m * n) * bpe);
  REQUIRE(origami::compute_arithmetic_intensity(m, n, k, bpe) == Approx(expected));
}

// ========================================================================
// String conversion
// ========================================================================

TEST_CASE("Categorization: to_string format", "[categorization]") {
  origami::problem_t p;
  p.size = {512, 128, 4096};
  p.a_transpose = origami::transpose_t::T;
  p.b_transpose = origami::transpose_t::N;
  p.mi_dtype = origami::data_type_t::BFloat16;
  p.batch = 1;

  auto cat = origami::categorize(p);
  auto str = cat.to_string();

  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("sz"));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_M["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("TN"));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("f16"));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("single"));
}

TEST_CASE("Categorization: layout_to_string", "[categorization]") {
  REQUIRE(std::string(origami::layout_to_string(origami::layout_t::NN)) == "NN");
  REQUIRE(std::string(origami::layout_to_string(origami::layout_t::NT)) == "NT");
  REQUIRE(std::string(origami::layout_to_string(origami::layout_t::TN)) == "TN");
  REQUIRE(std::string(origami::layout_to_string(origami::layout_t::TT)) == "TT");
}
