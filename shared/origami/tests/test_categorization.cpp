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

TEST_CASE("Categorization: NUM_GEMM_CATEGORIES is 100", "[categorization]") {
  REQUIRE(origami::NUM_GEMM_CATEGORIES == 100);
}

// ========================================================================
// classify_mn
// ========================================================================

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

// ========================================================================
// classify_k
// ========================================================================

TEST_CASE("Categorization: classify_k boundaries", "[categorization]") {
  REQUIRE(origami::classify_k(1) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2048) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2049) == origami::k_range_t::long_k);
  REQUIRE(origami::classify_k(65536) == origami::k_range_t::long_k);
}

// ========================================================================
// classify_batch
// ========================================================================

TEST_CASE("Categorization: classify_batch", "[categorization]") {
  REQUIRE(origami::classify_batch(0) == origami::batch_class_t::single);
  REQUIRE(origami::classify_batch(1) == origami::batch_class_t::single);
  REQUIRE(origami::classify_batch(2) == origami::batch_class_t::batched);
  REQUIRE(origami::classify_batch(128) == origami::batch_class_t::batched);
}

// ========================================================================
// ID uniqueness and round-trip
// ========================================================================

TEST_CASE("Categorization: id uniqueness and range", "[categorization]") {
  std::set<std::size_t> ids;
  for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi)
    for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni)
      for (int ki = 0; ki < static_cast<int>(origami::k_range_t::count); ++ki)
        for (int bi = 0; bi < static_cast<int>(origami::batch_class_t::count); ++bi) {
          origami::gemm_category_t cat{
              static_cast<origami::mn_range_t>(mi),
              static_cast<origami::mn_range_t>(ni),
              static_cast<origami::k_range_t>(ki),
              static_cast<origami::batch_class_t>(bi)};
          auto id = cat.id();
          REQUIRE(id < origami::NUM_GEMM_CATEGORIES);
          ids.insert(id);
        }
  REQUIRE(ids.size() == origami::NUM_GEMM_CATEGORIES);
}

TEST_CASE("Categorization: category_from_id round-trip", "[categorization]") {
  for (std::size_t id = 0; id < origami::NUM_GEMM_CATEGORIES; ++id) {
    auto cat = origami::category_from_id(id);
    REQUIRE(cat.id() == id);
  }
}

TEST_CASE("Categorization: category_from_id out-of-range", "[categorization]") {
  REQUIRE_THROWS_AS(origami::category_from_id(100), std::out_of_range);
  REQUIRE_THROWS_AS(origami::category_from_id(999), std::out_of_range);
}

// ========================================================================
// categorize from problem_t
// ========================================================================

TEST_CASE("Categorization: categorize from problem_t", "[categorization]") {
  auto problem = make_problem(2048, 4096, 1024);
  auto cat     = origami::categorize(problem);
  REQUIRE(cat.m_range == origami::mn_range_t::large);
  REQUIRE(cat.n_range == origami::mn_range_t::large);
  REQUIRE(cat.k_range == origami::k_range_t::short_k);
  REQUIRE(cat.batch == origami::batch_class_t::single);
}

TEST_CASE("Categorization: layout does NOT change category", "[categorization]") {
  origami::problem_t p1;
  p1.size = {1024, 2048, 4096};
  p1.batch = 1;
  p1.a_transpose = origami::transpose_t::T;
  p1.b_transpose = origami::transpose_t::N;
  p1.mi_dtype = origami::data_type_t::BFloat16;

  origami::problem_t p2 = p1;
  p2.a_transpose = origami::transpose_t::N;
  p2.b_transpose = origami::transpose_t::T;

  origami::problem_t p3 = p1;
  p3.a_transpose = origami::transpose_t::N;
  p3.b_transpose = origami::transpose_t::N;

  REQUIRE(origami::categorize(p1) == origami::categorize(p2));
  REQUIRE(origami::categorize(p1) == origami::categorize(p3));
}

TEST_CASE("Categorization: dtype does NOT change category", "[categorization]") {
  origami::problem_t p1;
  p1.size = {1024, 1024, 4096};
  p1.batch = 1;
  p1.a_transpose = origami::transpose_t::T;
  p1.b_transpose = origami::transpose_t::N;
  p1.mi_dtype = origami::data_type_t::BFloat16;

  origami::problem_t p2 = p1;
  p2.mi_dtype = origami::data_type_t::Float;
  p2.a_dtype  = origami::data_type_t::Float;
  p2.b_dtype  = origami::data_type_t::Float;

  REQUIRE(origami::categorize(p1) == origami::categorize(p2));
}

TEST_CASE("Categorization: batch changes category", "[categorization]") {
  origami::problem_t single_p;
  single_p.size = {1024, 1024, 1024};
  single_p.batch = 1;
  single_p.a_transpose = origami::transpose_t::T;
  single_p.b_transpose = origami::transpose_t::N;
  single_p.mi_dtype = origami::data_type_t::BFloat16;

  origami::problem_t batched_p = single_p;
  batched_p.batch = 16;

  auto single_cat  = origami::categorize(single_p);
  auto batched_cat = origami::categorize(batched_p);

  REQUIRE(single_cat.m_range == batched_cat.m_range);
  REQUIRE(single_cat.n_range == batched_cat.n_range);
  REQUIRE(single_cat.k_range == batched_cat.k_range);
  REQUIRE(single_cat.batch != batched_cat.batch);
  REQUIRE(single_cat.id() != batched_cat.id());
}

// ========================================================================
// categorize_mnk (size-only, batch=single)
// ========================================================================

TEST_CASE("Categorization: categorize_mnk", "[categorization]") {
  auto cat = origami::categorize_mnk(32, 128, 512);
  REQUIRE(cat.m_range == origami::mn_range_t::tiny);
  REQUIRE(cat.n_range == origami::mn_range_t::small);
  REQUIRE(cat.k_range == origami::k_range_t::short_k);
  REQUIRE(cat.batch == origami::batch_class_t::single);
}

// ========================================================================
// Boundary and corner cases
// ========================================================================

TEST_CASE("Categorization: boundary values", "[categorization]") {
  SECTION("id 0: tiny M, tiny N, short K, single") {
    auto cat = origami::categorize_mnk(1, 1, 1);
    REQUIRE(cat.id() == 0);
  }

  SECTION("max id: xlarge M, xlarge N, long K, batched") {
    origami::gemm_category_t cat{origami::mn_range_t::xlarge,
                                  origami::mn_range_t::xlarge,
                                  origami::k_range_t::long_k,
                                  origami::batch_class_t::batched};
    REQUIRE(cat.id() == 99);
  }

  SECTION("exact boundary: K=2048 vs K=2049") {
    auto a = origami::categorize_mnk(512, 512, 2048);
    auto b = origami::categorize_mnk(512, 512, 2049);
    REQUIRE(a.k_range == origami::k_range_t::short_k);
    REQUIRE(b.k_range == origami::k_range_t::long_k);
  }
}

TEST_CASE("Categorization: bound accessors", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 128, 4096);
  REQUIRE(cat.m_lower() == 257);
  REQUIRE(cat.m_upper() == 1024);
  REQUIRE(cat.n_lower() == 65);
  REQUIRE(cat.n_upper() == 256);
  REQUIRE(cat.k_lower() == 2049);
  REQUIRE(cat.k_upper() == SIZE_MAX);
}

// ========================================================================
// Full problem space coverage
// ========================================================================

TEST_CASE("Categorization: full problem space coverage", "[categorization]") {
  std::vector<std::size_t> dims = {1, 32, 64, 65, 128, 256, 257, 512, 1024,
                                   1025, 2048, 4096, 4097, 8192, 16384};
  std::vector<std::size_t> k_dims = {1, 128, 512, 1024, 2048,
                                     2049, 4096, 8192, 16384, 65536};

  for (auto m : dims)
    for (auto n : dims)
      for (auto k : k_dims) {
        auto cat = origami::categorize_mnk(m, n, k);
        REQUIRE(cat.id() < origami::NUM_GEMM_CATEGORIES);
        REQUIRE(m >= cat.m_lower());
        REQUIRE(n >= cat.n_lower());
        REQUIRE(k >= cat.k_lower());
        if (cat.m_upper() != SIZE_MAX) REQUIRE(m <= cat.m_upper());
        if (cat.n_upper() != SIZE_MAX) REQUIRE(n <= cat.n_upper());
        if (cat.k_upper() != SIZE_MAX) REQUIRE(k <= cat.k_upper());
      }
}

// ========================================================================
// Similar/different problems
// ========================================================================

TEST_CASE("Categorization: similar problems share category", "[categorization]") {
  REQUIRE(origami::categorize_mnk(2048, 2048, 4096) ==
          origami::categorize_mnk(3000, 3500, 5000));
}

TEST_CASE("Categorization: different regimes have different categories", "[categorization]") {
  REQUIRE(origami::categorize_mnk(32, 32, 32) != origami::categorize_mnk(4096, 4096, 4096));
  REQUIRE(origami::categorize_mnk(8192, 64, 1024) != origami::categorize_mnk(64, 8192, 1024));
}

// ========================================================================
// Arithmetic intensity
// ========================================================================

TEST_CASE("Categorization: arithmetic intensity formula", "[categorization]") {
  double m = 1024, n = 1024, k = 1024, bpe = 2.0;
  double expected = 2.0 * m * n * k / ((m * k + k * n + m * n) * bpe);
  REQUIRE(origami::compute_arithmetic_intensity(m, n, k, bpe) == Approx(expected));
}

TEST_CASE("Categorization: representative AI", "[categorization]") {
  SECTION("long_k has higher AI than short_k") {
    for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi)
      for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni) {
        origami::gemm_category_t short_cat{
            static_cast<origami::mn_range_t>(mi),
            static_cast<origami::mn_range_t>(ni),
            origami::k_range_t::short_k,
            origami::batch_class_t::single};
        origami::gemm_category_t long_cat{
            static_cast<origami::mn_range_t>(mi),
            static_cast<origami::mn_range_t>(ni),
            origami::k_range_t::long_k,
            origami::batch_class_t::single};
        REQUIRE(long_cat.representative_arithmetic_intensity() >
                short_cat.representative_arithmetic_intensity());
      }
  }

  SECTION("AI is positive for all categories") {
    for (std::size_t id = 0; id < origami::NUM_GEMM_CATEGORIES; ++id) {
      REQUIRE(origami::category_from_id(id).representative_arithmetic_intensity() > 0.0);
    }
  }
}

// ========================================================================
// to_string
// ========================================================================

TEST_CASE("Categorization: to_string format", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 128, 4096);
  auto str = cat.to_string();
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("cat"));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_M["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_N["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_K["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("single"));
}
