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

TEST_CASE("Categorization: NUM_GEMM_CATEGORIES is 50", "[categorization]") {
  REQUIRE(origami::NUM_GEMM_CATEGORIES == 50);
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
}

TEST_CASE("Categorization: classify_k boundaries", "[categorization]") {
  REQUIRE(origami::classify_k(1) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2048) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2049) == origami::k_range_t::long_k);
  REQUIRE(origami::classify_k(65536) == origami::k_range_t::long_k);
}

TEST_CASE("Categorization: id uniqueness", "[categorization]") {
  std::set<std::size_t> ids;
  for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi)
    for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni)
      for (int ki = 0; ki < static_cast<int>(origami::k_range_t::count); ++ki) {
        auto id = origami::categorize_mnk(
            origami::MN_RANGE_UPPER_BOUNDS[mi],
            origami::MN_RANGE_UPPER_BOUNDS[ni],
            origami::K_RANGE_UPPER_BOUNDS[ki]).id();
        REQUIRE(id < origami::NUM_GEMM_CATEGORIES);
        ids.insert(id);
      }
  REQUIRE(ids.size() == origami::NUM_GEMM_CATEGORIES);
}

TEST_CASE("Categorization: category_from_id round-trip", "[categorization]") {
  for (std::size_t id = 0; id < origami::NUM_GEMM_CATEGORIES; ++id) {
    REQUIRE(origami::category_from_id(id).id() == id);
  }
}

TEST_CASE("Categorization: category_from_id out-of-range", "[categorization]") {
  REQUIRE_THROWS_AS(origami::category_from_id(50), std::out_of_range);
}

TEST_CASE("Categorization: categorize from problem_t", "[categorization]") {
  auto problem = make_problem(2048, 4096, 1024);
  auto cat     = origami::categorize(problem);
  REQUIRE(cat.m_range == origami::mn_range_t::large);
  REQUIRE(cat.n_range == origami::mn_range_t::large);
  REQUIRE(cat.k_range == origami::k_range_t::short_k);
  REQUIRE(cat.batched == false);
}

TEST_CASE("Categorization: batch flag set correctly", "[categorization]") {
  auto single_problem = make_problem(1024, 1024, 1024, origami::transpose_t::T,
                                     origami::transpose_t::N, 1);
  auto batch_problem  = make_problem(1024, 1024, 1024, origami::transpose_t::T,
                                     origami::transpose_t::N, 16);

  auto single_cat = origami::categorize(single_problem);
  auto batch_cat  = origami::categorize(batch_problem);

  REQUIRE(single_cat.batched == false);
  REQUIRE(batch_cat.batched == true);
  REQUIRE(single_cat.id() == batch_cat.id());
}

TEST_CASE("Categorization: layout does NOT change id", "[categorization]") {
  origami::problem_t p1;
  p1.size = {1024, 2048, 4096};
  p1.a_transpose = origami::transpose_t::T;
  p1.b_transpose = origami::transpose_t::N;
  p1.mi_dtype = origami::data_type_t::BFloat16;

  origami::problem_t p2 = p1;
  p2.a_transpose = origami::transpose_t::N;
  p2.b_transpose = origami::transpose_t::T;

  REQUIRE(origami::categorize(p1).id() == origami::categorize(p2).id());
}

TEST_CASE("Categorization: dtype does NOT change id", "[categorization]") {
  origami::problem_t p1;
  p1.size = {1024, 1024, 4096};
  p1.mi_dtype = origami::data_type_t::BFloat16;

  origami::problem_t p2 = p1;
  p2.mi_dtype = origami::data_type_t::Float;

  REQUIRE(origami::categorize(p1).id() == origami::categorize(p2).id());
}

TEST_CASE("Categorization: boundary corners", "[categorization]") {
  REQUIRE(origami::categorize_mnk(1, 1, 1).id() == 0);
  REQUIRE(origami::categorize_mnk(99999, 99999, 99999).id() == 49);
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

TEST_CASE("Categorization: full problem space coverage", "[categorization]") {
  std::vector<std::size_t> dims = {1, 32, 64, 65, 128, 256, 257, 512, 1024,
                                   1025, 2048, 4096, 4097, 8192, 16384};
  std::vector<std::size_t> k_dims = {1, 512, 2048, 2049, 8192, 65536};

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

TEST_CASE("Categorization: AI formula", "[categorization]") {
  double m = 1024, n = 1024, k = 1024, bpe = 2.0;
  double expected = 2.0 * m * n * k / ((m * k + k * n + m * n) * bpe);
  REQUIRE(origami::compute_arithmetic_intensity(m, n, k, bpe) == Approx(expected));
}

TEST_CASE("Categorization: long_k AI > short_k AI", "[categorization]") {
  for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi)
    for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni) {
      origami::gemm_category_t sk{static_cast<origami::mn_range_t>(mi),
                                   static_cast<origami::mn_range_t>(ni),
                                   origami::k_range_t::short_k, false};
      origami::gemm_category_t lk{static_cast<origami::mn_range_t>(mi),
                                   static_cast<origami::mn_range_t>(ni),
                                   origami::k_range_t::long_k, false};
      REQUIRE(lk.representative_arithmetic_intensity() >
              sk.representative_arithmetic_intensity());
    }
}

TEST_CASE("Categorization: to_string", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 128, 4096);
  auto str = cat.to_string();
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("cat"));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_M["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("single"));
}
