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
  REQUIRE(origami::classify_mn(32) == origami::mn_range_t::tiny);
  REQUIRE(origami::classify_mn(64) == origami::mn_range_t::tiny);

  REQUIRE(origami::classify_mn(65) == origami::mn_range_t::small);
  REQUIRE(origami::classify_mn(128) == origami::mn_range_t::small);
  REQUIRE(origami::classify_mn(256) == origami::mn_range_t::small);

  REQUIRE(origami::classify_mn(257) == origami::mn_range_t::medium);
  REQUIRE(origami::classify_mn(512) == origami::mn_range_t::medium);
  REQUIRE(origami::classify_mn(1024) == origami::mn_range_t::medium);

  REQUIRE(origami::classify_mn(1025) == origami::mn_range_t::large);
  REQUIRE(origami::classify_mn(2048) == origami::mn_range_t::large);
  REQUIRE(origami::classify_mn(4096) == origami::mn_range_t::large);

  REQUIRE(origami::classify_mn(4097) == origami::mn_range_t::xlarge);
  REQUIRE(origami::classify_mn(8192) == origami::mn_range_t::xlarge);
  REQUIRE(origami::classify_mn(100000) == origami::mn_range_t::xlarge);
}

TEST_CASE("Categorization: classify_k boundaries", "[categorization]") {
  REQUIRE(origami::classify_k(1) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(128) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(1024) == origami::k_range_t::short_k);
  REQUIRE(origami::classify_k(2048) == origami::k_range_t::short_k);

  REQUIRE(origami::classify_k(2049) == origami::k_range_t::long_k);
  REQUIRE(origami::classify_k(4096) == origami::k_range_t::long_k);
  REQUIRE(origami::classify_k(8192) == origami::k_range_t::long_k);
  REQUIRE(origami::classify_k(65536) == origami::k_range_t::long_k);
}

TEST_CASE("Categorization: categorize_mnk basic", "[categorization]") {
  auto cat = origami::categorize_mnk(32, 128, 512);
  REQUIRE(cat.m_range == origami::mn_range_t::tiny);
  REQUIRE(cat.n_range == origami::mn_range_t::small);
  REQUIRE(cat.k_range == origami::k_range_t::short_k);
}

TEST_CASE("Categorization: categorize from problem_t", "[categorization]") {
  auto problem = make_problem(2048, 4096, 1024);
  auto cat     = origami::categorize(problem);
  REQUIRE(cat.m_range == origami::mn_range_t::large);
  REQUIRE(cat.n_range == origami::mn_range_t::large);
  REQUIRE(cat.k_range == origami::k_range_t::short_k);
}

TEST_CASE("Categorization: category id uniqueness and range", "[categorization]") {
  std::set<std::size_t> ids;

  for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi) {
    for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni) {
      for (int ki = 0; ki < static_cast<int>(origami::k_range_t::count); ++ki) {
        origami::gemm_category_t cat{
            static_cast<origami::mn_range_t>(mi),
            static_cast<origami::mn_range_t>(ni),
            static_cast<origami::k_range_t>(ki)};
        auto id = cat.id();
        REQUIRE(id < origami::NUM_GEMM_CATEGORIES);
        ids.insert(id);
      }
    }
  }

  REQUIRE(ids.size() == origami::NUM_GEMM_CATEGORIES);
}

TEST_CASE("Categorization: category_from_id round-trip", "[categorization]") {
  for (std::size_t id = 0; id < origami::NUM_GEMM_CATEGORIES; ++id) {
    auto cat = origami::category_from_id(id);
    REQUIRE(cat.id() == id);
  }
}

TEST_CASE("Categorization: category_from_id out-of-range throws", "[categorization]") {
  REQUIRE_THROWS_AS(origami::category_from_id(50), std::out_of_range);
  REQUIRE_THROWS_AS(origami::category_from_id(100), std::out_of_range);
  REQUIRE_THROWS_AS(origami::category_from_id(999), std::out_of_range);
}

TEST_CASE("Categorization: boundary values map correctly", "[categorization]") {
  SECTION("lower-left corner: tiny M, tiny N, short K") {
    auto cat = origami::categorize_mnk(1, 1, 1);
    REQUIRE(cat.id() == 0);
  }

  SECTION("upper-right corner: xlarge M, xlarge N, long K") {
    auto cat = origami::categorize_mnk(10000, 10000, 10000);
    REQUIRE(cat.id() == 49);
  }

  SECTION("exact boundary: M=64 (upper bound of tiny)") {
    auto a = origami::categorize_mnk(64, 64, 256);
    auto b = origami::categorize_mnk(65, 64, 256);
    REQUIRE(a.m_range == origami::mn_range_t::tiny);
    REQUIRE(b.m_range == origami::mn_range_t::small);
  }

  SECTION("exact boundary: K=2048 (upper bound of short_k)") {
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

TEST_CASE("Categorization: to_string format", "[categorization]") {
  auto cat = origami::categorize_mnk(512, 128, 4096);
  auto str = cat.to_string();
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("cat"));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_M["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_N["));
  REQUIRE_THAT(str, Catch::Matchers::ContainsSubstring("_K["));
}

TEST_CASE("Categorization: full problem space coverage", "[categorization]") {
  std::vector<std::size_t> test_dims = {1, 32, 64, 65, 128, 256, 257, 512, 1024,
                                        1025, 2048, 4096, 4097, 8192, 16384};
  std::vector<std::size_t> test_k_dims = {1, 128, 256, 512, 1024, 2048,
                                          2049, 4096, 8192, 16384, 65536};

  for (auto m : test_dims) {
    for (auto n : test_dims) {
      for (auto k : test_k_dims) {
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
  }
}

TEST_CASE("Categorization: similar problems share category", "[categorization]") {
  auto cat_a = origami::categorize_mnk(2048, 2048, 4096);
  auto cat_b = origami::categorize_mnk(3000, 3500, 5000);
  REQUIRE(cat_a == cat_b);
}

TEST_CASE("Categorization: different regimes have different categories", "[categorization]") {
  auto tiny_sq    = origami::categorize_mnk(32, 32, 32);
  auto large_sq   = origami::categorize_mnk(4096, 4096, 4096);
  auto tall_thin  = origami::categorize_mnk(8192, 64, 1024);
  auto short_wide = origami::categorize_mnk(64, 8192, 1024);
  auto deep_k     = origami::categorize_mnk(512, 512, 32768);

  REQUIRE(tiny_sq != large_sq);
  REQUIRE(tall_thin != short_wide);
  REQUIRE(large_sq != deep_k);
}

TEST_CASE("Categorization: equality operators", "[categorization]") {
  origami::gemm_category_t a{origami::mn_range_t::medium, origami::mn_range_t::large,
                             origami::k_range_t::short_k};
  origami::gemm_category_t b{origami::mn_range_t::medium, origami::mn_range_t::large,
                             origami::k_range_t::short_k};
  origami::gemm_category_t c{origami::mn_range_t::small, origami::mn_range_t::large,
                             origami::k_range_t::short_k};

  REQUIRE(a == b);
  REQUIRE(a != c);
}

TEST_CASE("Categorization: arithmetic intensity computation", "[categorization]") {
  SECTION("square problem AI matches formula") {
    double m = 1024, n = 1024, k = 1024;
    double bpe = 2.0;
    double expected = 2.0 * m * n * k / ((m * k + k * n + m * n) * bpe);
    double ai = origami::compute_arithmetic_intensity(m, n, k, bpe);
    REQUIRE(ai == Approx(expected));
  }

  SECTION("AI increases with K") {
    double ai_small_k = origami::compute_arithmetic_intensity(1024, 1024, 256);
    double ai_large_k = origami::compute_arithmetic_intensity(1024, 1024, 8192);
    REQUIRE(ai_large_k > ai_small_k);
  }

  SECTION("AI decreases when one dimension is small") {
    double ai_square = origami::compute_arithmetic_intensity(1024, 1024, 1024);
    double ai_skinny = origami::compute_arithmetic_intensity(32, 1024, 1024);
    REQUIRE(ai_square > ai_skinny);
  }

  SECTION("zero dimension returns zero") {
    double ai = origami::compute_arithmetic_intensity(0, 1024, 1024);
    REQUIRE(ai == 0.0);
  }
}

TEST_CASE("Categorization: representative AI", "[categorization]") {
  SECTION("larger categories have higher AI") {
    auto small_cat = origami::category_from_id(0);
    auto large_cat = origami::category_from_id(49);
    REQUIRE(large_cat.representative_arithmetic_intensity() >
            small_cat.representative_arithmetic_intensity());
  }

  SECTION("AI is positive for all categories") {
    for (std::size_t id = 0; id < origami::NUM_GEMM_CATEGORIES; ++id) {
      auto cat = origami::category_from_id(id);
      REQUIRE(cat.representative_arithmetic_intensity() > 0.0);
    }
  }

  SECTION("long_k categories have higher AI than short_k counterparts") {
    for (int mi = 0; mi < static_cast<int>(origami::mn_range_t::count); ++mi) {
      for (int ni = 0; ni < static_cast<int>(origami::mn_range_t::count); ++ni) {
        origami::gemm_category_t short_cat{
            static_cast<origami::mn_range_t>(mi),
            static_cast<origami::mn_range_t>(ni),
            origami::k_range_t::short_k};
        origami::gemm_category_t long_cat{
            static_cast<origami::mn_range_t>(mi),
            static_cast<origami::mn_range_t>(ni),
            origami::k_range_t::long_k};
        REQUIRE(long_cat.representative_arithmetic_intensity() >
                short_cat.representative_arithmetic_intensity());
      }
    }
  }
}
