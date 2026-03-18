// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"
#include "origami/logger.hpp"

using Catch::Approx;

// Split a CSV line into fields on commas (no quoting support needed here).
static std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

// Read the entire CSV file produced by CsvLogger into a header + data rows.
struct CsvContents {
  std::string header_line;
  std::vector<std::string> header_columns;
  std::vector<std::vector<std::string>> rows;

  bool load(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    if (!std::getline(f, header_line)) return false;
    header_columns = split_csv(header_line);

    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty()) {
        rows.push_back(split_csv(line));
      }
    }
    return true;
  }

  bool has_column(const std::string& name) const {
    return std::find(header_columns.begin(), header_columns.end(), name)
           != header_columns.end();
  }

  int column_index(const std::string& name) const {
    auto it = std::find(header_columns.begin(), header_columns.end(), name);
    if (it == header_columns.end()) return -1;
    return static_cast<int>(it - header_columns.begin());
  }
};

// ============================================================================

TEST_CASE("CsvLogger: singleton is accessible", "[logger][csv]") {
  auto& logger = origami::CsvLogger::instance();
  (void)logger;
  SUCCEED("CsvLogger::instance() returned without error");
}

TEST_CASE("CsvLogger: enabled state matches environment", "[logger][csv]") {
  auto& logger = origami::CsvLogger::instance();
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  bool env_set = (csv_path != nullptr && csv_path[0] != '\0');

  if (env_set) {
    REQUIRE(logger.is_enabled());
  } else {
    REQUIRE_FALSE(logger.is_enabled());
  }
}

TEST_CASE("CsvLogger: CSV header contains expected model parameter columns",
          "[logger][csv]") {
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  if (!csv_path || csv_path[0] == '\0') {
    SKIP("ORIGAMI_CSV_FILE not set");
  }

  auto hardware = make_hardware(942);
  auto problem  = make_problem(512, 512, 512);
  auto config   = make_config(128, 128, 64);
  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

  CsvContents csv;
  REQUIRE(csv.load(csv_path));
  REQUIRE(!csv.header_columns.empty());

  // Problem / config input columns
  CHECK(csv.has_column("M"));
  CHECK(csv.has_column("N"));
  CHECK(csv.has_column("K"));
  CHECK(csv.has_column("batch"));
  CHECK(csv.has_column("MT_M"));
  CHECK(csv.has_column("MT_N"));
  CHECK(csv.has_column("MT_K"));
  CHECK(csv.has_column("MI_M"));
  CHECK(csv.has_column("MI_N"));
  CHECK(csv.has_column("MI_K"));
  CHECK(csv.has_column("a_bits"));
  CHECK(csv.has_column("b_bits"));

  // CU occupancy columns
  CHECK(csv.has_column("num_wgs"));
  CHECK(csv.has_column("num_active_cus"));
  CHECK(csv.has_column("num_timesteps"));
  CHECK(csv.has_column("splitting_factor"));

  // Memory model columns
  CHECK(csv.has_column("Ld_CU_bytes"));
  CHECK(csv.has_column("total_Ld"));
  CHECK(csv.has_column("H_mem_l2"));
  CHECK(csv.has_column("H_mem_mall"));
  CHECK(csv.has_column("L_mem"));

  // Tile model columns
  CHECK(csv.has_column("L_compute"));
  CHECK(csv.has_column("L_prologue"));
  CHECK(csv.has_column("L_epilogue"));
  CHECK(csv.has_column("L_tile_single"));
  CHECK(csv.has_column("L_tile_total"));
  CHECK(csv.has_column("utilization"));
  CHECK(csv.has_column("occupancy_factor"));

  // Final output column
  CHECK(csv.has_column("total_latency"));
}

TEST_CASE("CsvLogger: all rows have same column count as header",
          "[logger][csv]") {
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  if (!csv_path || csv_path[0] == '\0') {
    SKIP("ORIGAMI_CSV_FILE not set");
  }

  auto hardware = make_hardware(950);
  auto problem  = make_problem(2048, 2048, 2048);
  auto config   = make_config(256, 128, 64);
  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

  CsvContents csv;
  REQUIRE(csv.load(csv_path));
  REQUIRE(!csv.rows.empty());

  const size_t expected_cols = csv.header_columns.size();
  for (size_t i = 0; i < csv.rows.size(); ++i) {
    INFO("Row " << i << " has " << csv.rows[i].size()
                << " columns, expected " << expected_cols);
    CHECK(csv.rows[i].size() == expected_cols);
  }
}

TEST_CASE("CsvLogger: integration - known inputs appear in CSV output",
          "[logger][csv]") {
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  if (!csv_path || csv_path[0] == '\0') {
    SKIP("ORIGAMI_CSV_FILE not set");
  }

  auto hardware = make_hardware(942);
  auto problem  = make_problem(7777, 3333, 4096);
  auto config   = make_config(128, 128, 64);
  double latency =
      origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

  CsvContents csv;
  REQUIRE(csv.load(csv_path));

  int col_m  = csv.column_index("M");
  int col_n  = csv.column_index("N");
  int col_k  = csv.column_index("K");
  int col_mt = csv.column_index("MT_M");
  int col_lat = csv.column_index("total_latency");
  REQUIRE(col_m >= 0);
  REQUIRE(col_n >= 0);
  REQUIRE(col_k >= 0);
  REQUIRE(col_mt >= 0);
  REQUIRE(col_lat >= 0);

  bool found = false;
  for (const auto& row : csv.rows) {
    if (row[col_m] == "7777" && row[col_n] == "3333" &&
        row[col_k] == "4096" && row[col_mt] == "128") {
      found = true;

      double csv_latency = std::stod(row[col_lat]);
      CHECK(csv_latency == Approx(latency).epsilon(1e-6));
      break;
    }
  }
  REQUIRE(found);
}

TEST_CASE("CsvLogger: latency model values are positive and finite",
          "[logger][csv]") {
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  if (!csv_path || csv_path[0] == '\0') {
    SKIP("ORIGAMI_CSV_FILE not set");
  }

  auto hardware = make_hardware(942);
  auto problem  = make_problem(4096, 4096, 4096);
  auto config   = make_config(128, 128, 128);
  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

  CsvContents csv;
  REQUIRE(csv.load(csv_path));

  // Check a few key double columns on the last row (the one we just generated)
  const auto& last_row = csv.rows.back();

  auto check_positive = [&](const std::string& col_name) {
    int idx = csv.column_index(col_name);
    REQUIRE(idx >= 0);
    double val = std::stod(last_row[idx]);
    INFO(col_name << " = " << val);
    CHECK(val > 0.0);
    CHECK(std::isfinite(val));
  };

  check_positive("L_compute");
  check_positive("L_mem");
  check_positive("L_prologue");
  check_positive("L_epilogue");
  check_positive("L_tile_single");
  check_positive("L_tile_total");
  check_positive("total_latency");
}

TEST_CASE("CsvLogger: begin_row clears previous state", "[logger][csv]") {
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  if (!csv_path || csv_path[0] == '\0') {
    SKIP("ORIGAMI_CSV_FILE not set");
  }

  auto hardware = make_hardware(950);

  // First call with M=9991
  auto p1 = make_problem(9991, 1024, 1024);
  auto c1 = make_config(128, 128, 64);
  origami::compute_total_latency(p1, hardware, c1, hardware.N_CU);

  // Second call with M=9992 — begin_row should clear previous fields
  auto p2 = make_problem(9992, 1024, 1024);
  origami::compute_total_latency(p2, hardware, c1, hardware.N_CU);

  CsvContents csv;
  REQUIRE(csv.load(csv_path));

  int col_m = csv.column_index("M");
  REQUIRE(col_m >= 0);

  // Find both rows and verify they have distinct M values
  bool found_9991 = false, found_9992 = false;
  for (const auto& row : csv.rows) {
    if (row[col_m] == "9991") found_9991 = true;
    if (row[col_m] == "9992") found_9992 = true;
  }
  CHECK(found_9991);
  CHECK(found_9992);
}

TEST_CASE("CsvLogger: multiple architectures produce valid output",
          "[logger][csv]") {
  const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
  if (!csv_path || csv_path[0] == '\0') {
    SKIP("ORIGAMI_CSV_FILE not set");
  }

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 2048);
      auto config   = make_config(128, 128, 64);
      double lat = origami::compute_total_latency(
          problem, hardware, config, hardware.N_CU);

      CHECK(lat > 0.0);
      CHECK(std::isfinite(lat));
    }
  }

  CsvContents csv;
  REQUIRE(csv.load(csv_path));

  // Verify the file is still well-formed after logging from multiple archs
  const size_t expected_cols = csv.header_columns.size();
  for (const auto& row : csv.rows) {
    CHECK(row.size() == expected_cols);
  }
}
