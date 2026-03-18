// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/logger.hpp"

#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace origami {

CsvLogger::CsvLogger() : enabled_(false), header_written_(false) {
    const char* csv_path = std::getenv("ORIGAMI_CSV_FILE");
    if (csv_path != nullptr && csv_path[0] != '\0') {
        csv_file_.open(csv_path, std::ios::out | std::ios::trunc);
        if (csv_file_.is_open()) {
            enabled_ = true;
        } else {
            std::cerr << "Warning: Failed to open CSV log file: " << csv_path << std::endl;
        }
    }
}

CsvLogger::~CsvLogger() {
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
}

CsvLogger& CsvLogger::instance() {
    static CsvLogger logger;
    return logger;
}

CsvLogger::RowBuffer& CsvLogger::row_buffer() {
    static thread_local RowBuffer buf;
    return buf;
}

void CsvLogger::begin_row() {
    row_buffer().clear();
}

void CsvLogger::flush_row() {
    if (!enabled_) return;

    auto& buf = row_buffer();
    if (buf.fields.empty()) return;

    std::lock_guard<std::mutex> lock(mutex_);

    if (!header_written_) {
        std::unordered_set<std::string> seen;
        for (const auto& [name, _] : buf.fields) {
            if (seen.insert(name).second) {
                column_order_.push_back(name);
            }
        }
        for (size_t i = 0; i < column_order_.size(); ++i) {
            if (i > 0) csv_file_ << ',';
            csv_file_ << column_order_[i];
        }
        csv_file_ << '\n';
        header_written_ = true;
    }

    std::unordered_map<std::string, std::string> lookup;
    for (const auto& [name, value] : buf.fields) {
        lookup[name] = value;
    }

    for (size_t i = 0; i < column_order_.size(); ++i) {
        if (i > 0) csv_file_ << ',';
        auto it = lookup.find(column_order_[i]);
        if (it != lookup.end()) {
            csv_file_ << it->second;
        }
    }
    csv_file_ << '\n';
    csv_file_.flush();

    buf.clear();
}

} // namespace origami
