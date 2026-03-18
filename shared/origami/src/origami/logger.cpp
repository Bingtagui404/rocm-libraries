// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/logger.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace origami {

Logger::Logger() : enabled_(false) {
    const char* log_file_path = std::getenv("ORIGAMI_LOG_FILE");
    
    if (log_file_path != nullptr && log_file_path[0] != '\0') {
        log_file_.open(log_file_path, std::ios::out | std::ios::app);
        
        if (log_file_.is_open()) {
            enabled_ = true;
            log(LogLevel::INFO, 
                "Origami logger initialized, writing to: " + std::string(log_file_path),
                __FILE__, __LINE__);
        } else {
            std::cerr << "Warning: Failed to open log file: " << log_file_path << std::endl;
        }
    }
}

Logger::~Logger() {
    if (enabled_ && log_file_.is_open()) {
        log(LogLevel::INFO, "Logger shutting down", __FILE__, __LINE__);
        log_file_.close();
    }
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line) {
    if (!enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    if (log_file_.is_open()) {
        const char* filename = file;
        for (const char* p = file; *p; p++) {
            if (*p == '/' || *p == '\\') {
                filename = p + 1;
            }
        }

        log_file_ << "[" << level_to_string(level) << "] "
                  << filename << ":" << line << " - "
                  << message << std::endl;
    }
}

void Logger::flush() {
    if (enabled_ && log_file_.is_open()) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_file_.flush();
    }
}

const char* Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKN ";
    }
}

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
