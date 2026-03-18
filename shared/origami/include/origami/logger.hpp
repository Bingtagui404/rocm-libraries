// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace origami {

class CsvLogger {
public:
    static CsvLogger& instance();
    bool is_enabled() const { return enabled_; }

    void begin_row();

    template<typename T>
    void set_field(const char* name, T value) {
        if (!enabled_) return;
        std::ostringstream oss;
        if constexpr (std::is_floating_point_v<T>) {
            oss << std::scientific << std::setprecision(6) << value;
        } else {
            oss << value;
        }
        row_buffer().fields.emplace_back(name, oss.str());
    }

    void flush_row();

    CsvLogger(const CsvLogger&) = delete;
    CsvLogger& operator=(const CsvLogger&) = delete;
    CsvLogger(CsvLogger&&) = delete;
    CsvLogger& operator=(CsvLogger&&) = delete;

private:
    CsvLogger();
    ~CsvLogger();

    struct RowBuffer {
        std::vector<std::pair<std::string, std::string>> fields;
        void clear() { fields.clear(); }
    };
    static RowBuffer& row_buffer();

    std::ofstream csv_file_;
    std::mutex mutex_;
    bool enabled_;
    bool header_written_;
    std::vector<std::string> column_order_;
};

} // namespace origami

#define OLOG_CSV_BEGIN() \
    if (origami::CsvLogger::instance().is_enabled()) \
        origami::CsvLogger::instance().begin_row()

#define OLOG_CSV(name, value) \
    if (origami::CsvLogger::instance().is_enabled()) \
        origami::CsvLogger::instance().set_field(name, value)

#define OLOG_CSV_FLUSH() \
    if (origami::CsvLogger::instance().is_enabled()) \
        origami::CsvLogger::instance().flush_row()
