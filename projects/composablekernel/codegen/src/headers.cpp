// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/headers.hpp"
#include "ck_headers.hpp"
#include "ck_tile_headers.hpp"
#include "ck_codegen_headers.hpp"

namespace ck {
namespace host {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
const std::string config_header = "";
#pragma clang diagnostic pop

std::unordered_map<std::string_view, std::string_view> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(std::make_pair("ck/config.h", config_header));
    return headers;
}

std::unordered_map<std::string_view, std::string_view> GetTileHeaders()
{
    auto headers = ck_tile_headers();
    auto codegen_hdrs = ck_codegen_headers();
    headers.insert(codegen_hdrs.begin(), codegen_hdrs.end());
    return headers;
}

} // namespace host
} // namespace ck
