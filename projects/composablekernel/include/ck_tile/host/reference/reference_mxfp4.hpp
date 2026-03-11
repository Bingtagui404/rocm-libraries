// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>

// CPU reference helpers for MXFP4 (E2M1) encoding/decoding.
//
// MXFP4 E2M1 representable values (positive half):
//   0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
//   Packing: one byte holds two FP4 values; lower nibble = element[even], upper = element[odd].
//   e8m0 scale: float = bit_cast<float>(uint32_t(scale_byte) << 23)
//
// Scale granularity: 32 elements per scale (kMxfp4Group = 32).
// Layout for quantized tensors:
//   hat   [B, H, rows, cols/2]     — packed nibbles
//   scale [B, H, rows, cols/32]    — e8m0 scale bytes

namespace ck_tile {
namespace reference {

static constexpr int kMxfp4Group = 32; // elements per MXFP4 scale group

// -------------------------------------------------------------------------
// Scalar encode / decode
// -------------------------------------------------------------------------

// Decode e8m0 scale byte to float32.
inline float mxfp4_e8m0_to_float(uint8_t scale_byte)
{
    uint32_t bits = static_cast<uint32_t>(scale_byte) << 23;
    float    f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

// Decode a 4-bit FP4 E2M1 nibble (0-15) to float32 (without scale).
inline float mxfp4_nibble_to_float(uint8_t nibble)
{
    static const float vals[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return vals[nibble & 0xF];
}

// Round a float32 value to the nearest FP4 E2M1 value (without scale).
// Returns the 4-bit nibble code (0-15).
inline uint8_t mxfp4_float_to_nibble(float x)
{
    static const float abs_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool         negative     = (x < 0.0f);
    const float        x_abs        = std::abs(x);

    int   best     = 0;
    float best_err = std::abs(x_abs - abs_vals[0]);
    for(int i = 1; i < 8; i++)
    {
        float err = std::abs(x_abs - abs_vals[i]);
        if(err < best_err)
        {
            best_err = err;
            best     = i;
        }
    }
    return static_cast<uint8_t>(negative ? (best | 8) : best);
}

// Compute e8m0 scale byte for a group given its max absolute value.
// Returns the smallest power-of-2 scale s such that max_abs / s <= 6.
inline uint8_t mxfp4_compute_scale_byte(float max_abs)
{
    constexpr float    rcp_dst_max = 1.0f / 6.0f;
    const float        scaled      = max_abs * rcp_dst_max;
    uint32_t           bits_in;
    __builtin_memcpy(&bits_in, &scaled, sizeof(float));
    constexpr uint32_t mant_mask = 0x007FFFFFu;
    constexpr uint32_t head_mask = 0xFF800000u;
    const uint32_t     bits_out  = (bits_in + mant_mask) & head_mask;
    return static_cast<uint8_t>(bits_out >> 23);
}

// -------------------------------------------------------------------------
// Quantize a flat array of `n` floats → packed FP4 + scale bytes.
//   n must be divisible by kMxfp4Group.
//   hat_out:   n/2 bytes
//   scale_out: n/kMxfp4Group bytes
// -------------------------------------------------------------------------
inline void mxfp4_quantize(const float* src, int n, uint8_t* hat_out, uint8_t* scale_out)
{
    const int num_groups = n / kMxfp4Group;
    for(int g = 0; g < num_groups; g++)
    {
        const int   d_start = g * kMxfp4Group;
        float       max_abs = 0.0f;
        for(int j = 0; j < kMxfp4Group; j++)
            max_abs = std::max(max_abs, std::abs(src[d_start + j]));

        const uint8_t scale_byte = mxfp4_compute_scale_byte(max_abs);
        const float   scale      = mxfp4_e8m0_to_float(scale_byte);
        scale_out[g]             = scale_byte;

        for(int j = 0; j < kMxfp4Group; j += 2)
        {
            const uint8_t n0        = mxfp4_float_to_nibble(src[d_start + j] / scale);
            const uint8_t n1        = mxfp4_float_to_nibble(src[d_start + j + 1] / scale);
            hat_out[(d_start + j) / 2] = static_cast<uint8_t>((n1 << 4) | (n0 & 0xF));
        }
    }
}

// -------------------------------------------------------------------------
// Dequantize MXFP4 tensor [B, H, rows, cols] from packed representation.
//
//   hat   layout: [B, H, rows, cols/2]
//   scale layout: [B, H, rows, cols/kMxfp4Group]
//   out   layout: [B, H, rows, cols]
// -------------------------------------------------------------------------
inline void reference_dequant_mxfp4(const uint8_t* hat,
                                     const uint8_t* scale_bytes,
                                     float*         out,
                                     int            B,
                                     int            H,
                                     int            rows,
                                     int            cols)
{
    const int num_groups = cols / kMxfp4Group;
    for(int b = 0; b < B; b++)
    {
        for(int h = 0; h < H; h++)
        {
            for(int r = 0; r < rows; r++)
            {
                for(int g = 0; g < num_groups; g++)
                {
                    const int   scale_off = b * H * rows * num_groups + h * rows * num_groups +
                                          r * num_groups + g;
                    const float scale     = mxfp4_e8m0_to_float(scale_bytes[scale_off]);

                    const int d_start = g * kMxfp4Group;
                    for(int j = 0; j < kMxfp4Group; j += 2)
                    {
                        const int     hat_off = b * H * rows * (cols / 2) +
                                            h * rows * (cols / 2) + r * (cols / 2) +
                                            (d_start + j) / 2;
                        const uint8_t byte = hat[hat_off];
                        out[b * H * rows * cols + h * rows * cols + r * cols + d_start + j] =
                            mxfp4_nibble_to_float(byte & 0xF) * scale;
                        out[b * H * rows * cols + h * rows * cols + r * cols + d_start + j + 1] =
                            mxfp4_nibble_to_float((byte >> 4) & 0xF) * scale;
                    }
                }
            }
        }
    }
}

} // namespace reference
} // namespace ck_tile
