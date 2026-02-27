/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#ifndef GUARD_REDUCTION_FUNCTIONS_HPP
#define GUARD_REDUCTION_FUNCTIONS_HPP

#include "configuration.hpp"
#include "static_unroll.hpp"

// NOTE: This header should be independent from batchnorm_functions.hpp
// Even is in OpenCL implementation, these functions are only enabled under
// certain condition. But now, these template will not be compiled before
// calling them.
// #include "batchnorm_functions.hpp"
namespace miopen {
namespace reduction {

namespace detail {
template <int N>
struct log2_floor
{
    constexpr static int value = log2_floor<(N >> 1)>::value + 1;
};
template <>
struct log2_floor<1>
{
    constexpr static int value = 0;
};
template <int N>
constexpr static int log2_floor_v = log2_floor<N>::value;

template <int N>
struct log2_ceil
{
    constexpr static int value = log2_floor_v<N> + ((1 << log2_floor_v<N>) == N ? 0 : 1);
};
template <int N>
constexpr static int log2_ceil_v = log2_ceil<N>::value;

} // namespace detail

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2_welford(FloatAccum& mean,
                                                    FloatAccum& variance,
                                                    FloatAccum& count,
                                                    FloatAccum scale,
                                                    FloatAccum (&lcl_data_mean)[SizeLclData],
                                                    FloatAccum (&lcl_data_variance)[SizeLclData],
                                                    FloatAccum (&lcl_data_count)[SizeLclData],
                                                    unsigned int lid)
{
    lcl_data_mean[lid]     = mean;
    lcl_data_variance[lid] = variance;
    lcl_data_count[lid]    = count;
    __syncthreads();

    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            FloatAccum delta = lcl_data_mean[lid + red] - lcl_data_mean[lid];
            FloatAccum n_a   = lcl_data_count[lid];
            FloatAccum n_b   = lcl_data_count[lid + red];
            FloatAccum n_new = n_a + n_b;
            FloatAccum n_new_rcp =
                n_new != 0.0f ? (1 / n_new)
                              : 0.f; // Calculates 1 / n_new; Done to handle the case where some of
                                     // the partitions of mean/variance calculation are zero
            lcl_data_mean[lid] =
                (lcl_data_mean[lid] * n_a + lcl_data_mean[lid + red] * n_b) * n_new_rcp;
            lcl_data_variance[lid] = lcl_data_variance[lid] + lcl_data_variance[lid + red] +
                                     delta * delta * (n_a * n_b * n_new_rcp);
            lcl_data_count[lid] = n_new;
        }
        __syncthreads();
    }

    mean     = lcl_data_mean[0];
    variance = lcl_data_variance[0] * scale;
}

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    lcl_data_x[lid] = x;
    lcl_data_y[lid] = y;
    __syncthreads();
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            lcl_data_x[lid] += lcl_data_x[lid + red];
            lcl_data_y[lid] += lcl_data_y[lid + red];
        }
        __syncthreads();
    }

    x = lcl_data_x[0] * scale;
    y = lcl_data_y[0] * scale;
}

template <typename FloatAccumC, typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2_2d(FloatAccumC& x,
                                               FloatAccumC& y,
                                               FloatAccum scale,
                                               FloatAccumC (&lcl_data)[SizeLclData],
                                               unsigned int xstride,
                                               unsigned int xlid,
                                               unsigned int ylid,
                                               unsigned int size)
{
    unsigned int offset1 = 2 * (xlid + ylid * xstride);
    // store the values by pairs (so the compiler will generate
    // one instruction to read/write them)
    lcl_data[offset1 + 0] = static_cast<FloatAccumC>(x);
    lcl_data[offset1 + 1] = static_cast<FloatAccumC>(y);

    __syncthreads();
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        unsigned int offset2 = offset1 + red * xstride * 2;
        if(ylid < red && offset2 < SizeLclData)
        {
            // make sure there is one read and one write
            x += lcl_data[offset2 + 0];
            y += lcl_data[offset2 + 1];
            lcl_data[offset1 + 0] = x;
            lcl_data[offset1 + 1] = y;
        }
        __syncthreads();
    }
    x = static_cast<FloatAccumC>(lcl_data[xlid * 2 + 0] * scale);
    y = static_cast<FloatAccumC>(lcl_data[xlid * 2 + 1] * scale);
}

template <typename FloatAccum>
__forceinline__ __device__ void dpp_interleaved_reduction(FloatAccum& temp_sum1,
                                                          FloatAccum& temp_sum2)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                     "s_nop 0"
                     : "=v"(temp_sum1), "=v"(temp_sum2)
                     : "0"(temp_sum1), "1"(temp_sum2));
}

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    const unsigned int ldsidx = lid >> 6;
    dpp_interleaved_reduction(x, y);
    // Last thread
    if((lid % 64) == 63)
    {
        lcl_data_x[ldsidx] = x;
        lcl_data_y[ldsidx] = y;
    }

    __syncthreads();

    x = y = 0;

    // This could be changed to clang loop unroll(full), because the size is small
    static_unroll_count<unsigned int, 0, SizeLclData, 1, 2>{[&](unsigned int i) {
        x += lcl_data_x[i];
        y += lcl_data_y[i];
    }};

    x *= scale;
    y *= scale;
}

template <typename FloatAccum>
__forceinline__ __device__ void dpp_interleaved_reduction_welford(volatile FloatAccum& temp_mean,
                                                                  volatile FloatAccum& temp_var,
                                                                  volatile FloatAccum& temp_count)
{
    FloatAccum delta;
    FloatAccum count_mult;
    FloatAccum n_rcp;

    // Disabling clang-format within this function, otherwise
    // the indentation gets very messy

    // clang-format off
    #define REDUCTION_STEP(dpp) \
            /* 1 */ "v_add_f32 %4 -%0 %0 " dpp "\n" /* store the deltas of the means, that's a scratch register we'll need for the math. Multiply src0 by -1 to get the proper sign of the delta. Note: the first delta will be incorrect! Will be non-zero! */ \
            /* 2 */ "v_mul_f32 %0 %0 %2\n" /* calculate mean times count */ \
            /* 3 */ "v_mul_f32 %3 %2 %2 " dpp "\n" /* get n_a * n_b */ \
            /* 4 */ "v_mul_f32 %4 %4 %4\n" /* square deltas, we need those for the variance calculation; this is put here in place of a NOP due to dependency between 2 and 5*/ \
            /* 5 */ "v_add_f32 %0 %0 %0 " dpp "\n" /* merge mean times count */ \
            /* 6 */ "v_add_f32 %2 %2 %2 " dpp "\n" /* sum up the new counts that correspond to the new mean times count values*/ \
            /* 7 */ "v_rcp_f32 %5 %2\n" /* prepare for division by n_a + n_b possibly also do v_div_scale_f32?*/ \
            /* 8 */ "v_cmp_eq_f32 vcc %2 0\n"/* Idea: move 0 if %2 is zero for specific lanes -- for the mean, the variance and the count */\
            /* 9 */ "v_cndmask_b32_e64 %5 %5 %2 vcc\n" \
            /*10 */ "v_mul_f32 %0 %0 %5\n"/* normalize mean; %5 is 1/(n_a + n_b), it's the updated counts */ \
            /*11 */ "v_add_f32 %1 %1 %1 " dpp "\n" /* part of the variance calculation -- sum up the two partitions, add the deltas in the next steps */ \
            /*12 */ "v_mul_f32 %5 %3 %5\n"\
            /*13 */ /* NOP is not necessary here, it's needed when the first instruction is non-DPP and the second one is, thus there should be no dependency between 11 and 14 or data corruption risk between 12 and 14 */ \
            /*14 */ "v_fma_f32 %1 %4 %5 %1\n" /* %4, %5 and %1 should already have been properly offset with the required number of lanes for the reduciton */ \
            /*15 */ "v_nop\n"\
            /*16 */ "v_nop\n"/* NOPs necessary because the next instr needs %4 and has DPP, dependency between 14 and 1 */\
            
    __asm__ volatile(
        "s_nop 4\n"
        REDUCTION_STEP("row_shr:1 bound_ctrl:0")
        REDUCTION_STEP("row_shr:2 bound_ctrl:0")
        REDUCTION_STEP("row_shr:4 bank_mask:0xe")
        REDUCTION_STEP("row_shr:8 bank_mask:0xc")
        REDUCTION_STEP("row_bcast:15 row_mask:0xa")
        REDUCTION_STEP("row_bcast:31 row_mask:0xc")


        ""
            : "=v"(temp_mean), "=v"(temp_var), "=v" (temp_count), "=v" (count_mult), "=v" (delta), "=v" (n_rcp)
            : "0"(temp_mean), "1"(temp_var), "2" (temp_count), "3" (count_mult), "4" (delta), "5" (n_rcp));
    // clang-format on
}

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2_welford(FloatAccum& mean,
                                                    FloatAccum& variance,
                                                    FloatAccum& count,
                                                    FloatAccum scale,
                                                    FloatAccum (&lcl_data_mean)[SizeLclData],
                                                    FloatAccum (&lcl_data_variance)[SizeLclData],
                                                    FloatAccum (&lcl_data_count)[SizeLclData],
                                                    unsigned int lid)
{
    const unsigned int ldsidx = lid >> 6;

    dpp_interleaved_reduction_welford<FloatAccum>(mean, variance, count);

    // Last thread
    if((lid % 64) == 63)
    {
        lcl_data_mean[ldsidx]     = mean;
        lcl_data_variance[ldsidx] = variance;
        lcl_data_count[ldsidx]    = count;
    }

    __syncthreads();

    mean     = lcl_data_mean[0];
    variance = lcl_data_variance[0];
    count    = lcl_data_count[0];

    // This could be changed to clang loop unroll(full), because the size is small
    // static_unroll_count<unsigned int, 1, SizeLclData, 1, 0>{[&](unsigned int i) {
    //     x += lcl_data_x[i];
    //     y += lcl_data_y[i];
    // }};
    for(unsigned int i = 1; i < SizeLclData; i++)
    {
        FloatAccum delta = lcl_data_mean[i] - mean;
        FloatAccum n_a   = count;
        FloatAccum n_b   = lcl_data_count[i];
        FloatAccum n_new = n_a + n_b;
        mean             = (mean * n_a + lcl_data_mean[i] * n_b) / (n_new);
        variance         = variance + lcl_data_variance[i] + (delta * delta * (n_a * n_b)) / n_new;
        count            = n_new;
    }
    // No scaling of the mean, that is already kept to scale as a requirement of Welford's variance
    // calculation
    variance *= scale;
}

} // namespace reduction
} // namespace miopen

#endif
