// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// End-to-end GPU correctness test for SageAttn preprocessing (SA3).
//
// Exercises the three-kernel pipeline launched by sageattn_v3_preprocess_run():
//   Launch 0: SageAttnKMeanKernel     — computes k_mean[b,h,d] = mean_n(K[b,h,n,d])
//   Launch 1: SageAttnV3PreprocessKernel — Q mean+quantize, K smooth+quantize, V transpose+quantize
//   Launch 2: BatchedGemmKernel        — delta_s = q_mean @ K'^T
//
// Verified outputs:
//   k_mean   — InputT, compared vs CPU reference (< 2e-3 abs error for fp16)
//   q_mean   — InputT, compared vs CPU reference (< 2e-3 abs error for fp16)
//   delta_s  — float,  compared vs CPU reference (< 1e-2 * hdim abs error)
//   Q hat    — MXFP4, dequantized and compared with Q_smooth reference (< 1.0 abs error)
//   Q scale  — e8m0 bytes, compared directly with CPU reference
//   K hat    — MXFP4, dequantized and compared with K_smooth reference (< 1.0 abs error)
//   K scale  — e8m0 bytes, compared directly with CPU reference
//   V hat    — MXFP4 (transposed), dequantized and compared with V reference (< 1.0 abs error)
//   V scale  — e8m0 bytes, compared directly with CPU reference

#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#ifdef CK_USE_NATIVE_MX_SUPPORT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_preprocess.hpp"
#include "ck_tile/host/reference/reference_sageattn_v3_preprocess.hpp"

namespace {

// Dispatch sageattn_v3_preprocess_run for runtime hdim (128 or 256).
template <typename InputT>
void run_sageattn_v3_preprocess(const ck_tile::SageAttnV3PreprocessArgs<InputT>& h,
                                float* delta_s_ptr,
                                InputT* k_mean_buf,
                                InputT* k_prime_buf,
                                float* k_mean_partial_buf,
                                int32_t* counter_buf,
                                hipStream_t stream)
{
    if(h.hdim == 128)
        ck_tile::sageattn_v3_preprocess_run<InputT, /*kRows=*/128, /*kCols=*/128>(
            h, delta_s_ptr, k_mean_buf, k_prime_buf, k_mean_partial_buf, counter_buf, stream);
    else if(h.hdim == 256)
        ck_tile::sageattn_v3_preprocess_run<InputT, /*kRows=*/128, /*kCols=*/256>(
            h, delta_s_ptr, k_mean_buf, k_prime_buf, k_mean_partial_buf, counter_buf, stream);
    else
        throw std::runtime_error("Unsupported hdim (must be 128 or 256)");
}

// Convert float vector to fp16, round-trip through fp16 so values match GPU.
std::vector<ck_tile::fp16_t> float_to_fp16_vec(const std::vector<float>& src)
{
    std::vector<ck_tile::fp16_t> dst(src.size());
    for(std::size_t i = 0; i < src.size(); i++)
        dst[i] = ck_tile::type_convert<ck_tile::fp16_t>(src[i]);
    return dst;
}

} // namespace

// ---------------------------------------------------------------------------
// Test fixture: parameterized by (B, H, seqlen_q, seqlen_k, hdim, use_fp16)
// ---------------------------------------------------------------------------
class SageAttnV3PreprocessTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, int, bool>>
{
    protected:
    int B() const { return std::get<0>(GetParam()); }
    int H() const { return std::get<1>(GetParam()); }
    int seqlen_q() const { return std::get<2>(GetParam()); }
    int seqlen_k() const { return std::get<3>(GetParam()); }
    int hdim() const { return std::get<4>(GetParam()); }
    bool use_fp16() const { return std::get<5>(GetParam()); }

    static constexpr int kM0 = 128; // Q/K tile rows (padding granularity)
    static constexpr int kG  = 32;  // MXFP4 scale granularity

    static std::vector<float> RandFloat(int n, float lo = -1.0f, float hi = 1.0f)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<float> v(n);
        for(auto& x : v)
            x = dist(rng);
        return v;
    }

    template <typename InputT>
    void RunGPUTest();
};

template <typename InputT>
void SageAttnV3PreprocessTest::RunGPUTest()
{
    const int b = B(), h = H(), sq = seqlen_q(), sk = seqlen_k(), hd = hdim();
    ASSERT_EQ(hd % kG, 0) << "hdim must be divisible by 32";

    // Use get_buffer_sizes to get padded dimensions and required buffer sizes.
    const auto bsz        = ck_tile::get_buffer_sizes<InputT, kM0>(b, h, sq, sk, hd);
    const int sq_pad      = static_cast<int>(bsz.seqlen_q_padded);
    const int sk_pad      = static_cast<int>(bsz.seqlen_k_padded);
    const int num_q_tiles = static_cast<int>(bsz.num_q_tiles);
    const int num_k_tiles = static_cast<int>(bsz.num_k_tiles);

    // ---- Generate float inputs ----
    auto Q_f32 = RandFloat(b * h * sq * hd);
    auto K_f32 = RandFloat(b * h * sk * hd);
    auto V_f32 = RandFloat(b * h * sk * hd);

    // For fp16: round-trip so CPU reference values match what the GPU sees.
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
    {
        auto round_trip = [](std::vector<float>& v) {
            for(auto& x : v)
                x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));
        };
        round_trip(Q_f32);
        round_trip(K_f32);
        round_trip(V_f32);
    }

    // ---- CPU reference: k_mean (float precision) ----
    std::vector<float> k_mean_ref_f32(b * h * hd, 0.0f);
    ck_tile::reference::reference_sageattn_v3_k_smooth(
        K_f32.data(), k_mean_ref_f32.data(), b, h, sk, hd);

    // If InputT = fp16, the GPU stores k_mean as fp16 → quantise the CPU reference too.
    // All downstream references (delta_s, K_smooth) must use the rounded k_mean.
    std::vector<float> k_mean_ref_rt = k_mean_ref_f32; // "round-tripped" version
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
    {
        for(auto& x : k_mean_ref_rt)
            x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));
    }

    // ---- CPU reference outputs (using rounded k_mean for delta_s / K_smooth) ----
    std::vector<float> q_mean_ref_f32(b * h * num_q_tiles * hd, 0.0f);
    std::vector<uint8_t> q_hat_ref(b * h * sq * (hd / 2), 0);
    std::vector<uint8_t> q_scale_ref(b * h * sq * (hd / kG), 0);
    ck_tile::reference::reference_sageattn_v3_q_preprocess(Q_f32.data(),
                                                           q_mean_ref_f32.data(),
                                                           q_hat_ref.data(),
                                                           q_scale_ref.data(),
                                                           b,
                                                           h,
                                                           sq,
                                                           hd,
                                                           kM0);

    // q_mean also stored as InputT on GPU → round-trip if fp16.
    std::vector<float> q_mean_ref_rt = q_mean_ref_f32;
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
    {
        for(auto& x : q_mean_ref_rt)
            x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));
    }

    std::vector<float> delta_s_ref(b * h * num_q_tiles * sk, 0.0f);
    ck_tile::reference::reference_sageattn_v3_delta_s(q_mean_ref_rt.data(),
                                                      K_f32.data(),
                                                      k_mean_ref_rt.data(),
                                                      delta_s_ref.data(),
                                                      b,
                                                      h,
                                                      num_q_tiles,
                                                      sk,
                                                      hd);

    std::vector<uint8_t> k_hat_ref(b * h * sk * (hd / 2), 0);
    std::vector<uint8_t> k_scale_ref(b * h * sk * (hd / kG), 0);
    ck_tile::reference::reference_sageattn_v3_k_preprocess(
        K_f32.data(), k_mean_ref_rt.data(), k_hat_ref.data(), k_scale_ref.data(), b, h, sk, hd);

    std::vector<uint8_t> v_hat_ref(b * h * hd * (sk / 2), 0);
    std::vector<uint8_t> v_scale_ref(b * h * hd * (sk / kG), 0);
    ck_tile::reference::reference_sageattn_v3_v_preprocess(
        V_f32.data(), v_hat_ref.data(), v_scale_ref.data(), b, h, sk, hd);

    // ---- Allocate and upload GPU input buffers ----
    const std::size_t elem_bytes = sizeof(InputT);
    ck_tile::DeviceMem q_dev(static_cast<std::size_t>(b * h * sq * hd) * elem_bytes);
    ck_tile::DeviceMem k_dev(static_cast<std::size_t>(b * h * sk * hd) * elem_bytes);
    ck_tile::DeviceMem v_dev(static_cast<std::size_t>(b * h * sk * hd) * elem_bytes);

    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
    {
        auto Q_fp16 = float_to_fp16_vec(Q_f32);
        auto K_fp16 = float_to_fp16_vec(K_f32);
        auto V_fp16 = float_to_fp16_vec(V_f32);
        q_dev.ToDevice(Q_fp16.data());
        k_dev.ToDevice(K_fp16.data());
        v_dev.ToDevice(V_fp16.data());
    }
    else
    {
        q_dev.ToDevice(Q_f32.data());
        k_dev.ToDevice(K_f32.data());
        v_dev.ToDevice(V_f32.data());
    }

    // ---- Output / scratch GPU buffers ----
    ck_tile::DeviceMem q_hat_dev(bsz.q_hat_bytes);
    ck_tile::DeviceMem q_scale_dev(bsz.q_scale_bytes);
    ck_tile::DeviceMem q_mean_dev(bsz.q_mean_bytes);
    ck_tile::DeviceMem k_hat_dev(bsz.k_hat_bytes);
    ck_tile::DeviceMem k_scale_dev(bsz.k_scale_bytes);
    ck_tile::DeviceMem delta_s_dev(bsz.delta_s_bytes);
    ck_tile::DeviceMem v_hat_dev(bsz.v_hat_bytes);
    ck_tile::DeviceMem v_scale_dev(bsz.v_scale_bytes);

    ck_tile::DeviceMem k_mean_buf(bsz.k_mean_bytes);
    ck_tile::DeviceMem k_prime_buf(bsz.k_prime_bytes);
    ck_tile::DeviceMem k_mean_partial_buf(bsz.k_mean_partial_bytes);
    ck_tile::DeviceMem counter_buf(bsz.counter_bytes);

    // ---- Fill Hargs (k_mean_ptr / k_prime_ptr filled by sageattn_v3_preprocess_run) ----
    ck_tile::SageAttnV3PreprocessArgs<InputT> hargs{};
    hargs.q_ptr                = static_cast<const InputT*>(q_dev.GetDeviceBuffer());
    hargs.seqlen_q             = sq;
    hargs.hdim                 = hd;
    hargs.stride_q             = hd;
    hargs.nhead_stride_q       = sq * hd;
    hargs.batch_stride_q       = h * sq * hd;
    hargs.q_hat_ptr            = static_cast<uint8_t*>(q_hat_dev.GetDeviceBuffer());
    hargs.stride_q_hat         = hd / 2;
    hargs.nhead_stride_q_hat   = sq_pad * (hd / 2);
    hargs.batch_stride_q_hat   = h * sq_pad * (hd / 2);
    hargs.q_scale_ptr          = static_cast<uint8_t*>(q_scale_dev.GetDeviceBuffer());
    hargs.stride_q_scale       = hd / kG;
    hargs.nhead_stride_q_scale = sq_pad * (hd / kG);
    hargs.batch_stride_q_scale = h * sq_pad * (hd / kG);
    hargs.q_mean_ptr           = static_cast<InputT*>(q_mean_dev.GetDeviceBuffer());
    hargs.q_tile_size          = kM0;
    hargs.stride_q_mean        = hd;
    hargs.nhead_stride_q_mean  = num_q_tiles * hd;
    hargs.batch_stride_q_mean  = h * num_q_tiles * hd;

    hargs.k_ptr                = static_cast<const InputT*>(k_dev.GetDeviceBuffer());
    hargs.seqlen_k             = sk;
    hargs.stride_k             = hd;
    hargs.nhead_stride_k       = sk * hd;
    hargs.batch_stride_k       = h * sk * hd;
    hargs.k_hat_ptr            = static_cast<uint8_t*>(k_hat_dev.GetDeviceBuffer());
    hargs.stride_k_hat         = hd / 2;
    hargs.nhead_stride_k_hat   = sk_pad * (hd / 2);
    hargs.batch_stride_k_hat   = h * sk_pad * (hd / 2);
    hargs.k_scale_ptr          = static_cast<uint8_t*>(k_scale_dev.GetDeviceBuffer());
    hargs.stride_k_scale       = hd / kG;
    hargs.nhead_stride_k_scale = sk_pad * (hd / kG);
    hargs.batch_stride_k_scale = h * sk_pad * (hd / kG);
    // k_mean_ptr and k_prime_ptr are set inside sageattn_v3_preprocess_run().

    hargs.v_ptr                = static_cast<const InputT*>(v_dev.GetDeviceBuffer());
    hargs.nhead_stride_v       = sk * hd;
    hargs.batch_stride_v       = h * sk * hd;
    hargs.v_hat_ptr            = static_cast<uint8_t*>(v_hat_dev.GetDeviceBuffer());
    hargs.stride_v_hat         = sk_pad / 2;
    hargs.nhead_stride_v_hat   = hd * (sk_pad / 2);
    hargs.batch_stride_v_hat   = h * hd * (sk_pad / 2);
    hargs.v_scale_ptr          = static_cast<uint8_t*>(v_scale_dev.GetDeviceBuffer());
    hargs.stride_v_scale       = sk_pad / kG;
    hargs.nhead_stride_v_scale = hd * (sk_pad / kG);
    hargs.batch_stride_v_scale = h * hd * (sk_pad / kG);

    hargs.batch       = b;
    hargs.nhead       = h;
    hargs.num_q_tiles = num_q_tiles;
    hargs.num_k_tiles = num_k_tiles;

    // ---- Launch ----
    run_sageattn_v3_preprocess<InputT>(hargs,
                                       static_cast<float*>(delta_s_dev.GetDeviceBuffer()),
                                       static_cast<InputT*>(k_mean_buf.GetDeviceBuffer()),
                                       static_cast<InputT*>(k_prime_buf.GetDeviceBuffer()),
                                       static_cast<float*>(k_mean_partial_buf.GetDeviceBuffer()),
                                       static_cast<int32_t*>(counter_buf.GetDeviceBuffer()),
                                       /*stream=*/nullptr);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // ---- Copy results back ----
    // q_mean and k_mean are stored as InputT on the GPU.
    std::vector<InputT> k_mean_gpu_raw(b * h * hd);
    std::vector<InputT> q_mean_gpu_raw(b * h * num_q_tiles * hd);
    k_mean_buf.FromDevice(k_mean_gpu_raw.data());
    q_mean_dev.FromDevice(q_mean_gpu_raw.data());

    std::vector<float> k_mean_gpu_f32(b * h * hd);
    std::vector<float> q_mean_gpu_f32(b * h * num_q_tiles * hd);
    for(std::size_t i = 0; i < k_mean_gpu_raw.size(); i++)
        k_mean_gpu_f32[i] = ck_tile::type_convert<float>(k_mean_gpu_raw[i]);
    for(std::size_t i = 0; i < q_mean_gpu_raw.size(); i++)
        q_mean_gpu_f32[i] = ck_tile::type_convert<float>(q_mean_gpu_raw[i]);

    std::vector<float> delta_s_gpu(b * h * num_q_tiles * sk_pad);
    std::vector<uint8_t> q_hat_gpu(b * h * sq_pad * (hd / 2));
    std::vector<uint8_t> q_scale_gpu(b * h * sq_pad * (hd / kG));
    std::vector<uint8_t> k_hat_gpu(b * h * sk_pad * (hd / 2));
    std::vector<uint8_t> k_scale_gpu(b * h * sk_pad * (hd / kG));
    std::vector<uint8_t> v_hat_gpu(b * h * hd * (sk_pad / 2));
    std::vector<uint8_t> v_scale_gpu(b * h * hd * (sk_pad / kG));

    delta_s_dev.FromDevice(delta_s_gpu.data());
    q_hat_dev.FromDevice(q_hat_gpu.data());
    q_scale_dev.FromDevice(q_scale_gpu.data());
    k_hat_dev.FromDevice(k_hat_gpu.data());
    k_scale_dev.FromDevice(k_scale_gpu.data());
    v_hat_dev.FromDevice(v_hat_gpu.data());
    v_scale_dev.FromDevice(v_scale_gpu.data());

    // Tolerance: fp16 introduces ~1/2^10 relative error vs float operations.
    const float mean_tol     = use_fp16() ? 2e-3f : 1e-4f;
    const int max_scale_diff = use_fp16() ? 1 : 0;

    // ======================== VERIFY k_mean =================================
    for(int bi = 0; bi < b; bi++)
        for(int hi = 0; hi < h; hi++)
            for(int d = 0; d < hd; d++)
            {
                const int off = bi * h * hd + hi * hd + d;
                EXPECT_NEAR(k_mean_gpu_f32[off], k_mean_ref_f32[off], mean_tol)
                    << "k_mean b=" << bi << " h=" << hi << " d=" << d;
            }

    // ======================== VERIFY q_mean =================================
    for(int bi = 0; bi < b; bi++)
        for(int hi = 0; hi < h; hi++)
            for(int qi = 0; qi < num_q_tiles; qi++)
                for(int d = 0; d < hd; d++)
                {
                    const int off = bi * h * num_q_tiles * hd + hi * num_q_tiles * hd + qi * hd + d;
                    EXPECT_NEAR(q_mean_gpu_f32[off], q_mean_ref_f32[off], mean_tol)
                        << "q_mean b=" << bi << " h=" << hi << " qi=" << qi << " d=" << d;
                }

    // ======================== VERIFY delta_s ================================
    // Tolerance accounts for GEMM with InputT accumulators over hdim additions.
    const float delta_s_tol = 1e-2f * static_cast<float>(hd);
    for(int bi = 0; bi < b; bi++)
        for(int hi = 0; hi < h; hi++)
            for(int qi = 0; qi < num_q_tiles; qi++)
                for(int kj = 0; kj < sk; kj++)
                {
                    const int gpu_off = bi * h * num_q_tiles * sk_pad + hi * num_q_tiles * sk_pad +
                                        qi * sk_pad + kj;
                    const int ref_off =
                        bi * h * num_q_tiles * sk + hi * num_q_tiles * sk + qi * sk + kj;
                    EXPECT_NEAR(delta_s_gpu[gpu_off], delta_s_ref[ref_off], delta_s_tol)
                        << "delta_s b=" << bi << " h=" << hi << " qi=" << qi << " kj=" << kj;
                }

    // ======================== VERIFY Q scale bytes ==========================
    for(int bi = 0; bi < b; bi++)
        for(int hi = 0; hi < h; hi++)
            for(int n = 0; n < sq; n++)
                for(int g = 0; g < hd / kG; g++)
                {
                    const int gpu_off =
                        bi * h * sq_pad * (hd / kG) + hi * sq_pad * (hd / kG) + n * (hd / kG) + g;
                    const int ref_off =
                        bi * h * sq * (hd / kG) + hi * sq * (hd / kG) + n * (hd / kG) + g;
                    const int diff = std::abs(static_cast<int>(q_scale_gpu[gpu_off]) -
                                              static_cast<int>(q_scale_ref[ref_off]));
                    EXPECT_LE(diff, max_scale_diff)
                        << "q_scale b=" << bi << " h=" << hi << " n=" << n << " g=" << g
                        << " gpu=" << static_cast<int>(q_scale_gpu[gpu_off])
                        << " ref=" << static_cast<int>(q_scale_ref[ref_off]);
                }

    // ======================== VERIFY Q hat (dequantized) ====================
    {
        // Extract valid rows from padded GPU buffer.
        std::vector<uint8_t> q_hat_real(b * h * sq * (hd / 2));
        std::vector<uint8_t> q_scale_real(b * h * sq * (hd / kG));
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sq; n++)
                {
                    const int src_hat =
                        bi * h * sq_pad * (hd / 2) + hi * sq_pad * (hd / 2) + n * (hd / 2);
                    const int dst_hat = bi * h * sq * (hd / 2) + hi * sq * (hd / 2) + n * (hd / 2);
                    std::copy(q_hat_gpu.begin() + src_hat,
                              q_hat_gpu.begin() + src_hat + (hd / 2),
                              q_hat_real.begin() + dst_hat);
                    const int src_sc =
                        bi * h * sq_pad * (hd / kG) + hi * sq_pad * (hd / kG) + n * (hd / kG);
                    const int dst_sc =
                        bi * h * sq * (hd / kG) + hi * sq * (hd / kG) + n * (hd / kG);
                    std::copy(q_scale_gpu.begin() + src_sc,
                              q_scale_gpu.begin() + src_sc + (hd / kG),
                              q_scale_real.begin() + dst_sc);
                }

        std::vector<float> q_dequant(b * h * sq * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            q_hat_real.data(), q_scale_real.data(), q_dequant.data(), b, h, sq, hd);

        std::vector<float> q_smooth_ref(b * h * sq * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int qi = 0; qi < num_q_tiles; qi++)
                {
                    const int rs = qi * kM0;
                    const int re = std::min(rs + kM0, sq);
                    for(int n = rs; n < re; n++)
                        for(int d = 0; d < hd; d++)
                        {
                            const float qval = Q_f32[bi * h * sq * hd + hi * sq * hd + n * hd + d];
                            const float mean = q_mean_ref_rt[bi * h * num_q_tiles * hd +
                                                             hi * num_q_tiles * hd + qi * hd + d];
                            q_smooth_ref[bi * h * sq * hd + hi * sq * hd + n * hd + d] =
                                qval - mean;
                        }
                }

        for(std::size_t i = 0; i < q_dequant.size(); i++)
            EXPECT_NEAR(q_dequant[i], q_smooth_ref[i], 1.0f) << "Q_hat dequant[" << i << "]";
    }

    // ======================== VERIFY K scale bytes ==========================
    for(int bi = 0; bi < b; bi++)
        for(int hi = 0; hi < h; hi++)
            for(int n = 0; n < sk; n++)
                for(int g = 0; g < hd / kG; g++)
                {
                    const int gpu_off =
                        bi * h * sk_pad * (hd / kG) + hi * sk_pad * (hd / kG) + n * (hd / kG) + g;
                    const int ref_off =
                        bi * h * sk * (hd / kG) + hi * sk * (hd / kG) + n * (hd / kG) + g;
                    const int diff = std::abs(static_cast<int>(k_scale_gpu[gpu_off]) -
                                              static_cast<int>(k_scale_ref[ref_off]));
                    EXPECT_LE(diff, max_scale_diff)
                        << "k_scale b=" << bi << " h=" << hi << " n=" << n << " g=" << g
                        << " gpu=" << static_cast<int>(k_scale_gpu[gpu_off])
                        << " ref=" << static_cast<int>(k_scale_ref[ref_off]);
                }

    // ======================== VERIFY K hat (dequantized) ====================
    {
        // Extract valid rows from padded GPU buffer.
        std::vector<uint8_t> k_hat_real(b * h * sk * (hd / 2));
        std::vector<uint8_t> k_scale_real(b * h * sk * (hd / kG));
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                {
                    const int src_hat =
                        bi * h * sk_pad * (hd / 2) + hi * sk_pad * (hd / 2) + n * (hd / 2);
                    const int dst_hat = bi * h * sk * (hd / 2) + hi * sk * (hd / 2) + n * (hd / 2);
                    std::copy(k_hat_gpu.begin() + src_hat,
                              k_hat_gpu.begin() + src_hat + (hd / 2),
                              k_hat_real.begin() + dst_hat);
                    const int src_sc =
                        bi * h * sk_pad * (hd / kG) + hi * sk_pad * (hd / kG) + n * (hd / kG);
                    const int dst_sc =
                        bi * h * sk * (hd / kG) + hi * sk * (hd / kG) + n * (hd / kG);
                    std::copy(k_scale_gpu.begin() + src_sc,
                              k_scale_gpu.begin() + src_sc + (hd / kG),
                              k_scale_real.begin() + dst_sc);
                }

        std::vector<float> k_dequant(b * h * sk * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            k_hat_real.data(), k_scale_real.data(), k_dequant.data(), b, h, sk, hd);

        std::vector<float> k_smooth_ref(b * h * sk * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                    for(int d = 0; d < hd; d++)
                    {
                        const float kval = K_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d];
                        const float mean = k_mean_ref_rt[bi * h * hd + hi * hd + d];
                        k_smooth_ref[bi * h * sk * hd + hi * sk * hd + n * hd + d] = kval - mean;
                    }

        for(std::size_t i = 0; i < k_dequant.size(); i++)
            EXPECT_NEAR(k_dequant[i], k_smooth_ref[i], 1.0f) << "K_hat dequant[" << i << "]";
    }

    // ======================== VERIFY V scale bytes ==========================
    for(int bi = 0; bi < b; bi++)
        for(int hi = 0; hi < h; hi++)
            for(int d = 0; d < hd; d++)
                for(int g = 0; g < sk / kG; g++)
                {
                    const int gpu_off = bi * h * hd * (sk_pad / kG) + hi * hd * (sk_pad / kG) +
                                        d * (sk_pad / kG) + g;
                    const int ref_off =
                        bi * h * hd * (sk / kG) + hi * hd * (sk / kG) + d * (sk / kG) + g;
                    const int diff = std::abs(static_cast<int>(v_scale_gpu[gpu_off]) -
                                              static_cast<int>(v_scale_ref[ref_off]));
                    EXPECT_LE(diff, max_scale_diff)
                        << "v_scale b=" << bi << " h=" << hi << " d=" << d << " g=" << g
                        << " gpu=" << static_cast<int>(v_scale_gpu[gpu_off])
                        << " ref=" << static_cast<int>(v_scale_ref[ref_off]);
                }

    // ======================== VERIFY V hat (dequantized, transposed layout) =
    {
        // V hat GPU layout: [B, H, hdim, sk_pad/2]. Extract valid sk columns.
        std::vector<uint8_t> v_hat_real(b * h * hd * (sk / 2));
        std::vector<uint8_t> v_scale_real(b * h * hd * (sk / kG));
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                {
                    const int src_hat =
                        bi * h * hd * (sk_pad / 2) + hi * hd * (sk_pad / 2) + d * (sk_pad / 2);
                    const int dst_hat = bi * h * hd * (sk / 2) + hi * hd * (sk / 2) + d * (sk / 2);
                    std::copy(v_hat_gpu.begin() + src_hat,
                              v_hat_gpu.begin() + src_hat + (sk / 2),
                              v_hat_real.begin() + dst_hat);
                    const int src_sc =
                        bi * h * hd * (sk_pad / kG) + hi * hd * (sk_pad / kG) + d * (sk_pad / kG);
                    const int dst_sc =
                        bi * h * hd * (sk / kG) + hi * hd * (sk / kG) + d * (sk / kG);
                    std::copy(v_scale_gpu.begin() + src_sc,
                              v_scale_gpu.begin() + src_sc + (sk / kG),
                              v_scale_real.begin() + dst_sc);
                }

        std::vector<float> v_dequant(b * h * hd * sk);
        ck_tile::reference::reference_dequant_mxfp4(
            v_hat_real.data(), v_scale_real.data(), v_dequant.data(), b, h, hd, sk);

        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                    for(int n = 0; n < sk; n++)
                    {
                        const float vref = V_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d];
                        const float vgpu = v_dequant[bi * h * hd * sk + hi * hd * sk + d * sk + n];
                        EXPECT_NEAR(vgpu, vref, 1.0f)
                            << "V_hat dequant b=" << bi << " h=" << hi << " d=" << d << " n=" << n;
                    }
    }
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------
TEST_P(SageAttnV3PreprocessTest, Fp16Input) { RunGPUTest<ck_tile::fp16_t>(); }

TEST_P(SageAttnV3PreprocessTest, Float32Input) { RunGPUTest<float>(); }

// ---------------------------------------------------------------------------
// Test instantiation: (B, H, seqlen_q, seqlen_k, hdim, unused_flag)
// hdim must be 128 or 256; seqlen_k must be divisible by 32 (V quantization group).
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    Shapes,
    SageAttnV3PreprocessTest,
    ::testing::Values(
        // --- aligned cases (seqlen multiples of kRows=128) ---
        std::make_tuple(1, 1, 256, 128, 128, true),
        std::make_tuple(2, 4, 128, 128, 128, true),
        std::make_tuple(1, 2, 128, 256, 128, true),
        std::make_tuple(1, 1, 128, 128, 256, true),
        std::make_tuple(1, 2, 128, 256, 256, true),
        // --- irregular seqlen (non-multiples of kRows=128) ---
        std::make_tuple(1, 1, 65, 96, 128, true),   // sq=65, sk=96 (both < kRows)
        std::make_tuple(1, 1, 127, 96, 128, true),  // sq=127=kRows-1, sk=96
        std::make_tuple(1, 2, 130, 96, 128, true),  // sq > kRows, sk < kRows
        std::make_tuple(2, 4, 300, 192, 128, true), // multi-tile, sk=192 (not multiple of 128)
        std::make_tuple(1, 1, 65, 96, 256, true),   // hdim=256 + misaligned
        std::make_tuple(1, 2, 300, 192, 256, true)  // hdim=256 + multi-tile
        ));

#else // CK_USE_NATIVE_MX_SUPPORT not defined

TEST(SageAttnV3PreprocessTest, SkippedOnNonGfx950)
{
    GTEST_SKIP() << "SageAttention V3 preprocessing requires gfx950 (CK_USE_NATIVE_MX_SUPPORT)";
}

#endif // CK_USE_NATIVE_MX_SUPPORT
