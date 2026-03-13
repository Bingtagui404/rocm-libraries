// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// End-to-end GPU correctness test for SageAttn preprocessing (SA3).
//
// Exercises the four-kernel pipeline launched by SageAttnV3Preprocess::run():
//   Launch 0: SageAttnV3KMeanKernel     -- computes k_mean[b,h,d] = mean_n(K[b,h,n,d])
//   Launch 1: SageAttnV3PreprocessKernel -- Q mean+quantize, K smooth+quantize
//   Launch 1b: SageAttnV3VPreprocessKernel -- V transpose+quantize
//   Launch 2: BatchedGemmKernel         -- delta_s = q_mean @ K'^T
//
// Verified outputs:
//   k_mean   -- InputT, compared vs CPU reference (< 2e-3 abs error for fp16)
//   q_mean   -- InputT, compared vs CPU reference (< 2e-3 abs error for fp16)
//   delta_s  -- float,  compared vs CPU reference (< 1e-2 * hdim abs error)
//   Q hat    -- MXFP4, dequantized and compared with Q_smooth reference (< 1.0 abs error)
//   Q scale  -- e8m0 bytes, compared directly with CPU reference
//   K hat    -- MXFP4, dequantized and compared with K_smooth reference (< 1.0 abs error)
//   K scale  -- e8m0 bytes, compared directly with CPU reference
//   V hat    -- MXFP4 (transposed), dequantized and compared with V reference (< 1.0 abs error)
//   V scale  -- e8m0 bytes, compared directly with CPU reference

#include "gtest/gtest.h"

#ifdef CK_USE_NATIVE_MX_SUPPORT

#include "ck_tile/host.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_preprocess.hpp"
#include "ck_tile/host/reference/reference_sageattn_v3_preprocess.hpp"

CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 11939)

namespace {

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

    static constexpr int kG = 32; // MXFP4 scale granularity

    template <typename InputT>
    void RunGPUTest();
};

template <typename InputT>
void SageAttnV3PreprocessTest::RunGPUTest()
{
    const int b = B(), h = H(), sq = seqlen_q(), sk = seqlen_k(), hd = hdim();
    ASSERT_EQ(hd % kG, 0) << "hdim must be divisible by 32";

    // fp16 + hdim=256 uses kRows=64 so the Q tile fits in LDS (kUseLdsQ=true).
    // All other combinations use kRows=128.
    const bool fp16_d256 = std::is_same_v<InputT, ck_tile::fp16_t> && (hd == 256);
    const int kM0 = fp16_d256 ? 64 : 128; // Q/K tile rows (padding granularity)

    // Use get_buffer_sizes to get padded dimensions and required buffer sizes.
    // Dispatch on hdim (and input type) to select the right instantiation.
    ck_tile::SageAttnV3PreprocessBufferSizes bsz{};
    if(hd == 64)
        bsz = ck_tile::SageAttnV3Preprocess<InputT, 128, 64>::get_buffer_sizes(b, h, sq, sk, hd);
    else if(hd == 128)
        bsz = ck_tile::SageAttnV3Preprocess<InputT, 128, 128>::get_buffer_sizes(b, h, sq, sk, hd);
    else if(hd == 256 && fp16_d256)
        bsz = ck_tile::SageAttnV3Preprocess<InputT, 64, 256>::get_buffer_sizes(b, h, sq, sk, hd);
    else if(hd == 256)
        bsz = ck_tile::SageAttnV3Preprocess<InputT, 128, 256>::get_buffer_sizes(b, h, sq, sk, hd);
    else
        FAIL() << "Unsupported hdim (must be 64, 128, or 256)";

    const int sq_pad      = static_cast<int>(bsz.seqlen_q_padded);
    const int sk_pad      = static_cast<int>(bsz.seqlen_k_padded);
    const int num_q_tiles = static_cast<int>(bsz.num_q_tiles);

    // ---- Generate input data using HostTensor + FillUniformDistribution ----
    const uint32_t seed = static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED)));

    ck_tile::HostTensor<InputT> q_host({b, h, sq, hd});
    ck_tile::HostTensor<InputT> k_host({b, h, sk, hd});
    ck_tile::HostTensor<InputT> v_host({b, h, sk, hd});

    ck_tile::FillUniformDistribution<InputT>{-1.f, 1.f, seed + 0}(q_host);
    ck_tile::FillUniformDistribution<InputT>{-1.f, 1.f, seed + 1}(k_host);
    ck_tile::FillUniformDistribution<InputT>{-1.f, 1.f, seed + 2}(v_host);

    // Convert to float for CPU reference (round-trip through InputT so values match GPU).
    auto to_f32 = [](const ck_tile::HostTensor<InputT>& t) {
        std::vector<float> v(t.get_element_space_size());
        for(std::size_t i = 0; i < v.size(); i++)
            v[i] = ck_tile::type_convert<float>(t.data()[i]);
        return v;
    };
    auto Q_f32 = to_f32(q_host);
    auto K_f32 = to_f32(k_host);
    auto V_f32 = to_f32(v_host);

    // ---- CPU reference: k_mean (float precision) ----
    std::vector<float> k_mean_ref_f32(b * h * hd, 0.0f);
    ck_tile::reference::reference_sageattn_v3_k_smooth(
        K_f32.data(), k_mean_ref_f32.data(), b, h, sk, hd);

    // If InputT = fp16, the GPU stores k_mean as fp16 -> quantise the CPU reference too.
    // All downstream references (delta_s, K_smooth) must use the rounded k_mean.
    std::vector<float> k_mean_ref_rt = k_mean_ref_f32;
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

    // q_mean also stored as InputT on GPU -> round-trip if fp16.
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

    // ---- Upload input tensors to GPU ----
    ck_tile::DeviceMem q_dev(q_host.get_element_space_size() * sizeof(InputT));
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size() * sizeof(InputT));
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size() * sizeof(InputT));
    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());

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

    // ---- Build kernel args ----
    ck_tile::SageAttnV3PreprocessArgs<InputT> hargs{};
    hargs.q_ptr          = static_cast<const InputT*>(q_dev.GetDeviceBuffer());
    hargs.seqlen_q       = sq;
    hargs.hdim           = hd;
    hargs.stride_q       = hd;
    hargs.nhead_stride_q = sq * hd;
    hargs.batch_stride_q = h * sq * hd;
    hargs.q_hat_ptr      = static_cast<uint8_t*>(q_hat_dev.GetDeviceBuffer());
    hargs.q_scale_ptr    = static_cast<uint8_t*>(q_scale_dev.GetDeviceBuffer());
    hargs.q_mean_ptr     = static_cast<InputT*>(q_mean_dev.GetDeviceBuffer());

    hargs.k_ptr          = static_cast<const InputT*>(k_dev.GetDeviceBuffer());
    hargs.seqlen_k       = sk;
    hargs.stride_k       = hd;
    hargs.nhead_stride_k = sk * hd;
    hargs.batch_stride_k = h * sk * hd;
    hargs.k_hat_ptr      = static_cast<uint8_t*>(k_hat_dev.GetDeviceBuffer());
    hargs.k_scale_ptr    = static_cast<uint8_t*>(k_scale_dev.GetDeviceBuffer());

    hargs.v_ptr          = static_cast<const InputT*>(v_dev.GetDeviceBuffer());
    hargs.nhead_stride_v = sk * hd;
    hargs.batch_stride_v = h * sk * hd;
    hargs.v_hat_ptr      = static_cast<uint8_t*>(v_hat_dev.GetDeviceBuffer());
    hargs.v_scale_ptr    = static_cast<uint8_t*>(v_scale_dev.GetDeviceBuffer());

    hargs.batch  = b;
    hargs.nhead  = h;

    // ---- Launch ----
    auto* ds_ptr    = static_cast<float*>(delta_s_dev.GetDeviceBuffer());
    auto* kmean_ptr = static_cast<InputT*>(k_mean_buf.GetDeviceBuffer());
    auto* kprime_ptr = static_cast<InputT*>(k_prime_buf.GetDeviceBuffer());
    if(hd == 64)
        ck_tile::SageAttnV3Preprocess<InputT, 128, 64>::run(hargs, ds_ptr, kmean_ptr, kprime_ptr,
                                                             /*stream=*/nullptr);
    else if(hd == 128)
        ck_tile::SageAttnV3Preprocess<InputT, 128, 128>::run(hargs, ds_ptr, kmean_ptr, kprime_ptr,
                                                              /*stream=*/nullptr);
    else if(fp16_d256)
        ck_tile::SageAttnV3Preprocess<InputT, 64, 256>::run(hargs, ds_ptr, kmean_ptr, kprime_ptr,
                                                             /*stream=*/nullptr);
    else
        ck_tile::SageAttnV3Preprocess<InputT, 128, 256>::run(hargs, ds_ptr, kmean_ptr, kprime_ptr,
                                                              /*stream=*/nullptr);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // ---- Copy results back ----
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

    // Tolerances
    const float mean_tol     = use_fp16() ? 2e-3f : 1e-4f;
    const float scl_atol     = use_fp16() ? 1.0f : 0.0f;
    const float delta_s_atol = 1e-2f * static_cast<float>(hd);

    // Helper: compact valid rows from a padded GPU buffer and build dequant+smooth reference.
    // For padded outputs (hat/scale), use ck_tile::check_err with gpu_stride overload directly.

    // ======================== VERIFY k_mean =================================
    ASSERT_TRUE(ck_tile::check_err(
        k_mean_gpu_f32.data(), 1, k_mean_ref_f32.data(), 1, k_mean_gpu_f32.size(), "k_mean",
        0.0, mean_tol));

    // ======================== VERIFY q_mean =================================
    ASSERT_TRUE(ck_tile::check_err(
        q_mean_gpu_f32.data(), 1, q_mean_ref_f32.data(), 1, q_mean_gpu_f32.size(), "q_mean",
        0.0, mean_tol));

    // ======================== VERIFY delta_s ================================
    // GPU: [b*h*num_q_tiles, sk_pad], valid: sk columns per row.
    ASSERT_TRUE(ck_tile::check_err(delta_s_gpu.data(),
                                   sk_pad,
                                   delta_s_ref.data(),
                                   b * h * num_q_tiles,
                                   sk,
                                   "delta_s",
                                   0.0,
                                   delta_s_atol));

    // ======================== VERIFY Q scale bytes ==========================
    // GPU: [b*h, sq_pad*(hd/kG)], valid: sq*(hd/kG) per outer row.
    ASSERT_TRUE(ck_tile::check_err(q_scale_gpu.data(),
                                   sq_pad * (hd / kG),
                                   q_scale_ref.data(),
                                   b * h,
                                   sq * (hd / kG),
                                   "q_scale",
                                   0.0,
                                   scl_atol));

    // ======================== VERIFY K scale bytes ==========================
    ASSERT_TRUE(ck_tile::check_err(k_scale_gpu.data(),
                                   sk_pad * (hd / kG),
                                   k_scale_ref.data(),
                                   b * h,
                                   sk * (hd / kG),
                                   "k_scale",
                                   0.0,
                                   scl_atol));

    // ======================== VERIFY V scale bytes ==========================
    // GPU: [b*h*hd, sk_pad/kG], valid: sk/kG per outer row.
    ASSERT_TRUE(ck_tile::check_err(v_scale_gpu.data(),
                                   sk_pad / kG,
                                   v_scale_ref.data(),
                                   b * h * hd,
                                   sk / kG,
                                   "v_scale",
                                   0.0,
                                   scl_atol));

    // Helper: compact valid rows from a padded GPU uint8 buffer.
    auto compact_u8 = [](const std::vector<uint8_t>& src,
                         std::size_t n_outer,
                         std::size_t gpu_stride,
                         std::size_t n_valid) {
        std::vector<uint8_t> out(n_outer * n_valid);
        for(std::size_t i = 0; i < n_outer; i++)
            std::copy(src.data() + i * gpu_stride,
                      src.data() + i * gpu_stride + n_valid,
                      out.data() + i * n_valid);
        return out;
    };

    // ======================== VERIFY Q hat (dequantized) ====================
    {
        const auto q_hat_c   = compact_u8(q_hat_gpu,   b * h, sq_pad * (hd / 2),  sq * (hd / 2));
        const auto q_scale_c = compact_u8(q_scale_gpu, b * h, sq_pad * (hd / kG), sq * (hd / kG));
        std::vector<float> q_dequant(b * h * sq * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            q_hat_c.data(), q_scale_c.data(), q_dequant.data(), b, h, sq, hd);

        std::vector<float> q_smooth(b * h * sq * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int qi = 0; qi < num_q_tiles; qi++)
                {
                    const int rs = qi * kM0, re = std::min(rs + kM0, sq);
                    for(int n = rs; n < re; n++)
                        for(int d = 0; d < hd; d++)
                            q_smooth[bi * h * sq * hd + hi * sq * hd + n * hd + d] =
                                Q_f32[bi * h * sq * hd + hi * sq * hd + n * hd + d] -
                                q_mean_ref_rt[bi * h * num_q_tiles * hd +
                                              hi * num_q_tiles * hd + qi * hd + d];
                }
        ASSERT_TRUE(ck_tile::check_err(q_dequant, q_smooth, "q_hat (dequant)", 0.0, 1.0));
    }

    // ======================== VERIFY K hat (dequantized) ====================
    {
        const auto k_hat_c   = compact_u8(k_hat_gpu,   b * h, sk_pad * (hd / 2),  sk * (hd / 2));
        const auto k_scale_c = compact_u8(k_scale_gpu, b * h, sk_pad * (hd / kG), sk * (hd / kG));
        std::vector<float> k_dequant(b * h * sk * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            k_hat_c.data(), k_scale_c.data(), k_dequant.data(), b, h, sk, hd);

        std::vector<float> k_smooth(b * h * sk * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                    for(int d = 0; d < hd; d++)
                        k_smooth[bi * h * sk * hd + hi * sk * hd + n * hd + d] =
                            K_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d] -
                            k_mean_ref_rt[bi * h * hd + hi * hd + d];
        ASSERT_TRUE(ck_tile::check_err(k_dequant, k_smooth, "k_hat (dequant)", 0.0, 1.0));
    }

    // ======================== VERIFY V hat (dequantized, transposed layout) ==
    {
        // V hat GPU layout: [b*h*hd, sk_pad/2], valid: sk/2 per outer row.
        const auto v_hat_c   = compact_u8(v_hat_gpu,   b * h * hd, sk_pad / 2,  sk / 2);
        const auto v_scale_c = compact_u8(v_scale_gpu, b * h * hd, sk_pad / kG, sk / kG);
        std::vector<float> v_dequant(b * h * hd * sk);
        ck_tile::reference::reference_dequant_mxfp4(
            v_hat_c.data(), v_scale_c.data(), v_dequant.data(), b, h, hd, sk);

        // v_dequant is [b,h,hd,sk]; build transposed ref from V_f32 [b,h,sk,hd].
        std::vector<float> v_ref_t(b * h * hd * sk);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                    for(int n = 0; n < sk; n++)
                        v_ref_t[bi * h * hd * sk + hi * hd * sk + d * sk + n] =
                            V_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d];
        ASSERT_TRUE(ck_tile::check_err(v_dequant, v_ref_t, "v_hat (dequant)", 0.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------
TEST_P(SageAttnV3PreprocessTest, Fp16Input) { RunGPUTest<ck_tile::fp16_t>(); }

TEST_P(SageAttnV3PreprocessTest, Float32Input) { RunGPUTest<float>(); }

// ---------------------------------------------------------------------------
// Test instantiation: (B, H, seqlen_q, seqlen_k, hdim, unused_flag)
// hdim must be 64, 128, or 256; seqlen_k must be divisible by 32 (V quantization group).
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    Shapes,
    SageAttnV3PreprocessTest,
    ::testing::Values(
        // --- hdim=64 ---
        std::make_tuple(1, 1, 128, 128, 64, true),
        std::make_tuple(1, 1, 65, 96, 64, true),    // misaligned seqlen with hdim=64
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

} // namespace

#else // CK_USE_NATIVE_MX_SUPPORT not defined

TEST(SageAttnV3PreprocessTest, SkippedOnNonGfx950)
{
    GTEST_SKIP() << "SageAttention V3 preprocessing requires gfx950 (CK_USE_NATIVE_MX_SUPPORT)";
}

#endif // CK_USE_NATIVE_MX_SUPPORT
