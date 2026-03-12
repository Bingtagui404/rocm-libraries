// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Benchmark / correctness verifier for SageAttnV3Preprocess -- the four-kernel
// SA3 preprocessing pipeline:
//   k_mean -> preprocess Q/K -> V tile-transpose -> delta_s GEMM
//
// Usage:
//   ./bin/tile_example_sageattn_v3_preprocess [options]
//
// Benchmark mode (default): reports per-kernel HBM bandwidth breakdown.
//
//   [0] KMean:      Read K, Write k_mean
//   [1] Prep Q/K:   Read Q, K, k_mean; Write q_mean, q_hat, q_scale, K', k_hat, k_scale
//   [1b] Prep V:    Read V; Write v_hat, v_scale
//   [2] delta_s:    Read q_mean, K'; Write delta_s (float32)
//
//   where G=32 (MXFP4 scale granularity), T_q = ceil(Sq/128).

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef CK_USE_NATIVE_MX_SUPPORT

#include "ck_tile/host.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_preprocess.hpp"
#include "ck_tile/host/reference/reference_sageattn_v3_preprocess.hpp"

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "4", "batch size")
        .insert("h", "16", "number of heads")
        .insert("q", "1024", "seqlen_q")
        .insert("k", "4096", "seqlen_k")
        .insert("d", "128", "hdim (128 or 256)")
        .insert("prec", "fp16", "data type: fp16 | fp32")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "50", "measurement iterations")
        .insert("csv", "0", "1: CSV output")
        .insert("verify", "0", "1: run correctness check vs CPU reference");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// ---------------------------------------------------------------------------
// Per-kernel timing result
// ---------------------------------------------------------------------------

struct BenchResult
{
    float kmean_ms        = 0.0f;
    float preprocess_ms   = 0.0f;
    float v_preprocess_ms = 0.0f;
    float delta_s_ms      = 0.0f;
    float total_ms        = 0.0f;
};

// ---------------------------------------------------------------------------
// Shape bundle (passed to templated run functions)
// ---------------------------------------------------------------------------

struct RunShape
{
    int batch, nhead, seqlen_q, seqlen_k, hdim;
    int warmup = 5, repeat = 50;
};

// ---------------------------------------------------------------------------
// Helper: build kernel args from shape + device buffers
// ---------------------------------------------------------------------------

template <typename InputT>
static ck_tile::SageAttnV3PreprocessArgs<InputT>
make_hargs(const RunShape& s,
           int sq_pad,
           int sk_pad,
           int num_q_tiles,
           int num_k_tiles,
           void* q_ptr,
           void* k_ptr,
           void* v_ptr,
           void* q_hat_ptr,
           void* q_scale_ptr,
           void* q_mean_ptr,
           void* k_hat_ptr,
           void* k_scale_ptr,
           void* v_hat_ptr,
           void* v_scale_ptr)
{
    constexpr int kG  = 32;
    constexpr int kM0 = 128;
    const int b = s.batch, h = s.nhead, sq = s.seqlen_q, sk = s.seqlen_k, hd = s.hdim;

    ck_tile::SageAttnV3PreprocessArgs<InputT> a{};
    a.q_ptr                = static_cast<const InputT*>(q_ptr);
    a.seqlen_q             = sq;
    a.hdim                 = hd;
    a.stride_q             = hd;
    a.nhead_stride_q       = sq * hd;
    a.batch_stride_q       = h * sq * hd;
    a.q_hat_ptr            = static_cast<uint8_t*>(q_hat_ptr);
    a.stride_q_hat         = hd / 2;
    a.nhead_stride_q_hat   = sq_pad * (hd / 2);
    a.batch_stride_q_hat   = h * sq_pad * (hd / 2);
    a.q_scale_ptr          = static_cast<uint8_t*>(q_scale_ptr);
    a.stride_q_scale       = hd / kG;
    a.nhead_stride_q_scale = sq_pad * (hd / kG);
    a.batch_stride_q_scale = h * sq_pad * (hd / kG);
    a.q_mean_ptr           = static_cast<InputT*>(q_mean_ptr);
    a.q_tile_size          = kM0;
    a.stride_q_mean        = hd;
    a.nhead_stride_q_mean  = num_q_tiles * hd;
    a.batch_stride_q_mean  = h * num_q_tiles * hd;
    a.k_ptr                = static_cast<const InputT*>(k_ptr);
    a.seqlen_k             = sk;
    a.stride_k             = hd;
    a.nhead_stride_k       = sk * hd;
    a.batch_stride_k       = h * sk * hd;
    a.k_hat_ptr            = static_cast<uint8_t*>(k_hat_ptr);
    a.stride_k_hat         = hd / 2;
    a.nhead_stride_k_hat   = sk_pad * (hd / 2);
    a.batch_stride_k_hat   = h * sk_pad * (hd / 2);
    a.k_scale_ptr          = static_cast<uint8_t*>(k_scale_ptr);
    a.stride_k_scale       = hd / kG;
    a.nhead_stride_k_scale = sk_pad * (hd / kG);
    a.batch_stride_k_scale = h * sk_pad * (hd / kG);
    a.v_ptr                = static_cast<const InputT*>(v_ptr);
    a.nhead_stride_v       = sk * hd;
    a.batch_stride_v       = h * sk * hd;
    a.v_hat_ptr            = static_cast<uint8_t*>(v_hat_ptr);
    a.stride_v_hat         = sk_pad / 2;
    a.nhead_stride_v_hat   = hd * (sk_pad / 2);
    a.batch_stride_v_hat   = h * hd * (sk_pad / 2);
    a.v_scale_ptr          = static_cast<uint8_t*>(v_scale_ptr);
    a.stride_v_scale       = sk_pad / kG;
    a.nhead_stride_v_scale = hd * (sk_pad / kG);
    a.batch_stride_v_scale = h * hd * (sk_pad / kG);
    a.batch                = b;
    a.nhead                = h;
    a.num_q_tiles          = num_q_tiles;
    a.num_k_tiles          = num_k_tiles;
    return a;
}

// ---------------------------------------------------------------------------
// Benchmark: templated on InputT and kCols (hdim)
// ---------------------------------------------------------------------------

template <typename InputT, int kRows, int kCols>
BenchResult run_benchmark(const RunShape& s)
{
    using SA3 = ck_tile::SageAttnV3Preprocess<InputT, kRows, kCols>;

    const int b = s.batch, h = s.nhead, sq = s.seqlen_q, sk = s.seqlen_k, hd = s.hdim;

    const auto bsz        = SA3::get_buffer_sizes(b, h, sq, sk, hd);
    const int sq_pad      = static_cast<int>(bsz.seqlen_q_padded);
    const int sk_pad      = static_cast<int>(bsz.seqlen_k_padded);
    const int num_q_tiles = static_cast<int>(bsz.num_q_tiles);
    const int num_k_tiles = static_cast<int>(bsz.num_k_tiles);

    ck_tile::DeviceMem q_dev(std::size_t(b * h * sq * hd) * sizeof(InputT));
    ck_tile::DeviceMem k_dev(std::size_t(b * h * sk * hd) * sizeof(InputT));
    ck_tile::DeviceMem v_dev(std::size_t(b * h * sk * hd) * sizeof(InputT));
    ck_tile::DeviceMem q_hat_dev(bsz.q_hat_bytes);
    ck_tile::DeviceMem q_scale_dev(bsz.q_scale_bytes);
    ck_tile::DeviceMem q_mean_dev(bsz.q_mean_bytes);
    ck_tile::DeviceMem k_hat_dev(bsz.k_hat_bytes);
    ck_tile::DeviceMem k_scale_dev(bsz.k_scale_bytes);
    ck_tile::DeviceMem v_hat_dev(bsz.v_hat_bytes);
    ck_tile::DeviceMem v_scale_dev(bsz.v_scale_bytes);
    ck_tile::DeviceMem delta_s_dev(bsz.delta_s_bytes);
    ck_tile::DeviceMem k_mean_buf(bsz.k_mean_bytes);
    ck_tile::DeviceMem k_prime_buf(bsz.k_prime_bytes);
    ck_tile::DeviceMem k_mean_partial_buf(bsz.k_mean_partial_bytes);
    ck_tile::DeviceMem counter_buf(bsz.counter_bytes);

    // Benchmark only -- inputs initialized to zero.
    q_dev.SetZero();
    k_dev.SetZero();
    v_dev.SetZero();

    auto hargs = make_hargs<InputT>(s,
                                    sq_pad,
                                    sk_pad,
                                    num_q_tiles,
                                    num_k_tiles,
                                    q_dev.GetDeviceBuffer(),
                                    k_dev.GetDeviceBuffer(),
                                    v_dev.GetDeviceBuffer(),
                                    q_hat_dev.GetDeviceBuffer(),
                                    q_scale_dev.GetDeviceBuffer(),
                                    q_mean_dev.GetDeviceBuffer(),
                                    k_hat_dev.GetDeviceBuffer(),
                                    k_scale_dev.GetDeviceBuffer(),
                                    v_hat_dev.GetDeviceBuffer(),
                                    v_scale_dev.GetDeviceBuffer());

    auto* k_mean_ptr    = static_cast<InputT*>(k_mean_buf.GetDeviceBuffer());
    auto* k_prime_ptr   = static_cast<InputT*>(k_prime_buf.GetDeviceBuffer());
    auto* k_partial_ptr = static_cast<float*>(k_mean_partial_buf.GetDeviceBuffer());
    auto* counter_ptr   = static_cast<int32_t*>(counter_buf.GetDeviceBuffer());
    auto* delta_s_ptr   = static_cast<float*>(delta_s_dev.GetDeviceBuffer());

    hipStream_t stream = nullptr;
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK_ERROR(hipEventCreate(&ev_start));
    HIP_CHECK_ERROR(hipEventCreate(&ev_stop));

    auto run_stages = [&](uint32_t stages) {
        SA3::run(hargs, delta_s_ptr, k_mean_ptr, k_prime_ptr, k_partial_ptr, counter_ptr, stream,
                 stages);
    };

    // Full-pipeline warmup (populates k_mean, k_prime for cache-warm per-stage timing).
    for(int i = 0; i < s.warmup; ++i)
        run_stages(ck_tile::kSA3StageAll);
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    auto time_ms = [&](uint32_t stage_mask) -> float {
        HIP_CHECK_ERROR(hipEventRecord(ev_start, stream));
        for(int i = 0; i < s.repeat; ++i)
            run_stages(stage_mask);
        HIP_CHECK_ERROR(hipEventRecord(ev_stop, stream));
        HIP_CHECK_ERROR(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, ev_start, ev_stop));
        return ms / static_cast<float>(s.repeat);
    };

    BenchResult res;
    res.total_ms        = time_ms(ck_tile::kSA3StageAll);
    res.kmean_ms        = time_ms(ck_tile::kSA3StageKMean);
    res.preprocess_ms   = time_ms(ck_tile::kSA3StagePreprocess);
    res.v_preprocess_ms = time_ms(ck_tile::kSA3StageVPreprocess);
    res.delta_s_ms      = time_ms(ck_tile::kSA3StageDeltaS);

    HIP_CHECK_ERROR(hipEventDestroy(ev_start));
    HIP_CHECK_ERROR(hipEventDestroy(ev_stop));

    return res;
}

// ---------------------------------------------------------------------------
// Correctness verification: templated on InputT and kCols (hdim)
// Returns true if all checks pass.
// ---------------------------------------------------------------------------

template <typename InputT, int kRows, int kCols>
bool run_verify(const RunShape& s)
{
    using SA3 = ck_tile::SageAttnV3Preprocess<InputT, kRows, kCols>;

    const int b = s.batch, h = s.nhead, sq = s.seqlen_q, sk = s.seqlen_k, hd = s.hdim;
    constexpr int kG = 32;

    const auto bsz        = SA3::get_buffer_sizes(b, h, sq, sk, hd);
    const int sq_pad      = static_cast<int>(bsz.seqlen_q_padded);
    const int sk_pad      = static_cast<int>(bsz.seqlen_k_padded);
    const int num_q_tiles = static_cast<int>(bsz.num_q_tiles);
    const int num_k_tiles = static_cast<int>(bsz.num_k_tiles);

    // ---- Generate inputs ----
    ck_tile::HostTensor<InputT> q_host({b, h, sq, hd});
    ck_tile::HostTensor<InputT> k_host({b, h, sk, hd});
    ck_tile::HostTensor<InputT> v_host({b, h, sk, hd});
    ck_tile::FillUniformDistribution<InputT>{-1.f, 1.f, 42u}(q_host);
    ck_tile::FillUniformDistribution<InputT>{-1.f, 1.f, 43u}(k_host);
    ck_tile::FillUniformDistribution<InputT>{-1.f, 1.f, 44u}(v_host);

    auto to_f32 = [](const ck_tile::HostTensor<InputT>& t) {
        std::vector<float> v(t.get_element_space_size());
        for(std::size_t i = 0; i < v.size(); i++)
            v[i] = ck_tile::type_convert<float>(t.data()[i]);
        return v;
    };
    auto Q_f32 = to_f32(q_host);
    auto K_f32 = to_f32(k_host);
    auto V_f32 = to_f32(v_host);

    // ---- CPU reference ----
    std::vector<float> k_mean_ref(std::size_t(b) * h * hd, 0.0f);
    ck_tile::reference::reference_sageattn_v3_k_smooth(
        K_f32.data(), k_mean_ref.data(), b, h, sk, hd);

    std::vector<float> k_mean_rt = k_mean_ref;
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
        for(auto& x : k_mean_rt)
            x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));

    std::vector<float> q_mean_ref(std::size_t(b) * h * num_q_tiles * hd, 0.0f);
    std::vector<uint8_t> q_hat_ref(std::size_t(b) * h * sq * (hd / 2), 0);
    std::vector<uint8_t> q_scale_ref(std::size_t(b) * h * sq * (hd / kG), 0);
    ck_tile::reference::reference_sageattn_v3_q_preprocess(
        Q_f32.data(), q_mean_ref.data(), q_hat_ref.data(), q_scale_ref.data(), b, h, sq, hd, kRows);

    std::vector<float> q_mean_rt = q_mean_ref;
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
        for(auto& x : q_mean_rt)
            x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));

    std::vector<float> delta_s_ref(std::size_t(b) * h * num_q_tiles * sk, 0.0f);
    ck_tile::reference::reference_sageattn_v3_delta_s(q_mean_rt.data(),
                                                      K_f32.data(),
                                                      k_mean_rt.data(),
                                                      delta_s_ref.data(),
                                                      b,
                                                      h,
                                                      num_q_tiles,
                                                      sk,
                                                      hd);

    std::vector<uint8_t> k_hat_ref(std::size_t(b) * h * sk * (hd / 2), 0);
    std::vector<uint8_t> k_scale_ref(std::size_t(b) * h * sk * (hd / kG), 0);
    ck_tile::reference::reference_sageattn_v3_k_preprocess(
        K_f32.data(), k_mean_rt.data(), k_hat_ref.data(), k_scale_ref.data(), b, h, sk, hd);

    std::vector<uint8_t> v_hat_ref(std::size_t(b) * h * hd * (sk / 2), 0);
    std::vector<uint8_t> v_scale_ref(std::size_t(b) * h * hd * (sk / kG), 0);
    ck_tile::reference::reference_sageattn_v3_v_preprocess(
        V_f32.data(), v_hat_ref.data(), v_scale_ref.data(), b, h, sk, hd);

    // ---- GPU buffers ----
    ck_tile::DeviceMem q_dev(q_host.get_element_space_size() * sizeof(InputT));
    ck_tile::DeviceMem k_dev(k_host.get_element_space_size() * sizeof(InputT));
    ck_tile::DeviceMem v_dev(v_host.get_element_space_size() * sizeof(InputT));
    q_dev.ToDevice(q_host.data());
    k_dev.ToDevice(k_host.data());
    v_dev.ToDevice(v_host.data());

    ck_tile::DeviceMem q_hat_dev(bsz.q_hat_bytes);
    ck_tile::DeviceMem q_scale_dev(bsz.q_scale_bytes);
    ck_tile::DeviceMem q_mean_dev(bsz.q_mean_bytes);
    ck_tile::DeviceMem k_hat_dev(bsz.k_hat_bytes);
    ck_tile::DeviceMem k_scale_dev(bsz.k_scale_bytes);
    ck_tile::DeviceMem v_hat_dev(bsz.v_hat_bytes);
    ck_tile::DeviceMem v_scale_dev(bsz.v_scale_bytes);
    ck_tile::DeviceMem delta_s_dev(bsz.delta_s_bytes);
    ck_tile::DeviceMem k_mean_buf(bsz.k_mean_bytes);
    ck_tile::DeviceMem k_prime_buf(bsz.k_prime_bytes);
    ck_tile::DeviceMem k_mean_partial_buf(bsz.k_mean_partial_bytes);
    ck_tile::DeviceMem counter_buf(bsz.counter_bytes);

    auto hargs = make_hargs<InputT>(s,
                                    sq_pad,
                                    sk_pad,
                                    num_q_tiles,
                                    num_k_tiles,
                                    q_dev.GetDeviceBuffer(),
                                    k_dev.GetDeviceBuffer(),
                                    v_dev.GetDeviceBuffer(),
                                    q_hat_dev.GetDeviceBuffer(),
                                    q_scale_dev.GetDeviceBuffer(),
                                    q_mean_dev.GetDeviceBuffer(),
                                    k_hat_dev.GetDeviceBuffer(),
                                    k_scale_dev.GetDeviceBuffer(),
                                    v_hat_dev.GetDeviceBuffer(),
                                    v_scale_dev.GetDeviceBuffer());

    SA3::run(hargs,
             static_cast<float*>(delta_s_dev.GetDeviceBuffer()),
             static_cast<InputT*>(k_mean_buf.GetDeviceBuffer()),
             static_cast<InputT*>(k_prime_buf.GetDeviceBuffer()),
             static_cast<float*>(k_mean_partial_buf.GetDeviceBuffer()),
             static_cast<int32_t*>(counter_buf.GetDeviceBuffer()),
             /*stream=*/nullptr);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // ---- Copy results ----
    std::vector<InputT> k_mean_gpu_raw(std::size_t(b) * h * hd);
    std::vector<InputT> q_mean_gpu_raw(std::size_t(b) * h * num_q_tiles * hd);
    k_mean_buf.FromDevice(k_mean_gpu_raw.data());
    q_mean_dev.FromDevice(q_mean_gpu_raw.data());

    std::vector<float> k_mean_gpu(k_mean_gpu_raw.size());
    std::vector<float> q_mean_gpu(q_mean_gpu_raw.size());
    for(std::size_t i = 0; i < k_mean_gpu_raw.size(); i++)
        k_mean_gpu[i] = ck_tile::type_convert<float>(k_mean_gpu_raw[i]);
    for(std::size_t i = 0; i < q_mean_gpu_raw.size(); i++)
        q_mean_gpu[i] = ck_tile::type_convert<float>(q_mean_gpu_raw[i]);

    std::vector<float> delta_s_gpu(std::size_t(b) * h * num_q_tiles * sk_pad);
    delta_s_dev.FromDevice(delta_s_gpu.data());

    std::vector<uint8_t> q_hat_gpu(std::size_t(b) * h * sq_pad * (hd / 2));
    std::vector<uint8_t> q_scale_gpu(std::size_t(b) * h * sq_pad * (hd / kG));
    std::vector<uint8_t> k_hat_gpu(std::size_t(b) * h * sk_pad * (hd / 2));
    std::vector<uint8_t> k_scale_gpu(std::size_t(b) * h * sk_pad * (hd / kG));
    std::vector<uint8_t> v_hat_gpu(std::size_t(b) * h * hd * (sk_pad / 2));
    std::vector<uint8_t> v_scale_gpu(std::size_t(b) * h * hd * (sk_pad / kG));
    q_hat_dev.FromDevice(q_hat_gpu.data());
    q_scale_dev.FromDevice(q_scale_gpu.data());
    k_hat_dev.FromDevice(k_hat_gpu.data());
    k_scale_dev.FromDevice(k_scale_gpu.data());
    v_hat_dev.FromDevice(v_hat_gpu.data());
    v_scale_dev.FromDevice(v_scale_gpu.data());

    const float mean_tol    = std::is_same_v<InputT, ck_tile::fp16_t> ? 2e-3f : 1e-4f;
    const float scl_atol    = std::is_same_v<InputT, ck_tile::fp16_t> ? 1.0f : 0.0f;
    const float delta_s_tol = 1e-2f * static_cast<float>(hd);

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

    bool all_pass = true;
    auto chk      = [&](bool ok, const char* name) {
        std::cout << "  " << std::left << std::setw(18) << name << (ok ? "  PASS" : "  FAIL")
                  << "\n";
        if(!ok)
            all_pass = false;
    };

    // k_mean / q_mean: compact layout, compare directly.
    chk(ck_tile::check_err(
            k_mean_gpu.data(), 1, k_mean_ref.data(), 1, k_mean_gpu.size(), "k_mean", 0.0, mean_tol),
        "k_mean");
    chk(ck_tile::check_err(
            q_mean_gpu.data(), 1, q_mean_ref.data(), 1, q_mean_gpu.size(), "q_mean", 0.0, mean_tol),
        "q_mean");

    // delta_s: GPU [b*h*num_q_tiles, sk_pad], valid: sk columns per row.
    chk(ck_tile::check_err(delta_s_gpu.data(),
                            sk_pad,
                            delta_s_ref.data(),
                            b * h * num_q_tiles,
                            sk,
                            "delta_s",
                            0.0,
                            delta_s_tol),
        "delta_s");

    // Q/K scale: GPU [b*h, seq_pad*(hd/kG)], valid: seq*(hd/kG) per outer row.
    chk(ck_tile::check_err(q_scale_gpu.data(),
                            sq_pad * (hd / kG),
                            q_scale_ref.data(),
                            b * h,
                            sq * (hd / kG),
                            "q_scale",
                            0.0,
                            scl_atol),
        "q_scale");
    chk(ck_tile::check_err(k_scale_gpu.data(),
                            sk_pad * (hd / kG),
                            k_scale_ref.data(),
                            b * h,
                            sk * (hd / kG),
                            "k_scale",
                            0.0,
                            scl_atol),
        "k_scale");

    // V scale: GPU [b*h*hd, sk_pad/kG], valid: sk/kG per outer row.
    chk(ck_tile::check_err(v_scale_gpu.data(),
                            sk_pad / kG,
                            v_scale_ref.data(),
                            b * h * hd,
                            sk / kG,
                            "v_scale",
                            0.0,
                            scl_atol),
        "v_scale");

    // Q hat dequant: compact hat+scale, dequantize, compare vs Q_smooth reference.
    {
        const auto q_hat_c   = compact_u8(q_hat_gpu,   b * h, sq_pad * (hd / 2),  sq * (hd / 2));
        const auto q_scale_c = compact_u8(q_scale_gpu, b * h, sq_pad * (hd / kG), sq * (hd / kG));
        std::vector<float> q_dequant(std::size_t(b) * h * sq * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            q_hat_c.data(), q_scale_c.data(), q_dequant.data(), b, h, sq, hd);
        std::vector<float> q_smooth(std::size_t(b) * h * sq * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int qi = 0; qi < num_q_tiles; qi++)
                {
                    const int rs = qi * kRows, re = std::min(rs + kRows, sq);
                    for(int n = rs; n < re; n++)
                        for(int d = 0; d < hd; d++)
                            q_smooth[bi * h * sq * hd + hi * sq * hd + n * hd + d] =
                                Q_f32[bi * h * sq * hd + hi * sq * hd + n * hd + d] -
                                q_mean_rt[bi * h * num_q_tiles * hd +
                                          hi * num_q_tiles * hd + qi * hd + d];
                }
        chk(ck_tile::check_err(q_dequant, q_smooth, "q_hat (dequant)", 0.0, 1.0), "q_hat (dequant)");
    }

    // K hat dequant.
    {
        const auto k_hat_c   = compact_u8(k_hat_gpu,   b * h, sk_pad * (hd / 2),  sk * (hd / 2));
        const auto k_scale_c = compact_u8(k_scale_gpu, b * h, sk_pad * (hd / kG), sk * (hd / kG));
        std::vector<float> k_dequant(std::size_t(b) * h * sk * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            k_hat_c.data(), k_scale_c.data(), k_dequant.data(), b, h, sk, hd);
        std::vector<float> k_smooth(std::size_t(b) * h * sk * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                    for(int d = 0; d < hd; d++)
                        k_smooth[bi * h * sk * hd + hi * sk * hd + n * hd + d] =
                            K_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d] -
                            k_mean_rt[bi * h * hd + hi * hd + d];
        chk(ck_tile::check_err(k_dequant, k_smooth, "k_hat (dequant)", 0.0, 1.0), "k_hat (dequant)");
    }

    // V hat dequant: GPU [b*h*hd, sk_pad/2], valid: sk/2 per outer row.
    {
        const auto v_hat_c   = compact_u8(v_hat_gpu,   b * h * hd, sk_pad / 2,  sk / 2);
        const auto v_scale_c = compact_u8(v_scale_gpu, b * h * hd, sk_pad / kG, sk / kG);
        std::vector<float> v_dequant(std::size_t(b) * h * hd * sk);
        ck_tile::reference::reference_dequant_mxfp4(
            v_hat_c.data(), v_scale_c.data(), v_dequant.data(), b, h, hd, sk);
        // v_dequant is [b,h,hd,sk]; build transposed ref from V_f32 [b,h,sk,hd].
        std::vector<float> v_ref_t(std::size_t(b) * h * hd * sk);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                    for(int n = 0; n < sk; n++)
                        v_ref_t[bi * h * hd * sk + hi * hd * sk + d * sk + n] =
                            V_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d];
        chk(ck_tile::check_err(v_dequant, v_ref_t, "v_hat (dequant)", 0.0, 1.0), "v_hat (dequant)");
    }

    return all_pass;
}

// ---------------------------------------------------------------------------
// Print benchmark result with per-kernel bandwidth breakdown
// ---------------------------------------------------------------------------

static void print_result(const RunShape& s,
                         const std::string& dtype,
                         const BenchResult& res,
                         bool csv_mode)
{
    const int b = s.batch, h = s.nhead, sq = s.seqlen_q, sk = s.seqlen_k, hd = s.hdim;
    constexpr int kG      = 32;
    constexpr int kM0     = 128;
    const int num_q_tiles = (sq + kM0 - 1) / kM0;

    const std::size_t elem_bytes = (dtype == "fp16") ? sizeof(ck_tile::fp16_t) : sizeof(float);

    const std::size_t bytes_kmean = std::size_t(b) * h * sk * hd * elem_bytes
                                    + std::size_t(b) * h * hd * elem_bytes;

    const std::size_t write_q_mean  = std::size_t(b) * h * num_q_tiles * hd * elem_bytes;
    const std::size_t write_k_prime = std::size_t(b) * h * sk * hd * elem_bytes;
    const std::size_t bytes_preprocess =
        std::size_t(b) * h * sq * hd * elem_bytes   // read Q
        + std::size_t(b) * h * sk * hd * elem_bytes // read K (second pass)
        + std::size_t(b) * h * hd * elem_bytes      // read k_mean
        + write_q_mean                               // write q_mean
        + std::size_t(b) * h * sq * (hd / 2)        // write q_hat
        + std::size_t(b) * h * sq * (hd / kG)       // write q_scale
        + write_k_prime                              // write K'
        + std::size_t(b) * h * sk * (hd / 2)        // write k_hat
        + std::size_t(b) * h * sk * (hd / kG);      // write k_scale

    const std::size_t bytes_v =
        std::size_t(b) * h * sk * hd * elem_bytes // read V
        + std::size_t(b) * h * hd * (sk / 2)      // write v_hat
        + std::size_t(b) * h * hd * (sk / kG);    // write v_scale

    const std::size_t bytes_delta_s =
        write_q_mean + write_k_prime
        + std::size_t(b) * h * num_q_tiles * sk * sizeof(float); // write delta_s

    const std::size_t bytes_total =
        bytes_kmean + bytes_preprocess + bytes_v + bytes_delta_s;

    auto gbs = [](std::size_t bytes, float ms) -> double {
        return static_cast<double>(bytes) / 1.0e6 / static_cast<double>(ms);
    };

    if(csv_mode)
    {
        auto row = [&](const char* kernel, float ms, std::size_t bytes) {
            std::cout << dtype << "," << b << "," << h << "," << sq << "," << sk << "," << hd
                      << "," << kernel << "," << std::fixed << std::setprecision(3) << ms << ","
                      << std::setprecision(1) << gbs(bytes, ms) << "\n";
        };
        row("kmean",        res.kmean_ms,        bytes_kmean);
        row("preprocess",   res.preprocess_ms,   bytes_preprocess);
        row("v_preprocess", res.v_preprocess_ms, bytes_v);
        row("delta_s_gemm", res.delta_s_ms,      bytes_delta_s);
        row("total",        res.total_ms,        bytes_total);
    }
    else
    {
        std::cout << "dtype=" << dtype << "  B=" << b << " H=" << h << " Sq=" << sq << " Sk=" << sk
                  << " D=" << hd << "\n";
        auto line = [&](const char* name, float ms, std::size_t bytes) {
            std::cout << "  " << std::left << std::setw(16) << name << std::right << std::fixed
                      << std::setprecision(3) << std::setw(8) << ms << " ms"
                      << "  " << std::setprecision(1) << std::setw(8) << gbs(bytes, ms) << " GB/s"
                      << "  (" << std::setprecision(0) << bytes / 1.0e6 << " MB)\n";
        };
        line("[0] KMean",        res.kmean_ms,        bytes_kmean);
        line("[1] Prep Q/K",     res.preprocess_ms,   bytes_preprocess);
        line("[1b] Prep V",      res.v_preprocess_ms, bytes_v);
        line("[2] delta_s GEMM", res.delta_s_ms,      bytes_delta_s);
        std::cout << "  " << std::string(50, '-') << "\n";
        line("Total",            res.total_ms,        bytes_total);
    }
}

// ---------------------------------------------------------------------------
// Dispatch: (dtype, hdim) -> template instantiation
// ---------------------------------------------------------------------------

static BenchResult dispatch_bench(const RunShape& s, const std::string& dtype)
{
    if(dtype == "fp16")
    {
        if(s.hdim == 128)
            return run_benchmark<ck_tile::fp16_t, 128, 128>(s);
        if(s.hdim == 256)
            return run_benchmark<ck_tile::fp16_t, 128, 256>(s);
    }
    else if(dtype == "fp32")
    {
        if(s.hdim == 128)
            return run_benchmark<float, 128, 128>(s);
        if(s.hdim == 256)
            return run_benchmark<float, 128, 256>(s);
    }
    throw std::runtime_error("Unsupported prec/hdim: " + dtype + " d=" + std::to_string(s.hdim));
}

static bool dispatch_verify(const RunShape& s, const std::string& dtype)
{
    if(dtype == "fp16")
    {
        if(s.hdim == 128)
            return run_verify<ck_tile::fp16_t, 128, 128>(s);
        if(s.hdim == 256)
            return run_verify<ck_tile::fp16_t, 128, 256>(s);
    }
    else if(dtype == "fp32")
    {
        if(s.hdim == 128)
            return run_verify<float, 128, 128>(s);
        if(s.hdim == 256)
            return run_verify<float, 128, 256>(s);
    }
    throw std::runtime_error("Unsupported prec/hdim: " + dtype + " d=" + std::to_string(s.hdim));
}

// ---------------------------------------------------------------------------
// Shape tables
// ---------------------------------------------------------------------------

static const std::vector<std::tuple<int, int, int, int, int>> kShapes = {
    // {batch, nhead, seqlen_q, seqlen_k, hdim}
    {1, 32, 1, 4096, 128},
    {1, 32, 1, 4096, 256},
    {1, 32, 1, 8192, 128},
    {1, 32, 1, 8192, 256},
    {1, 32, 1024, 1024, 128},
    {1, 32, 1024, 4096, 128},
    {1, 32, 4096, 4096, 128},
    {1, 32, 4096, 4096, 256},
    {4, 16, 1024, 4096, 128},
    {4, 16, 4096, 4096, 128},
    {8, 8, 2048, 2048, 128},
    {16, 32, 512, 4096, 128},
    {16, 32, 1024, 4096, 128},
};

static const std::vector<std::tuple<int, int, int, int, int>> kVerifyShapes = {
    {1, 1, 128, 128, 128},
    {1, 2, 256, 256, 128},
    {1, 1, 128, 128, 256},
    {1, 1, 65, 96, 128},
    {1, 1, 127, 96, 128},
    {2, 4, 300, 192, 128},
    {1, 1, 65, 96, 256},
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    try
    {
        auto [result, arg_parser] = create_args(argc, argv);
        if(!result)
            return -1;

        const std::string dtype = arg_parser.get_str("prec");
        const bool do_verify    = arg_parser.get_bool("verify");
        const bool csv_mode     = arg_parser.get_bool("csv");
        const int warmup        = arg_parser.get_int("warmup");
        const int repeat        = arg_parser.get_int("repeat");

        // Check if the user explicitly set shape parameters.
        bool explicit_shape = false;
        for(int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            if(s.rfind("-b=", 0) == 0 || s.rfind("-h=", 0) == 0 || s.rfind("-q=", 0) == 0 ||
               s.rfind("-k=", 0) == 0 || s.rfind("-d=", 0) == 0 || s.rfind("-prec=", 0) == 0)
                explicit_shape = true;
        }

        RunShape base{arg_parser.get_int("b"),
                      arg_parser.get_int("h"),
                      arg_parser.get_int("q"),
                      arg_parser.get_int("k"),
                      arg_parser.get_int("d"),
                      warmup,
                      repeat};

        if(do_verify)
        {
            bool suite_pass = true;

            auto run_one = [&](const RunShape& s, const std::string& dt) {
                std::cout << "verify  dtype=" << dt << "  B=" << s.batch << " H=" << s.nhead
                          << " Sq=" << s.seqlen_q << " Sk=" << s.seqlen_k << " D=" << s.hdim
                          << "\n";
                bool ok = dispatch_verify(s, dt);
                std::cout << "  --> " << (ok ? "PASS" : "FAIL") << "\n";
                if(!ok)
                    suite_pass = false;
            };

            if(explicit_shape)
            {
                run_one(base, dtype);
            }
            else
            {
                for(const std::string& dt : {"fp16", "fp32"})
                    for(auto [b, h, sq, sk, hd] : kVerifyShapes)
                        run_one(RunShape{b, h, sq, sk, hd, warmup, repeat}, dt);
            }

            return suite_pass ? 0 : 1;
        }

        // Benchmark mode
        if(csv_mode)
            std::cout << "dtype,batch,nhead,seqlen_q,seqlen_k,hdim,kernel,ms,GB_per_s\n";

        if(explicit_shape)
        {
            BenchResult res = dispatch_bench(base, dtype);
            print_result(base, dtype, res, csv_mode);
        }
        else
        {
            for(const std::string& dt : {"fp16", "fp32"})
            {
                for(auto [b, h, sq, sk, hd] : kShapes)
                {
                    RunShape s{b, h, sq, sk, hd, warmup, repeat};
                    BenchResult res = dispatch_bench(s, dt);
                    print_result(s, dt, res, csv_mode);
                    if(!csv_mode)
                        std::cout << "\n";
                }
            }
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return -2;
    }
}

#else // CK_USE_NATIVE_MX_SUPPORT not defined

#include <cstdio>
int main()
{
    puts("sageattn_v3_preprocess requires CK_USE_NATIVE_MX_SUPPORT (gfx950).");
    return 0;
}

#endif
