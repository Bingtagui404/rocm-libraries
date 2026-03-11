// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Benchmark / correctness verifier for sageattn_v3_preprocess_run() —
// the four-kernel SA3 preprocessing pipeline:
//   k_mean → preprocess Q/K → V tile-transpose → delta_s GEMM
//
// Usage:
//   ./bin/tile_example_sageattn_v3_preprocess [options]
//
//   -b <int>    batch size                   (default 4)
//   -h <int>    number of heads              (default 16)
//   -q <int>    seqlen_q                     (default 1024)
//   -k <int>    seqlen_k                     (default 4096)
//   -d <int>    hdim (128 or 256)            (default 128)
//   -t <str>    input type: fp16 | fp32      (default fp16)
//   -w <int>    warmup iterations            (default 5)
//   -r <int>    measurement iterations       (default 50)
//   --csv       print header+row in CSV format
//   --verify    run correctness check against CPU reference instead of bench
//
// HBM traffic counted (one-way, reads+writes):
//
//   Kernel 0 (KMean):
//     Read:  K  [B,H,Sk,D]                         -- first K pass
//     Write: k_mean [B,H,D]                         -- tiny
//
//   Kernel 1 (Preprocess Q/K):
//     Read:  Q  [B,H,Sq,D]
//            K  [B,H,Sk,D]                          -- second K pass
//            k_mean [B,H,D]                         -- tiny
//     Write: q_mean [B,H,T_q,D], q_hat [B,H,Sq,D/2], q_scale [B,H,Sq,D/G]
//            K'     [B,H,Sk,D],  k_hat [B,H,Sk,D/2], k_scale [B,H,Sk,D/G]
//
//   Kernel 1b (V Preprocess):
//     Read:  V  [B,H,Sk,D]
//     Write: v_hat [B,H,D,Sk/2], v_scale [B,H,D,Sk/G]
//
//   Kernel 2 (delta_s GEMM) -- excluded when --no-delta-s:
//     Read:  q_mean [B,H,T_q,D], K' [B,H,Sk,D]
//     Write: delta_s [B,H,T_q,Sk] (float32)
//
//   where G=32 (MXFP4 scale granularity), T_q = ceil(Sq/128).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CK_USE_NATIVE_MX_SUPPORT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/sageattn_v3_preprocess/sageattn_v3_preprocess.hpp"
#include "ck_tile/host/reference/reference_sageattn_v3_preprocess.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void hip_check(hipError_t err, const char* file, int line)
{
    if(err != hipSuccess)
    {
        std::ostringstream ss;
        ss << file << ":" << line << " HIP error: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}
#define HIP_CHECK(x) hip_check((x), __FILE__, __LINE__)

struct BenchArgs
{
    int batch         = 4;
    int nhead         = 16;
    int seqlen_q      = 1024;
    int seqlen_k      = 4096;
    int hdim          = 128;
    std::string dtype = "fp16";
    int warmup        = 5;
    int repeat        = 50;
    bool csv          = false;
    bool verify       = false;
    bool no_delta_s   = false; // exclude delta_s GEMM from BW accounting
};

static BenchArgs parse_args(int argc, char** argv)
{
    BenchArgs a;
    for(int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        if((s == "-b") && i + 1 < argc)
            a.batch = std::stoi(argv[++i]);
        else if((s == "-h") && i + 1 < argc)
            a.nhead = std::stoi(argv[++i]);
        else if((s == "-q") && i + 1 < argc)
            a.seqlen_q = std::stoi(argv[++i]);
        else if((s == "-k") && i + 1 < argc)
            a.seqlen_k = std::stoi(argv[++i]);
        else if((s == "-d") && i + 1 < argc)
            a.hdim = std::stoi(argv[++i]);
        else if((s == "-t") && i + 1 < argc)
            a.dtype = argv[++i];
        else if((s == "-w") && i + 1 < argc)
            a.warmup = std::stoi(argv[++i]);
        else if((s == "-r") && i + 1 < argc)
            a.repeat = std::stoi(argv[++i]);
        else if(s == "--csv")
            a.csv = true;
        else if(s == "--verify")
            a.verify = true;
        else if(s == "--no-delta-s")
            a.no_delta_s = true;
        else if(s == "--help" || s == "-?")
        {
            std::cout << "Usage: tile_example_sageattn_v3_preprocess [options]\n"
                         "  -b <batch>      default 4\n"
                         "  -h <nhead>      default 16\n"
                         "  -q <seqlen_q>   default 1024\n"
                         "  -k <seqlen_k>   default 4096\n"
                         "  -d <hdim>       default 128 (128 or 256)\n"
                         "  -t <dtype>      fp16 | fp32  (default fp16)\n"
                         "  -w <warmup>     default 5\n"
                         "  -r <repeat>     default 50\n"
                         "  --csv           CSV output\n"
                         "  --verify        correctness check vs CPU reference\n"
                         "  --no-delta-s    exclude delta_s GEMM from BW accounting\n";
            std::exit(0);
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Core benchmark: templated on InputT and kCols (hdim)
// ---------------------------------------------------------------------------

template <typename InputT, int kRows, int kCols>
float run_benchmark(const BenchArgs& a)
{
    const int b  = a.batch;
    const int h  = a.nhead;
    const int sq = a.seqlen_q;
    const int sk = a.seqlen_k;
    const int hd = a.hdim;

    constexpr int kG = 32; // MXFP4 scale granularity

    const int num_q_tiles = (sq + kRows - 1) / kRows;
    const int num_k_tiles = (sk + kRows - 1) / kRows;

    // ---- Allocate GPU buffers ----
    const std::size_t elem = sizeof(InputT);

    ck_tile::DeviceMem q_dev(std::size_t(b * h * sq * hd) * elem);
    ck_tile::DeviceMem k_dev(std::size_t(b * h * sk * hd) * elem);
    ck_tile::DeviceMem v_dev(std::size_t(b * h * sk * hd) * elem);

    ck_tile::DeviceMem q_hat_dev(std::size_t(b * h * sq * (hd / 2)));
    ck_tile::DeviceMem q_scale_dev(std::size_t(b * h * sq * (hd / kG)));
    ck_tile::DeviceMem q_mean_dev(std::size_t(b * h * num_q_tiles * hd) * elem);

    ck_tile::DeviceMem k_hat_dev(std::size_t(b * h * sk * (hd / 2)));
    ck_tile::DeviceMem k_scale_dev(std::size_t(b * h * sk * (hd / kG)));

    ck_tile::DeviceMem v_hat_dev(std::size_t(b * h * hd * (sk / 2)));
    ck_tile::DeviceMem v_scale_dev(std::size_t(b * h * hd * (sk / kG)));

    ck_tile::DeviceMem delta_s_dev(std::size_t(b * h * num_q_tiles * sk) * sizeof(float));

    ck_tile::DeviceMem k_mean_buf(std::size_t(b * h * hd) * elem);
    ck_tile::DeviceMem k_prime_buf(std::size_t(b * h * sk * hd) * elem);
    ck_tile::DeviceMem k_mean_partial_buf(std::size_t(b * h * hd) * sizeof(float));
    ck_tile::DeviceMem counter_buf(std::size_t(b * h) * sizeof(int32_t));

    // Initialise inputs with zeros (benchmark only; correctness tested separately).
    q_dev.SetZero();
    k_dev.SetZero();
    v_dev.SetZero();

    // ---- Build host args ----
    ck_tile::SageAttnV3PreprocessArgs<InputT> hargs{};

    hargs.q_ptr                = static_cast<const InputT*>(q_dev.GetDeviceBuffer());
    hargs.seqlen_q             = sq;
    hargs.hdim                 = hd;
    hargs.stride_q             = hd;
    hargs.nhead_stride_q       = sq * hd;
    hargs.batch_stride_q       = h * sq * hd;
    hargs.q_hat_ptr            = static_cast<uint8_t*>(q_hat_dev.GetDeviceBuffer());
    hargs.stride_q_hat         = hd / 2;
    hargs.nhead_stride_q_hat   = sq * (hd / 2);
    hargs.batch_stride_q_hat   = h * sq * (hd / 2);
    hargs.q_scale_ptr          = static_cast<uint8_t*>(q_scale_dev.GetDeviceBuffer());
    hargs.stride_q_scale       = hd / kG;
    hargs.nhead_stride_q_scale = sq * (hd / kG);
    hargs.batch_stride_q_scale = h * sq * (hd / kG);
    hargs.q_mean_ptr           = static_cast<InputT*>(q_mean_dev.GetDeviceBuffer());
    hargs.q_tile_size          = kRows;
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
    hargs.nhead_stride_k_hat   = sk * (hd / 2);
    hargs.batch_stride_k_hat   = h * sk * (hd / 2);
    hargs.k_scale_ptr          = static_cast<uint8_t*>(k_scale_dev.GetDeviceBuffer());
    hargs.stride_k_scale       = hd / kG;
    hargs.nhead_stride_k_scale = sk * (hd / kG);
    hargs.batch_stride_k_scale = h * sk * (hd / kG);

    hargs.v_ptr                = static_cast<const InputT*>(v_dev.GetDeviceBuffer());
    hargs.nhead_stride_v       = sk * hd;
    hargs.batch_stride_v       = h * sk * hd;
    hargs.v_hat_ptr            = static_cast<uint8_t*>(v_hat_dev.GetDeviceBuffer());
    hargs.stride_v_hat         = sk / 2;
    hargs.nhead_stride_v_hat   = hd * (sk / 2);
    hargs.batch_stride_v_hat   = h * hd * (sk / 2);
    hargs.v_scale_ptr          = static_cast<uint8_t*>(v_scale_dev.GetDeviceBuffer());
    hargs.stride_v_scale       = sk / kG;
    hargs.nhead_stride_v_scale = hd * (sk / kG);
    hargs.batch_stride_v_scale = h * hd * (sk / kG);

    hargs.batch       = b;
    hargs.nhead       = h;
    hargs.num_q_tiles = num_q_tiles;
    hargs.num_k_tiles = num_k_tiles;

    auto* k_mean_ptr    = static_cast<InputT*>(k_mean_buf.GetDeviceBuffer());
    auto* k_prime_ptr   = static_cast<InputT*>(k_prime_buf.GetDeviceBuffer());
    auto* k_partial_ptr = static_cast<float*>(k_mean_partial_buf.GetDeviceBuffer());
    auto* counter_ptr   = static_cast<int32_t*>(counter_buf.GetDeviceBuffer());
    auto* delta_s_ptr   = static_cast<float*>(delta_s_dev.GetDeviceBuffer());

    // ---- Timing setup ----
    hipStream_t stream = nullptr;
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    auto run_once = [&]() {
        ck_tile::sageattn_v3_preprocess_run<InputT, kRows, kCols>(
            hargs, delta_s_ptr, k_mean_ptr, k_prime_ptr, k_partial_ptr, counter_ptr, stream);
    };

    // Warmup
    for(int i = 0; i < a.warmup; ++i)
        run_once();
    HIP_CHECK(hipStreamSynchronize(stream));

    // Timed loop
    HIP_CHECK(hipEventRecord(ev_start, stream));
    for(int i = 0; i < a.repeat; ++i)
        run_once();
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev_start, ev_stop));

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return total_ms / static_cast<float>(a.repeat);
}

// ---------------------------------------------------------------------------
// Correctness verification: templated on InputT and kCols (hdim)
//
// Returns true if all checks pass, false otherwise.
// Prints a one-line summary per output tensor.
// ---------------------------------------------------------------------------

// Helper: max absolute error over a flat array
static float max_abs_err(const float* a, const float* b, std::size_t n)
{
    float err = 0.0f;
    for(std::size_t i = 0; i < n; i++)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

template <typename InputT, int kRows, int kCols>
bool run_verify(const BenchArgs& a)
{
    const int b      = a.batch;
    const int h      = a.nhead;
    const int sq     = a.seqlen_q;
    const int sk     = a.seqlen_k;
    const int hd     = a.hdim;
    constexpr int kG = 32;

    const auto bsz        = ck_tile::get_buffer_sizes<InputT, kRows>(b, h, sq, sk, hd);
    const int sq_pad      = static_cast<int>(bsz.seqlen_q_padded);
    const int sk_pad      = static_cast<int>(bsz.seqlen_k_padded);
    const int num_q_tiles = static_cast<int>(bsz.num_q_tiles);
    const int num_k_tiles = static_cast<int>(bsz.num_k_tiles);

    // ---- Generate random float inputs ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::size_t nq = std::size_t(b) * h * sq * hd;
    const std::size_t nk = std::size_t(b) * h * sk * hd;

    std::vector<float> Q_f32(nq), K_f32(nk), V_f32(nk);
    for(auto& x : Q_f32)
        x = dist(rng);
    for(auto& x : K_f32)
        x = dist(rng);
    for(auto& x : V_f32)
        x = dist(rng);

    // For fp16: round-trip values so CPU reference matches GPU arithmetic.
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
    {
        auto rt = [](std::vector<float>& v) {
            for(auto& x : v)
                x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));
        };
        rt(Q_f32);
        rt(K_f32);
        rt(V_f32);
    }

    // ---- CPU reference ----
    // k_mean (float precision)
    std::vector<float> k_mean_ref(std::size_t(b) * h * hd, 0.0f);
    ck_tile::reference::reference_sageattn_v3_k_smooth(
        K_f32.data(), k_mean_ref.data(), b, h, sk, hd);

    // Round-trip k_mean through InputT (GPU stores it as InputT)
    std::vector<float> k_mean_rt = k_mean_ref;
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
        for(auto& x : k_mean_rt)
            x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));

    // q_mean + Q hat/scale
    std::vector<float> q_mean_ref(std::size_t(b) * h * num_q_tiles * hd, 0.0f);
    std::vector<uint8_t> q_hat_ref(std::size_t(b) * h * sq * (hd / 2), 0);
    std::vector<uint8_t> q_scale_ref(std::size_t(b) * h * sq * (hd / kG), 0);
    ck_tile::reference::reference_sageattn_v3_q_preprocess(
        Q_f32.data(), q_mean_ref.data(), q_hat_ref.data(), q_scale_ref.data(), b, h, sq, hd, kRows);

    // Round-trip q_mean
    std::vector<float> q_mean_rt = q_mean_ref;
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
        for(auto& x : q_mean_rt)
            x = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp16_t>(x));

    // delta_s
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

    // K hat/scale
    std::vector<uint8_t> k_hat_ref(std::size_t(b) * h * sk * (hd / 2), 0);
    std::vector<uint8_t> k_scale_ref(std::size_t(b) * h * sk * (hd / kG), 0);
    ck_tile::reference::reference_sageattn_v3_k_preprocess(
        K_f32.data(), k_mean_rt.data(), k_hat_ref.data(), k_scale_ref.data(), b, h, sk, hd);

    // V hat/scale
    std::vector<uint8_t> v_hat_ref(std::size_t(b) * h * hd * (sk / 2), 0);
    std::vector<uint8_t> v_scale_ref(std::size_t(b) * h * hd * (sk / kG), 0);
    ck_tile::reference::reference_sageattn_v3_v_preprocess(
        V_f32.data(), v_hat_ref.data(), v_scale_ref.data(), b, h, sk, hd);

    // ---- GPU buffers ----
    const std::size_t elem = sizeof(InputT);

    ck_tile::DeviceMem q_dev(nq * elem);
    ck_tile::DeviceMem k_dev(nk * elem);
    ck_tile::DeviceMem v_dev(nk * elem);

    // Upload inputs
    if constexpr(std::is_same_v<InputT, ck_tile::fp16_t>)
    {
        std::vector<ck_tile::fp16_t> Q_h(nq), K_h(nk), V_h(nk);
        for(std::size_t i = 0; i < nq; i++)
            Q_h[i] = ck_tile::type_convert<ck_tile::fp16_t>(Q_f32[i]);
        for(std::size_t i = 0; i < nk; i++)
            K_h[i] = ck_tile::type_convert<ck_tile::fp16_t>(K_f32[i]);
        for(std::size_t i = 0; i < nk; i++)
            V_h[i] = ck_tile::type_convert<ck_tile::fp16_t>(V_f32[i]);
        q_dev.ToDevice(Q_h.data());
        k_dev.ToDevice(K_h.data());
        v_dev.ToDevice(V_h.data());
    }
    else
    {
        q_dev.ToDevice(Q_f32.data());
        k_dev.ToDevice(K_f32.data());
        v_dev.ToDevice(V_f32.data());
    }

    // Output buffers (padded)
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

    // ---- Build hargs ----
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
    hargs.q_tile_size          = kRows;
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
    hargs.batch                = b;
    hargs.nhead                = h;
    hargs.num_q_tiles          = num_q_tiles;
    hargs.num_k_tiles          = num_k_tiles;

    // ---- Launch ----
    ck_tile::sageattn_v3_preprocess_run<InputT, kRows, kCols>(
        hargs,
        static_cast<float*>(delta_s_dev.GetDeviceBuffer()),
        static_cast<InputT*>(k_mean_buf.GetDeviceBuffer()),
        static_cast<InputT*>(k_prime_buf.GetDeviceBuffer()),
        static_cast<float*>(k_mean_partial_buf.GetDeviceBuffer()),
        static_cast<int32_t*>(counter_buf.GetDeviceBuffer()),
        /*stream=*/nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    // ---- Copy results back ----
    // k_mean and q_mean are stored as InputT on GPU → convert to float
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

    // delta_s: GPU layout [B, H, num_q_tiles, sk_pad]; reference is [B, H, num_q_tiles, sk]
    std::vector<float> delta_s_gpu(std::size_t(b) * h * num_q_tiles * sk_pad);
    delta_s_dev.FromDevice(delta_s_gpu.data());

    // Quantized outputs (padded)
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

    // ---- Tolerances ----
    const float mean_tol    = std::is_same_v<InputT, ck_tile::fp16_t> ? 2e-3f : 1e-4f;
    const int max_scl_diff  = std::is_same_v<InputT, ck_tile::fp16_t> ? 1 : 0;
    const float delta_s_tol = 1e-2f * static_cast<float>(hd);

    bool all_pass = true;

    auto check = [&](const char* name, bool pass, float err, float tol) {
        std::cout << "  " << std::left << std::setw(18) << name << (pass ? "  PASS" : "  FAIL")
                  << "  max_err=" << std::scientific << std::setprecision(2) << err
                  << "  tol=" << tol << "\n";
        if(!pass)
            all_pass = false;
    };

    // --- k_mean ---
    {
        float err = max_abs_err(k_mean_gpu.data(), k_mean_ref.data(), k_mean_gpu.size());
        check("k_mean", err <= mean_tol, err, mean_tol);
    }

    // --- q_mean ---
    {
        float err = max_abs_err(q_mean_gpu.data(), q_mean_ref.data(), q_mean_gpu.size());
        check("q_mean", err <= mean_tol, err, mean_tol);
    }

    // --- delta_s (only valid columns, skip sk_pad padding) ---
    {
        float err = 0.0f;
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int qi = 0; qi < num_q_tiles; qi++)
                    for(int kj = 0; kj < sk; kj++)
                    {
                        const int gpu_off = bi * h * num_q_tiles * sk_pad +
                                            hi * num_q_tiles * sk_pad + qi * sk_pad + kj;
                        const int ref_off =
                            bi * h * num_q_tiles * sk + hi * num_q_tiles * sk + qi * sk + kj;
                        err = std::max(err, std::abs(delta_s_gpu[gpu_off] - delta_s_ref[ref_off]));
                    }
        check("delta_s", err <= delta_s_tol, err, delta_s_tol);
    }

    // --- Q scale (valid rows only) ---
    {
        int diff = 0;
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sq; n++)
                    for(int g = 0; g < hd / kG; g++)
                    {
                        const int gpu_off = bi * h * sq_pad * (hd / kG) + hi * sq_pad * (hd / kG) +
                                            n * (hd / kG) + g;
                        const int ref_off =
                            bi * h * sq * (hd / kG) + hi * sq * (hd / kG) + n * (hd / kG) + g;
                        diff = std::max(diff,
                                        std::abs(static_cast<int>(q_scale_gpu[gpu_off]) -
                                                 static_cast<int>(q_scale_ref[ref_off])));
                    }
        std::cout << "  " << std::left << std::setw(18) << "q_scale"
                  << (diff <= max_scl_diff ? "  PASS" : "  FAIL") << "  max_diff=" << diff
                  << "  tol=" << max_scl_diff << "\n";
        if(diff > max_scl_diff)
            all_pass = false;
    }

    // --- K scale (valid rows only) ---
    {
        int diff = 0;
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                    for(int g = 0; g < hd / kG; g++)
                    {
                        const int gpu_off = bi * h * sk_pad * (hd / kG) + hi * sk_pad * (hd / kG) +
                                            n * (hd / kG) + g;
                        const int ref_off =
                            bi * h * sk * (hd / kG) + hi * sk * (hd / kG) + n * (hd / kG) + g;
                        diff = std::max(diff,
                                        std::abs(static_cast<int>(k_scale_gpu[gpu_off]) -
                                                 static_cast<int>(k_scale_ref[ref_off])));
                    }
        std::cout << "  " << std::left << std::setw(18) << "k_scale"
                  << (diff <= max_scl_diff ? "  PASS" : "  FAIL") << "  max_diff=" << diff
                  << "  tol=" << max_scl_diff << "\n";
        if(diff > max_scl_diff)
            all_pass = false;
    }

    // --- V scale (valid groups only) ---
    {
        int diff = 0;
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                    for(int g = 0; g < sk / kG; g++)
                    {
                        const int gpu_off = bi * h * hd * (sk_pad / kG) + hi * hd * (sk_pad / kG) +
                                            d * (sk_pad / kG) + g;
                        const int ref_off =
                            bi * h * hd * (sk / kG) + hi * hd * (sk / kG) + d * (sk / kG) + g;
                        diff = std::max(diff,
                                        std::abs(static_cast<int>(v_scale_gpu[gpu_off]) -
                                                 static_cast<int>(v_scale_ref[ref_off])));
                    }
        std::cout << "  " << std::left << std::setw(18) << "v_scale"
                  << (diff <= max_scl_diff ? "  PASS" : "  FAIL") << "  max_diff=" << diff
                  << "  tol=" << max_scl_diff << "\n";
        if(diff > max_scl_diff)
            all_pass = false;
    }

    // --- Q hat dequant (valid rows only) ---
    {
        // Compact valid rows from padded buffer
        std::vector<uint8_t> q_hat_valid(std::size_t(b) * h * sq * (hd / 2));
        std::vector<uint8_t> q_scale_valid(std::size_t(b) * h * sq * (hd / kG));
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sq; n++)
                {
                    const int sg =
                        bi * h * sq_pad * (hd / 2) + hi * sq_pad * (hd / 2) + n * (hd / 2);
                    const int dg = bi * h * sq * (hd / 2) + hi * sq * (hd / 2) + n * (hd / 2);
                    std::copy(q_hat_gpu.begin() + sg,
                              q_hat_gpu.begin() + sg + (hd / 2),
                              q_hat_valid.begin() + dg);
                    const int ss =
                        bi * h * sq_pad * (hd / kG) + hi * sq_pad * (hd / kG) + n * (hd / kG);
                    const int ds = bi * h * sq * (hd / kG) + hi * sq * (hd / kG) + n * (hd / kG);
                    std::copy(q_scale_gpu.begin() + ss,
                              q_scale_gpu.begin() + ss + (hd / kG),
                              q_scale_valid.begin() + ds);
                }

        std::vector<float> q_dequant(std::size_t(b) * h * sq * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            q_hat_valid.data(), q_scale_valid.data(), q_dequant.data(), b, h, sq, hd);

        // Build CPU Q_smooth reference
        std::vector<float> q_smooth_ref(std::size_t(b) * h * sq * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int qi = 0; qi < num_q_tiles; qi++)
                {
                    const int rs = qi * kRows;
                    const int re = std::min(rs + kRows, sq);
                    for(int n = rs; n < re; n++)
                        for(int d = 0; d < hd; d++)
                        {
                            const float qv = Q_f32[bi * h * sq * hd + hi * sq * hd + n * hd + d];
                            const float qm = q_mean_rt[bi * h * num_q_tiles * hd +
                                                       hi * num_q_tiles * hd + qi * hd + d];
                            q_smooth_ref[bi * h * sq * hd + hi * sq * hd + n * hd + d] = qv - qm;
                        }
                }

        float err = max_abs_err(q_dequant.data(), q_smooth_ref.data(), q_dequant.size());
        check("q_hat (dequant)", err <= 1.0f, err, 1.0f);
    }

    // --- K hat dequant (valid rows only) ---
    {
        std::vector<uint8_t> k_hat_valid(std::size_t(b) * h * sk * (hd / 2));
        std::vector<uint8_t> k_scale_valid(std::size_t(b) * h * sk * (hd / kG));
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                {
                    const int sg =
                        bi * h * sk_pad * (hd / 2) + hi * sk_pad * (hd / 2) + n * (hd / 2);
                    const int dg = bi * h * sk * (hd / 2) + hi * sk * (hd / 2) + n * (hd / 2);
                    std::copy(k_hat_gpu.begin() + sg,
                              k_hat_gpu.begin() + sg + (hd / 2),
                              k_hat_valid.begin() + dg);
                    const int ss =
                        bi * h * sk_pad * (hd / kG) + hi * sk_pad * (hd / kG) + n * (hd / kG);
                    const int ds = bi * h * sk * (hd / kG) + hi * sk * (hd / kG) + n * (hd / kG);
                    std::copy(k_scale_gpu.begin() + ss,
                              k_scale_gpu.begin() + ss + (hd / kG),
                              k_scale_valid.begin() + ds);
                }

        std::vector<float> k_dequant(std::size_t(b) * h * sk * hd);
        ck_tile::reference::reference_dequant_mxfp4(
            k_hat_valid.data(), k_scale_valid.data(), k_dequant.data(), b, h, sk, hd);

        std::vector<float> k_smooth_ref(std::size_t(b) * h * sk * hd);
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int n = 0; n < sk; n++)
                    for(int d = 0; d < hd; d++)
                    {
                        const float kv = K_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d];
                        const float km = k_mean_rt[bi * h * hd + hi * hd + d];
                        k_smooth_ref[bi * h * sk * hd + hi * sk * hd + n * hd + d] = kv - km;
                    }

        float err = max_abs_err(k_dequant.data(), k_smooth_ref.data(), k_dequant.size());
        check("k_hat (dequant)", err <= 1.0f, err, 1.0f);
    }

    // --- V hat dequant (valid seqlen groups only) ---
    {
        // V GPU layout: [B, H, hdim, sk_pad/2] / [B, H, hdim, sk_pad/kG]
        // V ref layout: [B, H, hdim, sk/2]     / [B, H, hdim, sk/kG]
        std::vector<uint8_t> v_hat_valid(std::size_t(b) * h * hd * (sk / 2));
        std::vector<uint8_t> v_scale_valid(std::size_t(b) * h * hd * (sk / kG));
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                {
                    const int sg =
                        bi * h * hd * (sk_pad / 2) + hi * hd * (sk_pad / 2) + d * (sk_pad / 2);
                    const int dg = bi * h * hd * (sk / 2) + hi * hd * (sk / 2) + d * (sk / 2);
                    std::copy(v_hat_gpu.begin() + sg,
                              v_hat_gpu.begin() + sg + (sk / 2),
                              v_hat_valid.begin() + dg);
                    const int ss =
                        bi * h * hd * (sk_pad / kG) + hi * hd * (sk_pad / kG) + d * (sk_pad / kG);
                    const int ds = bi * h * hd * (sk / kG) + hi * hd * (sk / kG) + d * (sk / kG);
                    std::copy(v_scale_gpu.begin() + ss,
                              v_scale_gpu.begin() + ss + (sk / kG),
                              v_scale_valid.begin() + ds);
                }

        // Dequant in transposed layout [B, H, hdim, sk]
        std::vector<float> v_dequant(std::size_t(b) * h * hd * sk);
        ck_tile::reference::reference_dequant_mxfp4(
            v_hat_valid.data(), v_scale_valid.data(), v_dequant.data(), b, h, hd, sk);

        // V ref: V_f32[b, h, n, d] vs v_dequant[b, h, d, n]
        float err = 0.0f;
        for(int bi = 0; bi < b; bi++)
            for(int hi = 0; hi < h; hi++)
                for(int d = 0; d < hd; d++)
                    for(int n = 0; n < sk; n++)
                    {
                        const float vref = V_f32[bi * h * sk * hd + hi * sk * hd + n * hd + d];
                        const float vgpu = v_dequant[bi * h * hd * sk + hi * hd * sk + d * sk + n];
                        err              = std::max(err, std::abs(vgpu - vref));
                    }
        check("v_hat (dequant)", err <= 1.0f, err, 1.0f);
    }

    return all_pass;
}

// ---------------------------------------------------------------------------
// Print one benchmark result row
// ---------------------------------------------------------------------------

static void print_result(const BenchArgs& a, float ave_ms, bool csv_mode)
{
    const int b           = a.batch;
    const int h           = a.nhead;
    const int sq          = a.seqlen_q;
    const int sk          = a.seqlen_k;
    const int hd          = a.hdim;
    constexpr int kG      = 32;
    constexpr int kM0     = 128;
    const int num_q_tiles = (sq + kM0 - 1) / kM0;

    const std::size_t elem_bytes = (a.dtype == "fp16") ? sizeof(ck_tile::fp16_t) : sizeof(float);

    // ---- HBM bytes ----
    // Kernel 0: KMean -- reads K once, writes k_mean (tiny)
    const std::size_t read_K_0      = std::size_t(b) * h * sk * hd * elem_bytes;
    const std::size_t write_K_mean  = std::size_t(b) * h * hd * elem_bytes; // tiny

    // Kernel 1: Preprocess Q/K -- reads Q, K again, k_mean (tiny)
    const std::size_t read_Q        = std::size_t(b) * h * sq * hd * elem_bytes;
    const std::size_t read_K_1      = std::size_t(b) * h * sk * hd * elem_bytes;
    const std::size_t write_Q_hat   = std::size_t(b) * h * sq * (hd / 2);
    const std::size_t write_Q_scale = std::size_t(b) * h * sq * (hd / kG);
    const std::size_t write_Q_mean  = std::size_t(b) * h * num_q_tiles * hd * elem_bytes;
    const std::size_t write_K_hat   = std::size_t(b) * h * sk * (hd / 2);
    const std::size_t write_K_scale = std::size_t(b) * h * sk * (hd / kG);
    const std::size_t write_K_prime = std::size_t(b) * h * sk * hd * elem_bytes;

    // Kernel 1b: V Preprocess
    const std::size_t read_V        = std::size_t(b) * h * sk * hd * elem_bytes;
    const std::size_t write_V_hat   = std::size_t(b) * h * hd * (sk / 2);
    const std::size_t write_V_scale = std::size_t(b) * h * hd * (sk / kG);

    // Kernel 2: delta_s GEMM -- reads q_mean and K' back from HBM, writes delta_s
    const std::size_t read_Q_mean_gemm  = write_Q_mean;
    const std::size_t read_K_prime_gemm = write_K_prime;
    const std::size_t write_delta_s     = std::size_t(b) * h * num_q_tiles * sk * sizeof(float);

    // Preprocess-only total (memory-bound portion, no GEMM reads/write)
    const std::size_t total_preprocess = read_K_0 + write_K_mean + read_Q + read_K_1 +
                                         write_Q_hat + write_Q_scale + write_Q_mean +
                                         write_K_hat + write_K_scale + write_K_prime +
                                         read_V + write_V_hat + write_V_scale;

    // Full total including delta_s GEMM traffic
    const std::size_t total_bytes =
        a.no_delta_s ? total_preprocess
                     : (total_preprocess + read_Q_mean_gemm + read_K_prime_gemm + write_delta_s);

    const double gb_per_s = static_cast<double>(total_bytes) / 1.0e6 / ave_ms;

    if(csv_mode)
    {
        std::cout << a.dtype << "," << b << "," << h << "," << sq << "," << sk << "," << hd << ","
                  << (a.no_delta_s ? 1 : 0) << "," << std::fixed << std::setprecision(3) << ave_ms
                  << "," << std::setprecision(1) << gb_per_s << "\n";
    }
    else
    {
        std::cout << "dtype=" << a.dtype << "  B=" << b << " H=" << h << " Sq=" << sq
                  << " Sk=" << sk << " D=" << hd
                  << (a.no_delta_s ? "  [no-delta-s]" : "") << "  |  " << std::fixed
                  << std::setprecision(3) << ave_ms << " ms"
                  << "  " << std::setprecision(1) << gb_per_s << " GB/s"
                  << "  (total HBM " << std::setprecision(0) << total_bytes / 1.0e6 << " MB)\n";
    }
}

// ---------------------------------------------------------------------------
// Dispatch: (dtype, hdim) → template instantiation
// ---------------------------------------------------------------------------

static float dispatch_bench(const BenchArgs& a)
{
    if(a.dtype == "fp16")
    {
        if(a.hdim == 128)
            return run_benchmark<ck_tile::fp16_t, 128, 128>(a);
        if(a.hdim == 256)
            return run_benchmark<ck_tile::fp16_t, 128, 256>(a);
    }
    else if(a.dtype == "fp32")
    {
        if(a.hdim == 128)
            return run_benchmark<float, 128, 128>(a);
        if(a.hdim == 256)
            return run_benchmark<float, 128, 256>(a);
    }
    throw std::runtime_error("Unsupported dtype/hdim: " + a.dtype + " d=" + std::to_string(a.hdim));
}

static bool dispatch_verify(const BenchArgs& a)
{
    if(a.dtype == "fp16")
    {
        if(a.hdim == 128)
            return run_verify<ck_tile::fp16_t, 128, 128>(a);
        if(a.hdim == 256)
            return run_verify<ck_tile::fp16_t, 128, 256>(a);
    }
    else if(a.dtype == "fp32")
    {
        if(a.hdim == 128)
            return run_verify<float, 128, 128>(a);
        if(a.hdim == 256)
            return run_verify<float, 128, 256>(a);
    }
    throw std::runtime_error("Unsupported dtype/hdim: " + a.dtype + " d=" + std::to_string(a.hdim));
}

// ---------------------------------------------------------------------------
// Benchmark suite: a representative set of large-batch shapes
// ---------------------------------------------------------------------------

static const std::vector<std::tuple<int, int, int, int, int>> kShapes = {
    // {batch, nhead, seqlen_q, seqlen_k, hdim}
    // Realistic LLM decoding: small seqlen_q, large seqlen_k
    {1, 32, 1, 4096, 128},
    {1, 32, 1, 4096, 256},
    {1, 32, 1, 8192, 128},
    {1, 32, 1, 8192, 256},
    // Prefill: large seqlen_q and seqlen_k
    {1, 32, 1024, 1024, 128},
    {1, 32, 1024, 4096, 128},
    {1, 32, 4096, 4096, 128},
    {1, 32, 4096, 4096, 256},
    // Multi-batch
    {4, 16, 1024, 4096, 128},
    {4, 16, 4096, 4096, 128},
    {8, 8, 2048, 2048, 128},
    {16, 32, 512, 4096, 128},
    {16, 32, 1024, 4096, 128},
};

// Verify shapes: smaller to keep runtime manageable
static const std::vector<std::tuple<int, int, int, int, int>> kVerifyShapes = {
    // aligned (seqlen multiples of kRows=128)
    {1, 1, 128, 128, 128},
    {1, 2, 256, 256, 128},
    {1, 1, 128, 128, 256},
    // non-aligned (tail padding exercised)
    {1, 1, 65, 96, 128},
    {1, 1, 127, 96, 128},
    {2, 4, 300, 192, 128},
    {1, 1, 65, 96, 256},
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    BenchArgs a = parse_args(argc, argv);

    // Detect whether shape flags were given explicitly
    bool explicit_shape = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        if(s == "-b" || s == "-h" || s == "-q" || s == "-k" || s == "-d" || s == "-t")
            explicit_shape = true;
    }

    // ------------------------------------------------------------------ //
    // --verify mode
    // ------------------------------------------------------------------ //
    if(a.verify)
    {
        bool suite_pass = true;

        auto run_one_verify = [&](const BenchArgs& cur) {
            std::cout << "verify  dtype=" << cur.dtype << "  B=" << cur.batch << " H=" << cur.nhead
                      << " Sq=" << cur.seqlen_q << " Sk=" << cur.seqlen_k << " D=" << cur.hdim
                      << "\n";
            bool ok = dispatch_verify(cur);
            std::cout << "  --> " << (ok ? "PASS" : "FAIL") << "\n";
            if(!ok)
                suite_pass = false;
        };

        if(explicit_shape)
        {
            run_one_verify(a);
        }
        else
        {
            // Run built-in verify suite for both dtypes
            for(const std::string& dtype : {"fp16", "fp32"})
            {
                for(auto [b, h, sq, sk, hd] : kVerifyShapes)
                {
                    BenchArgs cur = a;
                    cur.batch     = b;
                    cur.nhead     = h;
                    cur.seqlen_q  = sq;
                    cur.seqlen_k  = sk;
                    cur.hdim      = hd;
                    cur.dtype     = dtype;
                    run_one_verify(cur);
                }
            }
        }

        return suite_pass ? 0 : 1;
    }

    // ------------------------------------------------------------------ //
    // Benchmark mode
    // ------------------------------------------------------------------ //
    if(a.csv)
        std::cout << "dtype,batch,nhead,seqlen_q,seqlen_k,hdim,no_delta_s,ms,GB_per_s\n";

    if(explicit_shape)
    {
        float ms = dispatch_bench(a);
        print_result(a, ms, a.csv);
    }
    else
    {
        for(const std::string& dtype : {"fp16", "fp32"})
        {
            for(auto [b, h, sq, sk, hd] : kShapes)
            {
                BenchArgs cur = a;
                cur.batch     = b;
                cur.nhead     = h;
                cur.seqlen_q  = sq;
                cur.seqlen_k  = sk;
                cur.hdim      = hd;
                cur.dtype     = dtype;
                float ms      = dispatch_bench(cur);
                print_result(cur, ms, a.csv);
            }
        }
    }

    return 0;
}

#else // CK_USE_NATIVE_MX_SUPPORT not defined

#include <cstdio>
int main()
{
    puts("sageattn_v3_preprocess requires CK_USE_NATIVE_MX_SUPPORT (gfx950).");
    return 0;
}

#endif
