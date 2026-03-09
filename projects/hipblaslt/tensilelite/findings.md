# HardwareMonitor collectOnce() Consolidation: Findings

## Problem

Profiling the TensileLite benchmarking client with small GEMM problems (M=255, N=255, K=15) revealed that `HardwareMonitor::collectOnce()` dominated benchmark runtime. The hardware monitor wait (`benchmark_runs.validate_enqueues.hwmonitor`) consumed 28.3% of wall time — far exceeding actual GPU kernel execution (1.6%).

Each `collectOnce()` invocation made 5 separate AMD SMI calls per iteration:

1. `amdsmi_get_temp_metric()` — edge temperature
2. `amdsmi_get_gpu_metrics_info()` — per-XCD GFX clocks (SYS)
3. `amdsmi_get_clk_freq(SOC)` — SOC clock
4. `amdsmi_get_clk_freq(MEM)` — memory clock
5. `amdsmi_get_gpu_fan_rpms()` — fan speed

Plus a one-time `amdsmi_get_clk_freq(SYS)` for max supported frequency (with up to 10 retries).

The per-iteration cost averaged 0.57ms — 19x longer than the average kernel execution (0.03ms). Over 3,366 benchmark runs this accumulated to 1,929ms.

## Why this matters

The hardware monitor runs in a background thread that collects data while GPU kernels execute. For large kernels (e.g., M=8000, N=8000, K=1000 at ~1.16ms/kernel), the monitoring cost is fully hidden — the background thread finishes before `wait()` is called, and `hwmonitor.wait` drops to ~0.001ms. But for small kernels, the GPU finishes in ~0.03ms while the monitor needs ~0.57ms for a single `collectOnce()`. The `wait()` call blocks for the remaining ~0.54ms, serializing the monitoring overhead onto the benchmark's critical path.

## Approach: Consolidate to a single AMD SMI call

The `amdsmi_gpu_metrics_t` struct (returned by `amdsmi_get_gpu_metrics_info`) already contains all the data fetched by the other 4 calls:

| Metric | Individual call | `gpu_metrics_t` field |
|--------|----------------|----------------------|
| Edge temp | `amdsmi_get_temp_metric()` | `temperature_edge` (°C) |
| SOC clock | `amdsmi_get_clk_freq(SOC)` | `current_socclks[0]` (MHz) |
| MEM clock | `amdsmi_get_clk_freq(MEM)` | `current_uclk` (MHz) |
| Fan speed | `amdsmi_get_gpu_fan_rpms()` | `current_fan_speed` (RPM) |

The code already called `amdsmi_get_gpu_metrics_info` for the SYS clock path. The other 4 calls were redundant.

### Changes made

**`client/include/HardwareMonitor.hpp`**: Added `m_gpuMetricsFailed` flag for consolidated error tracking.

**`client/src/HardwareMonitor.cpp`**:
- Rewrote `collectOnce()` for `AMDSMI_LIB_VERSION_MAJOR >= 25`: a single `amdsmi_get_gpu_metrics_info` call extracts all metrics, with unit conversions to match existing accumulator formats (temp × 1000 for millidegrees, clocks × 1e6 for Hz).
- Pre-v25 path left unchanged (no `gpu_metrics_info` available).
- Fallback to individual calls for unrecognized sensor/clock types (forward compatibility).
- `clearValues()` resets `m_gpuMetricsFailed` so each collection session starts fresh.
- On `amdsmi_get_gpu_metrics_info` failure, all metrics are permanently marked failed (matching existing per-metric error behavior).

## Results

Tested with M=255, N=255, K=15 (small kernels, 3,366 benchmark runs):

| Metric | Pre | Post | Delta |
|--------|-----|------|-------|
| hwmonitor total | 1,929ms | 1,795ms | -134ms (-6.9%) |
| hwmonitor mean | 0.57ms | 0.53ms | -0.04ms (-7.0%) |
| benchmark_runs total | 3,250ms | 2,630ms | -620ms (-19.1%) |
| Wall clock | 6,812ms | 6,488ms | -324ms (-4.8%) |

## Analysis

The improvement was ~7%, not the ~80% predicted. This reveals that:

1. **`amdsmi_get_gpu_metrics_info` is the dominant cost.** This single call reads the entire GPU metrics table from firmware via ioctl. It was already responsible for the bulk of the 0.57ms per-iteration cost.

2. **The eliminated calls were cheap.** `amdsmi_get_temp_metric`, `amdsmi_get_clk_freq`, and `amdsmi_get_gpu_fan_rpms` likely read from cached sysfs values rather than hitting firmware, contributing only ~0.04ms combined.

3. **The consolidation is still a net positive**: fewer syscalls, cleaner code, and a measurable 7% reduction in the hottest path. But the fundamental bottleneck is the firmware metrics read itself.

## Further optimization opportunities

To achieve a larger reduction, future work would need to address the `amdsmi_get_gpu_metrics_info` call cost directly:

- **Reduce collection frequency**: skip `collectOnce()` calls when kernel duration is below a threshold, or increase `minPeriod`.
- **Disable monitoring for small kernels**: add a mode that skips hardware monitoring when kernels are too short for meaningful samples (`--hardware-monitor=false` already exists but is all-or-nothing).
- **Sample less often**: instead of collecting every iteration, collect every N-th iteration and extrapolate.
