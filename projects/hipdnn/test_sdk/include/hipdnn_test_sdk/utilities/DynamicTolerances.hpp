// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>

namespace hipdnn_test_sdk::utilities::conv
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Calculates the expected tolerance for Convolution Backward Weights (WrW) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of weight gradients. It considers the accumulation of products of inputs and output
 * gradients over the batch and spatial dimensions.
 *
 * The tolerance is calculated by simulating the accumulation process using `ComputeType` precision
 * and adding the precision loss from casting the final result to `OutputType`.
 *
 * Error Estimation Strategy:
 * - High Precision (FP32, FP64): Uses a linear worst-case bound (Classical).
 *   Error ≈ n * epsilon * maxProduct
 * - Lower Precision (FP16, BF16): Uses a statistical/probabilistic bound.
 *   Error ≈ k * sqrt(n) * epsilon * maxProduct
 *
 * It also accounts for precision loss if inputs are cast to a lower precision ComputeType.
 *
 * @tparam OutputType The data type of the output (weight gradients).
 * @tparam InputType The data type of the input tensor.
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param inputMin The minimum value in the input tensor.
 * @param inputMax The maximum value in the input tensor.
 * @param dyMin The minimum value in the output gradient tensor.
 * @param dyMax The maximum value in the output gradient tensor.
 * @param dyDims The dimensions of the output gradient tensor (dy).
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateConvWrwTolerance(double inputMin,
                                double inputMax,
                                double dyMin,
                                double dyMax,
                                const std::vector<int64_t>& dyDims)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // dyDims: [N, K, Spatial...]
    // Accumulation for weights (dw) happens over N and Spatial dimensions.
    // dw[k, c, r, s] = sum_{n, h, w} dy[n, k, h, w] * x[n, c, h+r, w+s]

    if(dyDims.empty() || dyDims.size() < 2)
    {
        throw std::invalid_argument("dyDims must have at least 2 dimensions (N, K).");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(dyDims[0]); // Batch size N
    for(size_t i = 2; i < dyDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(dyDims[i]); // Spatial dimensions
    }

    double const maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    double const maxAbsDy = std::max(std::abs(dyMin), std::abs(dyMax));

    // Worst case product magnitude
    double const maxProduct = maxAbsInput * maxAbsDy;

    // Bound on sum(|x_i * y_i|)
    double const sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|x_i * y_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        double const nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        double const gamma = nU / (1.0 - nU);
        accumulatedTolerance = gamma * sumAbsProductBound;
    }
    else
    {
        // Lower Precision (FP16, BF16): Statistical bound (Probabilistic)
        // Error <= gamma_2n * sum(|x_i * y_i|)
        // gamma_n = sqrt(n) * u
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = sqrt(2n) * u
        // k_sigma = 6.0 for high confidence
        constexpr double K_SIGMA = 6.0;
        double const gamma
            = K_SIGMA * std::sqrt(2.0 * static_cast<double>(numberOfAccumulations)) * epsilon;
        accumulatedTolerance = gamma * sumAbsProductBound;
    }

    // Calculate input casting error
    // If InputType has higher precision (smaller epsilon) than ComputeType, we lose precision on load (downcasting).
    // Example: double -> float.
    // If InputType has lower precision (larger epsilon) than ComputeType, we preserve precision (upcasting).
    // Example: half -> float.
    // We only need to add tolerance if we are downcasting.
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    if(inputEpsilon < epsilon)
    {
        // Input precision is higher than compute precision, so we have casting error.
        // We add this to the tolerance.
        // Note: This is a worst-case bound.
        //
        // Derivation:
        // Let x_approx = x_true * (1 + d_x) and dy_approx = dy_true * (1 + d_dy)
        // where |d_x|, |d_dy| <= epsilon_compute (relative error bound).
        // Product P_approx = x_approx * dy_approx ≈ x_true * dy_true * (1 + d_x + d_dy)
        // Error_P = |P_approx - P_true| ≈ |P_true| * |d_x + d_dy|
        // Error_P <= |P_true| * (|d_x| + |d_dy|) <= |P_true| * (epsilon + epsilon)
        // Error_P <= 2 * |P_true| * epsilon
        // Summing over N accumulations: Total_Error <= 2 * sum(|x_i * y_i|) * epsilon
        double const castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    double const maxPossibleOutputValue = sumAbsProductBound;

    double castTolerance = 0.0;
    // Calculate precision loss due to casting from ComputeType to OutputType.
    // If OutputType has lower precision (larger epsilon) than ComputeType, we lose precision (downcasting).
    // Example: float -> half.
    // If OutputType has higher precision (smaller epsilon) than ComputeType, the value is exactly representable (upcasting).
    // Example: float -> double.
    // We only need to add tolerance if we are downcasting.
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    if(outputEpsilon > epsilon)
    {
        // The error is bounded by the precision of the OutputType at the final value.
        castTolerance = std::abs(maxPossibleOutputValue) * outputEpsilon;
    }

    // Total tolerance is the sum of accumulation error and cast error
    double const totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

/**
 * @brief Calculates the expected tolerance for Convolution Backward Data (DGrad) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of input gradients. It considers the accumulation of products of output gradients
 * and filter weights over the output channels and spatial filter dimensions.
 *
 * The tolerance is calculated by simulating the accumulation process using `ComputeType` precision
 * and adding the precision loss from casting the final result to `OutputType`.
 *
 * Error Estimation Strategy:
 * - High Precision (FP32, FP64): Uses a linear worst-case bound (Classical).
 *   Error ≈ n * epsilon * maxProduct
 * - Lower Precision (FP16, BF16): Uses a statistical/probabilistic bound.
 *   Error ≈ k * sqrt(n) * epsilon * maxProduct
 *
 * It also accounts for precision loss if inputs are cast to a lower precision ComputeType.
 *
 * @tparam OutputType The data type of the output (input gradients dx).
 * @tparam InputType The data type of the input tensors (dy and w).
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param dyMin The minimum value in the output gradient tensor (dy).
 * @param dyMax The maximum value in the output gradient tensor (dy).
 * @param wMin The minimum value in the filter weights tensor (w).
 * @param wMax The maximum value in the filter weights tensor (w).
 * @param wDims The dimensions of the filter weights tensor (w): [K, C, R, S] or [K, C, D, R, S].
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateConvDgradTolerance(
    double dyMin, double dyMax, double wMin, double wMax, const std::vector<int64_t>& wDims)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // wDims: [K, C, Spatial...] for 2D: [K, C, R, S], for 3D: [K, C, D, R, S]
    // Accumulation for input gradients (dx) happens over K (output channels) and Spatial filter dimensions.
    // dx[n, c, h, w] = sum_{k, r, s} dy[n, k, p, q] * w[k, c, r, s]

    if(wDims.empty() || wDims.size() < 4)
    {
        throw std::invalid_argument(
            "wDims must have at least 4 dimensions for 2D convolution [K, C, R, S].");
    }

    // Number of accumulations = K * (product of spatial dimensions)
    // For 2D: K * R * S
    // For 3D: K * D * R * S
    auto numberOfAccumulations = static_cast<uint64_t>(wDims[0]); // K (output channels)
    for(size_t i = 2; i < wDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(wDims[i]); // Spatial dimensions
    }

    double const maxAbsDy = std::max(std::abs(dyMin), std::abs(dyMax));
    double const maxAbsW = std::max(std::abs(wMin), std::abs(wMax));

    // Worst case product magnitude
    double const maxProduct = maxAbsDy * maxAbsW;

    // Bound on sum(|dy_i * w_i|)
    double const sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|dy_i * w_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        double const nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        double const gamma = nU / (1.0 - nU);
        accumulatedTolerance = gamma * sumAbsProductBound;
    }
    else
    {
        // Lower Precision (FP16, BF16): Statistical bound (Probabilistic)
        // Error <= gamma_2n * sum(|dy_i * w_i|)
        // gamma_n = sqrt(n) * u
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = sqrt(2n) * u
        // k_sigma = 6.0 for high confidence
        constexpr double K_SIGMA = 6.0;
        double const gamma
            = K_SIGMA * std::sqrt(2.0 * static_cast<double>(numberOfAccumulations)) * epsilon;
        accumulatedTolerance = gamma * sumAbsProductBound;
    }

    // Calculate input casting error
    // If InputType has higher precision (smaller epsilon) than ComputeType, we lose precision on load (downcasting).
    // Example: double -> float.
    // If InputType has lower precision (larger epsilon) than ComputeType, we preserve precision (upcasting).
    // Example: half -> float.
    // We only need to add tolerance if we are downcasting.
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    if(inputEpsilon < epsilon)
    {
        // Input precision is higher than compute precision, so we have casting error.
        // We add this to the tolerance.
        // Note: This is a worst-case bound.
        //
        // Derivation:
        // Let dy_approx = dy_true * (1 + d_dy) and w_approx = w_true * (1 + d_w)
        // where |d_dy|, |d_w| <= epsilon_compute (relative error bound).
        // Product P_approx = dy_approx * w_approx ≈ dy_true * w_true * (1 + d_dy + d_w)
        // Error_P = |P_approx - P_true| ≈ |P_true| * |d_dy + d_w|
        // Error_P <= |P_true| * (|d_dy| + |d_w|) <= |P_true| * (epsilon + epsilon)
        // Error_P <= 2 * |P_true| * epsilon
        // Summing over N accumulations: Total_Error <= 2 * sum(|dy_i * w_i|) * epsilon
        double const castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    double const maxPossibleOutputValue = sumAbsProductBound;

    double castTolerance = 0.0;
    // Calculate precision loss due to casting from ComputeType to OutputType.
    // If OutputType has lower precision (larger epsilon) than ComputeType, we lose precision (downcasting).
    // Example: float -> half.
    // If OutputType has higher precision (smaller epsilon) than ComputeType, the value is exactly representable (upcasting).
    // Example: float -> double.
    // We only need to add tolerance if we are downcasting.
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    if(outputEpsilon > epsilon)
    {
        // The error is bounded by the precision of the OutputType at the final value.
        castTolerance = std::abs(maxPossibleOutputValue) * outputEpsilon;
    }

    // Total tolerance is the sum of accumulation error and cast error
    double const totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

/**
 * @brief Calculates the expected tolerance for Convolution Forward Propagation (fprop) operations.
 *
 * This function estimates the maximum expected error due to floating-point accumulation during the
 * computation of forward convolution. It considers the accumulation of products of inputs and weights
 * over the input channels and filter spatial dimensions.
 *
 * The tolerance is calculated by simulating the accumulation process using `ComputeType` precision
 * and adding the precision loss from casting the final result to `OutputType`.
 *
 * Error Estimation Strategy:
 * - High Precision (FP32, FP64): Uses a linear worst-case bound (Classical).
 *   Error ≈ n * epsilon * maxProduct
 * - Lower Precision (FP16, BF16): Uses a statistical/probabilistic bound.
 *   Error ≈ k * sqrt(n) * epsilon * maxProduct
 *
 * It also accounts for precision loss if inputs are cast to a lower precision ComputeType.
 *
 * @tparam OutputType The data type of the output (forward convolution output).
 * @tparam InputType The data type of the input tensor.
 * @tparam ComputeType The data type used for accumulation (default: float).
 * @param inputMin The minimum value in the input tensor.
 * @param inputMax The maximum value in the input tensor.
 * @param wMin The minimum value in the weight/filter tensor.
 * @param wMax The maximum value in the weight/filter tensor.
 * @param wDims The dimensions of the weight/filter tensor (w).
 * @return The calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculateConvFpropTolerance(
    double inputMin, double inputMax, double wMin, double wMax, const std::vector<int64_t>& wDims)
{
    // Validate ComputeType
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    // wDims: [K, C, R, S]
    // Accumulation for output (y) happens over C (input channels) and R, S (filter spatial dimensions).
    // y[n, k, h, w] = sum_{c, r, s} x[n, c, h+r, w+s] * w[k, c, r, s]

    if(wDims.empty() || wDims.size() < 2)
    {
        throw std::invalid_argument("wDims must have at least 2 dimensions (K, C).");
    }

    auto numberOfAccumulations = static_cast<uint64_t>(wDims[1]); // Input channels C
    for(size_t i = 2; i < wDims.size(); ++i)
    {
        numberOfAccumulations *= static_cast<uint64_t>(wDims[i]); // Filter spatial dimensions R, S
    }

    double const maxAbsInput = std::max(std::abs(inputMin), std::abs(inputMax));
    double const maxAbsW = std::max(std::abs(wMin), std::abs(wMax));

    // Worst case product magnitude
    double const maxProduct = maxAbsInput * maxAbsW;

    // Bound on sum(|x_i * w_i|)
    double const sumAbsProductBound = static_cast<double>(numberOfAccumulations) * maxProduct;

    auto epsilon = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());
    double accumulatedTolerance = 0.0;

    if constexpr(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>)
    {
        // High Precision: Linear bound (Classical)
        // Error <= gamma_2n * sum(|x_i * w_i|)
        // gamma_n = n * u / (1 - n * u)
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = 2 * n * u / (1 - 2 * n * u)
        double const nU = 2.0 * static_cast<double>(numberOfAccumulations) * epsilon;

        if(nU >= 1.0)
        {
            throw std::overflow_error(
                "Number of accumulations is too large for the given precision. "
                "Error bound is undefined/infinite.");
        }

        double const gamma = nU / (1.0 - nU);
        accumulatedTolerance = gamma * sumAbsProductBound;
    }
    else
    {
        // Lower Precision (FP16, BF16): Statistical bound (Probabilistic)
        // Error <= gamma_2n * sum(|x_i * w_i|)
        // gamma_n = sqrt(n) * u
        // We assume NO FMAs are used, so factor is 2n.
        // gamma_2n = sqrt(2n) * u
        // k_sigma = 6.0 for high confidence
        constexpr double K_SIGMA = 6.0;
        double const gamma
            = K_SIGMA * std::sqrt(2.0 * static_cast<double>(numberOfAccumulations)) * epsilon;
        accumulatedTolerance = gamma * sumAbsProductBound;
    }

    // Calculate input casting error
    // If InputType has higher precision (smaller epsilon) than ComputeType, we lose precision on load (downcasting).
    // Example: double -> float.
    // If InputType has lower precision (larger epsilon) than ComputeType, we preserve precision (upcasting).
    // Example: half -> float.
    // We only need to add tolerance if we are downcasting.
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    if(inputEpsilon < epsilon)
    {
        // Input precision is higher than compute precision, so we have casting error.
        // We add this to the tolerance.
        // Note: This is a worst-case bound.
        //
        // Derivation:
        // Let x_approx = x_true * (1 + d_x) and w_approx = w_true * (1 + d_w)
        // where |d_x|, |d_w| <= epsilon_compute (relative error bound).
        // Product P_approx = x_approx * w_approx ≈ x_true * w_true * (1 + d_x + d_w)
        // Error_P = |P_approx - P_true| ≈ |P_true| * |d_x + d_w|
        // Error_P <= |P_true| * (|d_x| + |d_w|) <= |P_true| * (epsilon + epsilon)
        // Error_P <= 2 * |P_true| * epsilon
        // Summing over N accumulations: Total_Error <= 2 * sum(|x_i * w_i|) * epsilon
        double const castingError = 2.0 * sumAbsProductBound * epsilon;
        accumulatedTolerance += castingError;
    }

    // Calculate final accumulated value magnitude for casting error
    double const maxPossibleOutputValue = sumAbsProductBound;

    double castTolerance = 0.0;
    // Calculate precision loss due to casting from ComputeType to OutputType.
    // If OutputType has lower precision (larger epsilon) than ComputeType, we lose precision (downcasting).
    // Example: float -> half.
    // If OutputType has higher precision (smaller epsilon) than ComputeType, the value is exactly representable (upcasting).
    // Example: float -> double.
    // We only need to add tolerance if we are downcasting.
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    if(outputEpsilon > epsilon)
    {
        // The error is bounded by the precision of the OutputType at the final value.
        castTolerance = std::abs(maxPossibleOutputValue) * outputEpsilon;
    }

    // Total tolerance is the sum of accumulation error and cast error
    double const totalTolerance = accumulatedTolerance + castTolerance;

    // Check if totalTolerance exceeds the maximum representable value of OutputType
    if(totalTolerance > static_cast<double>(std::numeric_limits<OutputType>::max()))
    {
        throw std::overflow_error(
            "Calculated tolerance exceeds the maximum representable value of the output type.");
    }

    return static_cast<float>(totalTolerance);
}

} // namespace hipdnn_test_sdk::utilities::conv

namespace hipdnn_test_sdk::utilities::pointwise
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @brief Classification of pointwise operations by error characteristics.
 *
 * Each class has a distinct error multiplier C that reflects how many ULPs of
 * divergence are expected between reference and GPU implementations.
 * Backward ops get higher C values than forward due to chain rule amplification.
 *
 * Five categories (A-E), with D and E split into forward/backward:
 * - Bitwise (A): mathematically exact, errors from casts/canonicalization only
 * - Linear (B): single IEEE-rounded ops, error scales with operand magnitude
 * - Rational (C): Newton-Raphson-based ops, input conditioning sensitive
 * - Transcendental (D): polynomial/table approximations, implementation-dependent
 * - Composite (E): compositions of transcendentals, error from worst component
 */
enum class PointwiseErrorClass : uint8_t
{
    BITWISE, ///< Comparisons, select, relu, abs, neg, identity, floor, ceil
    LINEAR, ///< add, sub, mul, add_square
    RATIONAL, ///< div, reciprocal, sqrt, rsqrt
    TRANSCENDENTAL_FWD, ///< exp, log, sin, tan, tanh_fwd, sigmoid_fwd, erf, elu_fwd
    TRANSCENDENTAL_BWD, ///< tanh_bwd, sigmoid_bwd, elu_bwd
    COMPOSITE_FWD, ///< gelu_fwd, gelu_approx_tanh_fwd, softplus_fwd, swish_fwd
    COMPOSITE_BWD ///< gelu_bwd, gelu_approx_tanh_bwd, softplus_bwd, swish_bwd
};

namespace detail
{

/// Returns the error multiplier C for a given error class and precision.
/// C_high is for float/double compute; C_low is for half/bf16 compute.
/// C_low ≈ 2 × C_high because low-precision polynomial evaluations
/// accumulate proportionally more relative error per coefficient.
template <typename ComputeType>
constexpr double getErrorMultiplier(PointwiseErrorClass errorClass)
{
    constexpr bool IS_HIGH_PRECISION
        = std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>;

    switch(errorClass)
    {
    case PointwiseErrorClass::BITWISE:
        return IS_HIGH_PRECISION ? 1.0 : 2.0;
    case PointwiseErrorClass::LINEAR:
        return IS_HIGH_PRECISION ? 2.0 : 4.0;
    case PointwiseErrorClass::RATIONAL:
        return IS_HIGH_PRECISION ? 4.0 : 8.0;
    case PointwiseErrorClass::TRANSCENDENTAL_FWD:
        return IS_HIGH_PRECISION ? 8.0 : 16.0;
    case PointwiseErrorClass::TRANSCENDENTAL_BWD:
        return IS_HIGH_PRECISION ? 12.0 : 24.0;
    case PointwiseErrorClass::COMPOSITE_FWD:
        return IS_HIGH_PRECISION ? 16.0 : 32.0;
    case PointwiseErrorClass::COMPOSITE_BWD:
        return IS_HIGH_PRECISION ? 24.0 : 48.0;
    default:
        return 1.0;
    }
}

} // namespace detail

/**
 * @brief Calculates the expected tolerance for element-wise pointwise operations.
 *
 * Error model:
 * @code
 *   tolerance = C[class][precision] * epsilon_compute * scale
 *             + (inputDowncast  ? epsilon_compute * scale : 0)
 *             + (outputDowncast ? epsilon_output  * scale : 0)
 * @endcode
 *
 * The @p scale parameter is typically @c max(|input|) for unbounded-output ops
 * (exp, log, add, gelu), but should be @c 1.0 for bounded-output ops
 * (sigmoid -> [0,1], tanh -> [-1,1], erf -> [-1,1]). The caller decides
 * based on the operation being tested.
 *
 * @tparam OutputType  Data type of the output tensor.
 * @tparam InputType   Data type of the input tensor(s).
 * @tparam ComputeType Data type used for intermediate computation (default: float).
 * @param scale        Error scaling factor. Use max(|input|) for unbounded ops,
 *                     1.0 for bounded-output transcendentals (sigmoid, tanh, erf).
 * @param errorClass   Classification of the pointwise operation.
 * @return Calculated tolerance value as float.
 */
template <typename OutputType, typename InputType, typename ComputeType = float>
float calculatePointwiseTolerance(double scale, PointwiseErrorClass errorClass)
{
    static_assert(std::is_same_v<ComputeType, float> || std::is_same_v<ComputeType, double>
                      || std::is_same_v<ComputeType, half> || std::is_same_v<ComputeType, bfloat16>,
                  "ComputeType must be float, double, half, or bfloat16");

    auto epsilonCompute = static_cast<double>(std::numeric_limits<ComputeType>::epsilon());

    // Core error: C * epsilon_compute * scale
    double c = detail::getErrorMultiplier<ComputeType>(errorClass);
    double tolerance = c * epsilonCompute * scale;

    // Input casting error: added when InputType has higher precision than ComputeType.
    auto inputEpsilon = static_cast<double>(std::numeric_limits<InputType>::epsilon());
    if(inputEpsilon < epsilonCompute)
    {
        tolerance += epsilonCompute * scale;
    }

    // Output casting error: added when ComputeType has higher precision than OutputType.
    auto outputEpsilon = static_cast<double>(std::numeric_limits<OutputType>::epsilon());
    if(outputEpsilon > epsilonCompute)
    {
        tolerance += outputEpsilon * scale;
    }

    return static_cast<float>(tolerance);
}

} // namespace hipdnn_test_sdk::utilities::pointwise
