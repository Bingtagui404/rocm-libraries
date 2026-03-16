// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceBlockScaleDequantize
{
public:
    /// Block scale dequantize: Y[i] = X[i] * scale[block_of(i)]
    ///
    /// blockSize specifies the block size for each blocked dimension. Blocked dimensions
    /// are identified by comparing X and scale tensor shapes: dimension d is blocked when
    /// xDims[d] != scaleDims[d]. The i-th entry in blockSize corresponds to the i-th
    /// blocked dimension, and scale_index[d] = x_index[d] / blockSize[i].
    /// For unblocked dimensions, scale_index[d] = x_index[d].
    ///
    /// @param x                Input tensor (blocked low-precision data)
    /// @param scale            Per-block scale tensor
    /// @param y                Output tensor (dequantized, same shape as x)
    /// @param blockSize        Block size for each blocked dimension (from attributes)
    /// @param isNegativeScale  If true, scale represents negative exponent: Y = X * 2^(-scale)
    template <class XDataType, class ScaleDataType, class YDataType, class ComputeDataType = float>
    static void dequantize(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                           const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                           hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                           const std::vector<int32_t>& blockSize,
                           bool isNegativeScale = false)
    {
        const auto& xDims = x.dims();
        const auto& scaleDims = scale.dims();

        if(xDims.empty())
        {
            throw std::runtime_error("BlockScaleDequantize requires non-empty tensor dimensions.");
        }

        // Build per-dimension block sizes from the blockSize attribute.
        // blockSize entries map in order to dimensions where xDims[d] != scaleDims[d].
        std::vector<int64_t> effectiveBlockSize(xDims.size(), 1);
        size_t blockSizeIdx = 0;
        for(size_t d = 0; d < xDims.size() && d < scaleDims.size(); ++d)
        {
            if(xDims[d] != scaleDims[d])
            {
                if(blockSizeIdx < blockSize.size())
                {
                    effectiveBlockSize[d] = blockSize[blockSizeIdx];
                    ++blockSizeIdx;
                }
            }
        }

        auto dequantizeFunc = [&](const std::vector<int64_t>& xIndices) {
            // Compute scale indices by dividing by block size for blocked dims
            std::vector<int64_t> scaleIndices(xIndices.size());
            for(size_t d = 0; d < xIndices.size(); ++d)
            {
                if(d < scaleDims.size())
                {
                    scaleIndices[d] = xIndices[d] / effectiveBlockSize[d];
                }
                else
                {
                    scaleIndices[d] = 0;
                }
            }

            auto xVal = static_cast<ComputeDataType>(x.getHostValue(xIndices));
            auto scaleVal = static_cast<ComputeDataType>(scale.getHostValue(scaleIndices));

            ComputeDataType yVal;
            if(isNegativeScale)
            {
                // Negative scale: Y = X * 2^(-scale_value)
                yVal = xVal * std::pow(static_cast<ComputeDataType>(2.0), -scaleVal);
            }
            else
            {
                // Normal scale: Y = X * scale
                yVal = xVal * scaleVal;
            }

            y.setHostValue(static_cast<YDataType>(yVal), xIndices);
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(dequantizeFunc, xDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
    }
};

} // namespace hipdnn_test_sdk::utilities
