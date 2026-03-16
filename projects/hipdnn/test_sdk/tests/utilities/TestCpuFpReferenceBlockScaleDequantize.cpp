// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

using TypesBlockScaleDequantize
    = ::testing::Types<TypePair<float, float>, TypePair<half, float>, TypePair<bfloat16, float>>;

template <class T>
class CpuFpReferenceBlockScaleDequantizeTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceBlockScaleDequantizeTyped, TypesBlockScaleDequantize, );

TYPED_TEST(CpuFpReferenceBlockScaleDequantizeTyped, IdentityScale)
{
    using XType = typename TypeParam::First;
    using ScaleType = typename TypeParam::Second;

    // X: 2x4, Scale: 2x2 (block_size=2 along dim 1)
    Tensor<XType> xTensor({2, 4});
    Tensor<ScaleType> scaleTensor({2, 2});
    Tensor<float> yTensor({2, 4});

    xTensor.fillWithValue(static_cast<XType>(3.0f));
    scaleTensor.fillWithValue(static_cast<ScaleType>(1.0f));

    CpuFpReferenceBlockScaleDequantize::dequantize(
        xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    for(int b = 0; b < 2; ++b)
    {
        for(int c = 0; c < 4; ++c)
        {
            EXPECT_NEAR(yTensor.getHostValue(b, c), 3.0f, tolerance);
        }
    }
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NonTrivialScale)
{
    // X: 1x4, Scale: 1x2 => block_size=2 along dim 1
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(1.0f, 0, 0);
    xTensor.setHostValue(2.0f, 0, 1);
    xTensor.setHostValue(3.0f, 0, 2);
    xTensor.setHostValue(4.0f, 0, 3);

    scaleTensor.setHostValue(2.0f, 0, 0); // scales elements [0,1]
    scaleTensor.setHostValue(0.5f, 0, 1); // scales elements [2,3]

    CpuFpReferenceBlockScaleDequantize::dequantize(
        xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 1.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 2.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 3.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 4.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, MultiDimBlocking)
{
    // X: 2x64x32x32, Scale: 2x2x32x32 => block along dim 1, block_size=32
    Tensor<float> xTensor({2, 64, 32, 32});
    Tensor<float> scaleTensor({2, 2, 32, 32});
    Tensor<float> yTensor({2, 64, 32, 32});

    xTensor.fillWithValue(1.0f);
    scaleTensor.fillWithValue(3.0f);

    CpuFpReferenceBlockScaleDequantize::dequantize(
        xTensor, scaleTensor, yTensor, {32}, false);

    auto tolerance = 1e-5f;
    // Spot check a few values
    EXPECT_NEAR(yTensor.getHostValue(0, 0, 0, 0), 3.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(1, 63, 31, 31), 3.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, IsNegativeScaleFloat)
{
    // With is_negative_scale=true and float scale, Y = X * 2^(-scale)
    Tensor<float> xTensor({1, 4});
    Tensor<float> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(8.0f, 0, 0);
    xTensor.setHostValue(8.0f, 0, 1);
    xTensor.setHostValue(16.0f, 0, 2);
    xTensor.setHostValue(16.0f, 0, 3);

    scaleTensor.setHostValue(3.0f, 0, 0); // 2^(-3) = 0.125
    scaleTensor.setHostValue(1.0f, 0, 1); // 2^(-1) = 0.5

    CpuFpReferenceBlockScaleDequantize::dequantize(
        xTensor, scaleTensor, yTensor, {2}, true);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 8.0f * 0.125f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 8.0f * 0.125f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 16.0f * 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 16.0f * 0.5f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp32, NormalScaleMultiplication)
{
    // Normal (non-negative) scale: Y = X * scale
    Tensor<float> xTensor({1, 2});
    Tensor<float> scaleTensor({1, 1});
    Tensor<float> yTensor({1, 2});

    xTensor.setHostValue(5.0f, 0, 0);
    xTensor.setHostValue(10.0f, 0, 1);
    scaleTensor.setHostValue(0.1f, 0, 0);

    CpuFpReferenceBlockScaleDequantize::dequantize(
        xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 0.5f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 1.0f, tolerance);
}

TEST(TestCpuFpReferenceBlockScaleDequantizeFp8, E8M0ScaleDequantize)
{
    // Test with fp8_e8m0 scale: scale value is 2^(biased_exp - 127)
    // fp8_e8m0 with bits=127 => 2^0 = 1.0
    // fp8_e8m0 with bits=128 => 2^1 = 2.0
    Tensor<float> xTensor({1, 4});
    Tensor<fp8_e8m0> scaleTensor({1, 2});
    Tensor<float> yTensor({1, 4});

    xTensor.setHostValue(3.0f, 0, 0);
    xTensor.setHostValue(3.0f, 0, 1);
    xTensor.setHostValue(5.0f, 0, 2);
    xTensor.setHostValue(5.0f, 0, 3);

    // scale[0] = 1.0 (bits=127), scale[1] = 2.0 (bits=128)
    scaleTensor.setHostValue(fp8_e8m0::from_bits(127), 0, 0);
    scaleTensor.setHostValue(fp8_e8m0::from_bits(128), 0, 1);

    CpuFpReferenceBlockScaleDequantize::dequantize(
        xTensor, scaleTensor, yTensor, {2}, false);

    auto tolerance = 1e-5f;
    EXPECT_NEAR(yTensor.getHostValue(0, 0), 3.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 1), 3.0f * 1.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 2), 5.0f * 2.0f, tolerance);
    EXPECT_NEAR(yTensor.getHostValue(0, 3), 5.0f * 2.0f, tolerance);
}
