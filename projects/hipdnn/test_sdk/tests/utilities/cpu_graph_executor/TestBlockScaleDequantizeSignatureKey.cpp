// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizeSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;

TEST(TestBlockScaleDequantizeSignatureKey, EqualityOperator)
{
    BlockScaleDequantizeSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    BlockScaleDequantizeSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    BlockScaleDequantizeSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    BlockScaleDequantizeSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    BlockScaleDequantizeSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestBlockScaleDequantizeSignatureKey, HashFunction)
{
    BlockScaleDequantizeSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BlockScaleDequantizeSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BlockScaleDequantizeSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestBlockScaleDequantizeSignatureKey, Copy)
{
    BlockScaleDequantizeSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::BFLOAT16};
    BlockScaleDequantizeSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleDataType, DataType::HALF);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestBlockScaleDequantizeSignatureKey, CreateFromNodeAndTensorMap)
{
    BlockScaleDequantizeSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    auto builder = createValidBlockScaleDequantizeGraph();
    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    BlockScaleDequantizeSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
