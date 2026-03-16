// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBlockScaleDequantize.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizePlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::flatbuffer_utilities;

TEST(TestBlockScaleDequantizePlan, ExecutePlan)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graphWrapper.getNode(0);
    const auto& tensorMap = graphWrapper.getTensorMap();

    // Create two tensor bundles with same data for plan vs direct comparison
    unsigned int seed = getGlobalTestSeed();
    GraphTensorBundle planBundle(tensorMap);
    GraphTensorBundle directBundle(tensorMap);

    // Fill x and scale with same random data
    planBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    planBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);
    directBundle.getTensor(1).fillTensorWithRandomValues(0.0f, 1.0f, seed);
    directBundle.getTensor(2).fillTensorWithRandomValues(0.1f, 2.0f, seed);

    const auto* nodeAttributes = node.attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(nodeAttributes, nullptr);

    std::vector<int32_t> blockSize;
    if(nodeAttributes->block_size() != nullptr)
    {
        const auto* bs = nodeAttributes->block_size();
        blockSize.assign(bs->begin(), bs->end());
    }

    // Execute via plan
    BlockScaleDequantizeParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                      *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                      blockSize,
                                      nodeAttributes->is_negative_scale());

    // Direct execution for reference
    auto directXTensor
        = createShallowTensor<float>(params.xTensor, directBundle.getTensor(1).rawHostData());
    auto directScaleTensor
        = createShallowTensor<float>(params.scaleTensor, directBundle.getTensor(2).rawHostData());
    auto directYTensor
        = createShallowTensor<float>(params.yTensor, directBundle.getTensor(3).rawHostData());

    CpuFpReferenceBlockScaleDequantize::dequantize(
        *directXTensor, *directScaleTensor, *directYTensor, blockSize, false);

    // Plan execution
    auto variantPack = planBundle.toHostVariantPack();
    BlockScaleDequantizePlan<float, float, float, float> plan(std::move(params));
    plan.execute(variantPack);

    float tolerance = 1e-5f;
    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directBundle.getTensor(3), planBundle.getTensor(3)));
}

TEST(TestBlockScaleDequantizePlanBuilder, PlanConstruction)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    BlockScaleDequantizePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    bool result
        = dynamic_cast<BlockScaleDequantizePlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBlockScaleDequantizePlanBuilder, IsApplicable)
{
    auto builder = createValidBlockScaleDequantizeGraph();
    GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    BlockScaleDequantizePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    BlockScaleDequantizePlanBuilder<DataType::HALF, DataType::FLOAT, DataType::FLOAT,
                                    DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}
