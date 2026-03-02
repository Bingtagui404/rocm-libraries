// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/PointwiseOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

class TestPointwiseOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<PointwiseOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<PointwiseOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_1, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_2, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_AXIS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_axisDesc);
    }

    void setPointwiseParams() const
    {
        auto desc = getDescriptor();
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setPointwiseParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        auto operation = HIPDNN_POINTWISE_MODE_ADD;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in0Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _out0Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in1Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _in2Desc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _axisDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<PointwiseOperationDescriptor>();
        _in0Desc = createFinalizedTensor(40, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        _out0Desc = createFinalizedTensor(41, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        _in1Desc = createFinalizedTensor(3);
        _in2Desc = createFinalizedTensor(4);
        _axisDesc = createFinalizedTensor(5);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _in0Desc.reset();
        _out0Desc.reset();
        _in1Desc.reset();
        _in2Desc.reset();
        _axisDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutIn0Tensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_2, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_AXIS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_axisDesc);
    setPointwiseParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutOut0Tensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_2, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_AXIS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_axisDesc);
    setPointwiseParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setPointwiseParams();
    auto operation = HIPDNN_POINTWISE_MODE_ADD;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, FinalizeFailsWithoutPointwiseMode)
{
    setTensors();
    setPointwiseParams();
    auto computeType = HIPDNN_DATA_FLOAT;
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorIn0)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().in_0_tensor_uid, 40);
    ASSERT_NE(desc->getIn0Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorOut0)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_out0Desc));

    ASSERT_EQ(desc->getData().out_0_tensor_uid, 41);
    ASSERT_NE(desc->getOut0Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorIn1)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_1, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in1Desc));

    ASSERT_EQ(desc->getData().in_1_tensor_uid, 3);
    ASSERT_NE(desc->getIn1Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorIn2)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_IN_2, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in2Desc));

    ASSERT_EQ(desc->getData().in_2_tensor_uid, 4);
    ASSERT_NE(desc->getIn2Desc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorDescriptorAxis)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_POINTWISE_AXIS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_axisDesc));

    ASSERT_EQ(desc->getData().axis_tensor_uid, 5);
    ASSERT_NE(desc->getAxisDesc(), nullptr);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_INT64, 1, &_in0Desc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_in0Desc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetPointwiseMode)
{
    auto desc = getDescriptor();
    auto operation = HIPDNN_POINTWISE_MODE_ADD;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operation));

    ASSERT_EQ(desc->getData().operation, PointwiseMode::ADD);
}

TEST_F(TestPointwiseOperationDescriptor, SetPointwiseModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto operation = HIPDNN_POINTWISE_MODE_ADD;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 2, &operation),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPointwiseOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestPointwiseOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_POINTWISE_IN_0, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_in0Desc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPointwiseOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedIn0 = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       &retrievedIn0));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedIn0, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributePointwiseParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // operation
    hipdnnPointwiseMode_t operation = HIPDNN_POINTWISE_MODE_ABS;
    int64_t operationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 1, &operationCount, &operation));
    ASSERT_EQ(operationCount, 1);
    EXPECT_EQ(operation, HIPDNN_POINTWISE_MODE_ADD);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorIn0QueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorOut0QueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorIn1QueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_1,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorIn2QueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_2,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorAxisQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_AXIS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributePointwiseModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_POINTWISE_IN_0,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPointwiseOperationDescriptor, GetAttributePointwiseModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_POINTWISE_MODE, HIPDNN_TYPE_POINTWISE_MODE, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getIn0Desc(), nullptr);
    ASSERT_NE(desc->getOut0Desc(), nullptr);
    ASSERT_NE(desc->getIn1Desc(), nullptr);
    ASSERT_NE(desc->getIn2Desc(), nullptr);
    ASSERT_NE(desc->getAxisDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getIn0Desc()->getData().uid, 40);
    ASSERT_EQ(desc->getOut0Desc()->getData().uid, 41);
    ASSERT_EQ(desc->getIn1Desc()->getData().uid, 3);
    ASSERT_EQ(desc->getIn2Desc()->getData().uid, 4);
    ASSERT_EQ(desc->getAxisDesc()->getData().uid, 5);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("PointwiseOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("in_0_uid=40"), std::string::npos);
    ASSERT_NE(str.find("out_0_uid=41"), std::string::npos);
    ASSERT_NE(str.find("in_1_uid=3"), std::string::npos);
    ASSERT_NE(str.find("in_2_uid=4"), std::string::npos);
    ASSERT_NE(str.find("axis_uid=5"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestPointwiseOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 5);
    ASSERT_EQ(tensors[0]->getData().uid, 40);
    ASSERT_EQ(tensors[1]->getData().uid, 41);
    ASSERT_EQ(tensors[2]->getData().uid, 3);
    ASSERT_EQ(tensors[3]->getData().uid, 4);
    ASSERT_EQ(tensors[4]->getData().uid, 5);
}

TEST_F(TestPointwiseOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::PointwiseAttributes);

    auto* attrs = node->attributes.AsPointwiseAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->in_0_tensor_uid, 40);
    ASSERT_EQ(attrs->out_0_tensor_uid, 41);
    ASSERT_EQ(attrs->in_1_tensor_uid, 3);
    ASSERT_EQ(attrs->in_2_tensor_uid, 4);
    ASSERT_EQ(attrs->axis_tensor_uid, 5);
}

TEST_F(TestPointwiseOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_POINTWISE_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestPointwiseOperationDescriptor, GetTensorDescriptorsOrderIsIn0Out0In1In2Axis)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 5);
    // Verify ordering: [IN_0, OUT_0, IN_1, IN_2, AXIS] matches UIDs [40, 41, 3, 4, 5]
    EXPECT_EQ(tensors[0], desc->getIn0Desc());
    EXPECT_EQ(tensors[1], desc->getOut0Desc());
    EXPECT_EQ(tensors[2], desc->getIn1Desc());
    EXPECT_EQ(tensors[3], desc->getIn2Desc());
    EXPECT_EQ(tensors[4], desc->getAxisDesc());
}

TEST_F(TestPointwiseOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsInterface<IGraphOperation>();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 5);
    ASSERT_EQ(tensors[0]->getData().uid, 40);
}

TEST_F(TestPointwiseOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _in0Desc->tryAsInterface<IGraphOperation>();
    EXPECT_EQ(graphOp, nullptr);
}
