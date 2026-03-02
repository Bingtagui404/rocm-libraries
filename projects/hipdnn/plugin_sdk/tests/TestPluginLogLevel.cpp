// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>

// Unit tests for the hipdnnPluginSetLogLevel API definition
// These tests verify the API interface and data type compatibility

class TestPluginLogLevel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Save original log level
        _originalLevel = hipdnn_data_sdk::logging::getLogLevel();
    }

    void TearDown() override
    {
        // Restore original log level
        hipdnn_data_sdk::logging::setLogLevel(_originalLevel);
    }

    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};
};

// Test that hipdnnSeverity_t enum values are valid for use with hipdnnPluginSetLogLevel
TEST_F(TestPluginLogLevel, SeverityEnumValuesAreValid)
{
    // Verify all severity enum values are distinct
    EXPECT_NE(HIPDNN_SEV_INFO, HIPDNN_SEV_WARN);
    EXPECT_NE(HIPDNN_SEV_INFO, HIPDNN_SEV_ERROR);
    EXPECT_NE(HIPDNN_SEV_INFO, HIPDNN_SEV_FATAL);
    EXPECT_NE(HIPDNN_SEV_INFO, HIPDNN_SEV_OFF);

    EXPECT_NE(HIPDNN_SEV_WARN, HIPDNN_SEV_ERROR);
    EXPECT_NE(HIPDNN_SEV_WARN, HIPDNN_SEV_FATAL);
    EXPECT_NE(HIPDNN_SEV_WARN, HIPDNN_SEV_OFF);

    EXPECT_NE(HIPDNN_SEV_ERROR, HIPDNN_SEV_FATAL);
    EXPECT_NE(HIPDNN_SEV_ERROR, HIPDNN_SEV_OFF);

    EXPECT_NE(HIPDNN_SEV_FATAL, HIPDNN_SEV_OFF);
}

// Test that setLogLevel/getLogLevel work correctly with data SDK
TEST_F(TestPluginLogLevel, SetAndGetLogLevelWorksCorrectly)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_INFO);

    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);

    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);

    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_FATAL);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_FATAL);

    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF);
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_OFF);
}

// Test that isLogLevelEnabled respects severity ordering
TEST_F(TestPluginLogLevel, IsLogLevelEnabledRespectsSeverityOrdering)
{
    // When log level is INFO, all levels should be enabled
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL));

    // When log level is WARN, INFO should be disabled
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL));

    // When log level is ERROR, INFO and WARN should be disabled
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL));

    // When log level is FATAL, only FATAL should be enabled
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_FATAL);
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR));
    EXPECT_TRUE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL));

    // When log level is OFF, nothing should be enabled
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF);
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_WARN));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_ERROR));
    EXPECT_FALSE(hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_FATAL));
}

// Test that hipdnnPluginStatus_t values are correct for function return types
TEST_F(TestPluginLogLevel, PluginStatusValuesAreCorrect)
{
    // Verify HIPDNN_PLUGIN_STATUS_SUCCESS is zero (standard success value)
    EXPECT_EQ(HIPDNN_PLUGIN_STATUS_SUCCESS, 0);

    // Verify error status is non-zero
    EXPECT_NE(HIPDNN_PLUGIN_STATUS_BAD_PARAM, HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_NE(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, HIPDNN_PLUGIN_STATUS_SUCCESS);
}
