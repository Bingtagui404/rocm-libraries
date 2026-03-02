// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <array>
#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <test_plugins/TestPluginConstants.hpp>

// Integration tests for plugin log level synchronization
// Tests verify that log level changes propagate from backend to plugins

class IntegrationPluginLogLevel : public ::testing::Test
{
protected:
    hipdnnHandle_t _handle = nullptr;

    void SetUp() override
    {
        // Save original log level
        _originalLevel = hipdnn_data_sdk::logging::getLogLevel();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }

        // Restore original log level
        hipdnn_data_sdk::logging::setLogLevel(_originalLevel);
    }

    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};
};

// Test that hipdnnBackendSetGlobalLogLevel_ext returns success
TEST_F(IntegrationPluginLogLevel, SetGlobalLogLevelReturnsSuccess)
{
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_FATAL), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_OFF), HIPDNN_STATUS_SUCCESS);
}

// Test that hipdnnBackendGetGlobalLogLevel_ext returns the correct level
TEST_F(IntegrationPluginLogLevel, GetGlobalLogLevelReturnsCorrectLevel)
{
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_INFO);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_WARN);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_ERROR);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_FATAL), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_FATAL);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_OFF), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_OFF);
}

// Test that hipdnnBackendGetGlobalLogLevel_ext returns BAD_PARAM for null pointer
TEST_F(IntegrationPluginLogLevel, GetGlobalLogLevelReturnsErrorForNullPointer)
{
    EXPECT_EQ(hipdnnBackendGetGlobalLogLevel_ext(nullptr), HIPDNN_STATUS_BAD_PARAM);
}

// Test that log level is propagated when plugins are loaded
TEST_F(IntegrationPluginLogLevel, LogLevelIsPropagatedOnPluginLoad)
{
    // Set log level before loading plugins
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);

    // Load the test plugin
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    // Create handle to trigger plugin loading
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // Verify the log level is still set correctly
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_WARN);
}

// Test that changing log level after plugins are loaded works correctly
TEST_F(IntegrationPluginLogLevel, LogLevelChangeAfterPluginsLoaded)
{
    // Load the test plugin
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    // Create handle to trigger plugin loading
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // Change log level after plugins are loaded
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR), HIPDNN_STATUS_SUCCESS);

    // Verify the log level is updated
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_ERROR);
}

// Test that log level changes work when no plugins are loaded
TEST_F(IntegrationPluginLogLevel, LogLevelChangeWithNoPlugins)
{
    // Create empty plugin directory
    hipdnn_test_sdk::utilities::ScopedDirectory pluginDir("empty_plugins_log_level");
    auto pluginPath = pluginDir.path().string();
    const std::array<const char*, 1> paths = {pluginPath.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    // Set log level without any plugins
    EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    // Verify the log level is set
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;
    ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(level, HIPDNN_SEV_INFO);
}

// Test that log level persists across multiple set/get operations
TEST_F(IntegrationPluginLogLevel, LogLevelPersistsAcrossMultipleOperations)
{
    // Load the test plugin
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    // Create handle to trigger plugin loading
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    hipdnnSeverity_t level = HIPDNN_SEV_OFF;

    // Multiple set/get cycles
    for(int i = 0; i < 3; ++i)
    {
        ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(level, HIPDNN_SEV_INFO);

        ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_ERROR), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&level), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(level, HIPDNN_SEV_ERROR);
    }
}
