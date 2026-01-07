/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#include "SimpleTest.hpp"
#include "TestContext.hpp"
#include "common/SourceMatcher.hpp"
#include "rocRoller/Utilities/Logging.hpp"
#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions_detail.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>
#include <variant>

using namespace rocRoller;

namespace FuseExpressionsTest
{
    // Helper function to get all assign nodes of a particular expression type from the graph
    template <typename ExpressionType>
    std::vector<int> getAssignNodes(KernelGraph::KernelGraph const& graph)
    {
        using namespace rocRoller::KernelGraph;
        using namespace ControlGraph;

        std::vector<int> expressionNodes;
        for(auto const& node : graph.control.getNodes<Assign>())
        {
            auto assign = graph.control.get<Assign>(node);
            if(std::holds_alternative<ExpressionType>(*assign->expression))
            {
                expressionNodes.push_back(node);
            }
        }

        return expressionNodes;
    }

    TEST_CASE("FuseExpressions transformation works.", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        using namespace rocRoller::KernelGraph::FuseExpressionsDetail;

        auto context = TestContext::ForDefaultTarget();

        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 2;

        example.setTileSize(256, 64, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, true, false);
        example.setUnroll(2, 2);

        example.setPrefetch(true, 2, 2, false);

        auto graph  = example.getKernelGraph();
        auto params = example.getCommandParameters();

        // Apply transformations up to ConstantPropagation,
        // for some cases we will apply it before FuseExpressions and for others we will not
        graph = transform<IdentifyParallelDimensions>(graph);
        graph = transform<OrderMemory>(graph, true);
        graph = transform<UpdateParameters>(graph, params);
        graph = transform<AddLDS>(graph, params, context.get());
        graph = transform<LowerLinear>(graph, context.get());
        graph = transform<LowerTile>(graph, params, context.get());
        graph = transform<LowerTensorContraction>(graph, params, context.get());
        graph = transform<Simplify>(graph);
        auto graphBeforeConstantPropagation = graph;
        graph                               = transform<ConstantPropagation>(graph);

        SECTION("Find candidates")
        {
            auto candidates = FuseExpressionsDetail::findFuseCandidates(graph);
            CHECK(candidates.size() == 2);

            Candidate expectedCandidate1 = {
                /* tag */ 293, /* writingNode */ 184, /* readingNode */ 185, /* deleteTag */ true};
            Candidate expectedCandidate2 = {
                /* tag */ 295, /* writingNode */ 188, /* readingNode */ 185, /* deleteTag */ true};
            CHECK(candidates[0] == expectedCandidate1);
            CHECK(candidates[1] == expectedCandidate2);
        }

        SECTION("Apply graph transform with constant propagation")
        {
            // Count the Multiply and Add nodes before FuseExpressions
            auto multiplyNodesBefore = getAssignNodes<Expression::Multiply>(graph);
            CHECK(multiplyNodesBefore.size() == 3);
            auto addNodesBefore = getAssignNodes<Expression::Add>(graph);
            CHECK(addNodesBefore.size() == 8);

            // Apply FuseExpressions
            graph = transform<FuseExpressions>(graph);

            // Count the Multiply and Add nodes after FuseExpressions
            auto multiplyNodesAfter = getAssignNodes<Expression::Multiply>(graph);
            CHECK(multiplyNodesAfter.size() == 1);
            auto addNodesAfter = getAssignNodes<Expression::Add>(graph);
            CHECK(addNodesAfter.size() == 7);

            // Count the MultiplyAdd nodes after FuseExpressions
            auto multiplyAddNodes = getAssignNodes<Expression::MultiplyAdd>(graph);
            CHECK(multiplyAddNodes.size() == 1);
        }

        SECTION("Apply graph transform, no constant propagation")
        {
            graph = graphBeforeConstantPropagation;

            // Count the Multiply and Add nodes before FuseExpressions
            auto multiplyNodesBefore = getAssignNodes<Expression::Multiply>(graph);
            CHECK(multiplyNodesBefore.size() == 2);
            auto addNodesBefore = getAssignNodes<Expression::Add>(graph);
            CHECK(addNodesBefore.size() == 6);

            // Apply FuseExpressions
            graph = transform<FuseExpressions>(graph);

            // Count the Multiply and Add nodes after FuseExpressions
            auto multiplyNodesAfter = getAssignNodes<Expression::Multiply>(graph);
            CHECK(multiplyNodesAfter.size() == 0);
            auto addNodesAfter = getAssignNodes<Expression::Add>(graph);
            CHECK(addNodesAfter.size() == 5);

            // Count the MultiplyAdd nodes after FuseExpressions
            auto multiplyAddNodes = getAssignNodes<Expression::MultiplyAdd>(graph);
            CHECK(multiplyAddNodes.size() == 1);
        }
    }
}
