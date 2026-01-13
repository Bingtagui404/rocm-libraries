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
#include "catch2/matchers/catch_matchers_string.hpp"
#include "common/SourceMatcher.hpp"
#include "rocRoller/DataTypes/DataTypes.hpp"
#include "rocRoller/Expression_fwd.hpp"
#include "rocRoller/KernelGraph/Utils.hpp"
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

    TEST_CASE("FuseExpressions transformation works", "[kernel-graph][graph-transforms]")
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
                /* tag */ 293, /* writingNode */ 188, /* readingNode */ 185, /* deleteTag */ true};
            Candidate expectedCandidate2 = {
                /* tag */ 295, /* writingNode */ 184, /* readingNode */ 185, /* deleteTag */ true};
            CHECK(candidates[0] == expectedCandidate1);
            CHECK(candidates[1] == expectedCandidate2);
        }

        SECTION("Apply graph transform")
        {
            // Original:
            //         Node 188                 Node 184
            // (Tag293 = Tag24 * Tag291) (Tag295 = Tag33 * Tag35)
            //              \                      \
            //              \[seq]                 \[seq]
            //              \                      \
            //              ------> Node 185 <------
            //              (Tag_ = Tag293 + Tag295)

            // Expect expression DataFlowTag(293) = DataFlowTag(24) * DataFlowTag(291)
            auto multiplyIndex1 = 188;
            auto multiplyNode1  = graph.control.getNode<Assign>(multiplyIndex1);
            auto multiplyExpr1  = std::get<Expression::Multiply>(*multiplyNode1.expression);
            CHECK(std::get<Expression::DataFlowTag>(*multiplyExpr1.lhs).tag == 24);
            CHECK(std::get<Expression::DataFlowTag>(*multiplyExpr1.rhs).tag == 291);
            auto connections = graph.mapper.getConnections(multiplyIndex1);
            CHECK(connections.size() == 1);
            CHECK(connections.back().coordinate == 293);

            // Expect expression DataFlowTag(295) = DataFlowTag(33) * DataFlowTag(35)
            auto multiplyIndex2 = 184;
            auto multiplyNode2  = graph.control.getNode<Assign>(multiplyIndex2);
            auto multiplyExpr2  = std::get<Expression::Multiply>(*multiplyNode2.expression);
            CHECK(std::get<Expression::DataFlowTag>(*multiplyExpr2.lhs).tag == 33);
            CHECK(std::get<Expression::DataFlowTag>(*multiplyExpr2.rhs).tag == 35);
            connections = graph.mapper.getConnections(multiplyIndex2);
            CHECK(connections.size() == 1);
            CHECK(connections.back().coordinate == 295);

            // Expect expression DataFlowTag(293) + DataFlowTag(295)
            auto addIndex = 185;
            auto addNode  = graph.control.getNode<Assign>(addIndex);
            auto addExpr  = std::get<Expression::Add>(*addNode.expression);
            CHECK(std::get<Expression::DataFlowTag>(*addExpr.lhs).tag == 293);
            CHECK(std::get<Expression::DataFlowTag>(*addExpr.rhs).tag == 295);

            // Expect multiply nodes 184 and 188 are connected directly to add node 185 through sequence edges
            auto incomingNodes
                = graph.control.getInputNodeIndices<Sequence>(addIndex).to<std::vector>();
            CHECK(incomingNodes.size() == 2);
            CHECK(incomingNodes[0] == multiplyIndex1);
            CHECK(incomingNodes[1] == multiplyIndex2);

            // Apply FuseExpressions
            graph = transform<FuseExpressions>(graph);

            // After:
            //           Node 216               Node 213
            //             NOP                    NOP
            //              \                      \
            //              \[seq]                 \[seq]
            //              \                      \
            //              ------> Node 185 <------
            //     (Tag_ = (Tag24 * Tag291) + (Tag33 * Tag35))

            // Expect expression (DataFlowTag(24) * DataFlowTag(291)) + (DataFlowTag(33) * DataFlowTag(35))
            addNode  = graph.control.getNode<Assign>(addIndex);
            addExpr  = std::get<Expression::Add>(*addNode.expression);
            auto lhs = std::get<Expression::Multiply>(*addExpr.lhs);
            CHECK(std::get<Expression::DataFlowTag>(*lhs.lhs).tag == 24);
            CHECK(std::get<Expression::DataFlowTag>(*lhs.rhs).tag == 291);
            auto rhs = std::get<Expression::Multiply>(*addExpr.rhs);
            CHECK(std::get<Expression::DataFlowTag>(*rhs.lhs).tag == 33);
            CHECK(std::get<Expression::DataFlowTag>(*rhs.rhs).tag == 35);

            // Expect the multiply nodes that were connected to the add node have become NOPs
            incomingNodes = graph.control.getInputNodeIndices<Sequence>(addIndex).to<std::vector>();
            CHECK(incomingNodes.size() == 2);
            CHECK(graph.control.get<NOP>(incomingNodes[0]).has_value());
            CHECK(graph.control.get<NOP>(incomingNodes[1]).has_value());

            // Expect coordinates associated with tags 293 and 295 have been purged
            CHECK(!graph.coordinates.exists(293));
            CHECK(!graph.coordinates.exists(295));
        }

        SECTION("Write-read-write")
        {
            // Incrementing a variable
            // a = 2
            // a = a + 1
            // becomes...
            // NOP
            // a = 2 + 1

            // a = 2
            int    coord = graph.coordinates.addElement(Linear{});
            Assign writeNode;
            writeNode.expression = Expression::literal(2);
            int writeIdx         = graph.control.addElement(writeNode);
            graph.mapper.connect(writeIdx, coord, NaryArgument::DEST);

            // a = a + 1
            Assign incNode;
            auto   incLiteral = Expression::literal(1);
            auto   incVariable
                = std::make_shared<Expression::Expression>(Expression::DataFlowTag{coord});
            incNode.expression = std::make_shared<Expression::Expression>(
                Expression::Add{incVariable, incLiteral});
            int incIdx = graph.control.addElement(incNode);
            graph.mapper.connect(incIdx, coord, NaryArgument::DEST);

            // Insert nodes after node 186, which is a StoreTiled node with no outgoing edges
            insertAfter(graph, 186, writeIdx, writeIdx);
            insertAfter(graph, writeIdx, incIdx, incIdx);

            CHECK(graph.control.getOutputNodeIndices<Sequence>(186).to<std::vector>().back()
                  == writeIdx);
            CHECK(graph.control.getOutputNodeIndices<Sequence>(writeIdx).to<std::vector>().back()
                  == incIdx);

            // Apply FuseExpressions
            graph = transform<FuseExpressions>(graph);

            // Expect expression a = 2 + 1
            auto node = graph.control.getNode<Assign>(incIdx);
            auto expr = std::get<Expression::Add>(*node.expression);
            CHECK(std::get<int>(Expression::evaluate(*expr.lhs)) == 2);
            CHECK(std::get<int>(Expression::evaluate(*expr.rhs)) == 1);

            // Make sure everything is still connected correctly
            // Child of node 186 should be a NOP now
            int maybeNOPIdx
                = graph.control.getOutputNodeIndices<Sequence>(186).to<std::vector>().back();
            CHECK(graph.control.get<NOP>(maybeNOPIdx).has_value());
            // Child of NOP should be fused add expression
            CHECK(graph.control.getOutputNodeIndices<Sequence>(maybeNOPIdx).to<std::vector>().back()
                  == incIdx);

            // Make sure coordinate still exists and is mapped correctly
            CHECK(graph.coordinates.exists(coord));
            auto connections = graph.mapper.getConnections(incIdx);
            CHECK(connections.size() == 1);
            CHECK(connections.back().coordinate == coord);
        }

        SECTION("Write-read-read (should not be fused)")
        {
            // TODO:
            // Two reads after a write
            // a = 1
            // b = 2 * a
            // c = a + 1
        }

        SECTION("Write-read-write-read")
        {
            // TODO:
            // Multiple pairs of one write, one read
            // a = 1
            // b = a + 1
            // a = 2
            // c = a * 2
            // becomes...
            // NOP
            // b = 1 + 1
            // NOP
            // c = 2 * 2
        }
    }
}
