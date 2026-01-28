/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include "rocRoller/Graph/Hypergraph.hpp"
#include "rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge_fwd.hpp"
#include <rocRoller/KernelGraph/Transforms/InlineExpressions.hpp>
#include <rocRoller/KernelGraph/Transforms/InlineExpressions_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <unordered_map>
#include <variant>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;

    namespace InlineExpressionsDetail
    {
        /**
         * @brief Visitor to replace a DataFlowTag with a given expression
         *
         * This visitor traverses an expression tree, and if it finds a DataFlowTag that matches the given tag,
         * it replaces it with the given expression.
         */
        struct DataFlowTagReplacerVisitor
        {
            DataFlowTagReplacerVisitor(int tag, Expression::ExpressionPtr replacement)
                : m_tag(tag)
                , m_replacement(replacement)
            {
            }

            Expression::ExpressionPtr operator()(Expression::DataFlowTag const& expr)
            {
                if(expr.tag == m_tag)
                {
                    if(m_replacementDone)
                    {
                        Throw<FatalError>("More than one DataFlowTag matching given tag");
                    }
                    else
                    {
                        m_replacementDone = true;
                        return m_replacement;
                    }
                }
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr operator()(Expression::ScaledMatrixMultiply const& expr)
            {
                Expression::ScaledMatrixMultiply copy = expr;
                copy.matA                             = call(expr.matA);
                copy.matB                             = call(expr.matB);
                copy.matC                             = call(expr.matC);
                copy.scaleA                           = call(expr.scaleA);
                copy.scaleB                           = call(expr.scaleB);

                return std::make_shared<Expression::Expression>(copy);
            }

            template <Expression::CNary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr)
            {
                Expr copy = expr;
                std::ranges::for_each(copy.operands, [&](auto& op) { op = call(op); });
                return std::make_shared<Expression::Expression>(std::move(copy));
            }

            template <Expression::CTernary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr)
            {
                Expr copy = expr;
                copy.lhs  = call(expr.lhs);
                copy.r1hs = call(expr.r1hs);
                copy.r2hs = call(expr.r2hs);

                return std::make_shared<Expression::Expression>(copy);
            }

            template <Expression::CBinary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr)
            {
                Expr copy = expr;
                copy.lhs  = call(expr.lhs);
                copy.rhs  = call(expr.rhs);

                return std::make_shared<Expression::Expression>(copy);
            }

            template <Expression::CUnary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr)
            {
                Expr copy = expr;
                copy.arg  = call(expr.arg);

                return std::make_shared<Expression::Expression>(copy);
            }

            template <Expression::CValue Expr>
            Expression::ExpressionPtr operator()(Expr const& expr)
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr call(Expression::ExpressionPtr const& expr)
            {
                if(!expr)
                    return expr;
                return std::visit(*this, *expr);
            }

        private:
            bool                      m_replacementDone = false;
            int                       m_tag;
            Expression::ExpressionPtr m_replacement;
        };

        Expression::ExpressionPtr replaceDataFlowTag(Expression::ExpressionPtr const& expr,
                                                     int                              tag,
                                                     Expression::ExpressionPtr const& replacement)
        {
            DataFlowTagReplacerVisitor visitor(tag, replacement);
            return visitor.call(expr);
        }

        using RW     = ControlFlowRWTracer::ReadWrite;
        using Record = ControlFlowRWTracer::ReadWriteRecord;

        std::unordered_map<int, std::vector<Record>>
            sortRecordsByBodyParent(KernelGraph const& kgraph, ControlFlowRWTracer const& tracer)
        {
            auto trace = tracer.coordinatesReadWrite();

            std::unordered_map<int, std::vector<Record>> recordsByParent;
            for(auto record : trace)
            {
                if(record.rw != RW::Count)
                {
                    auto parent = bodyParents(record.control, kgraph).take(1).only();
                    if(parent.has_value())
                    {
                        recordsByParent[parent.value()].push_back(record);
                    }
                }
            }

            return recordsByParent;
        }

        std::vector<InliningCandidate> findInliningCandidates(KernelGraph const& kgraph)
        {
            std::vector<InliningCandidate> candidates;

            auto tracer = ControlFlowRWTracer(kgraph);
            auto trace  = tracer.coordinatesReadWrite();

            // Map from tag -> possible candidate
            std::unordered_map<int, std::optional<PossibleInliningCandidate>> possibleCandidates;
            for(auto record : trace)
            {
                bool isAssignNode = kgraph.control.get<Assign>(record.control).has_value();

                auto coordinate  = record.coordinate;
                auto maybeParent = bodyParents(record.control, kgraph).take(1).only();
                AssertFatal(
                    maybeParent.has_value(), "Node has no body parent", ShowValue(record.control));
                auto parent = maybeParent.value();

                // Possible candidate corresponding to the current body parent and coordinate, if one exists
                auto possibleCandidate = &possibleCandidates[coordinate];

                if(record.rw == RW::WRITE)
                {
                    // We only care about Assign nodes that write to a coordinate
                    if(!isAssignNode)
                    {
                        continue;
                    }

                    // We're about to create a new possible candidate, so if we already had one corresponding to this coordinate,
                    // and it satisfies the conditions for being a candidate, we can save it as such
                    if(possibleCandidate->has_value() && possibleCandidate->value().isCandidate())
                    {
                        candidates.push_back(
                            possibleCandidate->value().createCandidate(/*deleteTag=*/false));
                    }
                    // Create a new possible candidate, one that has been written to but not read from yet
                    possibleCandidates[coordinate]
                        = PossibleInliningCandidate(coordinate, parent, record.control);
                }
                else if(record.rw == RW::READ)
                {
                    // See if we have any active possible candidates corresponding to this coordinate
                    if(possibleCandidate->has_value())
                    {
                        // If this read would satisfy that possible candidate, add it
                        if(possibleCandidate->value().checkNewRead(parent, isAssignNode))
                        {
                            // Now that we've written to and read from this coordinate,
                            // this will be a candidate as long as we don't have any more reads to it
                            possibleCandidate->value().addRead(parent, record.control);
                        }
                        // Otherwise, we do not have a candidate
                        else
                        {
                            possibleCandidates[coordinate] = std::nullopt;
                        }
                    }
                }
                else if(record.rw == RW::READWRITE)
                {
                    // A READWRITE is composed of a read followed by a write, so we will treat this as two separate operations:

                    // See if we have any active possible candidates corresponding to this coordinate
                    if(possibleCandidate->has_value())
                    {
                        // If this read would satisfy that possible candidate, add it
                        if(possibleCandidate->value().checkNewRead(parent, isAssignNode))
                        {
                            // Now that we've written to and read from this coordinate,
                            // since we know we're about to write to it again, this is a candidate!
                            possibleCandidate->value().addRead(parent, record.control);
                            candidates.push_back(
                                possibleCandidate->value().createCandidate(/*deleteTag=*/false));
                        }
                    }

                    // We only care about Assign nodes that write to a coordinate
                    if(!isAssignNode)
                    {
                        continue;
                    }

                    // Create a new possible candidate, one that has been written to but not read from yet
                    possibleCandidates[coordinate]
                        = PossibleInliningCandidate(coordinate, parent, record.control);
                }
                else
                {
                    Throw<FatalError>("Invalid value for ControlFlowRWTracer::ReadWrite: {}",
                                      record.rw);
                }
            }

            for(const auto& [parentAndTag, possibleCandidate] : possibleCandidates)
            {
                if(possibleCandidate.has_value() && possibleCandidate.value().isCandidate())
                {
                    // Since we know there were no more reads or writes to this coordinate, we can delete it
                    candidates.push_back(
                        possibleCandidate.value().createCandidate(/*deleteCoordinate=*/true));
                }
            }

            return candidates;
        }
    }

    KernelGraph InlineExpressions::apply(KernelGraph const& original)
    {
        auto kgraph = original;

        auto candidates = InlineExpressionsDetail::findInliningCandidates(kgraph);
        for(auto candidate : candidates)
        {
            auto coordinateToReplace = candidate.m_coordinate;

            auto replacementExpr
                = kgraph.control.getNode<Assign>(candidate.m_writingNode).expression;

            auto readingNode = kgraph.control.getNode<Assign>(candidate.m_readingNode);
            auto readingExpr = readingNode.expression;

            // Replace the DataFlowTag in readingExpr corresponding to coordinateToReplace with replacementExpr
            auto newExpr = InlineExpressionsDetail::replaceDataFlowTag(
                readingExpr, coordinateToReplace, replacementExpr);

            // Replace the old expression with this new one
            readingNode.expression = newExpr;
            kgraph.control.setElement(candidate.m_readingNode, readingNode);

            // Replace writing node with NOP
            auto nop = kgraph.control.addElement(NOP());
            replaceWith(kgraph, candidate.m_writingNode, nop, false);
            purgeNodes(kgraph, {candidate.m_writingNode});

            // If necessary, delete the coordinate and connect any parents to any children
            if(candidate.m_deleteCoordinate)
            {
                auto incomingEdges = kgraph.coordinates.getNeighbours<Graph::Direction::Upstream>(
                    coordinateToReplace);
                std::vector<int> incomingNodes;
                for(auto inEdgeIdx : incomingEdges)
                {
                    if(kgraph.coordinates.getEdgeType(inEdgeIdx)
                       == CoordinateGraph::EdgeType::DataFlow)
                    {
                        for(auto inNodeIdx :
                            kgraph.coordinates.getNeighbours<Graph::Direction::Upstream>(inEdgeIdx))
                        {
                            incomingNodes.push_back(inNodeIdx);
                        }
                        kgraph.coordinates.deleteElement(inEdgeIdx);
                    }
                }
                auto outgoingEdges = kgraph.coordinates.getNeighbours<Graph::Direction::Downstream>(
                    coordinateToReplace);
                std::vector<int> outgoingNodes;
                for(auto outEdgeIdx : outgoingEdges)
                {
                    if(kgraph.coordinates.getEdgeType(outEdgeIdx)
                       == CoordinateGraph::EdgeType::DataFlow)
                    {
                        for(auto outNodeIdx :
                            kgraph.coordinates.getNeighbours<Graph::Direction::Downstream>(
                                outEdgeIdx))
                        {
                            outgoingNodes.push_back(outNodeIdx);
                        }
                        kgraph.coordinates.deleteElement(outEdgeIdx);
                    }
                }

                kgraph.coordinates.deleteElement(coordinateToReplace);
                kgraph.mapper.purgeMappingsTo(coordinateToReplace);

                if(incomingNodes.size() > 0 && outgoingNodes.size() > 0)
                    kgraph.coordinates.addElement(
                        CoordinateGraph::DataFlow(), incomingNodes, outgoingNodes);
            }
        }

        return kgraph;
    }
}
