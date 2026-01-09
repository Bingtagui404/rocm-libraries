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

#include "rocRoller/Expression_fwd.hpp"
#include "rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp"
#include "rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp"
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;

    namespace FuseExpressionsDetail
    {
        /**
         * @brief Visitor to replace a DataFlowTag with a given expression
         *
         * This visitor traverses an expression tree, and if it finds a DataFlowTag that matches the given tag,
         * it replaces it with the given expression.
         */
        struct DataFlowTagReplacerVisitor
        {
            bool                      replacementDone = false;
            int                       tag;
            Expression::ExpressionPtr exprToReplace;

            Expression::ExpressionPtr operator()(Expression::DataFlowTag const& expr)
            {
                if(expr.tag == tag)
                {
                    if(replacementDone)
                    {
                        Throw<FatalError>("More than one DataFlowTag matching given tag");
                    }
                    else
                    {
                        replacementDone = true;
                        return exprToReplace;
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
        };

        Expression::ExpressionPtr replaceDataFlowTag(Expression::ExpressionPtr const& expr,
                                                     int                              tag,
                                                     Expression::ExpressionPtr const& exprToReplace)
        {
            DataFlowTagReplacerVisitor visitor;
            visitor.tag           = tag;
            visitor.exprToReplace = exprToReplace;
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

        std::vector<Candidate> findFuseCandidates(KernelGraph const& kgraph)
        {

            std::vector<Candidate> candidates;

            auto tracer          = ControlFlowRWTracer(kgraph);
            auto recordsByParent = sortRecordsByBodyParent(kgraph, tracer);

            // Examine the reads and writes under each parent to find candidates
            for(const auto& [parent, records] : recordsByParent)
            {
                // Sort reads and writes within this body parent by coordinate
                std::unordered_map<int, std::vector<Record>> recordsByCoordinate;
                for(auto record : records)
                {
                    if(kgraph.control.get<Assign>(record.control).has_value())
                    {
                        // Add this record to the sequence corresponding to its coordinate
                        recordsByCoordinate[record.coordinate].push_back(record);
                    }
                }

                // For each coordinate, follow its sequence of reads and writes to find candidates
                for(const auto& [tag, records] : recordsByCoordinate)
                {
                    std::optional<int>       writingNode    = std::nullopt;
                    std::optional<int>       readingNode    = std::nullopt;
                    std::optional<Candidate> maybeCandidate = std::nullopt;
                    for(const auto& record : records)
                    {
                        auto node = record.control;

                        if(record.rw == RW::WRITE)
                        {
                            // If we have already written to and then read from this coordinate once each, we have found a candidate!
                            // We can save it as such and then start over with a new write
                            if(writingNode.has_value() && readingNode.has_value()
                               && maybeCandidate.has_value())
                            {
                                candidates.push_back(maybeCandidate.value());

                                readingNode    = std::nullopt;
                                maybeCandidate = std::nullopt;
                            }

                            // Begin the search for a candidate with a write
                            writingNode    = node;
                            readingNode    = std::nullopt;
                            maybeCandidate = std::nullopt;
                        }
                        else if(record.rw == RW::READ)
                        {
                            // If this coordinate has been written to but hasn't been read from yet, it may be a candidate
                            if(writingNode.has_value() && !readingNode.has_value())
                            {
                                maybeCandidate = {tag, writingNode.value(), node};
                            }
                            // Otherwise, this is not a candidate for fusing
                            else
                            {
                                maybeCandidate = std::nullopt;
                            }

                            readingNode = node;
                        }
                        else if(record.rw == RW::READWRITE)
                        {
                            // A READWRITE consists of a read followed by a write, in that order
                            // Thus, we will treat it as if it were split into two separate operations.

                            // READ
                            // If this tag has been written to but hasn't already been read from,
                            // and we know it is about to be written to again, we know that it's a candidate!
                            if(writingNode.has_value() && !readingNode.has_value())
                            {
                                candidates.push_back({tag, writingNode.value(), node});

                                readingNode    = std::nullopt;
                                maybeCandidate = std::nullopt;
                            }

                            // WRITE
                            writingNode    = node;
                            readingNode    = std::nullopt;
                            maybeCandidate = std::nullopt;
                        }
                        else
                        {
                            Throw<FatalError>(
                                "Invalid value for ControlFlowRWTracer::ReadWrite: {}", record.rw);
                        }
                    }

                    // Now that we've finished examining all of the reads and writes to this coordinate in this body parent,
                    // if we have a possible candidate, we can save it
                    if(writingNode.has_value() && readingNode.has_value()
                       && maybeCandidate.has_value())
                    {
                        Candidate candidate = maybeCandidate.value();

                        // Since we know that there won't be any more writes to this coordinate,
                        // we can delete this tag once we fuse our two expressions together
                        candidate.deleteTag = true;

                        candidates.push_back(candidate);
                    }
                    std::cout << std::endl;
                }
            }

            return candidates;
        }
    }

    /**
     * @brief Reconnect incoming and outgoing DataFlow edges from dimension.
     *
     * Inputs coming into the dimension are augmented with `other_inputs`.
     */
    void reflow(KernelGraph& graph, int dim, std::vector<int> const& other_inputs)
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        std::vector<int> inputs, outputs;
        for(auto const& tag : graph.coordinates.getNeighbours<Graph::Direction::Upstream>(dim))
        {
            auto df = graph.coordinates.get<CT::DataFlow>(tag);
            if(!df)
                continue;
            auto parents = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(tag);
            for(auto const parent : parents)
                inputs.push_back(parent);
            graph.coordinates.deleteElement(tag);
        }

        for(auto const& tag : graph.coordinates.getNeighbours<Graph::Direction::Downstream>(dim))
        {
            auto df = graph.coordinates.get<CT::DataFlow>(tag);
            if(!df)
                continue;
            auto children = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(tag);
            for(auto const child : children)
                outputs.push_back(child);
            graph.coordinates.deleteElement(tag);
        }

        std::copy(other_inputs.cbegin(), other_inputs.cend(), std::back_inserter(inputs));
        graph.coordinates.addElement(CT::DataFlow(), inputs, outputs);
    }

    KernelGraph FuseExpressions::apply(KernelGraph const& original)
    {
        auto kgraph = original;

        auto candidates = FuseExpressionsDetail::findFuseCandidates(kgraph);
        for(auto candidate : candidates)
        {
            auto tagToReplace = candidate.tag;

            auto writingNode  = candidate.writingNode;
            auto originalExpr = kgraph.control.getNode<Assign>(candidate.writingNode).expression;

            auto readingNodeTag = candidate.readingNode;
            auto readingNode    = kgraph.control.getNode<Assign>(readingNodeTag);
            auto readingExpr    = readingNode.expression;

            // Replace the DataFlowTag in readingExpr corresponding to tagToReplace with originalExpr
            auto newExpr = FuseExpressionsDetail::replaceDataFlowTag(
                readingExpr, tagToReplace, originalExpr);

            // Replace the old expression with this new one
            readingNode.expression = newExpr;
            kgraph.control.setElement(readingNodeTag, readingNode);

            // Replace writing node with NOP
            auto nop = kgraph.control.addElement(NOP());
            replaceWith(kgraph, writingNode, nop, false);
            purgeNodes(kgraph, {writingNode});

            // If necessary, delete tag
            if(candidate.deleteTag)
            {
                auto coords = CoordinateGraph::Transformer(&kgraph.coordinates);
                coords.removeCoordinate(tagToReplace);
            }
        }

        return kgraph;
    }
}
