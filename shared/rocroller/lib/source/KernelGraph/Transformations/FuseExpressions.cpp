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
#include <list>
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

        std::vector<Candidate> findFuseCandidates(KernelGraph const& kgraph)
        {
            std::vector<Candidate> candidates;

            auto tracer = ControlFlowRWTracer(kgraph);
            auto trace  = tracer.coordinatesReadWrite();

            // Map from <parent, tag> -> list of possible candidates
            std::map<std::pair<int, int>, std::vector<PossibleCandidate>> possibleCandidates;
            for(auto record : trace)
            {
                // We only care about assign nodes
                if(!kgraph.control.get<Assign>(record.control).has_value())
                {
                    continue;
                }

                auto tag         = record.coordinate;
                auto node        = kgraph.control.getNode<Assign>(record.control);
                auto maybeParent = bodyParents(record.control, kgraph).take(1).only();
                AssertFatal(maybeParent.has_value(), "Node has no body parent", ShowValue(node));
                auto parent = maybeParent.value();

                // List of possible candidates corresponding to the current body parent and tag
                auto listOfPossibleCandidates = &possibleCandidates[{parent, tag}];

                if(record.rw == RW::WRITE)
                {
                    // Create a new possible candidate
                    listOfPossibleCandidates->push_back({tag, record.control, std::nullopt});
                }
                else if(record.rw == RW::READ)
                {
                    // See if we have any active possible candidates corresponding to this tag and body parent
                    if(!listOfPossibleCandidates->empty())
                    {
                        // If there's an active possible candidate with a write but no read
                        auto possibleCandidate = &listOfPossibleCandidates->back();
                        if(possibleCandidate->hasWriteNoRead())
                        {
                            possibleCandidate->readingNode = record.control;
                        }
                        else
                        {
                            // Otherwise, this is not a possible candidate, so remove it
                            listOfPossibleCandidates->pop_back();
                        }
                    }
                }
                else if(record.rw == RW::READWRITE)
                {
                    // See if we have any active possible candidates corresponding to this tag and body parent
                    if(!listOfPossibleCandidates->empty())
                    {
                        // If there's an active possible candidate with a write but no read
                        auto possibleCandidate = &listOfPossibleCandidates->back();
                        if(possibleCandidate->hasWriteNoRead())
                        {
                            possibleCandidate->readingNode = record.control;
                        }
                        else
                        {
                            // Otherwise, this is not a possible candidate, so remove it
                            listOfPossibleCandidates->pop_back();
                        }
                    }

                    // Create a new possible candidate
                    listOfPossibleCandidates->push_back({tag, record.control, std::nullopt});
                }
                else
                {
                    Throw<FatalError>("Invalid value for ControlFlowRWTracer::ReadWrite: {}",
                                      record.rw);
                }
            }

            for(const auto& [key, listOfPossibleCandidates] : possibleCandidates)
            {
                for(const auto& possibleCandidate : listOfPossibleCandidates)
                {
                    if(possibleCandidate.isCandidate())
                    {
                        Candidate newCandidate = {possibleCandidate.tag,
                                                  possibleCandidate.writingNode.value(),
                                                  possibleCandidate.readingNode.value()};

                        // If this is the final possible candidate for this tag and body parent,
                        // that means there are no more reads or writes to this tag in this body parent and therefore the tag can be deleted
                        if(possibleCandidate == listOfPossibleCandidates.back())
                        {
                            newCandidate.deleteTag = true;
                        }

                        candidates.push_back(newCandidate);
                    }
                }
            }

            return candidates;
        }
    }

    KernelGraph FuseExpressions::apply(KernelGraph const& original)
    {
        auto kgraph = original;

        auto candidates = FuseExpressionsDetail::findFuseCandidates(kgraph);
        for(auto candidate : candidates)
        {
            auto tagToReplace = candidate.tag;

            auto replacementExpr = kgraph.control.getNode<Assign>(candidate.writingNode).expression;

            auto readingNode = kgraph.control.getNode<Assign>(candidate.readingNode);
            auto readingExpr = readingNode.expression;

            // Replace the DataFlowTag in readingExpr corresponding to tagToReplace with replacementExpr
            auto newExpr = FuseExpressionsDetail::replaceDataFlowTag(
                readingExpr, tagToReplace, replacementExpr);

            // Replace the old expression with this new one
            readingNode.expression = newExpr;
            kgraph.control.setElement(candidate.readingNode, readingNode);

            // Replace writing node with NOP
            auto nop = kgraph.control.addElement(NOP());
            replaceWith(kgraph, candidate.writingNode, nop, false);
            purgeNodes(kgraph, {candidate.writingNode});

            // If necessary, delete tag
            if(candidate.deleteTag)
            {
                kgraph.coordinates.deleteElement(tagToReplace);
                kgraph.mapper.purgeMappingsTo(tagToReplace);
                // auto coords = CoordinateGraph::Transformer(&kgraph.coordinates);
                // coords.removeCoordinate(tagToReplace);
            }
        }

        return kgraph;
    }
}
