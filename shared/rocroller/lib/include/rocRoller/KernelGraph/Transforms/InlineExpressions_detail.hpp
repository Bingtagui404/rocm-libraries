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

#pragma once

#include "rocRoller/Utilities/Error.hpp"
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/InlineExpressions.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace InlineExpressionsDetail
        {
            struct InliningCandidate
            {
                int  m_coordinate;
                int  m_writingNode;
                int  m_readingNode;
                bool m_deleteCoordinate;

                InliningCandidate(int coordinate,
                                  int writingNode,
                                  int readingNode,
                                  int deleteCoordinate)
                    : m_coordinate(coordinate)
                    , m_writingNode(writingNode)
                    , m_readingNode(readingNode)
                    , m_deleteCoordinate(deleteCoordinate)
                {
                }

                bool operator==(const InliningCandidate& rhs) const = default;
            };

            struct PossibleInliningCandidate
            {
                int m_coordinate;

                int m_writeParent;
                int m_writingNode;

                std::optional<int> m_readParent;
                std::optional<int> m_readingNode;

                PossibleInliningCandidate(int coordinate, int writeParent, int writingNode)
                    : m_coordinate(coordinate)
                    , m_writeParent(writeParent)
                    , m_writingNode(writingNode)
                    , m_readParent(std::nullopt)
                    , m_readingNode(std::nullopt)
                {
                }

                /*
                 * @brief Check if this possible candidate satisfies the conditions for being a true inlining candidate
                 * @return True if the candidate conditions are met and false otherwise
                 */
                bool isCandidate() const
                {
                    return m_readingNode.has_value() && m_readParent.has_value()
                           && m_writeParent == m_readParent.value();
                }

                /*
                 * @brief Assuming this possible candidate meets the conditions for an inlining candidate, create a new candidate struct
                 * @return a new InliningCandidate
                 */
                InliningCandidate createCandidate(bool deleteCoordinate) const
                {
                    AssertFatal(isCandidate(), "Cannot create candidate, conditions not met");
                    return InliningCandidate(
                        m_coordinate, m_writingNode, m_readingNode.value(), deleteCoordinate);
                }

                /*
                 * @brief Check if a new read will satisfy the conditions for an inlining candidate
                 * @return True if a new read to this coordinate will satisfy the conditions for a candidate, and false otherwise
                 */
                bool checkNewRead(int readParent, bool isAssignNode) const
                {
                    // We already know that we've written to this coordinate, so if we haven't yet read from it,
                    // and this new read comes from an assign node under the same body parent as the write, this will satisfy the conditions
                    return !m_readingNode.has_value() && m_writeParent == readParent
                           && isAssignNode;
                }

                /*
                 * @brief Add a read to this possible candidate
                 */
                void addRead(int readParent, int readingNode)
                {
                    AssertFatal(m_writeParent == readParent,
                                "Cannot add read to possible candidate, body parents must match",
                                ShowValue(m_writeParent),
                                ShowValue(readParent));
                    m_readParent  = readParent;
                    m_readingNode = readingNode;
                }

                bool operator==(const PossibleInliningCandidate& rhs) const = default;
            };

            /*
             * @brief Find DataFlowTag matching given tag in expression and replace it with a new expression
             * @return Pointer to new expression with replacement performed
             */
            Expression::ExpressionPtr
                replaceDataFlowTag(Expression::ExpressionPtr const& expr,
                                   int                              tag,
                                   Expression::ExpressionPtr const& exprToReplaceWith);

            /**
             * @brief Find candidates for expression inlining
             *
             * If a coordinate is:
             * 1. written to only once
             * 2. read only once within that same body parent
             *
             * then those two control nodes comprise a candidate for InlineExpressions.
             *
             * @return A list of candidates for InlineExpressions
             */
            std::vector<InliningCandidate> findInliningCandidates(KernelGraph const& kgraph);
        }
    }
}
