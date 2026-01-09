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

#include "rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp"
#include "rocRoller/KernelGraph/KernelGraph.hpp"
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace FuseExpressionsDetail
        {
            struct Candidate
            {
                int  tag;
                int  writingNode;
                int  readingNode;
                bool deleteTag = false;

                bool operator==(const Candidate& rhs) const = default;
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
             * @brief sort output from ControlFlowRWTracer by body parent
             *
             * @return Mapping from parent nodes to read-write records underneath them
             */
            std::unordered_map<int, std::vector<ControlFlowRWTracer::ReadWriteRecord>>
                sortRecordsByBodyParent(KernelGraph const&         kgraph,
                                        ControlFlowRWTracer const& tracer);

            /**
             * If a DataFlowTag is:
             * 1. written to only once
             * 2. read only once within that same body parent
             *
             * then those two control nodes comprise a candidate for FuseExpressions.
             */
            std::vector<Candidate> findFuseCandidates(KernelGraph const& kgraph);
        }
    }
}
