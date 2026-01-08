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

#include "rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp"
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
        std::vector<Candidate> findFuseCandidates(KernelGraph const& kgraph)
        {
            using RW     = ControlFlowRWTracer::ReadWrite;
            using Record = ControlFlowRWTracer::ReadWriteRecord;

            std::vector<Candidate> candidates;

            auto trace = ControlFlowRWTracer(kgraph).coordinatesReadWrite();

            // Create a map to hold all reads and writes under each body parent
            std::unordered_map<int, std::vector<ControlFlowRWTracer::ReadWriteRecord>> parents;
            for(auto record : trace)
            {
                // record: struct { int control, int coordinate, ReadWrite rw }
                if(record.rw != RW::Count)
                {
                    auto parent = bodyParents(record.control, kgraph).take(1).only();
                    AssertFatal(
                        parent.has_value(), "Node has no body parent", ShowValue(record.control));

                    parents[*parent].push_back(record);
                }
            }

            // For each parent, loop through reads and writes to find any data tags that are written to and read from exactly once
            for(const auto& [key, val] : parents)
            {
                // Maps DataFlowTags to the sequence of reads and writes that involve it within a single body parent
                std::unordered_map<int, std::vector<Record>> sequences;

                // Sort the trace by tag
                for(auto record : val)
                {
                    if(kgraph.control.get<Assign>(record.control).has_value())
                    {
                        // Add this record to the sequence corresponding to its tag
                        sequences[record.coordinate].push_back(record);
                    }
                }

                // For each tag, follow its sequence of reads and writes to find candidates
                for(const auto& [tag, sequence] : sequences)
                {
                    std::cout << "TAG " << tag << ":" << std::endl;
                    std::optional<int>       writingNode    = std::nullopt;
                    std::optional<int>       readingNode    = std::nullopt;
                    std::optional<Candidate> maybeCandidate = std::nullopt;
                    for(const auto& record : sequence)
                    {
                        auto node = record.control;

                        if(record.rw == RW::WRITE)
                        {
                            // If we have already written to and then read from this tag once each (indicating a possible candidate, see READ case),
                            // we have found a candidate!! We can save it as such and then start over with a new write
                            if(writingNode.has_value() && readingNode.has_value()
                               && maybeCandidate.has_value())
                            {
                                Candidate candidate = maybeCandidate.value();
                                std::cout << "Candidate found! Tag " << candidate.tag
                                          << " is written to by node " << candidate.writingNode
                                          << " and then read from by node " << candidate.readingNode
                                          << std::endl;

                                candidates.push_back(candidate);

                                readingNode    = std::nullopt;
                                maybeCandidate = std::nullopt;
                            }

                            // A write wipes the slate clean
                            writingNode    = node;
                            readingNode    = std::nullopt;
                            maybeCandidate = std::nullopt;
                            std::cout << "WRITE by node " << node << std::endl;
                        }
                        else if(record.rw == RW::READ)
                        {
                            std::cout << "READ by  node " << node << std::endl;

                            // If this tag has been written to but hasn't been read from yet, it may be a candidate for fusing
                            if(writingNode.has_value() && !readingNode.has_value())
                            {
                                maybeCandidate = {tag, writingNode.value(), node};
                            }
                            // Otherwise:
                            // - This tag may have been written to in an enclosing scope, or
                            // - This tag has already been read from
                            // In either case, this is not a candidate for fusing
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
                            std::cout << "READWRITE, splitting into two separate operations:\n";

                            // READ
                            std::cout << "READ by  node " << node << std::endl;
                            // If this tag has been written to but hasn't already been read from,
                            // and we know it is about to be written to again,
                            // we know that it's a candidate!
                            if(writingNode.has_value() && !readingNode.has_value())
                            {
                                Candidate candidate = {tag, writingNode.value(), node};
                                std::cout << "Candidate found! Tag " << candidate.tag
                                          << " is written to by node " << candidate.writingNode
                                          << " and then read from by node " << candidate.readingNode
                                          << std::endl;

                                candidates.push_back(candidate);

                                readingNode    = std::nullopt;
                                maybeCandidate = std::nullopt;
                            }

                            // WRITE
                            writingNode    = node;
                            readingNode    = std::nullopt;
                            maybeCandidate = std::nullopt;
                            std::cout << "WRITE by node " << node << std::endl;
                        }
                        else
                        {
                            Throw<FatalError>(
                                "Invalid value for ControlFlowRWTracer::ReadWrite: {}", record.rw);
                        }
                    }

                    // We've now finished iterating through all of the reads and writes of this tag in this body parent
                    // If we have a potential candidate, we can save it!
                    if(maybeCandidate.has_value())
                    {
                        Candidate candidate = maybeCandidate.value();
                        std::cout << "Candidate found! Tag " << candidate.tag
                                  << " is written to by node " << candidate.writingNode
                                  << " and then read from by node " << candidate.readingNode
                                  << ", tag can be deleted" << std::endl;

                        // Since we know that there won't be any more writes to this tag,
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
     * @brief Look for {Assign Multiply(., .)} --Sequence--> {Assign Add(., .)}.
     *
     * Look for
     *
     *   Assign Multiply(., .) -- Sequence --> Assign Add(., .)
     *
     * Make sure only one DF edge out of the result of the multiply.
     */
    std::optional<std::tuple<int, int>> findMultiplyAdd(KernelGraph const&             kgraph,
                                                        std::unordered_set<int> const& exclude)
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        for(auto const& parent : kgraph.control.getNodes<Assign>())
        {
            if(exclude.contains(parent))
                continue;

            // find Multiply
            auto multiplyAssign = kgraph.control.get<Assign>(parent);
            if(!std::holds_alternative<Expression::Multiply>(*multiplyAssign->expression))
                continue;

            auto child = only(kgraph.control.getOutputNodeIndices<Sequence>(parent));
            if(!child)
                continue;

            auto addAssign = kgraph.control.get<Assign>(*child);
            if(!addAssign)
                continue;

            auto dst = kgraph.mapper.get(parent, NaryArgument::DEST);
            AssertFatal(dst != -1, "Invalid connection.");
            auto dfs = only(kgraph.coordinates.getOutputNodeIndices(dst, CT::isEdge<CT::DataFlow>));
            if(!dfs)
                continue;

            if(!std::holds_alternative<Expression::Add>(*addAssign->expression))
                continue;

            return {{parent, *child}};
        }

        return {};
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

    /**
     * @brief Fuse Add(Multiply(., .), .) into MultiplyAdd(., ., .).
     */
    KernelGraph fuseMultiplyAdd(KernelGraph const& original)
    {
        using Expression::multiplyAdd;

        auto kgraph = original;

        std::unordered_set<int> excluded;

        for(;;)
        {
            auto candidate = findMultiplyAdd(kgraph, excluded);
            if(!candidate)
                break;

            auto const& [multiplyTag, addTag] = *candidate;
            excluded.emplace(multiplyTag);

            auto [addLHS, addLHSDF] = getBinaryLHS<Expression::Add>(kgraph, addTag);
            auto mulDST             = getDEST(kgraph, multiplyTag);

            if(addLHS != mulDST)
                continue;

            auto addDST             = getDEST(kgraph, addTag);
            auto [addRHS, addRHSDF] = getBinaryRHS<Expression::Add>(kgraph, addTag);
            auto [mulLHS, mulLHSDF] = getBinaryLHS<Expression::Multiply>(kgraph, multiplyTag);
            auto [mulRHS, mulRHSDF] = getBinaryRHS<Expression::Multiply>(kgraph, multiplyTag);
            auto fma                = multiplyAdd(mulLHSDF, mulRHSDF, addRHSDF);

            // Reuse register type and count from Multiply
            auto fmaAssign       = *kgraph.control.get<Assign>(multiplyTag);
            fmaAssign.expression = fma;

            auto fmaTag = kgraph.control.addElement(fmaAssign);

            // Connect FMA; delete Multiply and Add operations
            reconnect<Graph::Direction::Downstream>(kgraph, -1, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, addTag);
            reconnect<Graph::Direction::Downstream>(kgraph, fmaTag, addTag);

            kgraph.control.deleteElement(multiplyTag);
            kgraph.control.deleteElement(addTag);

            // Tidy up coordinate graph
            reflow(kgraph, addLHS, {addRHS});
            kgraph.coordinates.deleteElement(addLHS);

            // Tidy up connections
            kgraph.mapper.purge(multiplyTag);
            kgraph.mapper.purge(addTag);

            // Connect FMA
            kgraph.mapper.connect(fmaTag, addDST, NaryArgument::DEST);
        }

        return kgraph;
    }

    KernelGraph FuseExpressions::apply(KernelGraph const& original)
    {
        return fuseMultiplyAdd(original);
    }
}
