// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        template <std::predicate<CoordinateGraph::Dimension const&> NodePredicate,
                  std::predicate<CoordinateGraph::Edge const&>      EdgePredicate>
        std::set<int> getLeafNodesWithPredicates(CoordinateGraph::CoordinateGraph const& graph,
                                                 int                                     start,
                                                 NodePredicate nodePredicate,
                                                 EdgePredicate edgePredicate)
        {
            std::set<int> next;

            for(int node : graph.getOutputNodeIndices(start, edgePredicate))
            {
                if(nodePredicate(graph.getNode(node)))
                    next.insert(node);
            }

            if(next.empty())
            {
                return {start};
            }

            std::set<int> rv;

            for(int node : next)
            {
                auto nodeLeaves
                    = getLeafNodesWithPredicates(graph, node, nodePredicate, edgePredicate);
                rv.insert(nodeLeaves.begin(), nodeLeaves.end());
            }

            return rv;
        }

        std::set<int> loadNodesReachableWithoutDimensionModifyingNodes(
            ControlGraph::ControlGraph const& graph, int start)
        {
            auto isLoadTiled = [](ControlGraph::Operation const& op) {
                return std::holds_alternative<ControlGraph::LoadTiled>(op);
            };

            auto isSequenceEdge = [](ControlGraph::ControlEdge const& edge) {
                return std::holds_alternative<ControlGraph::Sequence>(edge);
            };

            auto isNotDimensionModifyingNode = [](ControlGraph::Operation const& op) {
                return !std::holds_alternative<ControlGraph::TensorContraction>(op);
            };

            auto sameDimensionLoadTiledNodes
                = reachableNodes<Graph::Direction::Upstream>(
                      graph, start, isNotDimensionModifyingNode, isSequenceEdge, isLoadTiled)
                      .to<std::set>();
            return sameDimensionLoadTiledNodes;
        }

        struct RedundantCommandArgsVisitor
        {

            template <typename Op>
            requires CIsAnyOf<Op,
                              ControlGraph::AssertOp,
                              ControlGraph::Assign,
                              ControlGraph::Barrier,
                              ControlGraph::Block,
                              ControlGraph::ConditionalOp,
                              ControlGraph::Deallocate,
                              ControlGraph::DoWhileOp,
                              ControlGraph::Exchange,
                              ControlGraph::ForLoopOp,
                              ControlGraph::Kernel,
                              ControlGraph::LoadLDSTile,
                              ControlGraph::LoadLinear,
                              ControlGraph::LoadSGPR,
                              ControlGraph::LoadTiled,
                              ControlGraph::LoadVGPR,
                              ControlGraph::LoadTileDirect2LDS,
                              ControlGraph::Multiply,
                              ControlGraph::NOP,
                              ControlGraph::Scope,
                              ControlGraph::SeedPRNG,
                              ControlGraph::SetCoordinate,
                              ControlGraph::StoreLDSTile,
                              ControlGraph::StoreLinear,
                              ControlGraph::StoreSGPR,
                              //   ControlGraph::StoreTiled,
                              ControlGraph::StoreVGPR,
                              //   ControlGraph::TensorContraction,
                              ControlGraph::UnrollOp,
                              ControlGraph::WaitZero>
            void operator()(int nodeID, Op const& op) {}

            void operator()(int nodeID, ControlGraph::StoreTiled const& op)
            {
                auto storeTile = graph.mapper.get<CoordinateGraph::MacroTile>(nodeID);
                auto isDestructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::DestructMacroTile>;
                auto storeDims
                    = graph.coordinates.getOutputNodeIndices(storeTile, isDestructMacroTile)
                          .to<std::vector>();

                auto isConstructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::ConstructMacroTile>;

                auto sameDimensionLoadTiledNodes
                    = loadNodesReachableWithoutDimensionModifyingNodes(graph.control, nodeID);

                Log::debug(
                    "IdentifyParallelDimensions::StoreTiled: store {} has {} reachable loads",
                    nodeID,
                    sameDimensionLoadTiledNodes.size());

                for(int loadID : sameDimensionLoadTiledNodes)
                {
                    auto loadTile = graph.mapper.get<CoordinateGraph::MacroTile>(loadID);
                    auto loadDims
                        = graph.coordinates.getInputNodeIndices(loadTile, isConstructMacroTile)
                              .to<std::vector>();

                    Log::debug("  pairing load {} dims (size={}) with store dims (size={})",
                               loadID,
                               loadDims.size(),
                               storeDims.size());

                    AssertFatal(loadDims.size() == storeDims.size(),
                                ShowValue(loadDims.size()),
                                ShowValue(storeDims.size()));

                    for(size_t i = 0; i < loadDims.size(); i++)
                    {
                        Log::debug("    dimension pair: load[{}]={} ↔ store[{}]={}",
                                   i,
                                   loadDims.at(i),
                                   i,
                                   storeDims.at(i));
                        redundantArgs.push_back({loadDims.at(i), storeDims.at(i)});
                    }
                }
            }

            void operator()(int nodeID, ControlGraph::TensorContraction const& op)
            {
                auto D = graph.mapper.get(nodeID, NaryArgument::DEST);
                AssertFatal(D > 0, ShowValue(D));

                auto A = graph.mapper.get(nodeID, NaryArgument::LHS);
                auto B = graph.mapper.get(nodeID, NaryArgument::RHS);
                AssertFatal(A > 0, ShowValue(A));
                AssertFatal(B > 0, ShowValue(B));

                auto isConstructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::ConstructMacroTile>;

                auto notFixedSize = [&](int tag) -> bool {
                    auto size = getSize(graph.coordinates.getNode(tag));
                    return !Expression::evaluationTimes(
                        size)[Expression::EvaluationTime::Translate];
                };

                auto aTileDims = graph.coordinates.getInputNodeIndices(A, isConstructMacroTile)
                                     .filter(notFixedSize)
                                     .to<std::vector>();
                auto bTileDims = graph.coordinates.getInputNodeIndices(B, isConstructMacroTile)
                                     .filter(notFixedSize)
                                     .to<std::vector>();

                AssertFatal(aTileDims.size() == bTileDims.size());

                AssertFatal(op.aDims.size() == op.bDims.size(),
                            ShowValue(op.aDims.size()),
                            ShowValue(op.bDims.size()));

                // Separate dimensions into free and contracted
                // For standard GEMM: aFreeDims=[M], aContractedDims=[K], bFreeDims=[N], bContractedDims=[K]
                std::set<size_t> aContractedIndices(op.aDims.begin(), op.aDims.end());
                std::set<size_t> bContractedIndices(op.bDims.begin(), op.bDims.end());

                std::vector<int> aFreeDims;
                std::vector<int> aContractedDims;
                std::vector<int> bFreeDims;
                std::vector<int> bContractedDims;

                for(size_t i = 0; i < aTileDims.size(); i++)
                {
                    if(aContractedIndices.contains(i))
                        aContractedDims.push_back(aTileDims[i]);
                    else
                        aFreeDims.push_back(aTileDims[i]);
                }

                for(size_t i = 0; i < bTileDims.size(); i++)
                {
                    if(bContractedIndices.contains(i))
                        bContractedDims.push_back(bTileDims[i]);
                    else
                        bFreeDims.push_back(bTileDims[i]);
                }

                // Match contracted dimensions between A and B
                AssertFatal(aContractedDims.size() == bContractedDims.size(),
                            ShowValue(aContractedDims.size()),
                            ShowValue(bContractedDims.size()));

                for(size_t i = 0; i < aContractedDims.size(); i++)
                    redundantArgs.push_back({aContractedDims[i], bContractedDims[i]});

                // Match free dimensions with output D
                auto isDataFlowEdge = CoordinateGraph::isEdge<CoordinateGraph::DataFlow>;
                auto isMacroTile    = [](CoordinateGraph::Dimension const& dim) {
                    return std::holds_alternative<CoordinateGraph::MacroTile>(dim);
                };

                auto finalDMacroTiles
                    = getLeafNodesWithPredicates(graph.coordinates, D, isMacroTile, isDataFlowEdge);

                AssertFatal(finalDMacroTiles.size() == 1, ShowValue(finalDMacroTiles.size()));

                auto isDestructMacroTile
                    = CoordinateGraph::isEdge<CoordinateGraph::DestructMacroTile>;
                for(int dTile : finalDMacroTiles)
                {
                    auto dTileDims
                        = graph.coordinates.getOutputNodeIndices(dTile, isDestructMacroTile)
                              .to<std::vector>();

                    size_t expectedDSize = aFreeDims.size() + bFreeDims.size();
                    AssertFatal(dTileDims.size() == expectedDSize,
                                ShowValue(dTileDims.size()),
                                ShowValue(expectedDSize),
                                ShowValue(aFreeDims.size()),
                                ShowValue(bFreeDims.size()));

                    // Match A's free dimensions to D's first dimensions
                    for(size_t i = 0; i < aFreeDims.size(); i++)
                    {
                        redundantArgs.push_back({aFreeDims[i], dTileDims[i]});
                    }

                    // Match B's free dimensions to D's remaining dimensions
                    for(size_t i = 0; i < bFreeDims.size(); i++)
                    {
                        redundantArgs.push_back({bFreeDims[i], dTileDims[aFreeDims.size() + i]});
                    }
                }

                // Handle block-scaled matrix multiplication scale tensors
                // ScaleA structure: [A_free_dims..., blocked_contracted_dims...]
                // ScaleB structure: [blocked_contracted_dims..., B_free_dims...]
                auto maybeScaleA = graph.mapper.get(nodeID, NaryArgument::LHS_SCALE);
                auto maybeScaleB = graph.mapper.get(nodeID, NaryArgument::RHS_SCALE);

                if(maybeScaleA > 0 || maybeScaleB > 0)
                {
                    std::vector<int> scaleADims, scaleBDims;

                    // Validate ScaleA dimensions if present
                    if(maybeScaleA > 0)
                    {
                        scaleADims = graph.coordinates
                                         .getInputNodeIndices(maybeScaleA, isConstructMacroTile)
                                         .to<std::vector>();

                        size_t expectedScaleASize = aFreeDims.size() + aContractedDims.size();
                        AssertFatal(scaleADims.size() == expectedScaleASize,
                                    ShowValue(scaleADims.size()),
                                    ShowValue(expectedScaleASize),
                                    ShowValue(aFreeDims.size()),
                                    ShowValue(aContractedDims.size()));

                        // Match ScaleA's free dimensions with A's free dimensions
                        for(size_t i = 0; i < aFreeDims.size(); i++)
                        {
                            redundantArgs.push_back({scaleADims[i], aFreeDims[i]});
                        }
                        Log::debug("IdentifyParallelDimensions: Matched {} ScaleA free dims with A",
                                   aFreeDims.size());
                    }

                    // Validate ScaleB dimensions if present
                    if(maybeScaleB > 0)
                    {
                        scaleBDims = graph.coordinates
                                         .getInputNodeIndices(maybeScaleB, isConstructMacroTile)
                                         .to<std::vector>();

                        size_t expectedScaleBSize = bContractedDims.size() + bFreeDims.size();
                        AssertFatal(scaleBDims.size() == expectedScaleBSize,
                                    ShowValue(scaleBDims.size()),
                                    ShowValue(expectedScaleBSize),
                                    ShowValue(bContractedDims.size()),
                                    ShowValue(bFreeDims.size()));

                        // Match ScaleB's free dimensions with B's free dimensions
                        for(size_t i = 0; i < bFreeDims.size(); i++)
                        {
                            size_t scaleBIdx = bContractedDims.size() + i;
                            redundantArgs.push_back({scaleBDims[scaleBIdx], bFreeDims[i]});
                        }
                        Log::debug("IdentifyParallelDimensions: Matched {} ScaleB free dims with B",
                                   bFreeDims.size());
                    }

                    // Match blocked contracted dimensions when both scales exist
                    if(maybeScaleA > 0 && maybeScaleB > 0)
                    {
                        for(size_t i = 0; i < aContractedDims.size(); i++)
                        {
                            size_t scaleAIdx = aFreeDims.size() + i;
                            size_t scaleBIdx = i;
                            redundantArgs.push_back({scaleADims[scaleAIdx], scaleBDims[scaleBIdx]});
                        }
                        Log::debug("IdentifyParallelDimensions: Matched {} blocked contracted dims "
                                   "between "
                                   "ScaleA and ScaleB",
                                   aContractedDims.size());
                    }
                }
            }

            void call(std::variant<int> nodeID, ControlGraph::Operation const& op)
            {
                std::visit(*this, nodeID, op);
            }

            KernelGraph const&         graph;
            std::vector<std::set<int>> redundantArgs;
        };

        std::vector<std::set<int>> identifyParallelDimensionSets(KernelGraph const& graph)
        {
            RedundantCommandArgsVisitor visitor{graph};

            for(auto nodeID : graph.control.getNodes())
            {
                visitor.call(nodeID, graph.control.getNode(nodeID));
            }

            return visitor.redundantArgs;
        }

        KernelGraph IdentifyParallelDimensions::apply(KernelGraph const& original)
        {
            auto copy = original;

            auto parallelDims = mergeSets(identifyParallelDimensionSets(copy));

            for(auto const& dimSet : parallelDims)
            {
                Expression::ExpressionPtr dimSize;

                for(int dim : dimSet)
                {
                    auto const& subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    if(subDim->size)
                    {
                        dimSize = subDim->size;
                        break;
                    }
                }

                AssertFatal(dimSize);

                for(int dim : dimSet)
                {
                    auto subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    subDim->size = dimSize;
                    copy.coordinates.setElement(dim, *subDim);
                }
            }

            // This pass recomputes User.size (tensor limit) using the merged SubDimension
            // sizes, eliminating redundant kernel arguments.
            for(auto userTag : copy.coordinates.getNodes())
            {
                auto user = copy.coordinates.get<CoordinateGraph::User>(userTag);
                if(!user || !user->size)
                    continue;

                Log::debug(
                    "IdentifyParallelDimensions: Checking User {} for SubDimension connections",
                    userTag);

                // Find SubDimensions connected to this User
                // Pattern 1 (input tensors): User → Split → SubDimensions
                // Pattern 2 (output tensors): SubDimensions → Join → User
                std::vector<int> subdims;

                // Try Split edges (input tensors: A, B, C)
                subdims = copy.coordinates
                              .getOutputNodeIndices(userTag,
                                                    CoordinateGraph::isEdge<CoordinateGraph::Split>)
                              .to<std::vector>();

                // Try Join edges (output tensors: D)
                if(subdims.empty())
                {
                    subdims = copy.coordinates
                                  .getInputNodeIndices(
                                      userTag, CoordinateGraph::isEdge<CoordinateGraph::Join>)
                                  .to<std::vector>();
                }

                if(subdims.empty())
                {
                    Log::debug("  No SubDimensions found via Split or Join edges");
                    Log::debug("  User may be scratch space (LDS tensors created later by AddLDS)");
                    continue;
                }

                Log::debug("  Found {} SubDimensions", subdims.size());

                // Recompute User size using merged SubDimension expressions
                std::vector<Expression::ExpressionPtr> sizes, strides;
                for(auto subdimTag : subdims)
                {
                    auto subdim = copy.coordinates.get<CoordinateGraph::SubDimension>(subdimTag);
                    AssertFatal(subdim && subdim->size && subdim->stride,
                                ShowValue(subdimTag),
                                "SubDimension node missing size or stride");
                    sizes.push_back(subdim->size);
                    strides.push_back(subdim->stride);
                }

                auto newSize = computeUserSize(sizes, strides);
                user->size   = newSize;
                copy.coordinates.setElement(userTag, *user);
                Log::debug("IdentifyParallelDimensions: Updated User {} size expression using {} "
                           "SubDimensions",
                           userTag,
                           subdims.size());
            }

            return copy;
        }
    }
}
