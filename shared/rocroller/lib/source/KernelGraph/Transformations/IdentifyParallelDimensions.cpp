// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>

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

                std::set<int> remainingADims(aTileDims.begin(), aTileDims.end());
                std::set<int> remainingBDims(bTileDims.begin(), bTileDims.end());

                for(size_t i = 0; i < op.aDims.size(); i++)
                {
                    auto aDim = aTileDims.at(op.aDims.at(i));
                    auto bDim = bTileDims.at(op.bDims.at(i));

                    redundantArgs.push_back({aDim, bDim});
                    remainingADims.erase(aDim);
                    remainingBDims.erase(bDim);
                }

                AssertFatal(remainingADims.size() == 1, ShowValue(remainingADims.size()));
                AssertFatal(remainingBDims.size() == 1, ShowValue(remainingBDims.size()));

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

                    AssertFatal(dTileDims.size() == 2, ShowValue(dTileDims.size()));

                    redundantArgs.push_back({*remainingADims.begin(), dTileDims.at(0)});
                    redundantArgs.push_back({*remainingBDims.begin(), dTileDims.at(1)});
                }

                // Handle block-scaled matrix multiplication scale tensors
                // ScaleA dimensions: [M, K/blockSize]
                // ScaleB dimensions: [K/blockSize, N]
                auto maybeScaleA = graph.mapper.get(nodeID, NaryArgument::LHS_SCALE);
                auto maybeScaleB = graph.mapper.get(nodeID, NaryArgument::RHS_SCALE);

                if(maybeScaleA > 0)
                {
                    // maybeScaleA is a MacroTile coordinate tag (same as A, B above)
                    auto scaleADims
                        = graph.coordinates.getInputNodeIndices(maybeScaleA, isConstructMacroTile)
                              .to<std::vector>();

                    Log::debug("IdentifyParallelDimensions: Found ScaleA with {} dims",
                               scaleADims.size());

                    // ScaleA[0] (M dimension) should match A[0] and D[0]
                    if(scaleADims.size() >= 1 && aTileDims.size() >= 1)
                    {
                        redundantArgs.push_back({scaleADims[0], aTileDims[0]});
                        Log::debug("  matched ScaleA[0] with A[0] (M dimension)");
                    }

                    // If both scale tensors exist, match their K/blockSize dimensions
                    if(maybeScaleB > 0 && scaleADims.size() >= 2)
                    {
                        auto scaleBDims
                            = graph.coordinates
                                  .getInputNodeIndices(maybeScaleB, isConstructMacroTile)
                                  .to<std::vector>();

                        Log::debug("IdentifyParallelDimensions: Found ScaleB with {} dims",
                                   scaleBDims.size());

                        // ScaleA[1] and ScaleB[0] both represent K/blockSize
                        if(scaleBDims.size() >= 1)
                        {
                            redundantArgs.push_back({scaleADims[1], scaleBDims[0]});
                            Log::debug(
                                "  matched ScaleA[1] with ScaleB[0] (K/blockSize dimension)");
                        }

                        // ScaleB[1] (N dimension) should match B[1] and D[1]
                        if(scaleBDims.size() >= 2 && bTileDims.size() >= 2)
                        {
                            redundantArgs.push_back({scaleBDims[1], bTileDims[1]});
                            Log::debug("  matched ScaleB[1] with B[1] (N dimension)");
                        }
                    }
                }
                else if(maybeScaleB > 0)
                {
                    // Only ScaleB exists (B is scaled, A is not)
                    auto scaleBDims
                        = graph.coordinates.getInputNodeIndices(maybeScaleB, isConstructMacroTile)
                              .to<std::vector>();

                    Log::debug("IdentifyParallelDimensions: Found ScaleB with {} dims (no ScaleA)",
                               scaleBDims.size());

                    // ScaleB[1] (N dimension) should match B[1] and D[1]
                    if(scaleBDims.size() >= 2 && bTileDims.size() >= 2)
                    {
                        redundantArgs.push_back({scaleBDims[1], bTileDims[1]});
                        Log::debug("  matched ScaleB[1] with B[1] (N dimension)");
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

            // Update User coordinates to use merged SubDimension sizes
            // After merging parallel SubDimensions above, tensors may have User size expressions
            // referencing SubDimension sizes that were merged. This pass recomputes User sizes
            // using the merged SubDimension sizes, eliminating redundant kernel arguments.
            //
            // Example: Matrix D's User initially references Tensor_D_size_0 and Tensor_D_size_1.
            //          After merging, D's SubDimensions now use Tensor_A_size_0 (M) and
            //          Tensor_B_size_1 (N). This pass updates D's User size expression to
            //          reference those merged sizes, allowing Tensor_D_size_* to be eliminated.
            for(auto userTag : copy.coordinates.getNodes())
            {
                auto user = copy.coordinates.get<CoordinateGraph::User>(userTag);
                if(!user || !user->size)
                    continue;

                Log::debug("IdentifyParallelDimensions: Checking User {} for SubDimension connections",
                           userTag);

                // Find SubDimensions connected to this User
                // Pattern 1 (input tensors): User → Split → SubDimensions
                // Pattern 2 (output tensors): SubDimensions → Join → User
                std::vector<int> subdims;

                // Try Split edges (input tensors: A, B, C)
                subdims = copy.coordinates
                              .getOutputNodeIndices(
                                  userTag, CoordinateGraph::isEdge<CoordinateGraph::Split>)
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
                    continue;
                }

                Log::debug("  Found {} SubDimensions", subdims.size());

                // Recompute User size using merged SubDimension expressions
                // Formula: 1 + Σ(stride[i] * (size[i] - 1))
                auto newSize  = Expression::literal(1u);
                bool allValid = true;

                for(auto subdimTag : subdims)
                {
                    auto subdim = copy.coordinates.get<CoordinateGraph::SubDimension>(subdimTag);
                    if(!subdim || !subdim->size || !subdim->stride)
                    {
                        Log::debug("  SubDimension {} missing size or stride", subdimTag);
                        allValid = false;
                        break;
                    }

                    auto contribution = subdim->stride * (subdim->size - Expression::literal(1u));
                    newSize           = newSize + contribution;
                }

                if(allValid)
                {
                    user->size = newSize;
                    copy.coordinates.setElement(userTag, *user);
                    Log::debug(
                        "IdentifyParallelDimensions: Updated User {} size expression using {} "
                        "SubDimensions",
                        userTag,
                        subdims.size());
                }
            }

            return copy;
        }
    }
}
