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

                // Handle block scaled tensors (gemm-spefic layout for when not pretiled)
                // ScaleA dimensions: [M, K/blockSize]
                // ScaleB dimensions: [K/blockSize, N]
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

                        // Only validate and match dimensions if scale has dimensions (i.e., not SingleScale)
                        // and is not pre tiled (4-dimensional)
                        if(scaleADims.size() == 2)
                        {
                            AssertFatal(aFreeDims.size() + aContractedDims.size() == 2,
                                        "ScaleA handling only supports GEMM tensor contraction",
                                        ShowValue(aFreeDims.size()),
                                        ShowValue(aContractedDims.size()));

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
                            Log::debug(
                                "IdentifyParallelDimensions: Matched {} ScaleA free dims with A",
                                aFreeDims.size());
                        }
                        else
                        {
                            Log::debug(
                                "IdentifyParallelDimensions: ScaleA is scalar (SingleScale mode), "
                                "or pre-tiled, skipping dimension matching");
                        }
                    }

                    // Validate ScaleB dimensions if present
                    if(maybeScaleB > 0)
                    {
                        scaleBDims = graph.coordinates
                                         .getInputNodeIndices(maybeScaleB, isConstructMacroTile)
                                         .to<std::vector>();

                        // Only validate and match dimensions if scale has dimensions (i.e., not SingleScale)
                        // and is not pre tiled (4-dimensional)
                        if(scaleBDims.size() == 2)
                        {
                            AssertFatal(bFreeDims.size() + bContractedDims.size() == 2,
                                        "ScaleB handling only supports GEMM tensor contraction",
                                        ShowValue(bFreeDims.size()),
                                        ShowValue(bContractedDims.size()));

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
                            Log::debug(
                                "IdentifyParallelDimensions: Matched {} ScaleB free dims with B",
                                bFreeDims.size());
                        }
                        else
                        {
                            Log::debug(
                                "IdentifyParallelDimensions: ScaleB is scalar (SingleScale mode), "
                                "or pre-tiled, skipping dimension matching");
                        }
                    }

                    // Match blocked contracted dimensions
                    if(maybeScaleA > 0 && maybeScaleB > 0 && scaleADims.size() == 2
                       && scaleBDims.size() == 2)
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

        /**
         * Visitor to recursively replace expression pointers with canonical ones
         */
        struct ReplaceExpressionsVisitor
        {
            std::vector<std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>> const&
                replacements;

            template <Expression::CUnary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CBinary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CTernary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CNary Expr>
            Expression::ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression::Expression>(std::move(cpy));
            }

            Expression::ExpressionPtr operator()(Expression::ScaledMatrixMultiply const& expr) const
            {
                Expression::ScaledMatrixMultiply cpy = expr;
                cpy.matA                             = call(expr.matA);
                cpy.matB                             = call(expr.matB);
                cpy.matC                             = call(expr.matC);
                cpy.scaleA                           = call(expr.scaleA);
                cpy.scaleB                           = call(expr.scaleB);
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <Expression::CValue Value>
            Expression::ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            Expression::ExpressionPtr call(Expression::ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                // Check if this entire expression should be replaced using identical()
                for(auto const& [target, replacement] : replacements)
                {
                    if(Expression::identical(expr, target))
                        return replacement;
                }

                // Otherwise, recursively process sub-expressions
                return std::visit(*this, *expr);
            }
        };

        /**
         * Visitor to apply expression replacements to all dimensions and operations in the graph
         */
        struct ReplaceInGraphVisitor
        {
            ReplaceExpressionsVisitor replaceVisitor;

            template <CoordinateGraph::CCoordinateTransformEdge T>
            CoordinateGraph::Edge visitCoordinateEdge(int tag, T const& edge)
            {
                return edge;
            }

            template <CoordinateGraph::CDataFlowEdge T>
            CoordinateGraph::Edge visitCoordinateEdge(int tag, T const& edge)
            {
                return edge;
            }

            template <CoordinateGraph::CDimension T>
            CoordinateGraph::Dimension visitDimension(int tag, T const& dim)
            {
                auto d   = dim;
                d.size   = replaceVisitor.call(dim.size);
                d.stride = replaceVisitor.call(dim.stride);
                d.offset = replaceVisitor.call(dim.offset);
                return d;
            }

            template <ControlGraph::COperation T>
            ControlGraph::Operation visitOperation(int tag, T const& op)
            {
                return op;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::Assign const& op)
            {
                auto newOp       = op;
                newOp.expression = replaceVisitor.call(op.expression);
                return newOp;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::ConditionalOp const& op)
            {
                auto newOp      = op;
                newOp.condition = replaceVisitor.call(op.condition);
                return newOp;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::AssertOp const& op)
            {
                auto newOp      = op;
                newOp.condition = replaceVisitor.call(op.condition);
                return newOp;
            }

            ControlGraph::Operation visitOperation(int tag, ControlGraph::ForLoopOp const& op)
            {
                auto newOp      = op;
                newOp.condition = replaceVisitor.call(op.condition);
                return newOp;
            }
        };

        KernelGraph IdentifyParallelDimensions::apply(KernelGraph const& original)
        {
            auto copy         = original;
            auto parallelDims = mergeSets(identifyParallelDimensionSets(copy));

            // Redundant expression to canonical expression
            std::vector<std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>>
                replacements;

            for(auto const& dimSet : parallelDims)
            {
                AssertFatal(dimSet.size() > 1,
                            "Parallel dimension sets should have at least 2 dimensions");

                Expression::ExpressionPtr canonicalSize = nullptr;

                for(int dim : dimSet)
                {
                    auto const& subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    if(subDim->size)
                    {
                        canonicalSize = subDim->size;
                        break;
                    }
                }

                for(int dim : dimSet)
                {
                    auto subDim = copy.coordinates.get<CoordinateGraph::SubDimension>(dim);
                    AssertFatal(subDim);

                    if(Expression::identical(subDim->size, canonicalSize))
                        continue;

                    replacements.push_back({subDim->size, canonicalSize});
                    subDim->size = canonicalSize;
                    copy.coordinates.setElement(dim, *subDim);
                }
            }

            // Replace all occurrences of non-canonical expressions in the graph
            if(!replacements.empty())
            {
                Log::debug(
                    "IdentifyParallelDimensions: Replacing {} expression(s) with canonical ones",
                    replacements.size());

                auto visitor = ReplaceInGraphVisitor{ReplaceExpressionsVisitor{replacements}};
                copy         = rewriteDimensions(copy, visitor);
            }

            return copy;
        }
    }
}
