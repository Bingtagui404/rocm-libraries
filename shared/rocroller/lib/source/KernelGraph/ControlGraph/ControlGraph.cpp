// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <bitset>
#include <cmath>
#include <iomanip>

namespace rocRoller::KernelGraph::ControlGraph
{
    void bitsetSet(std::vector<uint64_t>& bs, int id)
    {
        size_t word = static_cast<size_t>(id) >> 6;
        if(word >= bs.size())
            bs.resize(word + 1, 0);
        bs[word] |= uint64_t(1) << (id & 63);
    }

    void bitsetOr(std::vector<uint64_t>& dst, std::vector<uint64_t> const& src)
    {
        if(dst.size() < src.size())
            dst.resize(src.size(), 0);
        for(size_t i = 0; i < src.size(); ++i)
            dst[i] |= src[i];
    }

    bool bitsetAny(std::vector<uint64_t> const& bs)
    {
        return std::ranges::any_of(bs, [](auto const& w) { return w != 0; });
    }

    std::vector<int> bitsetToSortedVec(std::vector<uint64_t> const& bs)
    {
        std::vector<int> result;
        for(size_t w = 0; w < bs.size(); ++w)
        {
            uint64_t bits = bs[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                result.push_back(static_cast<int>(w * 64 + bit));
                bits &= bits - 1;
            }
        }
        return result;
    }

    std::unordered_map<int, std::unordered_map<int, NodeOrdering>>
        ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();
        std::unordered_map<int, std::unordered_map<int, NodeOrdering>> table;
        for(auto const& [node, orders] : m_orderCache)
        {
            for(int other : bitsetToSortedVec(orders.after))
            {
                if(node < other)
                    table[node][other] = NodeOrdering::LeftFirst;
            }
            for(int other : bitsetToSortedVec(orders.inBody))
            {
                if(node < other)
                    table[node][other] = NodeOrdering::RightInBodyOfLeft;
                else
                    table[other][node] = NodeOrdering::LeftInBodyOfRight;
            }
        }
        return table;
    }

    std::string ControlGraph::nodeOrderTableString(std::set<int> const& nodes) const
    {
        populateOrderCache();
        if(nodes.empty())
        {
            return "Empty order cache.\n";
        }

        std::ostringstream msg;

        int width = std::ceil(std::log10(static_cast<float>(*nodes.rbegin())));
        width     = std::max(width, 3);

        msg << std::setw(width) << " "
            << "\\";

        for(int n : nodes)
            msg << " " << std::setw(width) << n;

        for(int i : nodes)
        {
            msg << std::endl << std::setw(width) << i << "|";
            for(int j : nodes)
            {
                if(i == j)
                {
                    msg << " ";
                    auto oldFill = msg.fill('-');
                    msg << std::setw(width) << "-";
                    msg.fill(oldFill);
                }
                else
                {
                    msg << " " << std::setw(width) << abbrev(lookupOrder(CacheOnly, i, j));
                }
            }
            msg << " | " << std::setw(width) << i;
        }
        msg << std::endl
            << std::setw(width) << " "
            << "|";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;
        msg << std::endl;
        return msg.str();
    }

    std::string ControlGraph::nodeOrderTableString() const
    {
        populateOrderCache();
        TIMER(t, "nodeOrderTable");
        std::set<int> nodes;
        for(auto const& [node, orders] : m_orderCache)
        {
            if(bitsetAny(orders.after) || bitsetAny(orders.before) || bitsetAny(orders.inBody)
               || bitsetAny(orders.containing))
            {
                nodes.insert(node);
            }
            for(int n : bitsetToSortedVec(orders.after))
                nodes.insert(n);
            for(int n : bitsetToSortedVec(orders.before))
                nodes.insert(n);
            for(int n : bitsetToSortedVec(orders.inBody))
                nodes.insert(n);
            for(int n : bitsetToSortedVec(orders.containing))
                nodes.insert(n);
        }
        return nodeOrderTableString(nodes);
    }

    void ControlGraph::clearCache(Graph::GraphModification modification)
    {
        Hypergraph<Operation, ControlEdge, false>::clearCache(modification);

        if(modification == Graph::GraphModification::AddElement
           && m_cacheStatus != CacheStatus::Invalid)
        {
            // If adding a new element and order is non-empty (partial or valid)
            m_cacheStatus = CacheStatus::Partial;
        }
        else
        {
            m_orderCache.clear();
            m_cacheStatus = CacheStatus::Invalid;
        }

        m_descendentCache.clear();
    }

    bool ControlGraph::isModificationAllowed(int index) const
    {
        if(not Settings::getInstance()->get(Settings::EnforceGraphConstraints))
            return true;

        if(not m_changesRestricted)
            return true;

        auto const& el = getElement(index);

        if(std::holds_alternative<Operation>(el))
        {
            return std::visit(
                [](auto&& arg) {
                    using OpType = std::decay_t<decltype(arg)>;
                    return !(
                        std::is_same_v<OpType, ForLoopOp> or std::is_same_v<OpType, SetCoordinate>);
                },
                std::get<Operation>(el));
        }
        else
        {
            //
            // Theoretically, add/delete Body edge should be disallowed. But sometimes
            // delete Body edges is OK (e.g., Simplify), and currently there is no way
            // to know if this is called in a valid or invalid use case.
            //
            return true;
        }
    }

    void ControlGraph::validateOrderCache() const
    {
        auto bitsetOverlaps = [](std::vector<uint64_t> const& a, std::vector<uint64_t> const& b) {
            size_t n = std::min(a.size(), b.size());
            for(size_t i = 0; i < n; ++i)
                if(a[i] & b[i])
                    return true;
            return false;
        };

        for(auto const& [node, orders] : m_orderCache)
        {
            AssertFatal(!bitsetOverlaps(orders.after, orders.before),
                        "Node has conflicting orders (after & before)");
            AssertFatal(!bitsetOverlaps(orders.after, orders.inBody),
                        "Node has conflicting orders (after & inBody)");
            AssertFatal(!bitsetOverlaps(orders.after, orders.containing),
                        "Node has conflicting orders (after & containing)");
            AssertFatal(!bitsetOverlaps(orders.before, orders.inBody),
                        "Node has conflicting orders (before & inBody)");
            AssertFatal(!bitsetOverlaps(orders.before, orders.containing),
                        "Node has conflicting orders (before & containing)");
            AssertFatal(!bitsetOverlaps(orders.inBody, orders.containing),
                        "Node has conflicting orders (inBody & containing)");
        }
    }

    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");

        if(m_cacheStatus == CacheStatus::Valid)
            return;

        m_orderCache.clear();

        auto rootNodes = roots().to<std::vector>();

        populateOrderCache(rootNodes);
        validateOrderCache();

        m_cacheStatus = CacheStatus::Valid;
        //
        // m_descendentCache is only used to help build m_orderCache,
        // and it must be cleared after finish building m_orderCache
        // to ensure no stale data being used when building m_orderCache
        // next time.
        //
        m_descendentCache.clear();
    }

    template <CForwardRangeOf<int> Range>
    std::vector<uint64_t> ControlGraph::populateOrderCache(Range const& startingNodes) const
    {
        std::vector<uint64_t> rv;
        for(auto it = startingNodes.begin(); it != startingNodes.end(); ++it)
        {
            auto const nodes = populateOrderCache(*it);
            bitsetOr(rv, nodes);
        }
        return rv;
    }

    std::vector<uint64_t> ControlGraph::populateOrderCache(int startingNode) const
    {

        auto ccEntry = m_descendentCache.find(startingNode);
        if(ccEntry != m_descendentCache.end())
            return ccEntry->second;
        static_assert(std::variant_size_v<ControlEdge> == 5,
                      "Currently the available edge types are Sequence(0), Initialize(1), "
                      "ForLoopIncrement(2), Body(3) and Else(4)."
                      "If more edge types are added, this function has to be updated.");
        using GD = Graph::Direction;
        std::array<std::vector<int>, std::variant_size_v<ControlEdge>> directChildren;
        for(auto edge : getNeighbours<GD::Downstream>(startingNode))
        {
            auto edgeTypeIndex = getEdge(edge).index();
            for(auto child : getNeighbours<GD::Downstream>(edge))
                directChildren[edgeTypeIndex].push_back(child);
        }
        auto addDescendents = [this](std::vector<int> const& children) -> std::vector<uint64_t> {
            std::vector<uint64_t> descendents = populateOrderCache(children);
            std::vector<uint64_t> result      = descendents;
            for(int c : children)
                bitsetSet(result, c);
            return result;
        };
        auto initNodes = addDescendents(directChildren[1]);
        auto bodyNodes = addDescendents(directChildren[3]);
        auto elseNodes = addDescendents(directChildren[4]);
        auto incNodes  = addDescendents(directChildren[2]);
        auto seqNodes  = addDescendents(directChildren[0]);

        writeOrderCache(startingNode, initNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache(startingNode, bodyNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache(startingNode, elseNodes, NodeOrdering::RightInBodyOfLeft);
        writeOrderCache(startingNode, incNodes, NodeOrdering::RightInBodyOfLeft);

        writeOrderCache(startingNode, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, bodyNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, elseNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(initNodes, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, elseNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(bodyNodes, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(elseNodes, incNodes, NodeOrdering::LeftFirst);
        writeOrderCache(elseNodes, seqNodes, NodeOrdering::LeftFirst);
        writeOrderCache(incNodes, seqNodes, NodeOrdering::LeftFirst);

        std::vector<uint64_t> allNodes = std::move(initNodes);
        bitsetOr(allNodes, bodyNodes);
        bitsetOr(allNodes, elseNodes);
        bitsetOr(allNodes, incNodes);
        bitsetOr(allNodes, seqNodes);
        m_descendentCache[startingNode] = allNodes;
        return allNodes;
    }

    void ControlGraph::writeOrderCache(std::vector<uint64_t> const& nodesA,
                                       std::vector<uint64_t> const& nodesB,
                                       NodeOrdering                 order) const
    {
        if(!bitsetAny(nodesA) || !bitsetAny(nodesB))
            return;
        auto selectOrder = [](NodeOrders& orders, NodeOrdering order) -> std::vector<uint64_t>& {
            switch(order)
            {
            case NodeOrdering::LeftFirst:
                return orders.after;
            case NodeOrdering::RightFirst:
                return orders.before;
            case NodeOrdering::RightInBodyOfLeft:
                return orders.inBody;
            case NodeOrdering::LeftInBodyOfRight:
                return orders.containing;
            default:
                break;
            }
            AssertFatal(false, "Invalid order: ", ShowValue(order));
            return orders.after;
        };
        for(size_t w = 0; w < nodesA.size(); ++w)
        {
            uint64_t bits = nodesA[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int a   = static_cast<int>(w * 64 + bit);
                bitsetOr(selectOrder(m_orderCache[a], order), nodesB);
                bits &= bits - 1;
            }
        }
        auto oppositeOrder = opposite(order);
        for(size_t w = 0; w < nodesB.size(); ++w)
        {
            uint64_t bits = nodesB[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int b   = static_cast<int>(w * 64 + bit);
                bitsetOr(selectOrder(m_orderCache[b], oppositeOrder), nodesA);
                bits &= bits - 1;
            }
        }
    }

    void ControlGraph::writeOrderCache(int                          nodeA,
                                       std::vector<uint64_t> const& nodesB,
                                       NodeOrdering                 order) const
    {
        if(!bitsetAny(nodesB))
            return;
        auto selectOrder = [](NodeOrders& orders, NodeOrdering order) -> std::vector<uint64_t>& {
            switch(order)
            {
            case NodeOrdering::LeftFirst:
                return orders.after;
            case NodeOrdering::RightFirst:
                return orders.before;
            case NodeOrdering::RightInBodyOfLeft:
                return orders.inBody;
            case NodeOrdering::LeftInBodyOfRight:
                return orders.containing;
            default:
                break;
            }
            AssertFatal(false, "Invalid order: ", ShowValue(order));
            return orders.after;
        };
        bitsetOr(selectOrder(m_orderCache[nodeA], order), nodesB);
        auto oppositeOrder = opposite(order);
        for(size_t w = 0; w < nodesB.size(); ++w)
        {
            uint64_t bits = nodesB[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int b   = static_cast<int>(w * 64 + bit);
                bitsetSet(selectOrder(m_orderCache[b], oppositeOrder), nodeA);
                bits &= bits - 1;
            }
        }
    }

    void ControlGraph::writeOrderCache(int nodeA, int nodeB, NodeOrdering order) const
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            bitsetSet(m_orderCache[nodeA].after, nodeB);
            bitsetSet(m_orderCache[nodeB].before, nodeA);
            break;
        case NodeOrdering::RightFirst:
            bitsetSet(m_orderCache[nodeA].before, nodeB);
            bitsetSet(m_orderCache[nodeB].after, nodeA);
            break;
        case NodeOrdering::RightInBodyOfLeft:
            bitsetSet(m_orderCache[nodeA].inBody, nodeB);
            bitsetSet(m_orderCache[nodeB].containing, nodeA);
            break;
        case NodeOrdering::LeftInBodyOfRight:
            bitsetSet(m_orderCache[nodeA].containing, nodeB);
            bitsetSet(m_orderCache[nodeB].inBody, nodeA);
            break;
        default:
            break;
        }
    }

    NodeOrdering ControlGraph::lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        TIMER(t, "ControlGraph::lookupOrder");

        using GD = Graph::Direction;
        std::unordered_set<int> visited_nodes;

        // Decide order of A and B when A is parent of B
        auto const getOrderIfParent = [&](int edge) {
            return std::holds_alternative<Sequence>(getEdge(edge))
                       ? NodeOrdering::LeftFirst
                       : NodeOrdering::RightInBodyOfLeft;
        };

        // Decide order of A and B when A and B are descendants of a node
        // And order of edge type :
        // Initialize -> Body -> Else -> ForLoopIncrement -> Sequence.
        auto const getOrderOfDescendants = [&](int edgeA, int edgeB) {
            auto edgeAElem = getEdge(edgeA);
            auto edgeBElem = getEdge(edgeB);
            AssertFatal(edgeAElem.index() != edgeBElem.index(),
                        "edgeA and edgeB types should not be the same");

            return std::visit(
                rocRoller::overloaded{
                    [&](auto const&) {
                        AssertFatal(false, "Unhandled edge type in getOrderOfDescendants");
                        return NodeOrdering::Undefined;
                    },
                    [&](Body const&) {
                        return std::holds_alternative<Initialize>(edgeBElem)
                                   ? opposite(NodeOrdering::LeftFirst)
                                   : NodeOrdering::LeftFirst;
                    },
                    [&](Sequence const&) { return opposite(NodeOrdering::LeftFirst); },
                    [&](Initialize const&) { return NodeOrdering::LeftFirst; },
                    [&](ForLoopIncrement const&) {
                        return std::holds_alternative<Sequence>(edgeBElem)
                                   ? NodeOrdering::LeftFirst
                                   : opposite(NodeOrdering::LeftFirst);
                    },
                    [&](Else const&) {
                        return std::holds_alternative<Initialize>(edgeBElem)
                                       || std::holds_alternative<Body>(edgeBElem)
                                   ? opposite(NodeOrdering::LeftFirst)
                                   : NodeOrdering::LeftFirst;
                    }},
                edgeAElem);
        };

        // {key, value} = {node index, edge index}
        std::unordered_map<int, int> A_ancestors;
        // stack
        std::vector<int> stk{nodeA};

        // Traverse upstream from A to collect all ancestors of A
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();

            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeB)
                        return opposite(getOrderIfParent(edge));

                    A_ancestors.insert({parent, edge});

                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }

        AssertFatal(stk.empty());
        stk.push_back(nodeB);
        visited_nodes.clear();

        // Traverse upstream from B to find common ancestors to decide order
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();

            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeA)
                        return getOrderIfParent(edge);

                    if(A_ancestors.contains(parent))
                    {
                        // If this is a common ancestor, compare the types of both edges to
                        // know the order
                        if(getEdge(A_ancestors.at(parent)).index() != getEdge(edge).index())
                        {
                            return getOrderOfDescendants(A_ancestors.at(parent), edge);
                        }
                    }

                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }

        return NodeOrdering::Undefined;
    }

    static void validateNodes(ControlGraph const& control, int nodeA, int nodeB)
    {
        AssertFatal(nodeA != nodeB, ShowValue(nodeA));
        AssertFatal(control.getElementType(nodeA) == Graph::ElementType::Node
                        && control.getElementType(nodeB) == Graph::ElementType::Node,
                    ShowValue(control.getElementType(nodeA)),
                    ShowValue(control.getElementType(nodeB)));
    }

    NodeOrdering ControlGraph::compareNodes(CacheOnlyPolicy const, int nodeA, int nodeB) const
    {
        AssertFatal(m_cacheStatus == CacheStatus::Valid);

        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(CacheOnly, nodeA, nodeB);
    }

    NodeOrdering ControlGraph::compareNodes(UpdateCachePolicy const, int nodeA, int nodeB) const
    {
        if(m_cacheStatus != CacheStatus::Valid)
            populateOrderCache();

        return compareNodes(CacheOnly, nodeA, nodeB);
    }

    NodeOrdering
        ControlGraph::compareNodes(UseCacheIfAvailablePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);

        return m_cacheStatus == CacheStatus::Valid ? compareNodes(CacheOnly, nodeA, nodeB)
                                                   : compareNodes(IgnoreCache, nodeA, nodeB);
    }

    NodeOrdering ControlGraph::compareNodes(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(IgnoreCache, nodeA, nodeB);
    }
}
