from model.model import *
import heapq
from timer import SimpleTimer


class KListSimple:
    __slots__ = 'list', 'K', 'p_low'

    def __init__(self, K):
        self.list = []
        self.K = K
        self.p_low = 1e100

    def check_p(self, p):
        return len(self.list) < self.K or p < self.p_low

    def add(self, node: HypScan):
        self.list.append(node)
        self.p_low = min(self.p_low, node.probability)


class KList:
    __slots__ = 'min_heap', 'K', 'list', 'min_p'

    def __init__(self, K):
        self.min_heap = None
        self.K = K
        self.list = []
        self.min_p = 1e10

    def check_p(self, p):
        if self.min_heap is None:  # Same as len(self.list) < self.K
            return True
        else:
            return p > self.min_heap[0].probability

    def add(self, node: HypScan):
        if self.min_heap is None:
            self.list.append(node)
            if len(self.list) == self.K:
                heapq.heapify(self.list)
                self.min_heap = self.list
        else:
            heapq.heapreplace(self.min_heap, node)

    def n_largest(self):
        if self.min_heap is not None:
            sorted_smallest = []
            for i in range(0, self.K):
                sorted_smallest.append(heapq.heappop(self.min_heap))
            return list(reversed(sorted_smallest))
        else:
            return nlargest(self.list, self.K)

    def __len__(self):
        return len(self.list)


class Pruner:
    def __init__(self, N_scan, ratio_pruning, K_best):
        self.N_scan = N_scan
        self.K_best = K_best
        self.ratio_pruning = ratio_pruning
        self.calc_confidence = False

    def prune_ratio(self, nodes):
        if self.ratio_pruning > 0:
            best_prob = max(nodes, key=lambda h: h.probability).probability
            low_prob = best_prob / self.ratio_pruning
            keep = [x for x in nodes if x.probability >= low_prob]

            if self.calc_confidence:
                remove = [x for x in nodes if x.probability < low_prob]
                if len(remove) > 0:
                    best_removed = max(remove, key=lambda h: h.probability).probability
                    print(best_removed)

            return keep, best_prob
        else:
            return nodes, 0

    def prune_K_best(self, nodes):
        return nlargest(nodes, self.K_best)

    def prune(self, cluster, skip_n_scan=False):
        pre_n = len(cluster.leaves)

        cluster.leaves, best_prob = self.prune_ratio(cluster.leaves)
        ratio_n = len(cluster.leaves)

        #cluster.leaves = self.prune_K_best(cluster.leaves)
        #post_n = len(cluster.leaves)

        if not skip_n_scan:
            cluster.leaves = prune_N_scan(cluster.leaves, self.N_scan)
        N_scan_n = len(cluster.leaves)

        for idx, node in enumerate(cluster.leaves):
            node.K_max = max(node.K_max, idx)
            node.ratio_max = max(node.ratio_max, best_prob/node.probability)

        print("  -Pruning: {} (ratio) {} (N-scan) {}".format(pre_n, ratio_n, N_scan_n))

    def __repr__(self):
        return "Pruner: N_scan = {}, K_best = {}, ratio = {:.2g}".format(self.N_scan, self.K_best, self.ratio_pruning)


def nlargest(hypothesises, K):
    return heapq.nlargest(K, hypothesises, key=lambda x: x.probability)


class NScanFinder:
    __slots__ = 'i', 'parents', 'hash'

    def __init__(self):
        self.i = None
        self.parents = None
        self.hash = None

    def find_parents(self, node, N):
        self.parents = ()
        self.aux(node, N)  # Assume
        return self.parents

    def aux(self, node, rem):
        if node is None:
            return
        if rem == 0:
            self.parents += (node,)
            return
        if isinstance(node, HypScanJoin2):
            for parent in node.parents:
                self.aux(parent, rem)
        else:
            self.aux(node.parent, rem - 1)


_N_SCAN_FINDER = NScanFinder()


def prune_N_scan_stat(nodes, N_scan):
    if N_scan < 0:
        return nodes

    # Examine tree.
    node = nodes[0]
    parents = _N_SCAN_FINDER.find_parents(node, N_scan)
    if len(parents) == 0:
        return nodes

    # Sum probabilities for each level.
    prob_dict_tot = dict()  # key: parent_nodes (N-scans back), value: [prob_leaf_children, [leaf_children]]
    for N_i in range(0, N_scan):
        prob_dict = dict()
        for leaf_node in nodes:
            parents = _N_SCAN_FINDER.find_parents(leaf_node, N_i)
            val = prob_dict.setdefault(parents, [0, []])
            val[0] += leaf_node.probability
            val[1].append(leaf_node)

        items = prob_dict.items()
        sorted_items = sorted(items, key = lambda x: x[1][0])


def prune_N_scan(nodes, N_scan):
    # N_scan < 0: No pruning
    # N_scan == 0: MAP
    # Nodes are in same order as originally (sorted in -> sorted out)

    t = SimpleTimer("N-SCAN")

    if N_scan < 0:
        return nodes

    # Examine tree.
    node = nodes[0]
    parents = _N_SCAN_FINDER.find_parents(node, N_scan)
    if len(parents) == 0:
        return nodes

    # Sum probabilities:
    prob_dict = dict()  # key: parent_nodes (N-scans back), value: [prob_leaf_children, [leaf_children]]
    for leaf_node in nodes:
        parents = _N_SCAN_FINDER.find_parents(leaf_node, N_scan)
        val = prob_dict.setdefault(parents, [0, []])
        val[0] += leaf_node.probability
        val[1].append(leaf_node)

    # Sort items in dict on probability. item = (key, value):

    best_parents = heapq.nlargest(5, prob_dict.items(), key=lambda x: x[1][0])

    #best_parents_iter = iter(best_parents)
    # best1 = best_parents_iter.__next__()
    new_nodes = best_parents[0][1][1]
    #
    # n = len(best_parents)
    #
    # if n > 1:
    #      best2 = best_parents_iter.__next__()
    # #     new_nodes.extend(best2[1][1])
    # # for best in best_parents_iter:
    # #     if len(new_nodes) > 10:
    # #         break
    # #     new_nodes.extend(best[1][1])
    #
    # t.report()
    #
    # # Debug:
    # print('  N-SCAN: Ratio: {}'.format(best1[1][0]/best2[1][0] if n > 1 else 1e10))
    return new_nodes