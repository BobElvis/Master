from model.model import *
import heapq


class Pruner:
    def __init__(self, N_scan, ratio_pruning, K_best):
        self.N_scan = N_scan
        self.K_best = K_best
        self.ratio_pruning = ratio_pruning

    def prune(self, cluster):
        #prePruneHyp = len(cluster.leaves)
        cluster.leaves = prune_N_scan(cluster, self.N_scan)
        if self.ratio_pruning > 0:
            low_prob = max(cluster.leaves, key=lambda h: h.probability).probability / self.ratio_pruning
            cluster.leaves = [x for x in cluster.leaves if x.probability >= low_prob]
        # print("  Ratio pruning: {} to {}.".format(prePruneHyp, len(cluster.leaves)))
        cluster.leaves = nlargest(cluster.leaves, self.K_best)
        # print('  K-Hyp reduced {} to {}. Ratio: {:.3f}. {}.'
        #       .format(prePruneHyp, len(cluster.leaves),
        #               cluster.leaves[0].probability/cluster.leaves[-1].probability, cluster))
        # print("  Time used generating hyps: {:.3f}".format(time.time()-time_gen_hyps))


def nlargest(hypothesises, K):
    return heapq.nlargest(K, hypothesises, key=lambda x: x.probability)


def prune_N_scan(cluster, N_scan):
    # N_scan < 0: No pruning
    # N_scan == 0: MAP

    if N_scan < 0:
        return cluster.leaves

    # First check if cluster tree is N deep.
    i = 0
    node = cluster.leaves[0]
    while i < N_scan:
        if node.parent is None:
            print('N_scan: Cluster tree not N_scan deep.')
            return cluster.leaves
        if not isinstance(node, HypScanJoin):
            i += 1
        node = node.parent

    # Sum probabilities:
    prob_dict = dict()  # key: parent_node (N-scans back), value: [prob_leaf_children, [leaf_children]]
    for leaf_node in cluster.leaves:
        node = leaf_node
        i = 0
        while i < N_scan:
            if not isinstance(node, HypScanJoin):
                i += 1
            node = node.parent
        val = prob_dict.setdefault(node, [0, []])
        val[0] += leaf_node.probability
        val[1].append(leaf_node)

    # Sort items in dict on probability. item = (key, value):
    parent_sorted = sorted(prob_dict.items(), key=lambda x: -x[1][0])
    n = len(parent_sorted)
    best_children = parent_sorted[0][1][1]  # First, value, list of nodes.

    # Debug:
    print('N-SCAN: Best: {:.2g}%, Next Best: {:.2g}%...'.format(parent_sorted[0][1][0] * 100,
                                                                parent_sorted[1][1][0] * 100 if n > 1 else 0))
    print(" - Best: {}".format(leaf_node_tostr(parent_sorted[0][0])))
    if n > 1:
        print(" - Next: {}".format(leaf_node_tostr(parent_sorted[1][0])))
    print(' - Leaves: {} -> {}'.format(len(cluster.leaves), len(best_children)))

    return best_children