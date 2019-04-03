import time
import numpy as np
import math
from model.model import *
import sys
from mht.pruning import Pruner

INT_MAX = 10000000
P_CONSTANT = math.exp(-0.5)


class ClusterMerger:
    __slots__ = 't_nodes', 't_nodes_del', 'parents', 'clusters', 'n', 'leaves',\
                'low_K', 'min_p', 'pruner'

    def __init__(self, pruner):
        self.t_nodes = []
        self.t_nodes_del = []
        self.parents = []
        self.clusters = None
        self.n = 0
        self.leaves = []
        self.pruner = pruner
        self.low_K = None
        self.min_p = None

    def merge_clusters(self, clusters):
        c = clusters[0]

        # Calculate maximum probability (cluster leaves must be sorted):
        self.min_p = 1
        for cluster in clusters:
            self.min_p *= cluster.leaves[0].probability
        self.min_p /= self.pruner.ratio_pruning

        # Hypothesises:
        self.leaves = []
        self.low_K = 1e15
        self.clusters = clusters
        self.n = len(clusters)
        for hyp in c.leaves:
            assert len(self.t_nodes) == 0 and len(self.t_nodes_del) == 0
            self.__aux__(hyp, 1, 1.0)

        # Clusters:
        first_idx = c.first_idx
        targets = c.targets.copy()
        g_meas = c.gated_measurements.copy()
        for i in range(1, len(clusters)):
            c = clusters[i]
            first_idx = min(first_idx, c.first_idx)
            targets.extend(c.targets)
            g_meas.extend(c.gated_measurements)
        self.leaves, _ = self.pruner.prune_ratio(self.leaves)
        self.leaves = self.pruner.prune_K_best(self.leaves)
        return Cluster(first_idx, self.leaves, targets, g_meas)

    def __aux__(self, hyp, i, p):
        self.t_nodes.append(hyp.track_nodes)
        self.t_nodes_del.append(hyp.track_nodes_del)
        self.parents.append(hyp)

        p = p * hyp.probability
        if i == self.n:
            if p >= self.min_p:
                t_nodes, t_nodes_del = [], []
                for t_nodes_list in self.t_nodes:
                    t_nodes.extend(t_nodes_list)
                for t_nodes_list in self.t_nodes_del:
                    t_nodes_del.extend(t_nodes_list)
                node = HypScanJoin2(p, self.parents.copy(), t_nodes, t_nodes_del)
                self.leaves.append(node)
        else:
            for hyp2 in self.clusters[i].leaves:
                self.__aux__(hyp2, i+1, p)

        self.t_nodes.pop()
        self.t_nodes_del.pop()
        self.parents.pop()


def cluster_gating(clusters, measurements, track_gate_gamma, pruner: Pruner):
    # clusters are changed in place.

    cluster_merger = ClusterMerger(pruner)

    # --- Clear the gated measurements for the cluster: ---
    for cluster in clusters:
        cluster.gated_measurements.clear()

    # --- Temp arrays ---
    nu = np.empty((2,), dtype=np.float64)
    B1 = np.empty((2,), dtype=np.float64)

    # --- Gating measurements and merging of clusters if necessary. ---
    unassociated_measurements = []
    associated_clusters = []
    for meas in measurements:
        pos = meas.detection.pos
        associated_clusters.clear()
        for cluster in clusters:
            cluster_associated_w_meas = False
            for target in cluster.targets:
                for t_node in target.leaves:
                    assert not t_node.isPosterior, '{}'.format(t_node)

                    # Calculate gate value:
                    np.subtract(pos, t_node.z_hat, out=nu)
                    np.dot(t_node.Binv, nu, out=B1)
                    gate_value = np.dot(nu.T, B1)

                    # Gate:
                    if gate_value < track_gate_gamma:
                        if not cluster_associated_w_meas:
                            associated_clusters.append(cluster)
                            cluster_associated_w_meas = True
                        p = (P_CONSTANT ** gate_value) * t_node.mg
                        t_node.gated_measurements[meas] = p

        # Number of clusters associated for measurement:
        # - 0. Mark measurement to create new cluster.
        # - 1. No action needed.
        # - >1. Merge clusters.

        if len(associated_clusters) == 0:
            unassociated_measurements.append(meas)
        elif len(associated_clusters) == 1:
            associated_clusters[0].gated_measurements.append(meas)
        else:
            # Merge clusters and replace them with the new one:
            new_cluster = cluster_merger.merge_clusters(associated_clusters)
            new_cluster.gated_measurements.append(meas)

            for cluster in associated_clusters:
                clusters.remove(cluster)
            clusters.append(new_cluster)
    return unassociated_measurements
