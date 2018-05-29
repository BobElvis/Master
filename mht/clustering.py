import time
import numpy as np
import math
from model.model import *

P_CONSTANT = math.exp(-0.5)


def mergeClusters(clusters):
    # TODO: If more than two, combine all to one super-cluster instead of "stacked" double-clusters
    # TODO: Will probably depend of what is most compatible with pruning.
    cluster = None
    for c in clusters:
        if cluster is None:
            cluster = c
        else:
            cluster = Cluster.fromMerge(cluster, c)
    return cluster


class ClusterMerger:
    __slots__ = 't_nodes', 't_nodes_del', 'parents', 'clusters', 'n', 'leaves'

    def __init__(self):
        self.t_nodes = []
        self.t_nodes_del = []
        self.parents = []
        self.clusters = None
        self.n = 0
        self.leaves = []

    def merge_clusters(self, clusters):
        c_iter = iter(clusters)
        c = c_iter.__next__()

        # Hypothesises:
        self.leaves = []
        self.clusters = clusters
        self.n = len(clusters)
        for hyp in c.leaves:
            self.__aux__(hyp, 1, 1.0)

        # Clusters:
        targets = c.targets.copy()
        g_meas = c.gated_measurements.copy()
        for c in c_iter:
            targets.extend(c.targets)
            g_meas.extend(c.gated_measurements)

        return Cluster(self.leaves, targets, g_meas)

    def __aux__(self, hyp, i, p):
        self.t_nodes.append(hyp.track_nodes)
        self.t_nodes_del.append(hyp.track_nodes_del)
        self.parents.append(hyp.parent)

        p *= hyp.probability
        if i == self.n:
            t_nodes, t_nodes_del = [], []
            for t_nodes_list, t_nodes_del_list in zip(self.t_nodes, self.t_nodes_del):
                t_nodes.extend(t_nodes_list)
                t_nodes_del.extend(t_nodes_del_list)
            self.leaves.append(HypScanJoin2(p, self.parents.copy(), t_nodes, t_nodes_del))
        else:
            for hyp2 in self.clusters[i].leaves:
                self.__aux__(hyp2, i+1, p)

        self.t_nodes.pop()
        self.t_nodes_del.pop()
        self.parents.pop()


def cluster_gating(clusters, measurements, track_gate_gamma):
    # clusters are changed in place.
    print("Cluster management: Clusters: {}, Measurements: {}".format(len(clusters), len(measurements)))

    cluster_merger = ClusterMerger()

    # --- Clear the gated measurements for the cluster: ---
    for cluster in clusters:
        cluster.gated_measurements.clear()

    n_merges = 0
    n_merges_single = 0
    t_merge = 0
    t_gate = 0
    t_rem = 0

    # --- Temp arrays ---
    nu = np.empty((2,), dtype=np.float64)
    B1 = np.empty((2,), dtype=np.float64)

    # --- Gating measurements and merging of clusters if necessary. ---
    unassociated_measurements = []
    associated_clusters = []
    for meas in measurements:
        associated_clusters.clear()
        for cluster in clusters:
            cluster_associated_w_meas = False
            for target in cluster.targets:
                for t_node in target.leaves:
                    t = time.time()
                    assert not t_node.isPosterior, '{}'.format(t_node)

                    # Calculate gate value:
                    np.subtract(meas.value, t_node.z_hat, out=nu)
                    np.dot(t_node.Binv, nu, out=B1)
                    gate_value = np.dot(nu.T, B1)

                    # Gate:
                    if gate_value < track_gate_gamma:
                        if not cluster_associated_w_meas:
                            associated_clusters.append(cluster)
                            cluster_associated_w_meas = True
                        p = (P_CONSTANT ** gate_value) * t_node.mg * t_node.area_scale
                        t_node.gated_measurements[meas] = p
                    t_gate += time.time() - t

        # Number of clusters associated for measurement:
        # - 0. Mark measurement to create new cluster.
        # - 1. No action needed.
        # - >1. Merge clusters.

        if len(associated_clusters) == 0:
            unassociated_measurements.append(meas)
        elif len(associated_clusters) == 1:
            associated_clusters[0].gated_measurements.append(meas)
        else:
            t = time.time()
            n_merges += len(associated_clusters) - 1
            n_merges_single += 1

            # Merge clusters and replace them with the new one:
            #new_cluster = mergeClusters(associated_clusters)
            new_cluster = cluster_merger.merge_clusters(associated_clusters)
            new_cluster.gated_measurements.append(meas)

            t_merge += time.time() - t
            t = time.time()

            for cluster in associated_clusters:
                clusters.remove(cluster)
            clusters.append(new_cluster)

            t_rem += time.time() - t

    print("End cluster management. Number of merges: {}/{}".format(n_merges, n_merges_single))
    print("Time. Merge: {:.2f}. Gate: {:.2f}. Remove: {:.2f}".format(t_merge, t_gate, t_rem))
    return unassociated_measurements
