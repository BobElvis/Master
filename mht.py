from mhtprint import *
from model.model import *
from hypgen.mhtgen2 import MHTGen2
from hypgen.genmurty import GenMurty
from hypgen.basegen import nlargest
import time
import hypgen.basegen
import util
from multivariate import multi_gaussian

class MHT:
    """ Measurement possibilities. False (clutter), new target, previous target (several).
        Maximum number of possibilities. 1 + 1 + N.
        Possible addition. Combination of previous targets (N targets, max combined targets is r). 1 + 1 + N + nCr
    """

    def __init__(self, P_D, dt, clutter_density, measurement_model, track_gate, meas_init, mht_data):
        self.dt = dt
        self.tIdx = 1
        self.clusters = []
        self.dead_clusters = []
        self.scanIdx = -1

        self.trackGate = track_gate
        self.measurementModel = measurement_model

        self.densityFalseReports = clutter_density

        self.N_scan = -1
        self.ratio_pruning = 1e5
        self.K_best = 1000
        self.mht_data = mht_data
        self.meas_init = meas_init
        #self.mht_gen = MHTGen2(self, P_D, self.probability_deletion)
        self.mht_gen = GenMurty(self, self.K_best)

    def step(self, scan):
        self.scanIdx += 1
        # TODO: Pruning. (maintaining track to hyp relationship)
        # TODO: Death rate on tracks outside of land.
        # TODO: Save sequence of measurements corresponding to hypothesises for each cluster.
        # TODO: (Murty)
        print("{0} STEP {0}".format('-/\--\/-'*5))

        # Create measurement vector from scan:
        measurements = []
        for i in range(0, scan.n):
            measurements.append(Measurement(self.tIdx, scan.m[i, :], scan.time))
            self.tIdx += 1
        self.meas_init.init_measurements(measurements)

        # --- Clear the gated measurements for the cluster: ---
        for cluster in self.clusters:
            cluster.gated_measurements.clear()

        # --- Gating measurements and merging of clusters if necessary. ---
        unassociated_measurements = []
        for meas in measurements:
            associated_clusters = set()
            for cluster in self.clusters:
                for target in cluster.targets:
                    for t_node in target.leaves:
                        assert not t_node.isPosterior, '{}'.format(t_node)
                        if not self.trackGate.is_inside(t_node, meas):
                            continue
                        associated_clusters.add(cluster)
                        cluster.gated_measurements.add(meas)
                        p = multi_gaussian(meas.value - t_node.z_hat, t_node.B)*t_node.area_scale
                        t_node.gated_measurements[meas] = p

            # Number of clusters associated for measurement:
            # - 0. Mark measurement to create new cluster.
            # - 1. No action needed.
            # - >1. Merge clusters.

            if len(associated_clusters) == 0:
                unassociated_measurements.append(meas)
            elif len(associated_clusters) > 1:
                # Merge clusters and replace them with the new one:
                newCluster = self.mergeClusters(associated_clusters)
                for cluster in associated_clusters:
                    self.clusters.remove(cluster)
                self.clusters.append(newCluster)

        for cluster in self.clusters:
            time_gen_hyps = time.time()
            new_targets = self.mht_gen.gen_hyps(cluster)
            print('------- CLUSTER LEAVES:{}'.format(len(cluster.leaves)))

            # Prune:
            prePruneHyp = len(cluster.leaves)
            cluster.leaves = self.prune_N_scan(cluster, self.N_scan)
            if self.ratio_pruning > 0:
                low_prob = max(cluster.leaves, key=lambda h:h.probability).probability/self.ratio_pruning
                cluster.leaves = [x for x in cluster.leaves if x.probability >= low_prob]
            print("  Ratio pruning: {} to {}.".format(prePruneHyp, len(cluster.leaves)))
            cluster.leaves = nlargest(cluster.leaves, self.K_best)
            print('  K-Hyp reduced {} to {}. Ratio: {:.3f}%. {}.'
                  .format(prePruneHyp, len(cluster.leaves),
                          cluster.leaves[0].probability/cluster.leaves[-1].probability*100, cluster))
            print("  Time used generating hyps: {:.3f}".format(time.time()-time_gen_hyps))
            hypgen.basegen.normalize(cluster.leaves)

            # Update old target leaves.
            print('  NEW TARGETS: {}'.format(new_targets))
            for target in cluster.targets:
                new_leaves = []
                for old_leaf in target.leaves:
                    new_leaves.extend(old_leaf.children())
                target.leaves = new_leaves

            # Remove targets that did not get continued:
            cluster.targets = [t for t in cluster.targets if len(t.leaves) > 0]

            # Update cluster targets:
            cluster.targets.extend(new_targets)
            print("-------- END CLUSTER ---------")

        self.dead_clusters = self.dead_clusters + [c for c in self.clusters if len(c.targets) == 0]
        self.clusters = [c for c in self.clusters if len(c.targets) > 0]

        # Create new clusters from the unassociated measurements:
        self.clusters += [self.createNewCluster(m) for m in unassociated_measurements]

        # Debug and saving of data.
        self.mht_data.setData(scan, self.dead_clusters + self.clusters)

        # Predict all possible target locations. (measurement independent)
        for cluster in self.clusters:
            for target in cluster.targets:
                for t_node in target.leaves:
                    assert t_node.isPosterior, "{} + {}".format(t_node.getTrack(), list(t_node.children()))
                    self.measurementModel.predict(t_node)

    def createNewTarget(self, measurement):
        target = Target()
        mean, covariance = self.measurementModel.initialize(measurement)
        track_node = TrackNode(None, mean, covariance, target, measurement)
        target.init(track_node)
        return target

    def createNewCluster(self, meas):
        target = self.createNewTarget(meas)
        print("Creating new cluster")

        # Probabilities.
        falseProbability = self.densityFalseReports
        trueProbability = meas.density
        c = falseProbability + trueProbability  # Normalizing term

        hypFalse = HypScan(falseProbability/c, parent=None, track_nodes=[], track_nodes_del=[])
        hypTarget = HypScan(trueProbability/c, parent=None, track_nodes=[target.source], track_nodes_del=[])
        sources = sorted([hypFalse, hypTarget], key=lambda x: -x.probability)
        cluster = Cluster(sources=sources, targets=[target])
        return cluster

    @staticmethod
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

    @staticmethod
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
        best_children = parent_sorted[0][1][1] # First, value, list of nodes.

        # Debug:
        print('N-SCAN: Best: {:.2g}%, Next Best: {:.2g}%...'.format(parent_sorted[0][1][0]*100, parent_sorted[1][1][0]*100 if n > 1 else 0))
        print(" - Best: {}".format(leaf_node_tostr(parent_sorted[0][0])))
        if n > 1:
            print(" - Next: {}".format(leaf_node_tostr(parent_sorted[1][0])))
        print(' - Leaves: {} -> {}'.format(len(cluster.leaves), len(best_children)))

        return best_children

    # ---PRINT---:

    def printInfo(self):
        print('There are {} clusters:'.format(len(self.clusters)))
        for cluster in self.clusters:
            print(cluster)

    def printBest(self, K):
        print('//Best hypothesises: {} -- '.format(K))
        for cluster in self.clusters:
            print(' -Cluster {}:'.format(len(cluster.leaves)))
            for idx, node in enumerate(nlargest(cluster.leaves, K)):
                print("{}: {}".format(idx+1, leaf_node_tostr(node)))
        print('----')






