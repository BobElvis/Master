from mht.mhtprint import *
from model.model import *
from hypgen.mhtgen2 import MHTGen2
from hypgen.genmurty import GenMurty
import time
import hypgen.basegen
import mht.pruning as pruning
from mht.mhtdata import MhtData
from mht import clustering


class MHT:
    """ Measurement possibilities. False (clutter), new target, previous target (several).
        Maximum number of possibilities. 1 + 1 + N.
        Possible addition. Combination of previous targets (N targets, max combined targets is r). 1 + 1 + N + nCr
    """

    def __init__(self, dt, clutter_density, measurement_model, track_gate, meas_init, mht_data: MhtData, pruner):
        self.dt = dt
        self.tIdx = 1
        self.clusters = []

        self.trackGate = track_gate
        self.measurementModel = measurement_model
        self.densityFalseReports = clutter_density

        self.pruner = pruner
        self.mht_data = mht_data
        self.meas_init = meas_init
        #self.mht_gen = MHTGen2(self, 0.8, 0.15)
        self.mht_gen = GenMurty(self, pruner.K_best)

    def cluster_gating(self, measurements):
        return clustering.cluster_gating(self.clusters, measurements, self.trackGate.gamma)

    def step(self, scan):
        print("{0} STEP {0}".format('-/\--\/-'*5))

        # Create measurement vector from scan:
        measurements = []
        for i in range(0, scan.n):
            measurements.append(Measurement(self.tIdx, scan.m[i, :], scan.time))
            self.tIdx += 1
        self.meas_init.init_measurements(measurements)

        # Merge clusters:
        t1 = time.time()
        unassociated_measurements = self.cluster_gating(measurements)
        print("TIME: {:.2f}".format(time.time()-t1))

        for cluster in self.clusters:

            # Prune since merging may have generated too many hypothesises:
            self.pruner.prune(cluster)

            new_targets = self.mht_gen.gen_hyps(scan.time, cluster)
            #print('------- CLUSTER LEAVES:{}'.format(len(cluster.leaves)))

            self.pruner.prune(cluster)
            hypgen.basegen.normalize(cluster.leaves)

            # Update old target leaves.
            for target in cluster.targets:
                new_leaves = []
                for old_leaf in target.leaves:
                    new_leaves.extend(old_leaf.children())
                target.leaves = new_leaves

            # Remove targets that did not get continued:
            cluster.targets = [t for t in cluster.targets if len(t.leaves) > 0]

            # Update cluster targets:
            cluster.targets.extend(new_targets)
            #print("-------- END CLUSTER ---------")

        new_dead_clusters = [c for c in self.clusters if len(c.targets) == 0]
        self.clusters = [c for c in self.clusters if len(c.targets) > 0]

        # Create new clusters from the unassociated measurements:
        self.clusters += [self.createNewCluster(scan.time, m) for m in unassociated_measurements]

        # Debug and saving of data.
        self.mht_data.set_data(scan.time, self.clusters, new_dead_clusters)

        # Predict all possible target locations. (measurement independent)
        n_target_leaves = 0
        for cluster in self.clusters:
            for target in cluster.targets:
                for t_node in target.leaves:
                    n_target_leaves += 1
                    assert t_node.isPosterior, "{} + {}".format(t_node.getTrack(), list(t_node.children()))
                    self.measurementModel.predict(t_node)
        print("Number of target leaves: {}".format(n_target_leaves))

    def createNewTarget(self, scan_idx, measurement):
        target = Target()
        mean, covariance = self.measurementModel.initialize(measurement)
        track_node = TrackNode(scan_idx, None, mean, covariance, target, measurement)
        target.init(track_node)
        return target

    def createNewCluster(self, scan_idx, meas):
        target = self.createNewTarget(scan_idx, meas)

        # Probabilities:
        falseProbability = self.densityFalseReports
        trueProbability = meas.density
        c = falseProbability + trueProbability  # Normalizing term

        # Create nodes:
        hypFalse = HypScan(falseProbability/c, parent=None, track_nodes=[], track_nodes_del=[])
        hypTarget = HypScan(trueProbability/c, parent=None, track_nodes=[target.source], track_nodes_del=[])
        sources = sorted([hypFalse, hypTarget], key=lambda x: -x.probability)
        cluster = Cluster(scan_idx, sources=sources, targets=[target])
        return cluster


    # ---PRINT---:

    def printInfo(self):
        print('There are {} clusters:'.format(len(self.clusters)))
        for cluster in self.clusters:
            print(cluster)

    def printBest(self, K):
        print('//Best hypothesises: {} -- '.format(K))
        for cluster in self.clusters:
            print(' -Cluster {}:'.format(len(cluster.leaves)))
            for idx, node in enumerate(pruning.nlargest(cluster.leaves, K)):
                print("{}: {}".format(idx+1, leaf_node_tostr(node)))
        print('----')






