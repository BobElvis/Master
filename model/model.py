import numpy as np
from multivariate import multi_gaussian


class Scan(object):
    __slots__ = 'time', 'radar_img', 'camera_img', 'mx', 'my', 'm', 'n'

    def __init__(self, time, radar_img, camera_img, mx, my):
        self.time = time
        self.radar_img = radar_img
        self.camera_img = camera_img
        self.mx = mx
        self.my = my
        self.m = np.stack((self.mx, self.my), axis=1)
        self.n = len(mx)

    def __len__(self):
        return len(self.mx)


class Measurement:
    __slots__ = 'idx', 'value', 'time', 'init_speed', 'density', 'init_speed_var'

    def __init__(self, idx, value, time, density = None, init_speed = None):
        self.idx = idx
        self.value = value
        self.time = time
        self.init_speed = init_speed
        self.init_speed_var = None
        self.density = density

    def __repr__(self):
        meas_str = "M{}".format(self.idx)
        return meas_str

    def __getitem__(self, key):
        return self.value[key]


class Cluster:
    __slots__ = 'targets', 'leaves', 'sources', 'gated_measurements'

    def __init__(self, sources, targets, gated_measurements=None):
        self.targets = targets  # Targets defined in the cluster
        self.leaves = sources  # Last nodes in the tree (leaves)
        self.sources = sources  # First nodes in the cluster.
        self.gated_measurements = gated_measurements if gated_measurements is not None else set()

    @classmethod
    def fromMerge(cls, c1, c2):
        # Disjoint set of targets
        targets = sorted(c1.targets + c2.targets, key=lambda x:x.idx)
        gated_measurements = c1.gated_measurements.union(c2.gated_measurements)

        # Create new hypothesis jointly from both clusters.
        sources = []
        for h1 in c1.leaves:
            for h2 in c2.leaves:
                node = HypScanJoin.fromNodes(h1, h2)
                sources.append(node)
        return Cluster(sources, targets, gated_measurements)

    def __repr__(self):
        return 'C: {}, {}'.format(self.targets, self.gated_measurements)


class TrackNode:
    __slots__ = 'parent', 'measurement', 'gated_measurements', 'childrenDict', 'target',\
                'hyp_nodes', 'est_prior', 'cov_prior', 'est_posterior', 'cov_posterior',\
                'z_hat', 'B', 'isPosterior', 'inside_prior', 'PD', 'PX', 'area_scale', 'i_det'

    def __init__(self, parent, est_posterior, cov_posterior, target, measurement):
        # Node properties
        self.parent = parent
        self.measurement = measurement # May be none
        self.gated_measurements = {}
        self.childrenDict = dict()
        self.target = target
        self.hyp_nodes = []

        # Estimate properties
        self.est_prior = None  # [x, xdot, y, ydot]
        self.cov_prior = None
        self.est_posterior = est_posterior
        self.cov_posterior = cov_posterior
        self.z_hat = None
        self.B = None
        self.isPosterior = True

        self.PD = 1
        self.PX = 0
        self.i_det = None
        self.area_scale = 1

    def addGatedMeasurement(self, m):
        p = multi_gaussian(m.value - self.z_hat, self.B)
        self.gated_measurements[m] = p

    def isMeasurementGated(self, m):
        return m in self.gated_measurements

    def innovateTrack(self, measurement, innovator):
        # Returns the new track node corresponding to the innovation of the measurement.
        child_node = self.childrenDict.get(measurement)

        if child_node is None:
            # Track node does not have the measurement in already. Create it.
            if measurement is None:
                est_posterior, cov_posterior = self.est_prior, self.cov_prior
            else:
                est_posterior, cov_posterior = innovator.update(self, measurement.value)
            child_node = TrackNode(self, est_posterior, cov_posterior, self.target, measurement)
            self.childrenDict[measurement] = child_node
        return child_node

    def children(self):
        return self.childrenDict.values()

    def __repr__(self):
        return '{}-{}'.format(self.target, self.measurement)

    def getTrack(self):
        # Return the list of track nodes up to this one.
        return self.getTrackAux([])

    def getTrackAux(self, list):
        if self.parent is not None:
            self.parent.getTrackAux(list)
        list.append(self)
        return list


class Target:
    __slots__ = 'idx', 'source', 'leaves'

    def __init__(self):
        self.idx = None
        self.source = None
        self.leaves = None

    def init(self, track_node):
        self.idx = track_node.measurement.idx
        self.source = track_node
        self.leaves = [track_node]

    def __repr__(self):
        return 'T{}'.format(self.idx)


class HypScan:
    __slots__ = 'probability', 'parent', 'track_nodes', 'track_nodes_del', 'children', 'sol'

    def __init__(self, p, parent, track_nodes, track_nodes_del, sol=None):
        self.probability = p
        self.parent = parent
        self.track_nodes = track_nodes
        self.track_nodes_del = track_nodes_del
        self.children = []
        self.sol = sol


class HypScanJoin(HypScan):
    __slots__ = 'parent2'

    def __init__(self, p, parent1, parent2, track_nodes, track_nodes_del):
        super().__init__(p, parent1, track_nodes, track_nodes_del)
        self.parent2 = parent2

    @classmethod
    def fromNodes(cls, node1, node2):
        prob = node1.probability * node2.probability
        track_nodes = sorted(node1.track_nodes + node2.track_nodes, key=lambda x: x.target.idx)
        track_nodes_del = sorted(node1.track_nodes_del + node2.track_nodes_del, key=lambda x: x.target.idx)
        return cls(prob, node1, node2, track_nodes, track_nodes_del)
