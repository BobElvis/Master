import numpy as np
from multivariate import multi_gaussian
from typing import Iterable, List


class Scan(object):
    __slots__ = 'time', 'radar_img', 'camera_img', 'mx', 'my', 'm', 'n'

    def __init__(self, m, radar_img, camera_img = None, time = None):
        self.time = time
        self.radar_img = radar_img
        self.camera_img = camera_img
        self.m = m
        self.n = m.shape[0]

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


class TrackNode:
    __slots__ = 'scan_idx', 'parent', 'measurement', 'gated_measurements', 'childrenDict', 'target',\
                'est_prior', 'cov_prior', 'est_posterior', 'cov_posterior',\
                'z_hat', 'B', 'isPosterior', 'inside_prior', 'PD', 'PX', 'area_scale', 'i_det', 'Binv', 'mg'

    def __init__(self, scan_idx, parent, est_posterior, cov_posterior, target, measurement):
        # Node properties
        self.gated_measurements = {}
        self.childrenDict = dict()

        # Used for extraction:
        self.scan_idx = scan_idx
        self.parent = parent
        self.measurement = measurement  # May be none
        self.target = target

        # Estimate properties
        self.est_prior = None  # [x, xdot, y, ydot]
        self.cov_prior = None
        self.est_posterior = est_posterior
        self.cov_posterior = cov_posterior

        # Used for fast access:
        self.z_hat = None
        self.B = None
        self.Binv = None
        self.mg = None
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
            child_node = TrackNode(self.scan_idx + 1, self, est_posterior, cov_posterior, self.target, measurement)
            self.childrenDict[measurement] = child_node
        return child_node

    def children(self):
        return self.childrenDict.values()

    def __repr__(self):
        return '{}-{}-{}'.format(self.scan_idx, self.target, self.measurement)

    def getTrack(self, i_start=None, i_end=None):
        # Return the list of track nodes from i_start up to and including i_end
        if i_end is None:
            i_end = self.scan_idx
        if i_start is None or self.scan_idx >= i_start:
            return self.getTrackAux([], i_start, i_end)
        else:
            return []

    def getTrackAux(self, list, i_start, i_end):
        if (self.parent is not None) and (i_start is None or i_start <= self.scan_idx - 1):
            self.parent.getTrackAux(list, i_start, i_end)
        if i_end >= self.scan_idx:
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


class HypScan(object):
    __slots__ = 'probability', 'parent', 'track_nodes', 'track_nodes_del'

    def __init__(self, p: float, parent, track_nodes: List[TrackNode], track_nodes_del: List[TrackNode]):
        self.probability = p
        self.parent = parent
        self.track_nodes = track_nodes
        self.track_nodes_del = track_nodes_del


class HypScanJoin(HypScan):
    __slots__ = 'parent2'

    def __init__(self, p, parent1, parent2, track_nodes, track_nodes_del):
        super().__init__(p, parent1, track_nodes, track_nodes_del)
        self.parent2 = parent2


class HypScanJoin2(HypScan):
    __slots__ = 'parents'

    def __init__(self, prob, parents, track_nodes, track_nodes_del):
        super().__init__(prob, None, track_nodes, track_nodes_del)
        self.parents = parents


class Cluster:
    __slots__ = 'first_idx', 'last_idx', 'targets', 'leaves', 'sources', 'gated_measurements'

    def __init__(self, first_idx: int, sources: List[HypScan], targets: List[Target],
                 gated_measurements: List[Measurement] = None):
        self.first_idx = first_idx
        self.last_idx = None
        self.targets = targets  # Targets defined in the cluster
        self.leaves = sources  # Last nodes in the tree (leaves)
        self.sources = sources  # First nodes in the cluster.
        self.gated_measurements = gated_measurements if gated_measurements is not None else []

    def __repr__(self):
        return 'C: {}, {}'.format(self.targets, self.gated_measurements)


