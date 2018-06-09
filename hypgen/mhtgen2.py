from hypgen import mhtgen
import collections
from model.model import HypScan


class MHTGen2:
    def __init__(self, mht, prob_detect, prob_delete=0):
        self.mht = mht
        self.prob_detect = prob_detect
        self.delete_nodes = prob_delete > 0
        self.prob_delete = prob_delete

        self.meas = None
        self.new_targets = None
        self.n_meas = -1
        self.n_del = -1
        self.new_leaves = None
        self.current_parent = None

        self.sel = None
        self.av = None
        self.deleted = None
        self.scan_time = None

        self.prob_occlusion = 1 - self.prob_detect - self.prob_delete

        print('GEN: P_D: {}, P_X: {}, P_O: {}'.format(self.prob_detect, self.prob_delete, self.prob_occlusion))

    def gen_hyps(self, scan_time, cluster):
        self.scan_time = scan_time
        self.meas = sorted(cluster.gated_measurements, key=lambda x: x.idx)
        self.new_targets = [self.mht.createNewTarget(self.scan_time, m) for m in self.meas]
        self.n_meas = len(self.meas)
        self.new_leaves = []

        self.sel = []  # Normal list (supporting pop(right)/append (right) in O(1))
        self.deleted = []

        for scan_node in cluster.leaves:
            # Linked list: pop left/pop right/append left/append right in O(1)
            self.av = collections.deque(scan_node.track_nodes)
            self.current_parent = scan_node
            assert len(self.sel) == 0 and len(self.deleted) == 0
            self.gen_hyp_aux(0, scan_node.probability * (self.prob_occlusion ** len(scan_node.track_nodes)))

        cluster.leaves = self.new_leaves
        return self.new_targets

    def gen_hyp_aux(self, i, prob):
        if i == self.n_meas:
            if self.delete_nodes:
                self.n_del = len(self.av)
                self.gen_hyp_del(0, prob)
            else:
                new_node = HypScan(prob, self.current_parent, self.sel + mhtgen.MHTGen.innovate_track_nodes(self.av, self.mht.measurementModel), [])
                self.current_parent.children.append(new_node)
                self.new_leaves.append(new_node)
            return

        # Clutter:
        self.gen_hyp_aux(i+1, prob * self.mht.densityFalseReports)

        # Existing:
        n_t = len(self.av)
        p_ext = prob * self.prob_detect/self.prob_occlusion
        measurement = self.meas[i]
        for k in range(0, n_t):
            track_node = self.av.popleft()
            if track_node.isMeasurementGated(measurement):
                p_node = p_ext * track_node.gated_measurements[measurement]
                self.sel.append(track_node.innovateTrack(measurement, self.mht.measurementModel))
                self.gen_hyp_aux(i+1, p_node)
                self.sel.pop()
            self.av.append(track_node)

        # New Target:
        self.sel.append(self.new_targets[i].source)
        self.gen_hyp_aux(i+1, prob * measurement.density)
        self.sel.pop()

    def gen_hyp_del(self, i, prob):
        if i == self.n_del:
            new_node = HypScan(prob, self.current_parent,
                               self.sel.copy(), self.deleted + self.current_parent.track_nodes_del)
            self.new_leaves.append(new_node)
            return

        track_node = self.av.popleft()

        # Keep
        self.sel.append(track_node.innovateTrack(None, self.mht.measurementModel))
        self.gen_hyp_del(i+1, prob)
        self.sel.pop()

        # Delete (only delete tracks that had a measurement)
        if True:#not track_node.inside_prior: #track_node.measurement is not None:
            self.deleted.append(track_node)
            self.gen_hyp_del(i + 1, prob * self.prob_delete / self.prob_occlusion)
            self.deleted.pop()

        self.av.appendleft(track_node)






