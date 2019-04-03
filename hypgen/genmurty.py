from model.model import HypScan
import numpy as np
from python_example import Murty
import math
import time
import sys
import threading
import time
import heapq
import hypgen.murtyalg as murty
from mht.pruning import KList


_MAX = 100000000
_SCALE = 1e0


def murty_gen(C, K):
    m_gen = Murty(C)
    for i in range(0, K):
        res = m_gen.draw()  #ok, score, sol
        if not res[0]:
            raise StopIteration()
        yield res


class GenMurty:
    __slots__ = 'mht', 'K', 'best_K', 'num_hyp', 'ratio', 'new_targets',\
                'time_draw', 'time_hyp', 'current_cluster', 'current_scan', 'K_list'

    def __init__(self, mht, K, ratio=None):
        self.mht = mht
        self.K = K

        self.best_K = -1
        self.ratio = 1e20 if ratio is None else ratio  # current ratio is used (this is not a replacement for ratio pruning)

        # Hold state:
        self.new_targets = None
        self.time_draw = 0
        self.time_hyp = 0
        self.current_cluster = None
        self.K_list = KList(self.K)

        self.current_scan = -1

    def gen_hyps(self, scan_idx, cluster):
        self.current_scan = scan_idx
        meas = sorted(cluster.gated_measurements, key=lambda x:x.idx)
        self.new_targets = [self.mht.createNewTarget(scan_idx, m, self.mht.measurementModel) for m in meas]
        self.current_cluster = cluster
        self.time_draw, self.time_hyp = 0, 0
        self.best_K = 0

        self.K_list = KList(self.K)

        for idx, scan_node in enumerate(cluster.leaves):
            #print("  - Node {}:".format(idx))
            self.gen_hyps_single_del(scan_node, meas)

        #cluster.leaves = self.new_leaves
        cluster.leaves = self.K_list.n_largest()
        assert len(cluster.leaves) > 0

        print("  -TOT: Time hyp {:.3f}, {:.3f}".format(self.time_draw, self.time_hyp))
        return self.new_targets

    def gen_hyps_single_del(self, scan_node: HypScan, measurements):
        M = len(measurements)
        N = len(scan_node.track_nodes)
        #if scan_node.probability == 0.0010510982806177364:
        #    return

        prior = scan_node.probability
        for track_node in scan_node.track_nodes:
            prior *= (1 - track_node.PD - track_node.PX)

        if M == 0 and N == 0:
            if self.K_list.check_p(prior):
                self.K_list.add(HypScan(prior, scan_node, [], scan_node.track_nodes_del.copy()))
            return

        # Allocating array:
        C = np.full((N + M, N + 2*M + N), 1, dtype=float)
        Cset = np.full((N + M, N + 2*M + N), True)

        # Fill measurement to existing track nodes:
        for j, m in enumerate(measurements):
            for i, track_node in enumerate(scan_node.track_nodes):
                PD = track_node.PD
                PX = track_node.PX
                PO = 1 - PD - PX
                r = track_node.gated_measurements.get(m)
                if r is not None:
                    C[j, i] = r * PD / PO
                    Cset[j, i] = False

        for i, track_node in enumerate(scan_node.track_nodes):
            PD = track_node.PD
            PX = track_node.PX
            PO = 1 - PD - PX
            assert not track_node.isPosterior

            C[i+M, i] = PX/PO
            C[i+M, i + N + M + M] = 1.0
            Cset[i + M, i] = False
            Cset[i + M, i + N + M + M] = False
        for i, m in enumerate(measurements):
            #print("{:.4g} - {:.4g}".format(m.density, self.mht.densityFalseReports))
            C[i, N + i] = m.density
            C[i, N + M + i] = self.mht.densityFalseReports
            Cset[i, N + i] = False
            Cset[i, N + M + i] = False

        # Transform to -log(prob)
        np.negative(np.log(C, out=C), out=C)
        max_value = np.amax(C)
        np.putmask(C, Cset, _MAX)
        if max_value*(N+M) > _MAX*0.75:
            sys.exit(0)

        # Debug: Timing:
        time_draw = 1e-10
        time_hyp = 1e-10
        time_temp = time.time()

        # Temporary arrays to avoid allocating memory:
        meas_found = np.full(N, False, np.bool)
        delete_nodes = np.full(N, False, np.bool)
        meas_nodes = [None]*N

        for ok, score, solution in murty_gen(C, self.K):
            if score > _MAX*0.75:
                break

            time_draw += time.time() - time_temp
            time_temp = time.time()

            s = math.exp(-score)
            p = s*prior

            # Check if probability is worse than best than the K-hyp.
            if p*self.ratio < self.best_K or not self.K_list.check_p(p):
                break

            self.best_K = max(p, self.best_K)

            # Create hypothesis:
            new_track_nodes = []
            new_delete_nodes = scan_node.track_nodes_del.copy()

            meas_found.fill(False)
            delete_nodes.fill(False)

            # Measurement associations:
            for i in range(M):
                j = solution[i]
                if j < N:  # It is one of the targets
                    meas_found[j] = True
                    meas_nodes[j] = measurements[i]

            # Track deletion associations:
            for i in range(M, M+N):
                j = solution[i]
                if N > j == i - M:  # It is one of the targets:
                    delete_nodes[j] = True

            # Loop over tracks:
            for i, track_node in enumerate(scan_node.track_nodes):
                if delete_nodes[i]:
                    new_delete_nodes.append(track_node)
                elif meas_found[i]:
                    new_track_nodes.append(track_node.innovateTrack(meas_nodes[i].detection, self.mht.measurementModel))
                else:
                    new_track_nodes.append(track_node.innovateTrack(None, self.mht.measurementModel))

            for i in range(M):
                j = solution[i]
                if N <= j < N + M:# and i == j - (M + N):
                    new_track_nodes.append(self.new_targets[i].leaves[0])

            new_node = HypScan(p, scan_node, new_track_nodes, new_delete_nodes)

            self.K_list.add(new_node)
            time_hyp += time.time() - time_temp
            time_temp = time.time()

            # FIX
            if self.current_scan == 10781:
                print("{} | {} | {} | {}".format(score, p, self.current_cluster.first_idx, len(self.K_list)))
                #if p == 3.981229186661051e-24:
                #    print("RETURN")
                #    return

        self.time_draw += time_draw
        self.time_hyp += time_hyp

    # def gen_hyps_single(self, scan_node: HypScan, measurements):
    #     # Asserted to be working:
    #
    #     M = len(measurements)
    #     N = len(scan_node.track_nodes)
    #
    #     prior = scan_node.probability*(self.prob_occlusion**N)
    #
    #     if M == 0:
    #         new_track_nodes = [track_node.innovateTrack(None, self.mht.measurementModel)
    #                            for track_node in scan_node.track_nodes]
    #         self.new_leaves.append(HypScan(prior, scan_node, new_track_nodes, []))
    #         return
    #
    #     # Creating cost matrix:
    #     C = np.full((M, N+2*M), LOW)
    #     for i, track_node in enumerate(scan_node.track_nodes):
    #         C[range(M), i] = [track_node.gated_measurements.get(m, LOW*self.prob_occlusion/self.prob_detect)*self.prob_detect/self.prob_occlusion
    #                           for m in measurements]
    #     for i, m in enumerate(measurements):
    #         C[i, N + i] = m.density
    #         C[i, N + M + i] = self.mht.densityFalseReports
    #     np.negative(np.log(C, out=C), out=C)
    #
    #     for _, score, solution in murty_gen(C, self.K):
    #         p = math.exp(-score)*prior
    #         new_track_nodes = []
    #         # Check if solution is matching to track nodes. Update all track nodes:
    #         for idx, track_node in enumerate(scan_node.track_nodes):
    #             measurement = None
    #             for j in range(len(solution)):
    #                 if solution[j] == idx:
    #                     measurement = measurements[j]
    #                     break
    #             new_track_nodes.append(track_node.innovateTrack(measurement, self.mht.measurementModel))
    #         for j in range(len(solution)):
    #             if N <= solution[j] < M + N:
    #                 new_track_nodes.append(self.new_targets[j].source)
    #         new_node = HypScan(p, scan_node, new_track_nodes, [])
    #         self.new_leaves.append(new_node)

