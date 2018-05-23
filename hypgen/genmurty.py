from model.model import HypScan
import numpy as np
from python_example import Murty
import math
import time
import sys

LOW = sys.float_info.min


def murty_gen(C, K):
    m_gen = Murty(C)
    for i in range(0, K):
        res = m_gen.draw() #ok, score, sol
        if not res[0]:
            return None
        yield res


class GenMurty():
    def __init__(self, mht, K):
        self.mht = mht
        self.K = K

        self.low_K = -1
        self.num_hyp = 0
        self.useless = 0

        # Hold state:
        self.new_targets = None
        self.new_leaves = None
        self.time_draw = 0
        self.time_hyp = 0

    def gen_hyps(self, cluster):
        meas = sorted(cluster.gated_measurements, key=lambda x:x.idx)
        self.new_targets = [self.mht.createNewTarget(m) for m in meas]
        self.new_leaves = []
        self.time_draw, self.time_hyp = 0, 0
        self.low_K, self.num_hyp, self.useless = 1e15, 0, 0
        for scan_node in cluster.leaves:
            self.gen_hyps_single_del(scan_node, meas)

        cluster.leaves = self.new_leaves
        print("TOT: Time hyp {:.3f}, {:.3f}".format(self.time_draw, self.time_hyp))
        return self.new_targets

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

    def gen_hyps_single_del(self, scan_node: HypScan, measurements):
        M = len(measurements)
        N = len(scan_node.track_nodes)

        prior = scan_node.probability
        for track_node in scan_node.track_nodes:
            prior *= (1 - track_node.PD - track_node.PX)

        if M == 0 and N == 0:
            self.new_leaves.append(HypScan(prior, scan_node, [], scan_node.track_nodes_del.copy()))
            return

        # Allocating array:
        C = np.full((M + N, N + 2*M + N), LOW)

        # Fill measurement to existing track nodes:
        for i, track_node in enumerate(scan_node.track_nodes):
            PD = track_node.PD
            PX = max(track_node.PX, LOW)
            PO = 1 - PD - PX

            C[range(M), i] = [track_node.gated_measurements.get(m, LOW)*PD/PO
                              for m in measurements]
            C[i+M, i] = PX/PO
            C[i+M, i + N + M + M] = 1
        for i, m in enumerate(measurements):
            C[i, N + i] = m.density
            C[i, N + M + i] = self.mht.densityFalseReports

        # Transform to -log(prob)
        np.negative(np.log(C, out=C), out=C)

        # Debug: Timing:
        time_draw = 1e-10
        time_hyp = 1e-10
        time_temp = time.time()

        # Temporary arrays to avoid allocating memory:
        meas_found = np.full(N, False, np.bool)
        delete_nodes = np.full(N, False, np.bool)
        meas_nodes = [None]*N

        for _, score, solution in murty_gen(C, self.K):
            time_draw += time.time() - time_temp
            time_temp = time.time()

            # Hypothesis generated from LOW (or in reality zero probability).
            p = math.exp(-score)*prior
            if p < LOW/2:
                break

            # Check if probability is worse than best than the K-hyp.
            self.num_hyp += 1
            if self.num_hyp > self.K:
                if p < self.low_K:
                    break
                else:
                    self.low_K = p
            else:
                self.low_K = min(p, self.low_K)

            # Create hypothesis:
            new_track_nodes = []
            new_delete_nodes = scan_node.track_nodes_del.copy()

            meas_found.fill(False)
            delete_nodes.fill(False)

            # Measurement associations:
            for i in range(M):
                sol = solution[i]
                if sol < N:  # It is one of the targets
                    meas_found[sol] = True
                    meas_nodes[sol] = measurements[i]

            # Track deletion associations:
            for i in range(M, M+N):
                sol = solution[i]
                if sol < N: # It is one of the targets:
                    delete_nodes[sol] = True

            # Loop over tracks:
            for i, track_node in enumerate(scan_node.track_nodes):
                if delete_nodes[i]:
                    new_delete_nodes.append(track_node)
                elif meas_found[i]:
                    new_track_nodes.append(track_node.innovateTrack(meas_nodes[i], self.mht.measurementModel))
                else:
                    new_track_nodes.append(track_node.innovateTrack(None, self.mht.measurementModel))
            for i in range(M):
                if N <= solution[i] < N + M:
                    new_track_nodes.append(self.new_targets[i].source)

            new_node = HypScan(p, scan_node, new_track_nodes, new_delete_nodes)
            self.new_leaves.append(new_node)
            time_hyp += time.time() - time_temp
            time_temp = time.time()

        self.time_draw += time_draw
        self.time_hyp += time_hyp

