import copy
from model.model import *


class MhtData:
    # Intended to be used for both fast access and limit memory.

    def __init__(self, dataset):
        self.clusters_dead = []
        self.clusters = []
        self.first_idx = None
        self.last_idx = None
        self.dataset = dataset

    def set_data(self, idx, clusters, new_dead_clusters):
        if self.first_idx is None:
            self.first_idx = idx
        data_idx = idx - self.first_idx

        # Updating last_idx:
        for c in new_dead_clusters:
            c.last_idx = idx

        # Setting current clusters:
        clusters_copy = ([copy.copy(c) for c in clusters])
        if data_idx == len(self.clusters):
            self.clusters_dead.append(new_dead_clusters)
            self.clusters.append(clusters_copy)
        elif data_idx:
            self.clusters_dead[data_idx] = new_dead_clusters
            self.clusters[data_idx] = clusters_copy

    def get_data(self, mht_idx, first_idx=None):
        return self.get_clusters_dead(mht_idx, first_idx), self.get_clusters(mht_idx)

    def get_clusters(self, mht_idx):
        data_idx = mht_idx - self.first_idx
        if data_idx >= len(self.clusters):
            return None
        return self.clusters[data_idx]

    def get_clusters_dead(self, mht_idx, first_idx=None):
        if first_idx is None:
            first_idx = self.first_idx

        data_idx = mht_idx - self.first_idx
        if data_idx >= len(self.clusters):
            return None
        first_data_idx = first_idx - self.first_idx

        dead_clusters = []
        for i in range(first_data_idx, data_idx + 1):
            dead_clusters.extend(self.clusters_dead[i])
        return dead_clusters

    def save(self):
        pass


