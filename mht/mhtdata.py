import copy
import pickle
import sys
from dataset import ROOT_FOLDER_SAVE
from model.model import Cluster
import dataset
from os import path
import threading
from util import get_filename, save_data, load_data
from typing import List, Union


class Settings:
    def __str__(self):
        str = []
        for attr in dir(self):
            if attr.startswith("__") or callable(getattr(self, attr)):
                continue
            str.append("%s = %r\n" % (attr, getattr(self, attr)))
        return "".join(str)

    def write_to_file(self, name):
        f = open(name, "w")
        f.write(self.__str__())
        f.close()

    @staticmethod
    def get_file(filename):
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        return lines


class MhtData:
    # Intended to be used for both fast access and limit memory.
    # TODO: Fix dead.

    def __init__(self, dataset, settings):
        self.clusters_dead = []  # Clusters made dead at each timestep
        self.clusters = []  # Current clusters for each timestep
        self.first_idx = None
        self.dataset = dataset
        self.settings = settings
        self.max_leaves_cluster_alive = 10
        self.max_leaves_dead = 50

    def copy_cluster(self, cluster: Cluster):
        # Only a few hypotheses are saved for the cluster.
        return copy.copy(cluster)

    def last_idx(self):
        return len(self.clusters) + self.first_idx - 1

    def set_data(self, idx, clusters, new_dead_clusters):
        if self.first_idx is None:
            self.first_idx = idx
        data_idx = idx - self.first_idx
        assert data_idx == len(self.clusters), print("data_idx: {}, {}".format(data_idx, len(clusters)))

        # Updating last_idx:
        for c in new_dead_clusters:
            c.last_idx = idx
            c.leaves = c.leaves[:self.max_leaves_dead]
            c.targets = None

        # Setting current clusters:
        clusters_copy = ([self.copy_cluster(c) for c in clusters])
        self.clusters_dead.append(new_dead_clusters)
        self.clusters.append(clusters_copy)

        # Decrease size of previous cluster list:
        if len(self.clusters) > 1:
            for c in self.clusters[-2]:
                c.leaves = c.leaves[:self.max_leaves_cluster_alive]

    def get_data(self, mht_idx, first_idx=None):
        if self.first_idx is None:
            return None
        return self.get_clusters_dead(mht_idx, first_idx), self.get_clusters(mht_idx)

    def get_clusters(self, mht_idx) -> Union[List[Cluster], None]:
        data_idx = mht_idx - self.first_idx
        if data_idx >= len(self.clusters):
            return None
        return self.clusters[data_idx]

    def get_clusters_dead(self, mht_idx, first_idx=None) -> Union[List[Cluster], None]:
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

    def get_filename(self, newfile, number=None):
        return get_filename(ROOT_FOLDER_SAVE, self.dataset.__str__(), newfile, number)

    def save(self):
        print(threading.stack_size())
        threading.stack_size(201326592)  # 64*3 MB
        thread = threading.Thread(daemon=True, target=lambda: self.__save__())
        thread.start()
        thread.join()

    def __save__(self):
        filename = self.get_filename(True)
        save_data((self.clusters, self.clusters_dead, self.first_idx), filename)
        self.settings.write_to_file(filename + ".txt")

    def restore(self, number=None):
        filename = self.get_filename(False, number)
        self.clusters, self.clusters_dead, self.first_idx = load_data(filename)
        return Settings.get_file(filename + ".txt")



