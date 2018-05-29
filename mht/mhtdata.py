import copy

class MhtData:
    def __init__(self):
        self.data = dict()

    def getData(self, scan):
        return self.data.get(scan.time)

    def setData(self, scan, dead_clusters, clusters):
        self.data[scan.time] = ([copy.copy(c) for c in dead_clusters], [copy.copy(c) for c in clusters])

