class MhtData:
    def __init__(self):
        self.data = dict()

    def getData(self, scan):
        return self.data.get(scan.time)

    def setData(self, scan, clusters):
        self.data[scan.time] = [(cluster.leaves, cluster.targets) for cluster in clusters]

