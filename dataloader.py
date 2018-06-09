import cv2
from model.model import Scan
import util
import threading
import time
import abc


class DataSource:
    def load_data(self, idx) -> Scan:
        pass

    def __len__(self):
        pass


class MHTLoader(object):
    def __init__(self, mht, dataloader, i_start = 0, disabled=False):
        self.mht = mht
        self.mht_data = mht.mht_data
        self.dataloader = dataloader
        self.min_idx = i_start
        self.loaded_idx = i_start
        self.disabled = disabled
        if not disabled:
            mht.step(dataloader[i_start])

    def __getitem__(self, idx):
        if self.disabled:
            return None
        for i in range(self.loaded_idx + 1, idx + 1):
            self.mht.step(self.dataloader[i])
        self.loaded_idx = max(self.loaded_idx, idx)
        return self.mht_data.get_data(idx)

    def gen_to(self, idx):
        if self.disabled:
            return self.mht_data.get_data(idx)
        for i in range(self.loaded_idx + 1, idx + 1):
            self.mht.step(self.dataloader[i])
            self.loaded_idx = i
            yield i, self.mht_data.get_data(i)
        self.loaded_idx = max(self.loaded_idx, idx)


class SimpleDataloader(object):
    # Caches last retrieved item

    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.lastIdx = None
        self.lastItem = None

    def __getitem__(self, item):
        if item != self.lastIdx:
            self.lastIdx = item
            self.lastItem = self.load_item(item)
        return self.lastItem

    def load_item(self, idx):
        print("############# Loading item {} ###############".format(idx))
        scan = self.data_source.load_data(idx)
        scan.time = idx
        return scan

    def __len__(self):
        return len(self.data_source)


class AllDataloader(SimpleDataloader):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data = [None] * len(self)
        self.startThread()

    def getitem(self, idx):
        if self.data[idx] is not None:
            return self.data[idx]
        else:
            return super().__getitem__(idx)

    def startThread(self):
        t = threading.Thread(target=self.loadAllData, daemon=True)
        t.start()

    def loadAllData(self):
        for i in range(0, len(self)):
            self.data[i] = self.load_item(i)
