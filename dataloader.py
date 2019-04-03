import cv2
from model.model import Scan
import util
import threading
import time
import abc
from timer import SimpleTimer


class DataSource:
    def load_data(self, idx) -> Scan:
        pass

    def __len__(self):
        pass


class BaseLoader:
    # Caches last retrieved item

    def __init__(self):
        self.lastIdx = None
        self.lastItem = None

    def __getitem__(self, item):
        if item != self.lastIdx:
            self.lastIdx = item
            self.lastItem = self.load_item(item)
        return self.lastItem

    def load_item(self, idx):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class MHTLoader(BaseLoader):

    def __init__(self, mht, dataloader, i_start=0, disabled=False):
        super().__init__()
        self.mht = mht
        self.mht_data = mht.mht_data
        self.dataloader = dataloader
        self.loaded_idx = i_start - 1
        self.disabled = disabled

    def load_item(self, idx):
        if self.disabled:
            return self.mht_data.get_data(idx)
        t = SimpleTimer()
        t.set("Loading mht")
        for i in range(self.loaded_idx + 1, idx + 1):
            self.mht.step(self.dataloader[i])
        t.report(1)
        self.loaded_idx = max(self.loaded_idx, idx)
        return self.mht_data.get_data(idx)

    def __len__(self):
        len(self.dataloader)


class SimpleDataloader(BaseLoader):
    # Caches last retrieved item

    def __init__(self, data_source: DataSource):
        super().__init__()
        self.data_source = data_source
        self.lastIdx = None
        self.lastItem = None

    def load_item(self, idx):
        item = self.data_source.load_data(idx)
        return item

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
