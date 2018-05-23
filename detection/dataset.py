import os


def find_folders(dir):
    dates = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    dates.sort()
    return dates


def make_dataset(dir):
    dataset = []
    dates = find_folders(dir)
    for date in dates:
        items = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(dir, date)) if f.endswith(".png")]
        items.sort()
        dataset.append((date, items))
    return dataset


class Dataset(object):
    def __init__(self, root, dateidx = 0):
        self.root = root
        self.dates = find_folders(root)
        self.dataset = make_dataset(root)
        self.dateIdx = dateidx
    def __getitem__(self, index):
        return os.path.join(self.root, self.dates[self.dateIdx], self.dataset[self.dateIdx][1][index])

    def __len__(self):
        return len(self.dataset[self.dateIdx][1])
