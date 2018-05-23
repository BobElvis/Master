import dataloader


class SimulatorSource(dataloader.DataSource):
    def __init__(self):
        self.n_steps = 60
        pass

    def load_data(self, idx):
        pass

    def __len__(self):
        return self.n_steps