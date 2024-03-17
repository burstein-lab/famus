from app.sdfloader import SDFloader
from torch.utils.data import IterableDataset


class IterableTripletDataset(IterableDataset):
    def __init__(self, sdfloader: SDFloader):
        self.sdfloader = sdfloader

    def __iter__(self):
        return iter(self.sdfloader.triplet_batch_generator())

    def get_num_batches(self, batch_size: int):
        return self.sdfloader.get_num_batches(batch_size)
