from typing import List, Optional

import numpy as np
import torch

from datasets import Dataset
from flwr_datasets.partitioner import Partitioner
from torch.utils.data import Subset


class DirichletPartitioner(Partitioner):
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """

    def __init__(
            self,
            num_clients: int,
            alpha: float = 0.5,
            seed: int = 42,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.split_datasets: Optional[List[Dataset]] = None

    def partition(self):

        trainset = self.dataset
        return []
        # min_required_samples_per_client = 10
        # min_samples = 0
        # prng = np.random.default_rng(self.seed)
        #
        # # get the targets
        # tmp_t = self.dataset.targets
        # if isinstance(tmp_t, list):
        #     tmp_t = np.array(tmp_t)
        # if isinstance(tmp_t, torch.Tensor):
        #     tmp_t = tmp_t.numpy()
        # num_classes = len(set(tmp_t))
        # total_samples = len(tmp_t)
        # while min_samples < min_required_samples_per_client:
        #     idx_clients: List[List] = [[] for _ in range(self.num_clients)]
        #     for k in range(num_classes):
        #         idx_k = np.where(tmp_t == k)[0]
        #         prng.shuffle(idx_k)
        #         proportions = prng.dirichlet(np.repeat(self.alpha, self.num_clients))
        #         proportions = np.array(
        #             [
        #                 p * (len(idx_j) < total_samples / self.num_clients)
        #                 for p, idx_j in zip(proportions, idx_clients)
        #             ]
        #         )
        #         proportions = proportions / proportions.sum()
        #         proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        #         idx_k_split = np.split(idx_k, proportions)
        #         idx_clients = [
        #             idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
        #         ]
        #         min_samples = min([len(idx_j) for idx_j in idx_clients])
        #
        # trainsets_per_client = [Subset(self.dataset, idxs) for idxs in idx_clients]
        # return trainsets_per_client

    def load_partition(self, node_id: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        node_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        if self.split_datasets is None:
            self.split_datasets = self.partition()
        return self.split_datasets[node_id]
