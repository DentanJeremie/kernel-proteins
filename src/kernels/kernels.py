import numpy as np
from tqdm import tqdm

from src.utils.logs import logger
from src.utils.constants import *
from src.utils.graphs import graph_manager

class Kernel(object):

    def __init__(self, name='empty_kernel') -> None:
        self.name = name
        self._distances_matrix = None

    @property
    def features(self) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> np.ndarray:
        """Returns a view of the feature vector of the graph with index `idx`.
        Warning: this is view, so a modification in place affects the kernel.
        """
        assert type(idx) == int, "You can only get the feature vector by provinding the index (type int) of a graph!"
        return self.features[idx,:]

    def __call__(self, idx_0:int, idx_1:int) -> float:
        """Returns the kernel evaluation between two graphs IDs.

        :param key: A tuple (idx_0, idx_0) of graph indexes
        :returns: The value K(G_idx_0, G_idx_1)
        """
        assert type(idx_0) == int, "You can only get the kernel evaluation of a tuple of 2 indexes"
        assert type(idx_1) == int, "You can only get the kernel evaluation of a tuple of 2 indexes"
        return self[idx_0]@self[idx_1]

# ------------------ DISTANCES ------------------

    def dist(self, idx_0:int, idx_1:int) -> float:
        """Returns the distance between two graphs IDs.

        :param key: A tuple (idx_0, idx_0) of graph indexes
        :returns: The value dist(G_idx_0, G_idx_1) = K(G_idx_0, G_idx_0) + K(G_idx_1, G_idx_1) - 2*K(G_idx_0, G_idx_1)
        """
        assert type(idx_0) == int, "You can only get the distance of a tuple of 2 indexes"
        assert type(idx_1) == int, "You can only get the distance of a tuple of 2 indexes"
        return self(idx_0, idx_0) + self(idx_1, idx_1) - 2*self(idx_0, idx_1)

    @property
    def distances_matrix(self):
        if self._distances_matrix is None:
            self.build_distances_matrix()
        return self._distances_matrix

    def build_distances_matrix(self):
        self._distances_matrix = np.infty * np.ones((NUM_LABELED + NUM_TEST, NUM_LABELED))
        self.dist(0,0) # To build the kernel
        logger.info(f'Computing distances valid->train for kernel {self.name}')
        for _, _, _, idx_valid in tqdm(graph_manager.valid):
            for _, _, _, idx_train in graph_manager.train:
                self._distances_matrix[idx_valid, idx_train] = self.dist(idx_valid, idx_train)
        logger.info(f'Computing distances test->labeled for kernel {self.name}')
        for _, _, _, idx_test in tqdm(graph_manager.test):
            for _, _, grp, idx_labeled in graph_manager.full:
                if grp != 2:
                    self._distances_matrix[idx_test, idx_labeled] = self.dist(idx_test, idx_labeled)