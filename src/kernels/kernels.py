import typing as t

import numpy as np

class Kernel(object):

    def __init__(self, name='') -> None:
        self.name = name

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

    def dist(self, idx_0:int, idx_1:int) -> float:
        """Returns the distance between two graphs IDs.

        :param key: A tuple (idx_0, idx_0) of graph indexes
        :returns: The value dist(G_idx_0, G_idx_1) = K(G_idx_0, G_idx_0) + K(G_idx_1, G_idx_1) - 2*K(G_idx_0, G_idx_1)
        """
        assert type(idx_0) == int, "You can only get the distance of a tuple of 2 indexes"
        assert type(idx_1) == int, "You can only get the distance of a tuple of 2 indexes"
        return self(idx_0, idx_0) + self(idx_1, idx_1) - 2*self(idx_0, idx_1)