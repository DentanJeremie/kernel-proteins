import pickle

import numpy as np
from tqdm import tqdm

from src.utils.logs import logger
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.pathtools import project

class Kernel(object):

    def __init__(self, name:str='empty_kernel', force_from_scratch:bool=False) -> None:
        self.name = name
        self._distances_matrix = None
        self._features = None
        self._force_from_scratch = force_from_scratch

    @property
    def features(self):
        if self._features is None:
            self.load()
        if self._features is None:
            self.build_features()
        return self._features

    @property
    def distances_matrix(self):
        if self._distances_matrix is None:
            self.load()
        if self._distances_matrix is None:
            _ = self.features
            self.build_distances_matrix()
        return self._distances_matrix

    def build_features(self):
        raise NotImplementedError

# ------------------ GETTER, CALLER ------------------

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

    def build_distances_matrix(self):
        # Building the distance matrix without saving it for the moment
        _distances_matrix = np.infty * np.ones((NUM_LABELED + NUM_TEST, NUM_LABELED))
        logger.info(f'Computing distances valid->train for kernel {self.name}')
        for _, _, _, idx_valid in tqdm(graph_manager.valid):
            for _, _, _, idx_train in graph_manager.train:
                _distances_matrix[idx_valid, idx_train] = self.dist(idx_valid, idx_train)
        logger.info(f'Computing distances test->labeled for kernel {self.name}')
        for _, _, _, idx_test in tqdm(graph_manager.test):
            for _, _, grp, idx_labeled in graph_manager.full:
                if grp != 2:
                    _distances_matrix[idx_test, idx_labeled] = self.dist(idx_test, idx_labeled)

        # Now, saving the distance matrix
        self._distances_matrix = _distances_matrix
        self.save()

# ------------------ SAVING ------------------

    def save(self):
        """
        Saves the kernel and all its attributes.
        Please ensure consistency in attribute names with self.load.
        """
        logger.info(f'Saving kernel {self.name} at {project.as_relative(project.get_kernel_file_path(self.name))}')
        storable = {
            '_features':self._features,
            '_distances_matrix':self._distances_matrix,
        }
        with project.get_kernel_file_path(self.name).open('wb') as f:
            pickle.dump(storable, f)

    def load(self):
        """
        Loads the kernel and all its attributes.
        Returns None if the file is not found.
        Please ensure consistency in attribute names with self.save.
        """
        if self._force_from_scratch:
            logger.info('Skipping disk loading since `force_from_scratch=True`')
            return 

        logger.info(f'Loading kernel {self.name} from {project.as_relative(project.get_kernel_file_path(self.name))}')
        if not project.get_kernel_file_path(self.name).exists():
            logger.info(f'File not found, skipping loading from disk.')
            return

        with project.get_kernel_file_path(self.name).open('rb') as f:
            storable = pickle.load(f)

        if storable['_features'] is not None:
            self._features = storable['_features']
        if storable['_distances_matrix'] is not None:
            self._distances_matrix = storable['_distances_matrix']