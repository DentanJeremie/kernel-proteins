import pickle

import numpy as np

from src.utils.logs import logger
from src.utils.constants import *
from src.utils.pathtools import project

class BaseKernel(object):

    def __init__(self, name:str='empty_kernel', force_from_scratch:bool=False) -> None:
        self.name = name
        self._distances_matrix = None
        self._kernel_matrix = None
        self._force_from_scratch = force_from_scratch

    @property
    def kernel_matrix(self) -> np.ndarray:
        if self._kernel_matrix is None:
            self.load()
        if self._kernel_matrix is None:
            self.build_kernel_matrix()
        return self._kernel_matrix

    @property
    def distances_matrix(self) -> np.ndarray:
        if self._distances_matrix is None:
            self.load()
        if self._distances_matrix is None:
            _ = self.kernel_matrix
            self.build_distances_matrix()
        return self._distances_matrix

    def build_kernel_matrix(self):
        raise NotImplementedError

# ------------------ CALLER ------------------

    def __call__(self, idx_0:int, idx_1:int) -> float:
        """Returns the kernel evaluation between two graphs IDs.

        :param key: A tuple (idx_0, idx_0) of graph indexes
        :returns: The value K(G_idx_0, G_idx_1)
        """
        assert type(idx_0) == int, "You can only get the kernel evaluation of a tuple of 2 indexes"
        assert type(idx_1) == int, "You can only get the kernel evaluation of a tuple of 2 indexes"
        return self.kernel_matrix[idx_0, idx_1]

# ------------------ CENTER KERNEL MATRIX ------------------

    def _center_kernel_matrix(self):
        """Centers the kernel matrix.
        """
        logger.warning(f'Deprecated: we observed that centering the kernel decreases the performances.')
        N = NUM_LABELED + NUM_TEST
        centering_matrix = np.eye(N) - np.ones((N,N)) / N
        self._kernel_matrix = centering_matrix@self._kernel_matrix@centering_matrix

# ------------------ DISTANCES ------------------

    def dist(self, idx_0:int, idx_1:int) -> float:
        """Returns the distance between two graphs IDs.

        :param key: A tuple (idx_0, idx_0) of graph indexes
        :returns: The value dist(G_idx_0, G_idx_1) = K(G_idx_0, G_idx_0) + K(G_idx_1, G_idx_1) - 2*K(G_idx_0, G_idx_1)
        """
        assert type(idx_0) == int, "You can only get the distance of a tuple of 2 indexes"
        assert type(idx_1) == int, "You can only get the distance of a tuple of 2 indexes"
        return self.distances_matrix[idx_0, idx_1]

    def build_distances_matrix(self):
        # Building the distance matrix without saving it for the moment
        diag = np.diag(self.kernel_matrix)
        _distances_matrix = diag[:, np.newaxis] + diag - 2*self.kernel_matrix

        # Saving
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
            '_kernel_matrix':self._kernel_matrix,
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

        if storable['_kernel_matrix'] is not None:
            self._kernel_matrix = storable['_kernel_matrix']
        if storable['_distances_matrix'] is not None:
            self._distances_matrix = storable['_distances_matrix']


class EmptyKernel(BaseKernel):

    def __init__(self, force_from_scratch:bool=False):
        super().__init__(name='empty_kernel', force_from_scratch=force_from_scratch)

    def build_kernel_matrix(self) -> None:
        """Builds an empty kernel."""
        logger.info('Building an empty histograph kernel.')
        _kernel_matrix = np.zeros((NUM_LABELED + NUM_TEST, NUM_LABELED + NUM_TEST))
        logger.info('Empty kernel built.')

        # Saving
        self._kernel_matrix = _kernel_matrix
        self.save()
