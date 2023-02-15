import pickle
import typing as t

import networkx as nx

from src.utils.pathtools import project
from src.utils.logs import logger
from src.utils.constants import *

class GraphManager():

    def __init__(self) -> None:
        # Datasets promises
        self._train_data = None
        self._test_data = None
        self._labels_data = None

        # Iteration mode
        self._iterator_start = 0
        self._iterator_stop = NUM_TRAIN + NUM_TEST

    @property
    def train_data(self):
        if self._train_data is None:
            logger.info(f'Loading train data from {project.as_relative(project.train_data_path)}')
            with project.train_data_path.open('rb') as f:
                self._train_data = pickle.load(f)
        return self._train_data

    @property
    def test_data(self):
        if self._test_data is None:
            logger.info(f'Loading test data from {project.as_relative(project.test_data_path)}')
            with project.test_data_path.open('rb') as f:
                self._test_data = pickle.load(f)
        return self._test_data

    @property
    def labels_data(self):
        if self._labels_data is None:
            logger.info(f'Loading labels data from {project.as_relative(project.labels_data_path)}')
            with project.labels_data_path.open('rb') as f:
                self._labels_data = pickle.load(f)
        return self._labels_data

# ------------------ GETTERS ------------------

    def __getitem__(self, idx) -> t.Tuple[nx.Graph, int]:
        """
        The indexes are as follows:
        * indexes [0, 1, ..., 5999] are for the training graphs
        * indexes [6000, 6001, ... 7999] are for the testing graphs

        :param idx: The index of the graph to get
        :returns: A tupple `(graph, label)`. The graph is a `nx.Graph` and 
        the label is either 0, 1, or `None` for the testing graphs.
        """
        assert type(idx) == int, 'The indexes of graphs are integers.'
        assert idx>=0 and idx<NUM_TRAIN+NUM_TEST, 'Index not foud.'

        if idx<NUM_TRAIN:
            return self.train_data[idx], self.labels_data[idx]
        
        else:
            return self.test_data[idx - NUM_TRAIN], None

    def get_train(self, idx) -> t.Tuple[nx.Graph, int]:
        """
        Shortcut for self[idx]
        """
        return self[idx]

    def get_test(self, idx) -> t.Tuple[nx.Graph, int]:
        """
        Shortcut for self[idx - NUM_TRAIN]
        """
        return self[idx - NUM_TRAIN]

# ------------------ ITERATORS ------------------

    @property
    def train(self):
        """Returns an iterator over the train set"""
        self._iterator_start = 0
        self._iterator_stop = NUM_TRAIN
        return self

    @property
    def test(self):
        """Returns an iterator over the test set"""
        self._iterator_start = NUM_TRAIN
        self._iterator_stop = NUM_TRAIN + NUM_TEST
        return self

    @property
    def full(self):
        """Returns an iterator over the full set"""
        self._iterator_start = 0
        self._iterator_stop = NUM_TRAIN + NUM_TEST
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator_start < self._iterator_stop:
            self._iterator_start += 1
            return self[self._iterator_start - 1]
        
        raise StopIteration


graph_manager = GraphManager()