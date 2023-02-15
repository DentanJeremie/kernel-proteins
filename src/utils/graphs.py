import pickle
import typing as t

import networkx as nx
from numpy.random import RandomState

from src.utils.pathtools import project
from src.utils.logs import logger
from src.utils.constants import *

class GraphManager():

    def __init__(self) -> None:
        # Data promises read from disk
        self._labeled_data = None
        self._test_data = None
        self._labels_raw = None

        # Built datasets
        self._train_data = None
        self._valid_data = None
        self._labels_data = None

        # Iteration mode
        self._iterator_start = 0
        self._iterator_stop = NUM_TRAIN + NUM_TEST

# ------------------ PROPERTIES ------------------ 

    @property
    def labeled_data(self):
        if self._labeled_data is None:
            logger.info(f'Loading train data from {project.as_relative(project.train_data_path)}')
            with project.train_data_path.open('rb') as f:
                self._labeled_data = pickle.load(f)
        return self._labeled_data

    @property
    def test_data(self):
        if self._test_data is None:
            logger.info(f'Loading test data from {project.as_relative(project.test_data_path)}')
            with project.test_data_path.open('rb') as f:
                self._test_data = pickle.load(f)
        return self._test_data

    @property
    def labels_raw(self):
        if self._labels_raw is None:
            logger.info(f'Loading labels data from {project.as_relative(project.labels_data_path)}')
            with project.labels_data_path.open('rb') as f:
                self._labels_raw = pickle.load(f)
        return self._labels_raw

    @property
    def train_data(self):
        if self._train_data is None:
            self.split_train_valid()
        return self._train_data

    @property
    def valid_data(self):
        if self._valid_data is None:
            self.split_train_valid()
        return self._valid_data
    
    @property
    def labels_data(self):
        if self._labels_data is None:
            self.split_train_valid()
        return self._labels_data

# ------------------ SPLIT TRAIN / VALID ------------------ 

    def split_train_valid(self):
        random_state = RandomState(SEED)
        validation_indexes = random_state.choice(
            NUM_LABELED,
            NUM_VALID,
            replace=False,
        )

        self._train_data = [
            self.labeled_data[index]
            for index in range(NUM_LABELED)
            if index not in validation_indexes
        ]
        self._valid_data = [
            self.labeled_data[index]
            for index in range(NUM_LABELED)
            if index in validation_indexes
        ]
        self._labels_data = [
            self.labels_raw[index]
            for index in range(NUM_LABELED)
            if index not in validation_indexes
        ] + [
            self.labels_raw[index]
            for index in range(NUM_LABELED)
            if index in validation_indexes
        ]

# ------------------ GETTERS ------------------

    def __getitem__(self, idx) -> t.Tuple[nx.Graph, int, int, int]:
        """
        The indexes are as follows:
        * indexes [0, 1, ..., 4999] are for the training graphs
        * indexes [5000, 5001, ..., 5999] are for the validation graphs
        * indexes [6000, 6001, ... 7999] are for the testing graphs

        :param idx: The index of the graph to get
        :returns: A tupple `(graph, label, group, index)`. The graph is a `nx.Graph`, 
        the label is either 0, 1, or `None` for the testing graphs,
        , `group` is `0` for train, `1` for valid, and `2` for test, and `index` is ìdx.
        """
        assert type(idx) == int, 'The indexes of graphs are integers.'
        assert idx>=0 and idx<NUM_LABELED+NUM_TEST, 'Index not foud.'

        if idx<NUM_TRAIN:
            return self.train_data[idx], self.labels_data[idx], 0, idx
        elif idx<NUM_LABELED:
            return self.valid_data[idx - NUM_TRAIN], self.labels_data[idx], 1, idx
        else:
            return self.test_data[idx - NUM_LABELED], None, 2, idx

    def get_train(self, idx) -> t.Tuple[nx.Graph, int]:
        """
        Shortcut for self[idx]
        """
        return self[idx]

    def get_valid(self, idx) -> t.Tuple[nx.Graph, int]:
        """
        Shortcut for self[idx - NUM_TRAIN]
        """
        return self[idx - NUM_TRAIN]
    
    def get_test(self, idx) -> t.Tuple[nx.Graph, int]:
        """
        Shortcut for self[idx - NUM_LABELED]
        """
        return self[idx - NUM_LABELED]

# ------------------ ITERATORS ------------------

    @property
    def train(self):
        """Returns an iterator over the train set"""
        self._iterator_start = 0
        self._iterator_stop = NUM_TRAIN
        return self

    @property
    def valid(self):
        """Returns an iterator over the validation set"""
        self._iterator_start = NUM_TRAIN
        self._iterator_stop = NUM_LABELED
        return self

    @property
    def test(self):
        """Returns an iterator over the test set"""
        self._iterator_start = NUM_LABELED
        self._iterator_stop = NUM_LABELED + NUM_TEST
        return self

    @property
    def full(self):
        """Returns an iterator over the full set"""
        self._iterator_start = 0
        self._iterator_stop = NUM_LABELED + NUM_TEST
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator_start < self._iterator_stop:
            self._iterator_start += 1
            return self[self._iterator_start - 1]
        
        raise StopIteration


graph_manager = GraphManager()

def main():
    for index, item in enumerate(graph_manager.train):
        if index < 10 or index>4990:
            print(f'{index}: {item}')

    for index, item in enumerate(graph_manager.valid):
        if index < 10 or index>990:
            print(f'{index}: {item}')

    for index, item in enumerate(graph_manager.test):
        if index < 10 or index>1990:
            print(f'{index}: {item}')

    for index, item in enumerate(graph_manager.full):
        if index < 10 or index>7990:
            print(f'{index}: {item}')

if __name__ == '__main__':
    main()