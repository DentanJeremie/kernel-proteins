import networkx as nx
import numpy as np

from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger


class VertexHisto(object):

    def __init__(self):
        self._vertex_histographs = None

    @property
    def vertex_histographs(self):
        if self._vertex_histographs is None:
            self.build_histograms()
        return self._vertex_histographs

    def build_histograms(self) -> None:
        """Builds the vertex histograms."""
        self._vertex_histographs = np.zeros((NUM_TRAIN + NUM_TEST, NODE_TYPE_NUMBER))
        for index, (gph, label) in enumerate(graph_manager.full):
            raise NotImplementedError
