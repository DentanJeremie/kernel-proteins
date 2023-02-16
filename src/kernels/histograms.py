import numpy as np

from src.kernels.kernels import Kernel
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger


class EdgeHisto(Kernel):

    def __init__(self, force_from_scratch:bool=False):
        super().__init__(name='edge_histo', force_from_scratch=force_from_scratch)

    def build_features(self) -> None:
        """Builds the edge histograms."""
        logger.info('Building the edge histograph kernel.')
        _features = np.zeros((NUM_LABELED + NUM_TEST, EDGE_TYPE_NUMBER))
        for index, (gph, _, _, _) in enumerate(graph_manager.full):
            for edge in gph.edges:
                _features[
                    index,
                    gph.edges[edge][LAB_CLM][0]
                ] += 1
        logger.info('Edge histograph kernel built.')

        # Saving
        self._features = _features
        self.save()


class VertexHisto(Kernel):

    def __init__(self, force_from_scratch:bool=False):
        super().__init__(name='vertex_histo', force_from_scratch=force_from_scratch)

    def build_features(self) -> None:
        """Builds the vertex histograms."""
        logger.info('Building the vertex histograph kernel.')
        _features = np.zeros((NUM_LABELED + NUM_TEST, NODE_TYPE_NUMBER))
        for index, (gph, _, _, _) in enumerate(graph_manager.full):
            for node in gph.nodes:
                _features[
                    index,
                    gph.nodes[node][LAB_CLM][0]
                ] += 1
        logger.info('Vertex histograph kernel built.')

        # Saving
        self._features = _features
        self.save()


class EdgeVertexHisto(Kernel):

    def __init__(self, force_from_scratch:bool=False):
        super().__init__(name='edge_vertex_histo', force_from_scratch=force_from_scratch)

    def build_features(self) -> None:
        """Builds the edge and vertex histograms."""
        logger.info('Building the edge and vertex histograph kernel.')
        _features = np.zeros((NUM_LABELED + NUM_TEST, EDGE_TYPE_NUMBER + NODE_TYPE_NUMBER))
        for index, (gph, _, _, _) in enumerate(graph_manager.full):
            for edge in gph.edges:
                _features[
                    index,
                    gph.edges[edge][LAB_CLM][0]
                ] += 1
            for node in gph.nodes:
                _features[
                    index,
                    gph.nodes[node][LAB_CLM][0] + EDGE_TYPE_NUMBER
                ] += 1
        logger.info('Edge and vertex histograph kernel built.')

        # Saving
        self._features = _features
        self.save()


def main():
    print('Edge histo')
    vh = EdgeHisto()
    print(vh[0])
    for edge in graph_manager[0][0].edges:
        print(graph_manager[0][0].edges[edge][LAB_CLM][0])
    print(vh[1])
    for edge in graph_manager[1][0].edges:
        print(graph_manager[1][0].edges[edge][LAB_CLM][0])
    print(vh(0,1))
    print(vh(1,0))

    print('Vertex histo')
    vh = VertexHisto()
    print(vh[0])
    for node in graph_manager[0][0].nodes:
        print(graph_manager[0][0].nodes[node][LAB_CLM][0])
    print(vh[1])
    for node in graph_manager[1][0].nodes:
        print(graph_manager[1][0].nodes[node][LAB_CLM][0])
    print(vh(0,1))
    print(vh(1,0))

    print('Edge vertex histo')
    vh = EdgeVertexHisto()
    print(vh(0,1))
    print(vh(1,0))


if __name__ == '__main__':
    main()
