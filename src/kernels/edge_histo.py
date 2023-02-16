import numpy as np

from src.kernels.kernels import Kernel
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger


class EdgeHisto(Kernel):

    def __init__(self):
        super().__init__(name='edge_histo')
        self._features = None

    @property
    def features(self):
        if self._features is None:
            self.build_histograms()
        return self._features

    def build_histograms(self) -> None:
        """Builds the edge histograms."""
        logger.info('Building the edge histograph kernel.')
        self._features = np.zeros((NUM_LABELED + NUM_TEST, EDGE_TYPE_NUMBER))
        for index, (gph, _, _, _) in enumerate(graph_manager.full):
            for node in gph.edges:
                self._features[
                    index,
                    gph.nodes[node][LAB_CLM][0]
                ] += 1
        logger.info('Vertex histograph kernel built.')


def main():
    vh = EdgeHisto()
    print(vh[0])
    for node in graph_manager[0][0].nodes:
        print(graph_manager[0][0].nodes[node][LAB_CLM][0])
    print(vh[1])
    for node in graph_manager[1][0].nodes:
        print(graph_manager[1][0].nodes[node][LAB_CLM][0])
    print(vh(0,1))
    print(vh(1,0))


if __name__ == '__main__':
    main()
