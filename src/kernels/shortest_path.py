import collections
import typing as t

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.kernels.kernels import BaseKernel
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger


class ShortestPath(BaseKernel):
    
    def __init__(self, force_from_scratch:bool=False):
        super().__init__(name=f'shortest_path', force_from_scratch=force_from_scratch)
        # Cf. fo more details https://ysig.github.io/GraKeL/0.1a8/kernels/shortest_path.html
        # We use diract kernel both for the vertex label kernel and the path length kernel

    def build_kernel_matrix(self) -> None:
        """Builds the shortest path kernel."""
        logger.info('Building the shortest path kernel.')

        # Loading
        _ = graph_manager[0]
        _ = graph_manager[NUM_TRAIN]
        _ = graph_manager[NUM_LABELED]
        
        # Building the shortest paths lengths
        # shortest_path = dict{
        #    idx_graph -> dict{
        #       (label_node_source, label_node_target) -> set{sp_lengths of that type}
        # }
        # }
        logger.info(f'Computing all SP lengths')
        shortest_path: t.Dict[int, t.Dict[t.Tuple[int, int], t.Set[int]]] = dict()
        for gph, _, _, idx in tqdm(graph_manager.full):
            shortest_path[idx] = collections.defaultdict(set)
            sp_generator = nx.shortest_path_length(gph)
            for node_source, target_dict in sp_generator:
                for node_target, path_length in target_dict.items():
                    shortest_path[idx][(
                        gph.nodes[node_source][LAB_CLM][0],
                        gph.nodes[node_target][LAB_CLM][0],
                    )].add(path_length)

        # Building the kernle matrix
        # We use dirac kernels: value 1 iff equality
        logger.info('Building the kernel matrix')
        _kernel_matrix = np.zeros((NUM_LABELED+NUM_TEST, NUM_LABELED+NUM_TEST))
        for idx_0 in tqdm(range(NUM_LABELED+NUM_TEST)):
            for idx_1 in range(idx_0, NUM_LABELED+NUM_TEST):

                count = 0
                for node_label_tuple, lengths_set in shortest_path[idx_0].items():
                    count += len(shortest_path[idx_1][node_label_tuple].intersection(lengths_set))
                
                _kernel_matrix[idx_0, idx_1] = count
                _kernel_matrix[idx_1, idx_0] = count

        # Saving
        self._kernel_matrix = _kernel_matrix
        self._center_kernel_matrix()
        self.save()

def main():
    _ = ShortestPath().kernel_matrix

if __name__ == '__main__':
    main()