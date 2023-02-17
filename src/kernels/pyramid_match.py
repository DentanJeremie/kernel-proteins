import os
import typing as t

import networkx as nx
import numpy as np
from p_tqdm import p_map
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from tqdm import tqdm

from src.kernels.kernels import BaseKernel
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger

class PyramidMatch(BaseKernel):

    def __init__(self, d=6, l=4, force_from_scratch:bool=False):
        super().__init__(name=f'pyramid_match_l_{l}_d_{d}', force_from_scratch=force_from_scratch)
        # Parameter d is the dimension used for the pyramid match
        # Parameter l is the level
        # Cf. fo more details https://ysig.github.io/GraKeL/0.1a7/kernels/pyramid_match.html

        self.d = d
        self.l = l

    def build_kernel_matrix(self) -> None:
        """Builds the pyramid match kernel."""
        logger.info('Building the pyramid match kernel.')

        # Eigenvalue decomposition
        logger.info('Doing the eigenvalue decomposition of the adjacency matrixes')
        # Inspired from https://github.com/ysig/GraKeL/blob/65895874/grakel/kernels/pyramid_match.py
        # `embeddings_list[0]` will be a `np.ndarray` of shape (n,self.d) where n
        # is the number of nodes in graph 0.
        embeddings_list: t.List[np.ndarray] = list()
        node_labels_list: t.List[t.List[int]] = list()
        num_nodes_list: t.List[int] = list()
        for gph, _, _, _ in tqdm(graph_manager.full):
            adjacency_matrix:np.ndarray = nx.adjacency_matrix(gph)
            if adjacency_matrix.shape[0] > self.d+1:
                lmbda, eigs_matrix = eigs(
                    csr_matrix(adjacency_matrix, dtype=float),
                    k=self.d,
                    ncv=10*self.d,
                )
                idx = lmbda.argsort()[::-1]
                eigs_matrix = eigs_matrix[:, idx]
            else:
                lmbda, eigs_matrix = np.linalg.eig(adjacency_matrix.toarray())
                idx = lmbda.argsort()[::-1]
                eigs_matrix = eigs_matrix[:, idx]
                eigs_matrix = eigs_matrix[:, :self.d]
                # If size of graph smaller than d, pad with zeros
                if self.d - eigs_matrix.shape[1] > 0:
                    eigs_matrix = np.pad(eigs_matrix, [(0,0), (0, self.d - eigs_matrix.shape[1])])

            if eigs_matrix.shape[1] != self.d:
                logger.warn('checkpoint 1')
            if adjacency_matrix.shape[0] != len(gph.nodes):
                logger.warn('checkpoint 2')    
            if eigs_matrix.shape[0] != len(gph.nodes):
                logger.warn('checkpoint 3')        

            # Replace all components by their absolute value
            eigs_matrix = np.absolute(eigs_matrix)
            embeddings_list.append(eigs_matrix)
            node_labels_list.append([gph.nodes[node]['labels'][0] for node in gph.nodes])
            num_nodes_list.append(len(gph.nodes))

        # Building the histograms
        logger.info('Building the histograms for pyramid match kernel.')
        # Inspired from https://github.com/ysig/GraKeL/blob/65895874/grakel/kernels/pyramid_match.py
        histograms_by_graph: t.List[t.List[np.ndarray]] = list()
        for embeddings, labels, num_nodes in tqdm(zip(embeddings_list, node_labels_list, num_nodes_list), total=len(embeddings_list)):
            histograms_by_level = list()
            for level in range(self.l):
                # Number of cells along each dimension at level `level``
                num_cells = 2**level
                D = np.zeros((self.d*NODE_TYPE_NUMBER, num_cells))
                T = np.floor(embeddings*num_cells)
                T[T==num_cells] = num_cells-1
                for p in range(embeddings.shape[0]):
                    if p > num_nodes:
                        break
                    for q in range(embeddings.shape[1]):
                        # Identify the cell into which the i-th
                        # vertex lies and increase its value by 1
                        D[labels[p]*self.d + q, int(T[p, q])]  += 1
                histograms_by_level.append(D)
            histograms_by_graph.append(histograms_by_level)

        # Building the histograms
        logger.info(f'Building the kernel matrix for pyramid match kernel')
        # Inspired from https://github.com/ysig/GraKeL/blob/65895874/grakel/kernels/pyramid_match.py
        _kernel_matrix = np.zeros((NUM_LABELED+NUM_TEST, NUM_LABELED+NUM_TEST))
        for idx_0 in tqdm(range(NUM_LABELED+NUM_TEST)):
            for idx_1 in range(NUM_LABELED+NUM_TEST):
                count = 0
                x, y = histograms_by_graph[idx_0], histograms_by_graph[idx_1]
                if len(x) != 0 and len(y) != 0:
                    intersec = np.zeros(self.l)
                    for (p, xp, yp) in zip(range(self.l), x, y):
                        # Calculate histogram intersection
                        # (eq. 6 in :cite:`nikolentzos2017matching`)
                        if xp.shape[0] < yp.shape[0]:
                            xpp, ypp = xp, yp[:xp.shape[0], :]
                        elif yp.shape[0] < xp.shape[0]:
                            xpp, ypp = xp[:yp.shape[0], :], yp
                        else:
                            xpp, ypp = xp, yp
                        intersec[p] = np.sum(np.minimum(xpp, ypp))
                        count += intersec[self.l-1]
                        for p in range(self.l-1):
                            # Computes the new matches that occur at level p.
                            # These matches weight less than those that occur at
                            # higher levels (e.g. p+1 level)
                            count += (1.0/(2**(self.l-p-1)))*(intersec[p]-intersec[p+1])

            _kernel_matrix[idx_0, idx_1] = count
        logger.info('Pyramid match kernel built.')

        # Saving
        self._kernel_matrix = _kernel_matrix
        self.save()