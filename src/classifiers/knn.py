import numpy as np
from tqdm import tqdm

from src.classifiers.classifiers import Classifier
from src.kernels.kernels import Kernel
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger


class KNN(Classifier):

    def __init__(self, kernel:Kernel, num_neighbors:int) -> None:
        logger.info(f'Initializing a KNN classifier with num_neighbors={num_neighbors}')
        self.kernel = kernel    
        self.num_neighbors = num_neighbors

        # Distance promise
        self._distances_matrix = None
        
        # Note for the submission
        self.note = f'knn_{self.num_neighbors}'

    @property
    def distances_matrix(self):
        if self._distances_matrix is None:
            self.build_distances_matrix()
        return self._distances_matrix

    def build_distances_matrix(self):
        self._distances_matrix = np.zeros((NUM_LABELED + NUM_TEST, NUM_TRAIN))
        self.kernel.dist(0,0) # To build the kernel
        logger.info('Computing distances valid->train')
        for _, _, _, idx_valid in tqdm(graph_manager.valid):
            for _, _, _, idx_train in graph_manager.train:
                self._distances_matrix[idx_valid, idx_train] = self.kernel.dist(idx_valid, idx_train)
        logger.info('Computing distances test->train')
        for _, _, _, idx_test in tqdm(graph_manager.test):
            for _, _, _, idx_train in graph_manager.train:
                self._distances_matrix[idx_test, idx_train] = self.kernel.dist(idx_test, idx_train)

    def predict(self, idx: int) -> int:
        argsort = np.argsort(self.distances_matrix[idx,:])
        labels = [
            graph_manager[idx_neighbor][1]
            for idx_neighbor in argsort[:self.num_neighbors]
        ]
        if np.mean(labels) < 0.5:
            return 0
        return 1

def main():
    from src.kernels.vertex_histo import VertexHisto
    knn = KNN(VertexHisto(),num_neighbors=5)
    knn.evaluate()
    knn.make_submission()

if __name__ == '__main__':
    main()


    

