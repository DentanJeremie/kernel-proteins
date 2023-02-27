import numpy as np
from tqdm import tqdm

from src.classifiers.classifiers import BaseClassifier
from src.kernels.kernels import BaseKernel
from src.utils.constants import *
from src.utils.graphs import graph_manager
from src.utils.logs import logger

EPSILON = 10e-4


class KNN(BaseClassifier):

    def __init__(self, kernel:BaseKernel, num_neighbors:int) -> None:
        logger.info(f'Initializing a KNN classifier with num_neighbors={num_neighbors}') 
        super().__init__(kernel, name=f'knn_{num_neighbors}')
        self.num_neighbors = num_neighbors

    def predict(self, idx: int) -> int:
        argsort = np.argsort(self.kernel.distances_matrix[idx,:NUM_TRAIN])
        labels = [
            graph_manager[idx_neighbor][1]
            for idx_neighbor in argsort[:self.num_neighbors]
        ]
        return np.mean(labels)


class FixedWeightedKNN(BaseClassifier):

    def __init__(self, kernel:BaseKernel, num_neighbors:int) -> None:
        logger.info(f'Initializing a FixedWeightedKNN classifier with num_neighbors={num_neighbors}') 
        super().__init__(kernel, name=f'fixed_weighted_knn_{num_neighbors}')
        self.num_neighbors = num_neighbors

    def predict(self, idx: int) -> int:
        argsort = np.argsort(self.kernel.distances_matrix[idx,:NUM_TRAIN])
        labels = np.array([
            graph_manager[idx_neighbor][1]
            for idx_neighbor in argsort[:self.num_neighbors]
        ])
        weights = np.array([
            1 - index/self.num_neighbors
            for index in range(self.num_neighbors)
        ])
        return np.sum(labels * weights) / np.sum(weights)
    

class InverseWeightedKNN(BaseClassifier):

    def __init__(self, kernel:BaseKernel, num_neighbors:int) -> None:
        logger.info(f'Initializing a InverseWeightedKNN classifier with num_neighbors={num_neighbors}') 
        super().__init__(kernel, name=f'inverse_weighted_knn_{num_neighbors}')
        self.num_neighbors = num_neighbors

    def predict(self, idx: int) -> int:
        argsort = np.argsort(self.kernel.distances_matrix[idx,:NUM_TRAIN])
        labels = np.array([
            graph_manager[idx_neighbor][1]
            for idx_neighbor in argsort[:self.num_neighbors]
        ])
        weights = np.array([
            1/(self.kernel.distances_matrix[idx, idx_neighbor] + EPSILON)
            for idx_neighbor in argsort[:self.num_neighbors]
        ])
        return np.sum(labels * weights) / np.sum(weights)

def main():
    from src.kernels.histograms import VertexHisto
    knn = KNN(VertexHisto(),num_neighbors=5)
    knn.evaluate()
    knn.make_submission()
    knn = FixedWeightedKNN(VertexHisto(),num_neighbors=5)
    knn.evaluate()
    knn.make_submission()
    knn = InverseWeightedKNN(VertexHisto(),num_neighbors=5)
    knn.evaluate()
    knn.make_submission()

if __name__ == '__main__':
    main()


    

