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
        super().__init__(kernel, note=f'knn_{num_neighbors}')
        self.num_neighbors = num_neighbors

    def predict(self, idx: int) -> int:
        argsort = np.argsort(self.kernel.distances_matrix[idx,:])
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


    

