import typing as t

from src.utils.logs import logger
from src.kernels.vertex_histo import VertexHisto
from src.kernels.edge_histo import EdgeHisto
from src.classifiers.knn import KNN
from src.classifiers.classifiers import Classifier, DummyClassifier

kernels = [
    VertexHisto(),
    EdgeHisto(),

]
classifiers = [
    (DummyClassifier, [
        {}
    ]),
    (KNN, [
        {'num_neighbors':3},
        {'num_neighbors':5},
        {'num_neighbors':7},
    ])
]

def main():
    for kernel in kernels:
        for classifier, kwargs_list in classifiers:
            for kwargs in kwargs_list:
                clf: Classifier = classifier(kernel=kernel, **kwargs)
                logger.info(f'Evaluating kernel={kernel.name}, classifier={clf.note}, kwargs={kwargs}')
                clf.evaluate()
                clf.make_submission()

if __name__ == '__main__':
    main()