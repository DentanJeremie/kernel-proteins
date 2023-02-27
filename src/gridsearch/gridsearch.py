import typing as t

from src.utils.logs import logger
from src.utils.pathtools import project
from src.kernels.kernels import BaseKernel, EmptyKernel
from src.kernels.histograms import EdgeHisto, VertexHisto, EdgeVertexHisto
from src.kernels.pyramid_match import PyramidMatch
from src.classifiers.knn import KNN
from src.classifiers.classifiers import BaseClassifier, DummyClassifier

kernels: t.List[BaseKernel] = [
    PyramidMatch(),
    EmptyKernel(),
    EdgeHisto(),
    VertexHisto(),
    EdgeVertexHisto(),
]
classifiers = [
    (DummyClassifier, [
        {}
    ]),
    (KNN, [
        {'num_neighbors':1},
        {'num_neighbors':2},
        {'num_neighbors':3},
        {'num_neighbors':5},
        {'num_neighbors':7},
    ])
]

def main():
    best_auc = -1
    best_kernel = None
    best_classifier = None
    best_kwargs = None
    best_output_path = None

    for kernel in kernels:
        for classifier, kwargs_list in classifiers:
            for kwargs in kwargs_list:
                clf: BaseClassifier = classifier(kernel=kernel, **kwargs)
                logger.info(f'Evaluating kernel={kernel.name}, classifier={clf.name}, kwargs={kwargs}')
                auc_score = clf.evaluate()
                output_path = clf.make_submission()

                if auc_score > best_auc:
                    best_auc = auc_score
                    best_kernel = kernel.name
                    best_classifier = clf.name
                    best_kwargs = kwargs
                    best_output_path = project.as_relative(output_path)

    logger.info(f'Best auc score: {best_auc:.3f}')
    logger.info(f'Best config: kernel={best_kernel}, classifier={best_classifier}, kwargs={best_kwargs}')
    logger.info(f'Best submission stored at {best_output_path}')

if __name__ == '__main__':
    main()