import typing as t

from src.utils.logs import logger
from src.utils.pathtools import project
from src.kernels.kernels import BaseKernel
from src.kernels.histograms import EdgeHisto, VertexHisto, EdgeVertexHisto
from src.kernels.pyramid_match import PyramidMatch
from src.kernels.shortest_path import ShortestPath
from src.classifiers.knn import KNN, FixedWeightedKNN, InverseWeightedKNN
from src.classifiers.svm import SVM
from src.classifiers.classifiers import BaseClassifier

kernels: t.List[BaseKernel] = [
    ShortestPath(),
    PyramidMatch(),
    PyramidMatch(d=4),
    EdgeHisto(),
    VertexHisto(),
    EdgeVertexHisto(),
]
classifiers = [
    (KNN, [
        {'num_neighbors':k}
        for k in range(3, 40)
    ]),
    (FixedWeightedKNN, [
        {'num_neighbors':k}
        for k in range(3, 40)
    ]),
    (InverseWeightedKNN, [
        {'num_neighbors':k}
        for k in range(3, 40)
    ]),
    (SVM, [
        {'c':35, 'num_train':100},
        {'c':40, 'num_train':100},
        {'c':45, 'num_train':100},
        {'c':50, 'num_train':100},
        {'c':80, 'num_train':200},
        {'c':83, 'num_train':200},
        {'c':85, 'num_train':200},
        {'c':88, 'num_train':200},
        {'c':195, 'num_train':400},
        {'c':200, 'num_train':400},
        {'c':205, 'num_train':400},
    ])

]

def main():
    best_auc = -1
    best_kernel = None
    best_classifier = None
    best_kwargs = None
    best_output_path = None

    logger.info(f'Starting grid search over the kernels and the classifiers.')
    
    logger.info('Building or loading the kernels')
    for kernel in kernels:
        _ = kernel.kernel_matrix

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