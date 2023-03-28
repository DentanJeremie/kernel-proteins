from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from src.utils.logs import logger
from src.utils.pathtools import project
from src.utils.graphs import graph_manager
from src.utils.auc import roc_auc_score
from src.kernels.kernels import BaseKernel

ROC_VERBOSE = 100


class BaseClassifier():

    def __init__(self, kernel:BaseKernel, name='empty_classifier') -> None:
        self.name = name
        self.kernel = kernel

    def predict(self, idx:int) -> int:
        raise NotImplementedError

    def evaluate(self) -> float:
        """Returns the performances according to the validation set, equals to the AUC score."""
        logger.info(f'Computing the AUC score of kernel {self.kernel.name} with classifier {self.name}')
        y_true = [
            lab
            for _, lab, _, _ in graph_manager.valid 
        ]
        y_pred = [
            self.predict(idx)
            for _, _, _, idx in graph_manager.valid 
        ]
        auc_score = roc_auc_score(y_true, y_pred)

        logger.info(f'Final AUC score {auc_score:.3f} for clf {self.name} and kernel {self.kernel.name}')
        return auc_score
                
    def make_submission(self) -> Path:
        """Makes a submission file and stores it. Returns the storing path."""
        output_path = project.get_new_prediction_file(note = f'{self.kernel.name}_{self.name}')

        data = {
            'Id':list(range(1, 2001)),
            'Predicted':[
                self.predict(idx)
                for _, _, _, idx in graph_manager.test
            ]
        }
        pd.DataFrame(data).to_csv(output_path, index=False)
        logger.info(f'Created a submission file at {project.as_relative(output_path)}')

        return output_path


class DummyClassifier(BaseClassifier):

    def __init__(self, kernel:BaseKernel, name='dummy') -> None:
        super().__init__(kernel, name)

    def predict(self, idx: int) -> int:
        return np.random.random()


def main():
    from src.kernels.kernels import EmptyKernel
    clf = DummyClassifier(EmptyKernel)
    clf.predict(0)
    clf.evaluate()
    clf.make_submission()

if __name__ == '__main__':
    main()
