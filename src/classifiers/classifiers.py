from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from src.utils.logs import logger
from src.utils.pathtools import project
from src.utils.graphs import graph_manager
from src.kernels.kernels import BaseKernel

class BaseClassifier():

    def __init__(self, kernel:BaseKernel, name='empty_classifier') -> None:
        self.name = name
        self.kernel = kernel

    def predict(self, idx:int) -> int:
        raise NotImplementedError

    def evaluate(self) -> t.Tuple[float, float, float]:
        """Returns the performances according to the validation set,
        as a tuple (acc_0, acc_1, weighted_acc)."""
        correct_0 = 0
        incorrect_0 = 0
        correct_1 = 0
        incorrect_1 = 0

        for _, lab, _, idx in graph_manager.valid:
            pred = self.predict(idx)

            if (lab, pred) == (0, 0):
                correct_0 += 1
            if (lab, pred) == (0, 1):
                incorrect_0 += 1
            if (lab, pred) == (1, 0):
                incorrect_1 += 1
            if (lab, pred) == (1, 1):
                correct_1 += 1

        logger.debug(f'correct_0={correct_0}, incorrect_0={incorrect_0}, correct_1={correct_1}, incorrect_1={incorrect_1}')
        acc_0 = correct_0 / (correct_0 + incorrect_0)
        acc_1 = correct_1 / (correct_1 + incorrect_1)
        weighted_acc = (acc_0 + acc_1) / 2

        logger.info(f'Performances: acc_0={acc_0:.3f}, acc_1={acc_1:.3f}, weighted_acc={weighted_acc:.3f}')
        return acc_0, acc_1, weighted_acc

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
        return np.random.randint(0, 2)


def main():
    from src.kernels.kernels import EmptyKernel
    clf = DummyClassifier(EmptyKernel)
    clf.predict(0)
    clf.evaluate()
    clf.make_submission()

if __name__ == '__main__':
    main()
