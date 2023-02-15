import typing as t

import pandas as pd

from src.utils.logs import logger
from src.utils.pathtools import project
from src.utils.graphs import graph_manager

class Classifier():

    def predict(idx:int) -> int:
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

        acc_0 = correct_0 / (correct_0 + incorrect_0)
        acc_1 = correct_1 / (correct_1 + incorrect_1)
        weighted_acc = (acc_0 + acc_1) / 2

        logger.debug(f'correct_0={correct_0}, incorrect_0={incorrect_0}, correct_1={correct_1}, incorrect_1={incorrect_1}')
        logger.info(f'Performances: acc_0={acc_0:.1f}, acc_1={acc_1:.1f}, weighted_acc={weighted_acc:.1f}')
        return acc_0, acc_1, weighted_acc

    def make_submission(self, note:str = ''):
        """Makes a submission file and stores it."""
        output_path = project.get_new_prediction_file(note = note)

        data = {
            'Id':list(range(1, 2001)),
            'Predicted':[
                self.predict(idx)
                for _, _, _, idx in graph_manager.test
            ]
        }
        pd.DataFrame(data).to_csv(output_path, index=False)
        logger.info(f'Created a submission file at {project.as_relative(output_path)}')
