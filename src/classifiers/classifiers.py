from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from src.utils.logs import logger
from src.utils.pathtools import project
from src.utils.graphs import graph_manager
from src.kernels.kernels import BaseKernel

ROC_VERBOSE = 100

class BaseClassifier():

    def __init__(self, kernel:BaseKernel, name='empty_classifier') -> None:
        self.name = name
        self.kernel = kernel

    def predict(self, idx:int) -> int:
        raise NotImplementedError

    def evaluate(self) -> float:
        """Returns the performances according to the validation set,
        as a tuple (acc_0, acc_1, weighted_acc)."""
        logger.info(f'Computing the AUC score of kernel {self.kernel.name} with classifier {self.name}')
        # Sorting the labels by prediction
        idx_to_logit_and_label = {
            idx:(
                self.predict(idx),
                lab,
            )
            for _, lab, _, idx in graph_manager.valid
        }
        sorted_idx = sorted(
            idx_to_logit_and_label,
            key = lambda idx: (idx_to_logit_and_label[idx][0], 1-idx_to_logit_and_label[idx][1])
        )
        sorted_label = [
            idx_to_logit_and_label[idx][1]
            for idx in sorted_idx
        ]


        # idx <= split_index -> predict 0 ; idx > split_index -> predict 1
        # At the beginning, everybody is classified as 1
        split_index = -1
        ok_classified_0 = 0
        misclassified_0 = np.sum([1 for _, lab, _, _ in graph_manager.valid if lab == 0])
        ok_classified_1 = np.sum([1 for _, lab, _, _ in graph_manager.valid if lab == 1])
        misclassified_1 = 0

        sum_0 = ok_classified_0 + misclassified_0
        sum_1 = ok_classified_1 + misclassified_1
        tp_rate = ok_classified_1 / sum_1
        fp_rate = misclassified_0 / sum_0

        # Building the ROC curve
        roc_points = [(tp_rate, fp_rate)]
        logger.debug(f'(tp, fp) = {tp_rate:.3f}, {fp_rate:.3f}')
        for split_index, new_label in enumerate(sorted_label):
            if new_label == 0:
                ok_classified_0 += 1
                misclassified_0 -= 1
            elif new_label == 1:
                ok_classified_1 -= 1
                misclassified_1 += 1
            else:
                raise ValueError(f'Labels must be 0 or 1: {new_label}')
            
            assert sum_0 == ok_classified_0 + misclassified_0
            assert sum_1 == ok_classified_1 + misclassified_1
            
            tp_rate = ok_classified_1 / sum_1
            fp_rate = misclassified_0 / sum_0
            roc_points.append((tp_rate, fp_rate))

            if split_index % ROC_VERBOSE == 0:
                logger.debug(f'(tp, fp) = {tp_rate:.3f}, {fp_rate:.3f}')
        logger.debug(f'(tp, fp) = {tp_rate:.3f}, {fp_rate:.3f}')

        # Computing the auc score (x-axis: fp_rate, y_axis: tp_rate)
        auc_score = 0
        last_tp_rate, last_fp_rate = roc_points[0]

        for tp_rate, fp_rate in roc_points[1:]:
            assert tp_rate <= last_tp_rate and fp_rate <= last_fp_rate, 'Non-decresing points in ROC curve!'

            # Vertical moove 
            if fp_rate == last_fp_rate:
                assert tp_rate < last_tp_rate, 'Uncorrect vertical moove'
                last_tp_rate, last_fp_rate = tp_rate, fp_rate
            # Horizontal moove
            else:
                assert tp_rate == last_tp_rate, 'Uncorrect horizontal moove'
                auc_score += tp_rate * (last_fp_rate - fp_rate)
                last_tp_rate, last_fp_rate = tp_rate, fp_rate

        # Final score
        assert last_tp_rate == 0, f'Non-zero final tp_rate value: {last_tp_rate}'
        assert last_fp_rate == 0, f'Non-zero final fp_rate_value: {last_fp_rate}'
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
