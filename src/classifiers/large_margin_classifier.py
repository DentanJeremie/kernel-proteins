import typing as t

import numpy as np
from scipy.optimize import newton

from src.classifiers.classifiers import BaseClassifier
from src.kernels.kernels import BaseKernel
from src.utils.logs import logger
from src.utils.constants import *
from src.utils.graphs import graph_manager

class LargeMarginClassifier(BaseClassifier):

    def __init__(
        self,
        kernel:BaseKernel,
        lmbda:float = 1,
        name:str='',
    ) -> None:
        """
        Base class for Large Margin Classifiers.
        """
        logger.info(f'Initializing a LMC classifier with phi={name}') 
        super().__init__(kernel, name=f'lmc_{name}_lambda_{lmbda}')
        self.lmbda = lmbda

        # Weights
        self._alpha_i = np.random.random(NUM_TRAIN)
        self._fitted = False

# ------------------ EVALUATION ------------------

    def __call__(self, idx:int):
        if not self._fitted:
            self.fit()

        return np.sum(self._alpha_i @ self.kernel.kernel_matrix[idx,:])

    def predict(self, idx: int) -> int:
        if self(idx) >= 0.5:
            return 1
        else:
            return 0
        
# ------------------ OPTIMIZATION ------------------

    def phi(self, vector:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def phi_prime(self, vector:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def phi_prime2(self, vector:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self):

        # Setting the objective function and its derivatives
        labels_train = np.array([
            item[1]
            for item in graph_manager.train
        ])

        def obj(array:np.ndarray) -> np.ndarray:
            y_K_alpha = labels_train * self.kernel.kernel_matrix @ array
            logger.debug(f'labels_train.shape: {labels_train.shape}')
            logger.debug(f'(self.kernel.kernel_matrix @ self._alpha_i).shape: {(self.kernel.kernel_matrix @ array).shape}')
            logger.debug(f'y_K_alpha.shape: {y_K_alpha.shape}')
            return (
                1/NUM_TRAIN * np.sum(self.phi(y_K_alpha))
                + self.lmbda/2 * array.transpose() @ self.kernel.kernel_matrix @ array
            )

        self._alpha_i = newton(
            self.objective_function,
            x0=self._alpha_i,
        )
