import typing as t

import numpy as np
from scipy.optimize import minimize

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
        """
        Fits the alpha_i according to the train data.
        """
        logger.info(f'Starting fitting of {self.name} with method {MINIMIZATION_METHOD}')

        # Setting the objective function and its derivatives
        labels_train = np.array([
            item[1]
            for item in graph_manager.train
        ])
        kernel_matrix = self.kernel.kernel_matrix[:NUM_TRAIN,:NUM_TRAIN]

        def fun(array:np.ndarray) -> np.ndarray:
            """The objective function to minimize."""
            y_K_alpha = labels_train * kernel_matrix @ array
            logger.debug(f'labels_train.shape: {labels_train.shape}')
            logger.debug(f'(kernel_matrix @ self._alpha_i).shape: {(kernel_matrix @ array).shape}')
            logger.debug(f'y_K_alpha.shape: {y_K_alpha.shape}')
            return (
                1/NUM_TRAIN * np.sum(self.phi(y_K_alpha))
                + self.lmbda/2 * array.transpose() @ kernel_matrix @ array
            )

        def jac(array:np.ndarray) -> np.ndarray:
            """The gradient of the objective function."""
            y_K_alpha = labels_train * kernel_matrix @ array
            P = np.diag(self.phi_prime(y_K_alpha))
            return (
                1/NUM_TRAIN * kernel_matrix @ P @ labels_train
                + self.lmbda * kernel_matrix @ array
            )

        def hess(array:np.ndarray) -> np.ndarray:
            """The hessian of the objective function."""
            y_K_alpha = labels_train * kernel_matrix @ array
            W = np.diag(self.phi_prime2(y_K_alpha))
            return (
                1/NUM_TRAIN * kernel_matrix @ W @ kernel_matrix
                + self.lmbda * kernel_matrix
            )

        self._alpha_i = minimize(
            fun=fun,
            x0=self._alpha_i,
            jac=jac,
            hess=hess,
            method=MINIMIZATION_METHOD,
        )['x']

        # Finishing
        self._fitted = True
        logger.info('Minimization completed!')


class LogisticLMC(LargeMarginClassifier):

    def __init__(self, kernel: BaseKernel, lmbda: float = 1) -> None:
        super().__init__(kernel, lmbda, name='logistic')

    def phi(self, vector: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(-1*vector))

    def phi_prime(self, vector: np.ndarray) -> np.ndarray:
        return -1/(1 + np.exp(vector))

    def phi_prime2(self, vector: np.ndarray) -> np.ndarray:
        return -1*np.exp(vector) / (1 + np.exp(vector)) / (1 + np.exp(vector))


def main():
    from src.kernels.histograms import VertexHisto
    knn = LogisticLMC(VertexHisto())
    knn.evaluate()
    knn.make_submission()

if __name__ == '__main__':
    main()