import typing as t

import numpy as np
from scipy import optimize

from src.classifiers.classifiers import BaseClassifier
from src.kernels.kernels import BaseKernel
from src.utils.logs import logger
from src.utils.constants import *
from src.utils.graphs import graph_manager

EPSILON_EQUALITY = 10e-3
MINIMIZATION_METHOD = 'SLSQP'


class SVM(BaseClassifier):

    def __init__(
            self,
            kernel:BaseKernel,
            c:float,
            num_train: int,
        ) -> None:
        logger.info(f'Initializing a SVM classifier with c={c}, num_train={num_train}') 
        super().__init__(kernel, name=f'svm_c_{c}_n_{num_train}')
        self.c = c

        # Getting the train indexes
        self.num_train = num_train
        self.index_train = list()
        count_0 = 0
        count_1 = 0
        for _, lab, _, idx in graph_manager.train:
            if lab == 0 and count_0 < self.num_train // 2:
                self.index_train.append(idx)
                count_0 += 1
            if lab == 1 and count_1 < self.num_train - self.num_train // 2:
                self.index_train.append(idx)
                count_1 += 1

        # Computation constants
        self._alpha = None
        self._support = None
        self._is_support = None
        self._support_label = None
        self._b = None

        # Evaluations
        self._fitted = False
        self._evaluations = None

    @property
    def evaluations(self):
        if self._evaluations is None:
            self.compute_separating_function()
        return self._evaluations

    def fit(self):
        """
        Fits the SVM with the train set.
        """
        labels = np.array([
            lab
            for _, lab, _, _ in graph_manager.train 
        ])[self.index_train]
        labels = 2*labels - 1 # Labels in {-1,1} instead of {0, 1}
        train_kernel_matrix = self.kernel.kernel_matrix[:, self.index_train][self.index_train, :]

        # Logging
        logger.info(f'Fitting SVM with c={self.c}, num_train={self.num_train} for kernel {self.kernel.name}')

        # Lagrange dual problem
        def loss(alpha):
            '''Dual loss of the KSVM problem'''
            alpha_tilde = np.diag(labels)@alpha
            return 1/2 * alpha_tilde.T@train_kernel_matrix@alpha_tilde - np.sum(alpha)
        
        # Partial derivate of dual loss on alpha
        def grad_loss(alpha):
            '''partial derivative of the dual loss wrt alpha'''
            return np.diag(labels)@train_kernel_matrix@np.diag(labels)@alpha - 1
        
        # Constraints on alpha
        fun_eq = lambda alpha: np.sum(alpha*labels) 
        jac_eq = lambda alpha:  labels
        fun_ineq = lambda alpha: np.concatenate((alpha, self.c - alpha))
        jac_ineq = lambda alpha: np.vstack((np.eye(self.num_train), -1*np.eye(self.num_train)))

        constraints = (
            {
                'type': 'eq', 
                'fun': fun_eq,
                'jac': jac_eq
            },
            {
                'type': 'ineq', 
                'fun': fun_ineq , 
                'jac': jac_ineq
            }
        )

        optRes = optimize.minimize(
            fun=lambda alpha: loss(alpha),
            x0=np.ones(self.num_train), 
            method=MINIMIZATION_METHOD, 
            jac=lambda alpha: grad_loss(alpha), 
            constraints=constraints
        )
    
        self._alpha = optRes.x

        # Assign the required attributes
        # M = number of support vectors
        # Support: alpha>0
        self._is_support = self._alpha>EPSILON_EQUALITY
        self._support_label = labels[self._is_support] # Shape M
        # Margin: alpha>0 and alpha <C
        is_margin = (self._alpha>EPSILON_EQUALITY) * (self._alpha < self.c - EPSILON_EQUALITY)
        margin_label = labels[is_margin] 
        # Computing b
        evaluations = train_kernel_matrix @ np.diag(labels) @ self._alpha
        margin_evaluations = evaluations[is_margin]
        self._b = np.median(1/margin_label - margin_evaluations)

        # Fitted
        self._fitted =  True

    def compute_separating_function(self):
        """
        Fills `self.evaluations` with the evaluation of the separating function on all points.
        """
        # Fitted ?
        if not self._fitted:
            self.fit()

        logger.info(f'Computing the separation function with c={self.c}, num_train={self.num_train}  of kernel {self.kernel.name}')
        # We call M the number of margin points
        predict_kernel_matrix = self.kernel.kernel_matrix[:,self.index_train] # Shape NUM_LABELED + NUM_TEST, self.num_train
        predict_kernel_matrix = predict_kernel_matrix[:,self._is_support] # Shape NUM_LABELED + NUM_TEST, M
        alpha_support = self._alpha[self._is_support] # shape M
        alpha_tilde = alpha_support * self._support_label # shape M
        self._evaluations = predict_kernel_matrix @ alpha_tilde # shape NUM_LABELED + NUM_TEST

    def predict(self, idx: int) -> int:
        return self.evaluations[idx] 

def main():
    from src.kernels.histograms import EdgeVertexHisto
    svm = SVM(EdgeVertexHisto(),c=100)
    svm.evaluate()
    svm.make_submission()

if __name__ == '__main__':
    main()