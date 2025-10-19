from typing import Callable

import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from .base_solver import BaseLSPrimalSolver, AbstractPrimalSolver  # Assuming this is in the same directory
from ..features.signature_calculator import calculate_signatures


class HilbertOperatorPrimalSolver(BaseLSPrimalSolver):
    r"""
    A primal solver based on the principles of operator learning in a
    Reproducing Kernel Hilbert Space (RKHS), as proposed in arXiv:2507.21189v1.

    This solver approximates the continuation value by solving a regularized
    operator estimation problem at each time step. This is practically
    implemented using Kernel Ridge Regression, which finds a non-linear
    function in an RKHS that minimizes a squared-error loss with a norm penalty.
    """

    def __init__(
            self,
            risk_free_rate: float,
            alpha: float = 0.02,
            kernel: str = 'rbf',
            gamma: float = None
    ):
        """
        Initializes the HilbertOperatorSolver.

        Args:
            truncation_level (int): The signature truncation level.
            risk_free_rate (float): The risk-free interest rate (r).
            alpha (float): The regularization strength (corresponds to lambda in the paper).
            kernel (str): The kernel to use (e.g., 'rbf', 'linear', 'polynomial').
            gamma (float): The kernel coefficient for 'rbf' and other kernels.
        """
        super().__init__(risk_free_rate)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma

    def _create_regressor(self):
        """
        Creates the scikit-learn pipeline for the Kernel Ridge Regression.

        This pipeline first standardizes the input signature features and then
        applies the KernelRidge regressor.
        """
        # 1. Define the base estimator (the model we want to tune)
        kernel_ridge_estimator = KernelRidge(kernel=self.kernel, gamma=self.gamma, alpha=self.alpha)

        # 2. Define the pipeline
        pipeline = make_pipeline(StandardScaler(), kernel_ridge_estimator)

        return pipeline