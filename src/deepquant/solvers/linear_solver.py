import torch
from sklearn.linear_model import LinearRegression
from typing import Callable

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base_solver import AbstractPrimalSolver, AbstractDualSolver, BaseLSPrimalSolver, BaseDLSolver

class LinearPrimalSolver(BaseLSPrimalSolver):
    """
    A concrete implementation of the Longstaff-Schwartz primal solver that
    uses a simple Linear Regression model to approximate the continuation value.

    This class serves as the baseline model in the solver hierarchy. It inherits
    the main backward induction algorithm from `BaseLSPrimalSolver` and simply
    plugs in `sklearn.linear_model.LinearRegression` as its function approximator.
    This corresponds to finding the coefficients $\beta_k$ in the base class
    documentation using Ordinary Least Squares.
    """
    def _create_regressor(self) -> make_pipeline:
        """
        Provides the scikit-learn LinearRegression model.

        This method fulfills the requirement of the `BaseLSPrimalSolver` template
        by returning an unfitted instance of the linear regression model.

        Returns:
            An instance of `sklearn.linear_model.LinearRegression`.
        """
        scaler = StandardScaler()
        lin_regression = LinearRegression()
        return make_pipeline(scaler, lin_regression)


class LinearDualSolver(BaseDLSolver):
    """Dual solver using a linear combination of basis martingales."""

    def _prepare_features(self, signatures: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the basis martingales and uses them directly as features.
        """
        mean = signatures.mean(dim=0, keepdim=True)
        std = signatures.std(dim=0, keepdim=True) + 1e-8
        return (signatures - mean) / std
