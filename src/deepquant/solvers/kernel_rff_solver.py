import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base_solver import BaseLSPrimalSolver, BaseDLSolver


class KernelRFFPrimalSolver(BaseLSPrimalSolver):
    r"""
    A concrete implementation of the Longstaff-Schwartz primal solver that
    uses a powerful, non-linear Kernel Ridge Regression model to approximate
    the continuation value.

    This solver improves upon the baseline linear model by its ability to capture
    complex, non-linear relationships between the path signature and the option's
    continuation value. It does this by approximating a Gaussian (RBF) kernel
    using the **Random Fourier Features (RFF)** technique. This provides the power
    of kernel methods with the speed of linear models.

    The model pipeline consists of three steps:
    1.  **StandardScaler**: Standardizes the input signature features to have a mean of 0
        and a variance of 1. This is crucial for the performance of kernel methods.
    2.  **RBFSampler**: Applies the RFF transformation to map the features into a
        higher-dimensional space where non-linear relationships can be represented linearly.
    3.  **Ridge Regression**: Performs a regularized linear regression in the new
        feature space to prevent overfitting.

    Using a more expressive regressor should lead to a more accurate exercise
    strategy and a tighter (higher) lower bound, helping to shrink the duality gap.
    """

    def __init__(self, truncation_level: int, risk_free_rate: float, n_rff: int, gamma: str = 'scale'):
        """
        Initializes the KernelRFFPrimalSolver.

        Args:
            truncation_level (int): The signature truncation level.
            risk_free_rate (float): The risk-free interest rate (r).
            n_rff (int): The number of Random Fourier Features to use. This is the
                         dimensionality of the randomized feature space. Higher values
                         provide a better kernel approximation at the cost of computation.
            gamma (str or float): The bandwidth parameter for the RBF kernel being
                                  approximated. Using the string 'scale' is a robust
                                  heuristic that sets gamma to 1 / (n_features * Var(X)).
        """
        # Initialize the base class with shared parameters
        super().__init__(truncation_level, risk_free_rate)
        self.n_rff = n_rff
        self.gamma = gamma

    def _create_regressor(self) -> make_pipeline:
        """
        Provides the scikit-learn Scaler -> RFF -> Ridge regression pipeline.

        This method fulfills the requirement of the `BaseLSPrimalSolver` template.
        It constructs a pipeline that first standardizes the input signatures,
        applies the non-linear RFF transformation, and then fits a Ridge regressor
        in the new, higher-dimensional feature space.

        Returns:
            A scikit-learn pipeline instance that acts as a single regressor.
        """
        scaler = StandardScaler()
        rff_sampler = RBFSampler(n_components=self.n_rff, gamma=self.gamma, random_state=42)
        ridge_regressor = Ridge(alpha=1.0) # alpha is the L2 regularization strength
        return make_pipeline(scaler, rff_sampler, ridge_regressor)


class KernelRFFDualSolver(BaseDLSolver):
    r"""
    A concrete implementation of the dual solver that uses a non-linear martingale
    constructed from **Random Fourier Features (RFF)**.

    This solver enhances the baseline dual method by first mapping the basis
    martingales into a high-dimensional feature space using RFF. The final,
    optimized martingale is then constructed as a linear combination of these new,
    non-linear features.

    The resulting martingale is a more expressive function, defined as:
    $$ M_t(\alpha) = \sum_{k=1}^{N_{RFF}} \alpha_k \cdot \phi(\text{BasisMartingales}_t)_k $$
    where $\phi(\cdot)_k$ represents the $k$-th component of the RFF transformation.

    This allows the martingale to better approximate the true option payoff surface,
    which should result in a tighter (lower) and more accurate upper bound,
    helping to shrink the duality gap compared to the linear model.
    """

    def __init__(self, truncation_level: int, learning_rate: float, max_epochs: int, patience: int, tolerance: float,
                 n_rff: int, gamma: str = 'scale'):
        """
        Initializes the KernelRFFDualSolver.

        Args:
            truncation_level (int): The signature truncation level.
            learning_rate (float): The learning rate for the Adam optimizer.
            max_epochs (int): The maximum number of training epochs.
            patience (int): Number of epochs to wait for improvement before stopping.
            tolerance (float): The minimum change in loss for early stopping.
            n_rff (int): The dimensionality of the RFF feature space.
            gamma (str or float): The bandwidth parameter for the RBF kernel being
                                  approximated. Using 'scale' is a robust heuristic.
        """
        # Initialize the base class with shared parameters
        super().__init__(truncation_level, learning_rate, max_epochs, patience, tolerance)
        self.n_rff = n_rff
        self.gamma = gamma

    def _prepare_features(self, signatures: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the basis martingales, then applies a non-linear RFF
        transformation to create the final feature set.

        This method fulfills the requirement of the `BaseDLSolver` template by
        defining the specific feature engineering for this solver.

        Args:
            basis_martingales (torch.Tensor): The raw basis martingales calculated
                                              in the base class.

        Returns:
            A torch.Tensor of shape (num_paths, num_steps, n_rff) containing
            the transformed features.
        """
        scaler = StandardScaler()
        rff_sampler = RBFSampler(n_components=self.n_rff, gamma=self.gamma, random_state=42)
        pipeline = make_pipeline(scaler, rff_sampler)

        signatures_np = signatures.cpu().numpy()
        features_np = pipeline.fit_transform(signatures_np)

        return torch.from_numpy(features_np).to(signatures.device, dtype=torch.float32)
