import time
from abc import ABC, abstractmethod

import numpy
import torch
from typing import Dict, Callable, Optional, Tuple
from sklearn.base import RegressorMixin


# AbstractPrimalSolver and AbstractDualSolver remain the same...
class AbstractPrimalSolver(ABC):
    @abstractmethod
    def solve(self, precomputed_vars: dict, option_type: str, strike_price: float, payoff_fn: Callable, **kwargs) -> float:
        pass


def count_in_money_paths(
        asset_prices_at_t: torch.Tensor,
        strike_price: float,
        option_type: str = 'put'
) -> int:
    """
    Calculates the number of in-the-money paths at a specific time step.

    Args:
        asset_prices_at_t (torch.Tensor): A 1D tensor of asset prices for all paths at a single time t.
        strike_price (float): The strike price of the option.
        option_type (str): The type of the option, either 'put' or 'call'.

    Returns:
        int: The number of paths that are in-the-money.
    """
    if option_type == 'put':
        intrinsic_values = torch.clamp(strike_price - asset_prices_at_t, min=0)
    elif option_type == 'call':
        intrinsic_values = torch.clamp(asset_prices_at_t - strike_price, min=0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    # Create a boolean mask where intrinsic value > 0
    in_money_mask = intrinsic_values > 0

    # Sum the boolean mask to count the number of 'True' values
    num_in_money = torch.sum(in_money_mask)

    return num_in_money.item()

class BaseLSPrimalSolver(AbstractPrimalSolver):
    r"""
    A template base class for Longstaff-Schwartz (LS) primal solvers.

    This class implements the entire backward induction algorithm which is the
    core of the LS method for pricing American options. The price of an American
    option is the solution to an optimal stopping problem:

    $$ V_t = \sup_{\tau \in \mathcal{T}_{t,T}} \mathbb{E}\left[e^{-r(\tau-t)} \text{Payoff}(S_\tau) \mid \mathcal{F}_t\right] $$

    The LS algorithm solves this by discretizing time and applying dynamic
    programming. At each time step $t$, the option's value is determined by the
    Bellman equation, which is the maximum of either exercising immediately or holding:

    $$ V_t = \max(\text{Payoff}(S_t), C_t) $$

    where $C_t$ is the **continuation value**, defined as the discounted expected
    future value of the option, conditional on the information up to time $t$:

    $$ C_t = \mathbb{E}\left[e^{-r\Delta t} V_{t+\Delta t} \mid \mathcal{F}_t\right] $$

    The key challenge is estimating $C_t$. This algorithm approximates it using a
    least-squares regression on a set of basis functions of the state. In our
    case, the basis functions are the components of the path signature:

    $$ C_t \approx \sum_{k=1}^{D} \beta_k \cdot \text{sig}(\text{Path}_t)_k $$

    Subclasses only need to provide the specific regression model to estimate the
    coefficients $\beta_k$.
    """

    def __init__(self, risk_free_rate: float):
        """
        Initializes the base Longstaff-Schwartz solver.

        Args:
            risk_free_rate (float): The risk-free interest rate for discounting.
        """
        self.r = risk_free_rate

    @abstractmethod
    def _create_regressor(self) -> RegressorMixin:
        """
        Abstract method for creating the regression model.

        Subclasses must implement this method to provide a scikit-learn
        compatible regression model (i.e., an object with .fit() and .predict() methods).

        Returns:
            An unfitted regression model instance.
        """
        pass

    def solve(self, precomputed_vars: dict, option_type: str, strike_price: float, payoff_fn: Callable, **kwargs) -> float:
        """
        Executes the signature-based Longstaff-Schwartz algorithm.

        Args:
            paths (torch.Tensor): A 3D tensor of simulated asset paths.
            payoff_fn (Callable): The option's payoff function.
            **kwargs: Must contain 'T' (float), the time to maturity.

        Returns:
            The calculated lower bound price of the American option.
        """
        # --- 1. Unpack precomputed variables ---
        paths = precomputed_vars["paths"]
        payoffs = precomputed_vars["payoffs"]
        signatures = precomputed_vars["signatures"]
        dt = precomputed_vars["dt"]

        device, num_steps = paths.device, paths.shape[1]
        discount_factor = torch.exp(torch.tensor(-self.r * dt, device=device))

        # --- 2. Initialization ---
        # Get initial option values from the precomputed payoff matrix
        option_values = payoffs[:, -1]

        i = 0
        # --- 3. Backward Induction Loop ---
        # The loop is now much cleaner
        for t in range(num_steps - 2, -1, -1):
            intrinsic_value = payoffs[:, t]  # Use precomputed payoff
            in_the_money_mask = intrinsic_value > 0
            if not torch.any(in_the_money_mask):
                option_values *= discount_factor
                continue

            # Get the precomputed signatures for this time step
            signatures_t = signatures[:, t, :]

            final_mask = in_the_money_mask.squeeze()
            if final_mask.dim() > 1:
                final_mask = final_mask[:, 0]

            training_signatures = signatures_t[final_mask]
            training_Y = option_values[final_mask] * discount_factor

            regressor = self._create_regressor()
            regressor.fit(training_signatures.cpu().numpy(), training_Y.cpu().numpy())

            continuation_value = torch.from_numpy(regressor.predict(signatures_t.cpu().numpy())).to(device)

            option_values = torch.where(
                (intrinsic_value > continuation_value) & in_the_money_mask,
                intrinsic_value,
                option_values * discount_factor
            )

            i += 1

        final_price = torch.mean(option_values)
        return final_price.item()

class AbstractDualSolver(ABC):
    @abstractmethod
    def solve(self, precomputed_vars: dict, payoff_fn, **kwargs) -> float:
        pass


class BaseDLSolver(AbstractDualSolver):
    r"""
    A template base class for dual (upper bound) solvers.

    This class implements the entire optimization workflow for the dual method.
    The dual approach to American option pricing is based on the principle of
    martingale duality for optimal stopping. This principle states that the price
    of an American option can be bounded from above by any suitable martingale.
    The price is the infimum over all martingales $M_t$:

    $$ V_0 \le \inf_{M \in \mathcal{M}} \mathbb{E}\left[\sup_{t \in [0,T]} (\text{Payoff}_t - M_t)\right] $$

    This algorithm seeks to find the tightest possible upper bound by constructing
    a family of martingales parameterized by a vector of coefficients $\alpha$ and
    then finding the optimal $\alpha$ that minimizes the expectation. The optimization
    problem is:

    $$ \alpha^* = \arg\min_{\alpha} \mathbb{E}\left[\max_{t=0,\dots,N} (\text{Payoff}_t - M_t(\alpha))\right] $$

    The martingale, $M_t(\alpha)$, is constructed as a linear combination of "basis
    martingales," which are stochastic integrals against the components of the
    path signature:

    $$ M_t(\alpha) = \sum_{j=1}^{D} \alpha_j \int_0^t \text{sig}(\text{Path}_s)_j \, dW_s $$

    Subclasses only need to implement the `_prepare_features` method, which defines
    how the basis martingales are transformed before the final linear combination.
    """
    def __init__(self, truncation_level: int, learning_rate: float, max_epochs: int, patience: int, tolerance: float):
        """
        Initializes the base dual solver.

        Args:
            truncation_level (int): The signature truncation level.
            learning_rate (float): The learning rate for the Adam optimizer.
            max_epochs (int): The maximum number of training epochs.
            patience (int): Number of epochs to wait for improvement before stopping.
            tolerance (float): The minimum change in loss to be considered an improvement.
        """
        self.truncation_level = truncation_level
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance

    @abstractmethod
    def _prepare_features(self, signatures: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for preparing features from basis martingales.

        Subclasses must implement this method to transform the calculated basis
        martingales into the final set of features that will be linearly combined
        to form the optimized martingale.

        Args:
            basis_martingales (torch.Tensor): The raw, calculated basis martingales.

        Returns:
            A torch.Tensor containing the final features for optimization.
        """
        pass

    def solve(self, paths: torch.Tensor, dW: torch.Tensor, payoff_fn: Callable, **kwargs) -> float:
        """
        Executes the signature-based dual pricing algorithm.

        Args:
            paths (torch.Tensor): A 3D tensor of simulated asset paths.
            dW (torch.Tensor): The Brownian increments used to generate the paths.
            payoff_fn (Callable): The option's payoff function.
            **kwargs: Placeholder for additional arguments.

        Returns:
            The calculated upper bound price of the American option.
        """
        device = paths.device
        num_steps, num_paths = paths.shape[1], paths.shape[0]
        from ..features.signature_calculator import calculate_signatures  # Avoid circular import

        # **REFACTOR**: We need to compute features at each time step inside the loop.
        # First, get the dimension of the final features.
        sig_T = calculate_signatures(paths, self.truncation_level)
        feature_dim = self._prepare_features(sig_T).shape[1]

        # Now, build the basis martingales using the transformed features.
        basis_martingales = torch.zeros(num_paths, num_steps, feature_dim, device=device, dtype=torch.float32)

        for t in range(num_steps - 1):
            paths_slice = paths[:, :t + 1, :]
            if paths_slice.shape[1] > 1:
                # 1. Calculate signatures at time t
                current_signatures = calculate_signatures(paths_slice, self.truncation_level)
                # 2. Transform signatures into integrand features
                integrand_features = self._prepare_features(current_signatures)
                # 3. Calculate the martingale increment
                increment = integrand_features * dW[:, t].unsqueeze(1)
                basis_martingales[:, t + 1, :] = basis_martingales[:, t, :] + increment

        # The rest of the optimization logic is the same, but it now operates on a
        # a mathematically correct martingale.
        alpha = torch.randn(feature_dim, 1, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=self.lr)
        # ... (the rest of the optimization loop is unchanged) ...
        asset_paths = paths[:, :, 0]
        payoffs = torch.stack([payoff_fn(asset_paths[:, t]) for t in range(num_steps)], dim=1)
        best_loss = float('inf')
        patience_counter = 0

        print(f"Starting {self.__class__.__name__} optimization with early stopping...")
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            M = torch.matmul(basis_martingales, alpha).squeeze(-1)
            loss = torch.mean(torch.max(payoffs - M, dim=1)[0])
            loss.backward()
            optimizer.step()
            if best_loss - loss.item() > self.tolerance:
                best_loss, patience_counter = loss.item(), 0
            else:
                patience_counter += 1
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch + 1}/{self.max_epochs}], Loss (Upper Bound): {loss.item():.6f}")
            if patience_counter >= self.patience:
                print(f"  -> Early stopping triggered at epoch {epoch + 1}.")
                break

        return best_loss