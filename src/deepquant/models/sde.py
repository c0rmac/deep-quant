from typing import Tuple

import torch
import torchsde
from abc import ABC, abstractmethod

# An abstract base class defining the common interface for SDE models
class SDEModel(ABC):
    """Abstract base class for SDE models."""

    @abstractmethod
    def simulate_paths(self, num_paths: int, num_steps: int, T: float, **kwargs) -> torch.Tensor:
        """
        Simulates asset paths.

        Args:
            num_paths (int): The number of paths to simulate.
            num_steps (int): The number of time steps.
            T (float): The total time to maturity.
            **kwargs: Additional model-specific parameters.

        Returns:
            torch.Tensor: A 3D tensor of shape (num_paths, num_steps, num_dimensions)
                          representing the simulated paths.
        """
        pass


class HestonModel(SDEModel):
    """
    Concrete implementation of the Heston stochastic volatility model.

    The Heston model is a fundamental tool in quantitative finance that describes
    the evolution of an asset price whose volatility is also a random process.
    Unlike simpler models like Black-Scholes, it can capture key market phenomena
    such as volatility smiles and skews.

    In this model, the variance follows a Cox-Ingersoll-Ross (CIR) mean-reverting
    process, ensuring that volatility tends to return to a long-term average.
    It corresponds to the special case of a non-rough volatility model where the
    Hurst parameter H = 0.5.

    Attributes:
        s0 (float): The initial stock price.
        v0 (float): The initial variance.
        kappa (float): The rate of mean reversion for the variance process.
        theta (float): The long-term mean of the variance.
        xi (float): The volatility of the variance process ("vol of vol").
        rho (float): The correlation between the asset and variance processes.
        r (float): The risk-free interest rate.
    """

    def __init__(self, s0: float, v0: float, kappa: float, theta: float, xi: float, rho: float, r: float):
        """
        Initializes the HestonModel with its core parameters.

        Args:
            s0 (float): Initial stock price, S_0.
            v0 (float): Initial variance, V_0.
            kappa (float): Rate of mean reversion (κ).
            theta (float): Long-term variance mean (θ).
            xi (float): Volatility of variance (ξ).
            rho (float): Correlation between the two Brownian motions (ρ).
            r (float): Risk-free interest rate.
        """
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r

    def simulate_paths(self, num_paths: int, num_steps: int, T: float, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Simulates asset and variance paths according to the Heston model.

        The simulation uses a mixed scheme for discretization:
        1.  **Stock Price Process ($S_t$)**: A Log-Euler scheme is used for stability and
            to better preserve the log-normal property of the asset price.
            $$ dS_t = r S_t dt + \sqrt{V_t} S_t dW^s_t $$

        2.  **Variance Process ($V_t$)**: An Euler-Maruyama scheme is used for the
            mean-reverting Cox-Ingersoll-Ross (CIR) process.
            $$ dV_t = \kappa(\theta - V_t)dt + \xi\sqrt{V_t}dW^v_t $$

        The two Brownian motions, $W^s_t$ and $W^v_t$, are correlated such that
        $E[dW^s_t dW^v_t] = \rho dt$. This correlation is crucial for capturing
        the leverage effect observed in equity markets.

        **Numerical Stability**:
        To ensure the variance process $V_t$ remains non-negative, this
        implementation uses an **absorption boundary condition**. Any calculated negative
        variance is immediately clamped to zero.

        Args:
            num_paths (int): The number of independent paths to simulate.
            num_steps (int): The number of time steps for the simulation.
            T (float): The total time to maturity, in years.
            **kwargs: Placeholder for additional arguments to match a common interface.

        Returns:
            A tuple containing:
            - **paths** (torch.Tensor): A 3D tensor of shape (num_paths, num_steps + 1, 2)
              representing the simulated paths for (asset price, variance).
            - **dW_s** (torch.Tensor): A 2D tensor of shape (num_paths, num_steps) containing
              the Brownian motion increments `dW_s` used for the asset price.
        """
        dt = T / num_steps
        paths = torch.zeros(num_paths, num_steps + 1, 2)
        paths[:, 0, 0] = self.s0
        paths[:, 0, 1] = self.v0

        # Generate correlated Brownian motions
        dW_v = torch.randn(num_paths, num_steps) * torch.sqrt(torch.tensor(dt))
        dW_s_ind = torch.randn(num_paths, num_steps) * torch.sqrt(torch.tensor(dt))
        dW_s = self.rho * dW_v + torch.sqrt(torch.tensor(1.0 - self.rho ** 2)) * dW_s_ind

        for t in range(num_steps):
            S_t = paths[:, t, 0]
            v_t = paths[:, t, 1]
            v_t_clamped = torch.clamp(v_t, min=0.0)
            v_t_sqrt = torch.sqrt(v_t_clamped)

            # Use the more stable Log-Euler scheme for the price process
            S_t_updated = S_t * torch.exp((self.r - 0.5 * v_t_clamped) * dt + v_t_sqrt * dW_s[:, t])
            paths[:, t + 1, 0] = S_t_updated

            # Use the Euler scheme for the variance process
            v_t_updated = v_t + self.kappa * (self.theta - v_t_clamped) * dt + self.xi * v_t_sqrt * dW_v[:, t]
            paths[:, t + 1, 1] = torch.clamp(v_t_updated, min=0.0)

        return paths, dW_s


class BergomiModel(SDEModel):
    r"""
    Concrete implementation of the two-factor rough Bergomi (2fBS) SDE model.

    This model is a cornerstone of modern quantitative finance for its ability to
    capture the "rough" nature of volatility, as indicated by a **Hurst parameter (H)
    less than 0.5**. It models the spot variance as an exponential of a Gaussian
    Volterra process, which introduces long-range dependence and realistic term
    structures for volatility derivatives that are not captured by classic models
    like Heston.

    The simulation is based on the hybrid scheme, which provides an efficient
    and accurate numerical approximation for the core Volterra process.

    **Model Equations:**
    1.  **Stock Price Process ($S_t$)**: The asset price follows a standard geometric
        Brownian motion, but driven by the stochastic variance $V_t$.
        $$ dS_t = r S_t dt + \sqrt{V_t} S_t dB_t $$

    2.  **Variance Process ($V_t$)**: The spot variance is an exponential of the
        Volterra process $Y_t$, which gives the model its rough characteristics.
        $$ V_t = V_0 \exp(\eta Y_t - \frac{1}{2}\eta^2 t^{2H}) $$
        where the Volterra process $Y_t$ is a fractional integral of a Brownian
        motion $W^v_t$:
        $$ Y_t = \int_0^t (t-s)^{H-1/2} \, dW^v_s $$

    3.  **Correlated Noise**: The Brownian motion for the price, $B_t$, is
        correlated with the Brownian motion for the volatility, $W^v_t$:
        $$ dB_t = \rho \, dW^v_t + \sqrt{1-\rho^2} \, dW^s_t $$

    Attributes:
        s0 (float): The initial stock price.
        v0 (float): The initial forward variance.
        r (float): The risk-free interest rate.
        H (float): The Hurst parameter, must be in (0, 0.5) for rough volatility.
        eta (float): The volatility of volatility parameter.
        rho (float): The correlation between the volatility and price processes.
    """

    def __init__(self, s0: float, v0: float, r: float, H: float, eta: float, rho: float):
        """
        Initializes the BergomiModel with its core parameters.
        """
        if not (0 < H < 0.5):
            raise ValueError(f"Hurst parameter H={H} must be in (0, 0.5) for rough volatility.")
        self.s0 = s0
        self.v0 = v0
        self.r = r
        self.H = H
        self.eta = eta
        self.rho = rho

    def _get_covariance_matrix(self, alpha: float, dt: float) -> torch.Tensor:
        """
        Calculates the covariance matrix for the hybrid simulation scheme.
        This matrix defines the correlation structure for the 2D Brownian
        motion used to approximate the fractional Brownian motion increments.
        """
        cov = torch.zeros(2, 2)
        cov[0, 0] = dt
        cov[0, 1] = dt ** (alpha + 1) / (alpha + 1)
        cov[1, 1] = dt ** (2 * alpha + 1) / (2 * alpha + 1)
        cov[1, 0] = cov[0, 1]
        return cov

    def simulate_paths(self, num_paths: int, num_steps: int, T: float, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates asset and variance paths according to the rough Bergomi model.
        """
        dt = T / num_steps
        alpha = self.H - 0.5

        # --- 1. Simulate the Volterra process for volatility ---
        cov = self._get_covariance_matrix(alpha, dt)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), cov)
        dW_vol_components = dist.sample((num_paths, num_steps))

        # Hybrid scheme to construct the Volterra process Y
        Y1 = torch.zeros(num_paths, num_steps + 1);
        Y2 = torch.zeros(num_paths, num_steps + 1)
        for i in range(1, num_steps + 1):
            Y1[:, i] = Y1[:, i - 1] * torch.exp(torch.tensor(-dt, dtype=torch.float32)) + dW_vol_components[:, i - 1,
                                                                                          1] - (
                                   cov[1, 0] / cov[0, 0]) * dW_vol_components[:, i - 1, 0]
            Y2[:, i] = Y2[:, i - 1] + dW_vol_components[:, i - 1, 0]
        Y = Y1 + (cov[1, 0] / cov[0, 0]) * Y2

        # --- 2. Construct the variance process V from Y ---
        time_grid = torch.linspace(0, T, num_steps + 1, dtype=torch.float32)
        variance_factor = 0.5 * self.eta ** 2 * time_grid ** (2 * self.H)
        V = self.v0 * torch.exp(self.eta * Y - variance_factor)

        # **Crucial Stability Fix**: Clamp the variance to prevent numerical overflow.
        # The exponential function can produce `inf` values for extreme random paths,
        # which breaks the simulation. We cap the variance at a large but finite value.
        V = torch.clamp(V, min=0.0, max=5.0)

        # --- 3. Simulate the stock price process S with correlated noise ---
        sqrt_dt = torch.sqrt(torch.tensor(dt, dtype=torch.float32))
        dW_price_ind = torch.randn(num_paths, num_steps, dtype=torch.float32) * sqrt_dt
        dW_v = dW_vol_components[:, :, 0]  # This is dW^v from the math
        dB = self.rho * dW_v + torch.sqrt(torch.tensor(1.0 - self.rho ** 2, dtype=torch.float32)) * dW_price_ind

        S = torch.zeros(num_paths, num_steps + 1)
        S[:, 0] = self.s0

        # Use log-Euler scheme for stability of the price process.
        increments = (self.r - 0.5 * V[:, :-1]) * dt + torch.sqrt(torch.clamp(V[:, :-1], min=0.0)) * dB
        S[:, 1:] = self.s0 * torch.exp(torch.cumsum(increments, dim=1))

        # --- 4. Combine paths and return ---
        paths = torch.stack([S, V], dim=2)
        return paths, dB


class SDEFactory:
    """A factory class that dynamically creates the appropriate SDE model."""

    def create_model(self, **kwargs) -> SDEModel:
        H = kwargs.get('H')
        if H is None:
            raise ValueError("Hurst parameter 'H' must be provided to the SDEFactory.")

        if H >= 0.5:
            print("SDEFactory: H >= 0.5, creating HestonModel.")
            # **THE FIX**: Explicitly define the required parameters for the Heston model.
            heston_keys = ['s0', 'v0', 'kappa', 'theta', 'xi', 'rho', 'r']
            # Filter the kwargs to only pass the necessary parameters.
            heston_params = {key: kwargs[key] for key in heston_keys if key in kwargs}
            return HestonModel(**heston_params)

        elif 0 < H < 0.5:
            print(f"SDEFactory: H={H}<0.5, creating BergomiModel.")
            # **THE FIX**: Explicitly define the required parameters for the Bergomi model.
            bergomi_keys = ['s0', 'v0', 'r', 'H', 'eta', 'rho']
            # Filter the kwargs to only pass the necessary parameters.
            bergomi_params = {key: kwargs[key] for key in bergomi_keys if key in kwargs}
            return BergomiModel(**bergomi_params)

        else:
            raise ValueError(f"Hurst parameter H={H} is not supported.")