import math
from collections import deque
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from abc import ABC, abstractmethod
from scipy.stats import kstest, stats
from torch.quasirandom import SobolEngine


# An abstract base class defining the common interface for SDE models
class SDEModel(ABC):
    """Abstract base class for SDE models."""

    @abstractmethod
    def simulate_paths(self, num_paths: int, num_steps: int, T: float, Z: Optional[Tuple[torch.Tensor, ...]] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates asset paths.

        Args:
            num_paths (int): The number of paths to simulate.
            num_steps (int): The number of time steps.
            T (float): The total time to maturity.
            **kwargs: Additional model-specific parameters.

        Returns:
            A tuple containing:
            - **paths** (torch.Tensor): A 3D tensor of shape (num_paths, num_steps + 1, 2)
              representing the simulated paths for (asset price, variance).
            - **dW_s** (torch.Tensor): A 2D tensor of shape (num_paths, num_steps) containing
              the Brownian motion increments `dW_s` used for the asset price.
        """
        pass

    def simulate_paths_qmc_antithetic(
            self,
            num_paths: int,
            num_steps: int,
            T: float,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates paths using Quasi-Monte Carlo (Sobol sequence) combined with
        the antithetic variates variance reduction technique.

        Args:
            num_paths (int): Total paths to generate (should be an even number).
            num_steps (int): The number of time steps for each path.
            T (float): Total time to maturity.

        Returns:
            A tuple containing the combined tensors of all simulated paths and dWs.
        """
        if num_paths % 2 != 0:
            raise ValueError("num_paths must be an even number for antithetic variates.")

        half_paths = num_paths // 2

        # 1. Determine the dimensionality needed for the random numbers.
        # For Heston, we need two sets of random numbers per time step.
        sobol_dim = num_steps * 2

        # 2. Initialize the SobolEngine to generate quasi-random numbers in [0, 1].
        sobol_engine = SobolEngine(dimension=sobol_dim, scramble=True)

        # 3. Draw half the paths from the Sobol sequence.
        # Shape: (half_paths, num_steps * 2)
        uniform_draws = sobol_engine.draw(half_paths)

        epsilon = 1e-7
        uniform_draws = torch.clamp(uniform_draws, min=epsilon, max=1.0 - epsilon)

        # 4. Transform uniform draws to standard normal and create antithetic partners.
        # We use the inverse CDF (i-CDF) for the transformation.
        # The antithetic partner for a uniform draw `u` is `1 - u`.
        normal_draws = torch.distributions.Normal(0, 1).icdf(uniform_draws)
        antithetic_normal_draws = torch.distributions.Normal(0, 1).icdf(1 - uniform_draws)

        # 5. Reshape the draws into the (Z1, Z2) tuple format expected by Heston.
        # Shape of each Z: (half_paths, num_steps)
        Z_base = (normal_draws[:, :num_steps], normal_draws[:, num_steps:])
        Z_antithetic = (antithetic_normal_draws[:, :num_steps], antithetic_normal_draws[:, num_steps:])

        # 6. Simulate both sets of paths.
        paths1, dW1 = self.simulate_paths(half_paths, num_steps, T, Z=Z_base)
        paths2, dW2 = self.simulate_paths(half_paths, num_steps, T, Z=Z_antithetic)

        # 7. Concatenate and return the final, variance-reduced set of paths.
        return torch.cat([paths1, paths2], dim=0), torch.cat([dW1, dW2], dim=0)

    def find_best_adaptive_simulation(
            self,
            num_steps: int,
            T: float,
            num_runs: int = 3,
            adaptive_params: Dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the adaptive RQMC simulation multiple times and returns the single
        best set of paths that resulted in the minimum confidence interval width.

        This "best-of-N" approach is a robust way to ensure the highest quality
        paths are used for the final pricing, accounting for the stochastic nature
        of the RQMC simulation.

        Args:
            sde_model (SDEModel): The SDE model (e.g., Bergomi) to simulate with.
            num_steps (int): The number of time steps for the simulation.
            T (float): The time to maturity.
            num_runs (int): The number of independent adaptive simulations to run.
            adaptive_params (Dict): A dictionary of parameters to pass to the
                                    `simulate_paths_adaptive_rqmc` function.

        Returns:
            The single best tuple of (paths, dWs, martingale_values) found.
        """
        print(f"--- Starting Best-of-{num_runs} Adaptive Simulation Search ---")

        best_ci_width = float('inf')
        best_run_results = None

        # --- 1. Loop for Multiple Independent Runs ---
        i = 0
        while i < num_runs:
            print(f"\n--- Starting Ensemble Run {i + 1}/{num_runs} ---")

            # --- a. Run the full adaptive RQMC simulation once ---
            # The adaptive function will run its own internal loop until it converges.
            current_paths, current_dWs, current_martingales = self.simulate_paths_adaptive_rqmc(
                num_steps=num_steps,
                T=T,
                **(adaptive_params or {})
            )

            # --- b. Calculate the final confidence interval for this run ---
            final_S_T = current_paths[:, -1, 0]
            final_std_dev = torch.std(final_S_T)
            final_std_err = final_std_dev / (final_S_T.shape[0] ** 0.5)
            final_ci_width = 1.96 * final_std_err * 2

            print(f"--- Run {i + 1} Complete. Final CI Width: {final_ci_width:.6f} ---")

            # --- c. Check if this run is the best so far ---
            if final_ci_width < best_ci_width:
                print(
                    f"*** New best result found! CI Width improved from {best_ci_width:.6f} to {final_ci_width:.6f} ***")
                best_ci_width = final_ci_width
                best_run_results = (current_paths, current_dWs, current_martingales)

            i += 1

        # --- 2. Return the Single Best Result ---
        if best_run_results is None:
            raise RuntimeError("All simulation runs failed to produce a valid result.")

        print(f"\n--- Best-of-{num_runs} Search Complete ---")
        print(f"Returning best path set with a final CI width of {best_ci_width:.6f}")

        return best_run_results

    def simulate_paths_adaptive_rqmc(
            self,
            num_steps: int,
            T: float,
            improvement_tolerance: float = 0.01,  # e.g., require at least a 1% improvement
            patience: int = 10,  # Stop after 5 batches with no significant improvement
            initial_batch_size: int = 1024 * 5,  # Powers of 2 are good for Sobol
            step_batch_size: int = 1024 * 2,
            max_paths: int = 1024 * 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively simulates paths using Randomized Quasi-Monte Carlo (RQMC) and
        stops when the convergence of the mean stagnates.

        This version includes a "revert to best" logic, returning the set of paths
        that produced the minimum confidence interval, ensuring the most stable
        result is returned even if the simulation continues due to the patience window.

        Args:
            num_steps (int): The number of time steps for each path simulation.
            T (float): The total time to maturity.
            improvement_tolerance (float): The minimum relative improvement in the
                confidence interval width required to reset the patience counter.
            patience (int): The number of consecutive batches with no significant
                improvement before stopping the simulation.
            initial_batch_size (int): The number of paths in the first batch (should be a power of 2).
            step_batch_size (int): The number of paths to add in each subsequent iteration (should be a power of 2).
            max_paths (int): The maximum total number of paths to simulate.

        Returns:
            A tuple containing the combined tensors of the best-found simulated paths,
            their corresponding dWs, and their martingale values.
        """
        all_paths: List[torch.Tensor] = []
        all_dWs: List[torch.Tensor] = []
        total_paths = 0

        # --- Variables for Stagnation and "Revert to Best" Logic ---
        best_ci_width = float('inf')
        patience_counter = 0

        # Variables to store the state of the simulation that produced the best result
        best_paths_so_far = []
        best_dWs_so_far = []
        best_martingales_so_far = []

        # --- Initialize the Sobol Engine ---
        sobol_dim = num_steps * 2
        sobol_engine = SobolEngine(dimension=sobol_dim, scramble=True)
        normal_dist = torch.distributions.Normal(0, 1)

        # --- Adaptive Loop ---
        batch_size = initial_batch_size
        while total_paths < max_paths:
            # --- a. Generate one RQMC + Antithetic Batch ---
            half_batch = batch_size // 2
            uniform_draws = sobol_engine.draw(half_batch)

            # Apply a single random shift to the entire batch (the "R" in RQMC)
            random_shift = torch.rand(sobol_dim)
            shifted_draws = (uniform_draws + random_shift) % 1.0

            # Clamp to prevent infinities from the inverse CDF
            epsilon = 1e-7
            shifted_draws_clamped = torch.clamp(shifted_draws, min=epsilon, max=1.0 - epsilon)
            antithetic_draws_clamped = 1.0 - shifted_draws_clamped

            # Convert both sets to standard normal
            normal_draws = normal_dist.icdf(shifted_draws_clamped)
            antithetic_normal_draws = normal_dist.icdf(antithetic_draws_clamped)

            # Reshape for the simulation function
            Z_base = (normal_draws[:, :num_steps], normal_draws[:, num_steps:])
            Z_antithetic = (antithetic_normal_draws[:, :num_steps], antithetic_normal_draws[:, num_steps:])

            # Simulate both the base and antithetic paths
            paths1, dW1 = self.simulate_paths(half_batch, num_steps, T, Z=Z_base)
            paths2, dW2 = self.simulate_paths(half_batch, num_steps, T, Z=Z_antithetic)

            # Append the new, full batch of paths to our collection for this iteration
            all_paths.append(torch.cat([paths1, paths2], dim=0))
            all_dWs.append(torch.cat([dW1, dW2], dim=0))
            total_paths += batch_size

            # --- b. Calculate Current Confidence Interval ---
            combined_paths = torch.cat(all_paths, dim=0)
            S_T = combined_paths[:, -1, 0]
            current_std_dev = torch.std(S_T)
            std_err = current_std_dev / (total_paths ** 0.5)
            current_ci_width = 1.96 * std_err * 2

            current_mean = torch.mean(S_T)

            # --- c. Check for Stagnation (with NaN fix) ---
            improvement_ratio = (best_ci_width - current_ci_width) / best_ci_width if best_ci_width != float('inf') else float('inf')

            if improvement_ratio > improvement_tolerance:
                # If there's significant improvement, we have a new "best" result.
                print(f"Paths: {total_paths:,} | CI Width improved to: {current_ci_width:.4f}. Resetting patience.")
                best_ci_width = current_ci_width
                patience_counter = 0

                # Save the current state as the best state so far
                best_paths_so_far = list(all_paths)
                best_dWs_so_far = list(all_dWs)
            else:
                # If no significant improvement, increment patience.
                patience_counter += 1
                print(f"Paths: {total_paths:,} | No significant improvement. CI Width: {current_ci_width:.4f}. Patience: {patience_counter}/{patience}.")

            # If patience has run out, we've reached the stable oscillation phase.
            if patience_counter >= patience:
                print(f"Convergence stagnated for {patience} batches. Stopping simulation.")
                break

            # Use the smaller step size for all subsequent iterations
            batch_size = step_batch_size

        if total_paths >= max_paths:
            print(f"Max paths ({max_paths:,}) reached without meeting stagnation criteria.")

        # --- d. Return the Best Result ---
        # Instead of returning the final (potentially noisier) set of paths, we
        # return the saved "best" set that produced the minimum confidence interval.
        if not best_paths_so_far:
            # Handle edge case where the loop finishes on the very first iteration
            return torch.cat(all_paths, dim=0), torch.cat(all_dWs, dim=0)

        print(f"Reverting to the best simulation state with CI Width: {best_ci_width:.4f}")
        return torch.cat(best_paths_so_far, dim=0), torch.cat(best_dWs_so_far, dim=0)

    def obtain_optimal_number_paths(
            self,
            num_steps: int,
            T: float,
            mean_relative_tolerance: float = 0.005,
            std_dev_relative_tolerance: float = 0.001,
            initial_batch_size: int = 512,  # Must be a power of 2 for Sobol
            step_batch_size: int = 512,  # Must be a power of 2
            max_paths: int = 200000
    ) -> int:
        path_lengths = []
        for attempt in range(100):
            best = self.simulate_paths_adaptive_rqmc2(num_steps, T, mean_relative_tolerance, std_dev_relative_tolerance, initial_batch_size, step_batch_size, max_paths)[0]
            path_lengths.append(best.shape[0])

        return math.ceil(np.array(path_lengths).mean() / 2) * 2

    def simulate_paths_adaptive_rqmc2(
            self,
            num_steps: int,
            T: float,
            mean_relative_tolerance: float = 0.005,
            std_dev_relative_tolerance: float = 0.001,
            initial_batch_size: int = 512,  # Must be a power of 2 for Sobol
            step_batch_size: int = 512,  # Must be a power of 2
            max_paths: int = 200000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively simulates paths using Randomized Quasi-Monte Carlo (RQMC)
        with antithetic variates until higher moments converge.

        This method generates batches of paths that are both low-discrepancy (due to QMC)
        and random (due to the shift), allowing for valid statistical convergence
        testing with dramatically faster convergence.
        """
        all_paths: List[torch.Tensor] = []
        all_dWs: List[torch.Tensor] = []
        total_paths = 0
        previous_std_dev = -1.0

        # --- 1. Initialize the Sobol Engine ---
        # The dimension is determined by the number of random draws needed per path.
        # For Heston, this is 2 (for dW_S and dW_V) * num_steps.
        sobol_dim = num_steps * 2
        sobol_engine = SobolEngine(dimension=sobol_dim, scramble=True)

        # We need a normal distribution to convert uniform draws via the inverse CDF
        normal_dist = torch.distributions.Normal(0, 1)

        # --- 2. Adaptive Loop ---
        batch_size = initial_batch_size
        while total_paths < max_paths:
            # --- a. Generate one RQMC + Antithetic Batch ---
            half_batch = batch_size // 2

            # Draw a batch of low-discrepancy points from the Sobol sequence
            uniform_draws = sobol_engine.draw(half_batch)

            # Apply a single random shift to the entire batch (the "R" in RQMC)
            random_shift = torch.rand(sobol_dim)
            shifted_draws = (uniform_draws + random_shift) % 1.0

            # Clamp to prevent infinities from the inverse CDF
            epsilon = 1e-7
            shifted_draws_clamped = torch.clamp(shifted_draws, min=epsilon, max=1.0 - epsilon)

            # Create the antithetic partners
            antithetic_draws_clamped = 1.0 - shifted_draws_clamped

            # Convert both sets to standard normal
            normal_draws = normal_dist.icdf(shifted_draws_clamped)
            antithetic_normal_draws = normal_dist.icdf(antithetic_draws_clamped)

            # Reshape into the (Z1, Z2) tuple format for the Heston model
            Z_base = (normal_draws[:, :num_steps], normal_draws[:, num_steps:])
            Z_antithetic = (antithetic_normal_draws[:, :num_steps], antithetic_normal_draws[:, num_steps:])

            # Simulate both the base and antithetic paths
            paths1, dW1 = self.simulate_paths(half_batch, num_steps, T, Z=Z_base)
            paths2, dW2 = self.simulate_paths(half_batch, num_steps, T, Z=Z_antithetic)

            # Append the new, full batch of paths to our collection
            all_paths.append(torch.cat([paths1, paths2], dim=0))
            all_dWs.append(torch.cat([dW1, dW2], dim=0))
            total_paths += batch_size

            # --- b. Check for Convergence (same logic as before) ---
            combined_paths = torch.cat(all_paths, dim=0)
            S_T = combined_paths[:, -1, 0]
            current_mean = torch.mean(S_T)
            current_std_dev = torch.std(S_T)

            # Check mean convergence
            mean_ci_width = 1.96 * (current_std_dev / (total_paths ** 0.5)) * 2
            mean_converged = mean_ci_width < (current_mean * mean_relative_tolerance)

            # Check standard deviation convergence
            std_dev_converged = False
            if previous_std_dev > 0:
                std_dev_relative_change = abs(current_std_dev - previous_std_dev) / previous_std_dev
                std_dev_converged = std_dev_relative_change < std_dev_relative_tolerance

            if mean_converged and std_dev_converged:
                print(f"RQMC Convergence achieved at {total_paths:,} paths.")
                break

            print(
                f"Paths: {total_paths:,} | Mean Converged: {mean_converged} | Std Dev Converged: {std_dev_converged}")
            previous_std_dev = current_std_dev
            batch_size = step_batch_size  # Use the smaller step size for subsequent iterations

        if total_paths >= max_paths:
            print(f"Max paths ({max_paths:,}) reached without meeting full convergence criteria.")

        return torch.cat(all_paths, dim=0), torch.cat(all_dWs, dim=0)

    def find_convergence_steps(
            self,
            T: float,
            num_paths: int = 80_000,
            initial_steps: int = 100,
            step_increment: int = 100,
            max_tests: int = 300,
            tolerance: float = 0.0021,
            stability_window: int = 5,
            trim_percentile: float = 0.5  # New parameter: % to trim from each tail
    ) -> int:
        """
        Finds the minimum number of steps for the path distribution to converge
        using a robust, trimmed Kolmogorov-Smirnov test.

        This method addresses the slow convergence of higher moments (skew, kurtosis)
        by trimming the extreme tails of the distributions before comparing them.
        This focuses the test on the stable, central part of the distribution.

        Args:
            T (float): The total time to maturity.
            num_paths (int): The number of paths to use for the test.
            initial_steps (int): The starting number of steps.
            step_increment (int): The amount to increase steps by for each test.
            max_tests (int): The number of different resolutions to test.
            tolerance (float): The tolerance for the KS statistic.
            stability_window (int): The number of previous distributions to compare against.
            trim_percentile (float): The percentage of data to trim from each tail
                                     (e.g., 0.5 means trim 0.5% from the top and 0.5%
                                     from the bottom). Set to 0 to disable.

        Returns:
            int: The recommended number of steps for convergence.
        """
        # --- 1. Validate Input Parameters ---
        if max_tests <= stability_window:
            raise ValueError("max_tests must be greater than stability_window.")
        if not (0 <= trim_percentile < 50):
            raise ValueError("trim_percentile must be between 0 and 50.")

        steps_to_test = [initial_steps + i * step_increment for i in range(max_tests)]
        max_steps = steps_to_test[-1]

        print(f"--- Starting Trimmed KS Test (k={stability_window}, trim={trim_percentile}%) for T={T:.2f} ---")
        print(f"Testing step counts: {steps_to_test}")

        # --- 2. Initialize Rolling Window ---
        distribution_history = deque(maxlen=stability_window)

        # --- 3. Main Forward-Iterating Loop ---
        for i, num_steps in enumerate(steps_to_test):
            print(f"\nGenerating distribution for {num_steps} steps...")

            # --- a. Simulate paths and get the path statistic ---
            paths, _ = self.simulate_paths_qmc_antithetic(num_paths, num_steps, T)
            dt = T / num_steps
            variance_paths = paths[:, :, 1]
            path_statistic = torch.sum(variance_paths[:, 1:], dim=1) * dt
            current_distribution = path_statistic.cpu().numpy()

            # --- c. Check for Stability (if window is full) ---
            if i >= stability_window:
                is_stable = True
                print(f"  -> Comparing current distribution against the last {stability_window} distributions...")

                for historical_distribution in distribution_history:
                    # --- NEW: Trim both distributions to remove unstable tails ---
                    low_bound = trim_percentile
                    high_bound = 100 - trim_percentile

                    # Find the percentile values for the current distribution
                    p_low_curr, p_high_curr = np.percentile(current_distribution, [low_bound, high_bound])
                    # Find the percentile values for the historical distribution
                    p_low_hist, p_high_hist = np.percentile(historical_distribution, [low_bound, high_bound])

                    # Trim the arrays based on their respective percentile bounds
                    trimmed_current = current_distribution[
                        (current_distribution >= p_low_curr) & (current_distribution <= p_high_curr)]
                    trimmed_historical = historical_distribution[
                        (historical_distribution >= p_low_hist) & (historical_distribution <= p_high_hist)]

                    # --- Perform the KS test on the trimmed, robust data ---
                    ks_statistic, _ = kstest(trimmed_current, trimmed_historical)
                    print(f"     - Trimmed KS Stat: {ks_statistic:.4f} (Tolerance: {tolerance})")

                    if ks_statistic > tolerance:
                        is_stable = False
                        break

                if is_stable:
                    stable_step_count = steps_to_test[i - stability_window]
                    print(f"\n--- STABLE CONVERGENCE ACHIEVED at {stable_step_count} steps. ---")
                    return stable_step_count

            # --- d. Update History ---
            # We still store the full, untrimmed distribution in the history
            distribution_history.append(current_distribution)

        # --- 4. Handle Failure to Converge ---
        raise RuntimeError(
            f"Convergence failed. The path distribution did not stabilize within the "
            f"tested range up to {max_steps} steps."
        )


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
        self.H = 0.5

    def simulate_paths(self, num_paths: int, num_steps: int, T: float, Z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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
        S = torch.zeros(num_paths, num_steps + 1)
        V = torch.zeros(num_paths, num_steps + 1)
        S[:, 0] = self.s0
        V[:, 0] = self.v0

        if Z is None:
            # Generate random numbers if they are not provided
            Z1 = torch.randn(num_paths, num_steps)
            Z2 = torch.randn(num_paths, num_steps)
        else:
            # Use the provided random numbers
            Z1, Z2 = Z

        dW_S = Z1 * (dt ** 0.5)
        dW_V = (self.rho * Z1 + (1 - self.rho ** 2) ** 0.5 * Z2) * (dt ** 0.5)

        # Full Truncation Scheme for numerical stability
        for t in range(num_steps):
            V_t_positive = torch.clamp(V[:, t], min=0.0)
            S[:, t + 1] = S[:, t] * torch.exp((self.r - 0.5 * V_t_positive) * dt + (V_t_positive ** 0.5) * dW_S[:, t])
            V[:, t + 1] = V[:, t] + self.kappa * (self.theta - V_t_positive) * dt + self.xi * (
                        V_t_positive ** 0.5) * dW_V[:, t]

        paths = torch.stack([S, V], dim=2)
        dW = torch.stack([dW_S, dW_V], dim=2)
        return paths, dW_S


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

        self.alpha = H - 0.5

    def _get_fbm_covariance(self, num_steps: int, T: float) -> torch.Tensor:
        """
        Calculates the covariance matrix for the increments of a fractional
        Brownian motion (fBm) with Hurst parameter H in a highly efficient,
        vectorized manner.

        Args:
            num_steps (int): The number of time steps (increments).
            T (float): The total time to maturity.

        Returns:
            torch.Tensor: The (num_steps x num_steps) covariance matrix.
        """
        dt = T / num_steps
        H2 = 2 * self.H

        # --- 1. Create a grid of integer indices ---
        # The formula depends on the integer indices i and j, not the time points.
        indices = torch.arange(1, num_steps + 1)
        i, j = torch.meshgrid(indices, indices, indexing='ij')

        # --- 2. Apply the correct covariance formula for increments ---
        # The formula is E[ (W_i - W_{i-1}) * (W_j - W_{j-1}) ]
        # which simplifies to the expression below based on integer indices.
        cov_matrix = 0.5 * (
                torch.abs(i - j + 1).pow(H2) +
                torch.abs(i - j - 1).pow(H2) -
                2 * torch.abs(i - j).pow(H2)
        )

        # --- 3. Scale the result by dt^(2H) ---
        # The scaling factor is applied at the end.
        cov_matrix *= (dt ** H2)

        return cov_matrix

    def simulate_paths(self, num_paths: int, num_steps: int, T: float, Z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates asset and variance paths according to the rough Bergomi model.
        """
        dt = T / num_steps
        t_values = torch.linspace(0, T, num_steps + 1)

        # --- 1. Generate Fractional Brownian Motion Increments ---
        fbm_cov = self._get_fbm_covariance(num_steps, T)
        cholesky_transform = torch.linalg.cholesky(fbm_cov)

        # --- 2. Probe Simulation for Adaptive Variance Clamp ---
        probe_paths = int(num_paths * 0.1)
        probe_Z1 = torch.randn(probe_paths, num_steps)
        probe_dW_H = (cholesky_transform @ probe_Z1.T).T
        Y_probe = torch.cumsum(probe_dW_H, dim=1)
        probe_V_unclamped = self.v0 * torch.exp(
            self.eta * Y_probe - 0.5 * self.eta ** 2 * (t_values[1:] ** (2 * self.H)))

        valid_variances = probe_V_unclamped[torch.isfinite(probe_V_unclamped)]
        # v_max = torch.quantile(valid_variances, 0.999) if valid_variances.numel() > 0 else torch.tensor(2.0)
        v_max = torch.max(valid_variances)

        # --- 3. Main Simulation Setup ---
        if Z is None:
            Z1 = torch.randn(num_paths, num_steps)
            Z2 = torch.randn(num_paths, num_steps)
        else:
            Z1, Z2 = Z

        dW_H = (cholesky_transform @ Z1.T).T
        dW_perp = Z2 * (dt ** 0.5)
        dB = self.rho * (Z1 * (dt ** 0.5)) + torch.sqrt(torch.tensor(1 - self.rho ** 2)) * dW_perp

        # --- 4. Vectorized Variance Path Generation ---
        V = torch.zeros(num_paths, num_steps + 1)
        V[:, 0] = self.v0
        Y_main = torch.cumsum(dW_H, dim=1)
        V_unclamped = self.v0 * torch.exp(self.eta * Y_main - 0.5 * self.eta ** 2 * (t_values[1:] ** (2 * self.H)))
        # v_max = torch.quantile(V_unclamped, 0.99)

        V[:, 1:] = torch.clamp(V_unclamped, min=0.0)  # Clamp is still a robust safety measure

        # --- 5. Iterative Stock Price Simulation (The Stable Scheme) ---
        S = torch.zeros(num_paths, num_steps + 1)
        S[:, 0] = self.s0
        log_S = torch.zeros_like(S)
        log_S[:, 0] = torch.log(torch.tensor(self.s0))

        # This loop provides the numerical stability for the S path
        for t in range(num_steps):
            sqrt_V_t = torch.sqrt(V[:, t])

            log_S_increment = (self.r - 0.5 * V[:, t]) * dt + sqrt_V_t * dB[:, t]
            log_S[:, t + 1] = log_S[:, t] + log_S_increment

        S = torch.exp(log_S)

        # --- 6. Finalize and Return ---
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