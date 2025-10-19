import joblib
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List

from joblib import delayed, Parallel
from scipy import stats

from ..features.signature_calculator import calculate_signatures, \
    parallel_calculate_all_signatures_batched
from ..heuristics import get_num_steps
from ..models.batch_path_gen import generate_paths_in_batches, generate_paths_in_batches_individual
from ..models.sde import SDEModel, HestonModel, BergomiModel
from ..solvers.base_solver import AbstractPrimalSolver, AbstractDualSolver
from ..utils import payoff_factory
from ..workflows.european_pricer import EuropeanPricerBase, HestonEuropeanPricer, BergomiEuropeanPricer


class PricingEngine:
    """
    Orchestrates the execution of modular primal and dual solvers to price an American option.

    This class encapsulates the end-to-end workflow for a single pricing run.
    It is initialized with a specific SDE model and concrete solver implementations
    (e.g., LinearPrimalSolver, LinearDualSolver). Its main `run` method simulates
    the required paths and then invokes the primal (lower bound) and dual (upper bound)
    solvers to compute the final price interval and duality gap.

    This modular design, based on the Strategy Pattern, allows for easy comparison
    of different solver methodologies (linear, kernel, deep learning) without
    changing the core engine logic.
    """

    def __init__(
            self,
            sde_model: SDEModel,
            primal_solver: AbstractPrimalSolver,
            dual_solver: AbstractDualSolver,
            option_type: str,
            strike: float,
            r: float,
            device: Optional[str] = None
    ):
        """
        Initializes the pricing engine with modular solver components.

        Args:
            sde_model (SDEModel): An instantiated SDE model (e.g., HestonModel) that
                                  can simulate paths and Brownian increments.
            primal_solver (AbstractPrimalSolver): A concrete implementation of a primal solver.
            dual_solver (AbstractDualSolver): A concrete implementation of a dual solver.
            option_type (str): The type of option to price, either 'put' or 'call'.
            strike (float): The strike price of the option.
            device (Optional[str]): The PyTorch device to use ('cuda', 'cpu', or 'mps').
                                    If None, it's auto-detected.
        """
        self.model = sde_model
        self.primal_solver = primal_solver
        self.dual_solver = dual_solver
        self.payoff_fn = payoff_factory(option_type, strike)

        self.option_type = option_type
        self.strike = strike
        self.r = r

        self.european_pricer = self._get_european_pricer()

        # Auto-detect device if not explicitly provided
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.backends.mps.is_available(): self.device = 'mps'
        else:
            self.device = device

        print("Primal-Dual Engine Initialized.")
        print(f"  -> Using device: {self.device}")

    def _get_european_pricer(self) -> Optional[EuropeanPricerBase]:
        """Selects the appropriate European pricer based on the SDE model type."""
        if isinstance(self.model, HestonModel):
            return HestonEuropeanPricer()
        elif isinstance(self.model, BergomiModel):
            return BergomiEuropeanPricer()
        else:
            print("No specific European pricer found for this SDE model. Control variates will be disabled.")
            return None

    def _get_random_subsample_indices(
            self,
            num_paths: int,
            subsample_size: Optional[int],
            device: torch.device
    ) -> torch.Tensor:
        """
        Selects a random subsample of indices.

        Args:
            num_paths (int): The total number of available samples.
            subsample_size (Optional[int]): The desired number of samples.
            device (torch.device): The device to create the index tensor on.

        Returns:
            A 1D tensor of randomly selected indices.
        """
        # If no subsampling is needed, return all indices
        if not subsample_size or subsample_size >= num_paths:
            return torch.arange(num_paths, device=device)

        # Generate a random permutation of all indices and take the first `subsample_size`
        permuted_indices = torch.randperm(num_paths, device=device)
        return permuted_indices[:subsample_size]

    def precompute(
            self,
            paths: torch.Tensor,
            dW: torch.Tensor,
            T: float,
            subsample_size: int = 30_000,
            lookback: int = 500,
            truncation_level: int = 2
    ) -> dict:
        """
        Precomputes shared variables using a simple random subsample.
        """
        device, num_steps, num_paths = paths.device, paths.shape[1], paths.shape[0]

        # --- 1. Subsample the data first using random sampling ---
        # MODIFICATION START
        subsample_indices = self._get_random_subsample_indices(
            num_paths, subsample_size, device
        )
        # MODIFICATION END

        # Slice all tensors down to the fixed subset for all future calculations
        paths_subsampled = paths[subsample_indices]
        asset_paths_subsampled = paths_subsampled[:, :, 0]  # Derived from the subsampled paths

        # --- 2. Precompute variables for the subsampled paths ---

        # Payoff matrix for all time steps
        payoffs_subsampled = torch.stack(
            [self.payoff_fn(asset_paths_subsampled[:, t]) for t in range(num_steps)],
            dim=1
        )

        # Signatures (using your parallel function)
        from ..features.signature_calculator import calculate_signatures
        signatures_subsampled = parallel_calculate_all_signatures_batched(
            paths_subsampled, lookback, truncation_level
        )

        # Discount factors
        dt = np.float32(T / (num_steps - 1))
        discount_factors = torch.exp(-self.r * torch.arange(num_steps, device=device) * dt)

        # --- 3. Return everything in a dictionary ---
        return {
            "paths": paths_subsampled,
            "asset_paths": asset_paths_subsampled,
            "payoffs": payoffs_subsampled,
            "signatures": signatures_subsampled,
            "discount_factors": discount_factors,
            "dW": dW,
            "T": T,
            "dt": dt
        }

    def run(self,
            T: float,
            primal_uncertainty: float,
            max_num_paths: int = 100_000,
            max_num_steps: int = 100_000,
            max_num_simulation_steps: int = 10_000,
            ) -> Dict[str, float]:
        """
        Runs a full pricing simulation to calculate the primal and dual bounds.

        Args:
            T (float): The time to maturity of the option.
            max_num_paths (int): The number of Monte Carlo paths to simulate.
            max_num_steps (int): The number of time steps in each simulated path.

        Returns:
            A dictionary containing the calculated 'lower_bound', 'upper_bound',
            and the resulting 'duality_gap'.
        """
        X_0_2 = 10_000  # N(0.2)
        X_0_4 = 7200  # N(0.4)
        X_0_5 = 1800  # N(0.5)

        num_steps = get_num_steps(self.model.H, max_num_steps, X_0_2, X_0_4, X_0_5)
        num_simulations = joblib.cpu_count() * max_num_simulation_steps
        simulation_batch_size = joblib.cpu_count()

        # num_steps = self.model.find_convergence_steps(T, 40_000)
        num_paths_per_sim = min(self.model.obtain_optimal_number_paths(150, T), max_num_paths)

        all_lower_bounds: List[float] = []

        # --- Variables to track the single best simulation ---
        best_simulation_data = None
        min_distance_to_mean = float('inf')
        best_run_index = -1

        # Calculate how many main batches are needed to complete all simulations
        num_main_batches = (num_simulations + simulation_batch_size - 1) // simulation_batch_size

        for i in range(num_main_batches):
            # Determine how many simulations to run in this specific batch
            sims_in_this_batch = min(simulation_batch_size, num_simulations - len(all_lower_bounds))
            if sims_in_this_batch <= 0:
                continue

            print(f"\n--- Starting Batch {i + 1}/{num_main_batches} with {sims_in_this_batch} simulations ---")

            # --- 1. Generate a batch of path sets in parallel (Corrected) ---
            start_index_for_batch = len(all_lower_bounds)

            simulations_batch = generate_paths_in_batches_individual(
                sde=self.model,
                num_paths=(num_paths_per_sim),
                num_steps=num_steps,
                T=T,
                num_batches=sims_in_this_batch
            )

            # --- 2. Precompute variables for each simulation in the batch (Now also parallelized) ---
            print("Precomputing variables for the batch...")
            precompute_tasks = [
                delayed(self.precompute)(sim[0], sim[1], T) for sim in simulations_batch
            ]
            precomputed_vars_batch = Parallel(n_jobs=-1)(precompute_tasks)

            # --- 3. Execute the Primal Solver in parallel for the entire batch ---
            print("Executing Primal Solver in parallel for the batch...")
            primal_tasks = [
                delayed(self.primal_solver.solve)(
                    precomputed_vars, self.option_type, self.strike, self.payoff_fn
                ) for precomputed_vars in precomputed_vars_batch
            ]
            lower_bounds_batch = Parallel(n_jobs=-1)(primal_tasks)

            # --- 4. Aggregate results and find the new best simulation ---
            all_lower_bounds.extend(lower_bounds_batch)
            current_lower_bound = np.mean(all_lower_bounds)

            # Check if any simulation in this new batch is a better "best so far"
            for j, primal_value in enumerate(lower_bounds_batch):
                distance = abs(primal_value - current_lower_bound)
                if distance < min_distance_to_mean:
                    min_distance_to_mean = distance
                    current_best_index = start_index_for_batch + j
                    # Store the path data for this new best simulation, overwriting the old one
                    best_simulation_data = {
                        "primal_value": primal_value,
                        "precomputed_vars": precomputed_vars_batch[j]  # This now includes the signatures
                    }
                    best_run_index = current_best_index
                    print(
                        f"  -> New best simulation found: #{best_run_index + 1} (Primal: {primal_value:.4f}, Distance: {distance:.6f})")

            # --- Live Updates ---
            current_ci = self.calculate_confidence_interval(all_lower_bounds)
            print(f"\n--- Batch {i + 1} Complete ---")
            print(f"Total simulations processed: {len(all_lower_bounds)}/{num_simulations}")
            print(f"Current Lower Bound (Average): {current_lower_bound:.6f}")
            print(f"Current 95% Confidence Interval: {current_ci}")

            if current_lower_bound - current_ci[0] < primal_uncertainty:
                break

        # --- 5. Final Lower Bound Calculation ---
        print(all_lower_bounds)
        final_lower_bound = np.mean(all_lower_bounds)
        final_lower_ci = self.calculate_confidence_interval(all_lower_bounds)
        print("\n--- All Primal Runs Complete ---")
        print(f"Final Primal Price (Lower Bound): {final_lower_bound:.6f}")
        print(f"Final 95% Confidence Interval: {final_lower_ci}")

        # --- 6. Execute Dual Solver using the stored best simulation ---
        print(f"\nExecuting Dual Solver using stored best simulation (#{best_run_index + 1})...")

        paths = best_simulation_data["precomputed_vars"]["paths"]
        paths = paths.to(self.device, dtype=torch.float32)
        best_simulation_data["precomputed_vars"]["paths"] = paths

        upper_bound_single_run = self.dual_solver.solve(
            precomputed_vars=best_simulation_data["precomputed_vars"],
            payoff_fn=self.payoff_fn
        )

        # Calculate the stable duality gap from this single representative run
        primal_for_gap_calc = best_simulation_data["primal_value"]
        duality_gap_estimate = upper_bound_single_run - primal_for_gap_calc

        # --- 7. Calculate Final Hybrid Results ---
        final_upper_bound = final_lower_bound + duality_gap_estimate

        print(f"  -> Single Dual Price (from best run): {upper_bound_single_run:.6f}")
        print(f"  -> Stable Duality Gap Estimate: {duality_gap_estimate:.6f}")
        print(f"  -> Final Hybrid Upper Bound: {final_upper_bound:.6f}")

        return {
            "lower_bound": final_lower_bound,
            "upper_bound": final_upper_bound,
            "duality_gap": duality_gap_estimate,
            "lower_bound_ci": final_lower_ci
        }

    def calculate_confidence_interval(self, data, confidence=0.95):
        """
        Calculates the confidence interval for the mean of a sample.

        This function uses the t-distribution, which is suitable for cases where
        the population standard deviation is unknown.

        Parameters:
        ----------
        data : array-like
            The sample data. It should be a list, numpy array, or pandas Series
            of numerical data. Must contain at least two data points.
        confidence : float, optional
            The desired confidence level (e.g., 0.95 for 95%).
            The default is 0.95.

        Returns:
        -------
        tuple
            A tuple containing the (lower_bound, upper_bound) of the confidence interval.
            Returns (nan, nan) if the input data has fewer than two values.
        """
        # Convert data to a numpy array to ensure it's iterable and numerical
        data_array = np.asarray(data)
        n = len(data_array)

        # A confidence interval requires at least 2 data points
        if n < 2:
            return (np.nan, np.nan)

        # Calculate sample mean and standard error of the mean (SEM)
        mean = np.mean(data_array)
        sem = stats.sem(data_array)

        # Calculate the confidence interval using the t-distribution
        # The stats.t.interval function handles the critical value calculation
        interval = stats.t.interval(
            confidence=confidence,
            df=n - 1,
            loc=mean,
            scale=sem
        )

        return interval

    def _calculate_clt_interval(
            self,
            mean_price: float,
            std_dev: float,
            num_paths: int
    ) -> Tuple[float, float]:
        """Calculates the 95% confidence interval using the CLT."""
        standard_error = std_dev / (num_paths ** 0.5)
        half_width = 1.96 * standard_error

        return (mean_price - half_width, mean_price + half_width)